"""Teacher labeling: one CV page image -> one CVRecord (schema.py) via a teacher VLM.

In distillation terms the large model is the teacher and the small fine-tuned models are students.
Default teacher: ``glm-5.3-flash`` through any OpenAI-compatible chat endpoint. One request per page.

Transport findings (verified on the endpoint used for the labels):
  - The image part MUST come BEFORE the text part in the content array, or the proxy
    drops the image.
  - Calls occasionally come back empty, refused ("no image"), or all-empty: retried.
  - Reasoning tokens count against ``max_tokens`` (GLM-5.3-Flash used ~2.3k of ~3.7k
    on one page), so the limit is generous.

A reply that parses but fails schema validation is sent back in the same conversation
with the validation errors, so the teacher fixes its own answer.

Output: one JSONL line per page, appended:
  {cv_id, page, image, status, record, attempts: [...], model, ts, latency_s}
``attempts`` keeps every raw reply and error for audit. A rerun skips pages already
``ok`` in the output file; readers take the last line per (cv_id, page).

Settings come from the environment, else from .env at the repository root: LABEL_BASE_URL (the endpoint,
e.g. https://host/v1), LABEL_API_KEY, and optionally LABEL_MODEL.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from pydantic import ValidationError

from .data import DATA, ROOT
from .schema import entry_templates, template, to_record


def _load_env(path: Path) -> None:
    """Read KEY=VALUE lines from the repository's .env; variables already in the environment win."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env(ROOT / ".env")
DOCUMENTS_PATH = DATA / "processed" / "documents.jsonl"

DEFAULT_BASE_URL = os.environ.get("LABEL_BASE_URL")
DEFAULT_MODEL = os.environ.get("LABEL_MODEL", "glm-5.3-flash")

_REFUSAL_MARKERS = (
    "no image",
    "cannot see",
    "forgot to attach",
    "forgot to include",
    "please provide",
    "please upload",
    "upload the image",
    "i don't see an image",
    "didn't receive",
)

PROMPT = f"""You are labeling one page of a CV (résumé) for an applicant tracking system.
Return ONLY one JSON object, no prose, no code fences. It has exactly the keys of this template, every key present:
{json.dumps(template(), ensure_ascii=False)}
Each list holds entries of this shape:
{json.dumps(entry_templates(), ensure_ascii=False)}

Rules:
1. Copy values in the page's language. Do not translate, summarise, reorder or reword. Fix spelling and grammar mistakes in ordinary words and sentences: a missing, extra or swapped letter; a wrong or missing accent or cedilla; a wrong gender or plural agreement; a wrong verb form. Fix only when the intended text is certain. Change only the letters or ending of the wrong word; never add, remove or replace any other word. Fix an agreement only when the word it agrees with is certain; when it could agree with more than one word of the phrase, keep it as written. Spacing and punctuation are copied as written; they are not fixes. Correct spellings stay as written: capital letters without accents ("Ecole", "Etudiant") and spellings French also accepts without a circumflex are correct. Agreement that depends on the writer's own gender stays as written. Never change these; copy them letter by letter with their capitalisation, even when they look misspelled and even when the same name is written differently elsewhere on the page: names of people, companies, brands, schools, products and places (inside a name, fix only a misspelled ordinary word such as Université or Faculté); abbreviations, acronyms and shortened words or names ("Univ.", "Ing.", "M2", "ESSEC"); technology, software and programming language names; email addresses, URLs, usernames and phone numbers; codes, IDs, certificate and licence numbers, version numbers and barcodes. List every fix in corrections (rule 15). Only dates, "type" fields and ratings are converted (rules 5, 6, 11).
2. Use only what is written on this page. Never infer a value that is not written: no city or country that is not written, no age computed from a birth date. A missing value is null, a missing list is []. Never write "", "N/A" or "-".
3. This may be one page of a multi-page CV. Extract only what is on this page. If a section continues from the previous page without its heading, still put its entries in the matching list.
4. Text broken across lines is one piece of text: a word hyphenated at a line end ("mo-" + "lestie") is written whole ("molestie"); a URL or email split over two lines is joined with no space.
5. Dates: YYYY, YYYY-MM or YYYY-MM-DD, as precise as the page states. Examples: "Juillet 2022" -> "2022-07"; "12/10/1999" -> "1999-10-12" (day/month/year); "Été 2021" -> "2021". Ongoing ("présent", "à ce jour", "en cours", "aujourd'hui", "current", "now") -> endDate "present". If an entry has only one date, put it in endDate.
6. type fields are null unless the entry or its own heading states the type in words. A general heading such as "Expérience professionnelle", "Experiences pratique", "Work history" or "Projets" states no type.
   work.type: "internship" when it says stage, stagiaire, intern, internship, PFE or projet de fin d'études; "freelance" when it says freelance; "job" when it names an employment contract (CDI, CDD, full-time, part-time, employé); otherwise null.
   projects.type: "final_year" only for PFE / projet de fin d'études / end-of-studies project; "academic" when it says academic, académique, universitaire, PFA, projet de fin d'année or mini-projet ("projet de fin d'année" is NOT final_year); "personal" when it says personal or personnel; "professional" when it says professional or professionnel; otherwise null.
   skills.type: always classify: "technical", "soft" (qualités, savoir-être, soft skills) or "other".
7. Put each entry in the list that matches the meaning of the heading it is under: "Expérience académique" -> projects, "Vie associative" -> volunteer, "Certifications" -> certificates (even a hackathon attestation). Only when the heading matches no list: hackathons, competitions and prizes -> awards; clubs and associations -> volunteer.
8. Mapping: the headline under the name -> basics.label; every email -> basics.emails; every phone number -> basics.phones (as written); LinkedIn, GitHub and similar -> basics.profiles (network = the site's name, url as written); language tests (TOEIC, TOEFL, IELTS, DELF...) -> languages[].certificate; spoken languages go to languages, not skills; mention -> education.honors; GPA / moyenne -> education.score.
9. Skills: one skills entry per heading: name = the heading as written, keywords = each item under it. Never split one heading's items into separate entries. Items with no heading form one entry with name null.
10. Locations are never split. basics.location: the whole written location is one string in address, and city, region, postalCode and country stay null: "Sousse, Tunisie" -> {{"address": "Sousse, Tunisie", "city": null, "country": null, ...}}; "London, England, W1T 1JY" -> address "London, England, W1T 1JY". Fill city, region, postalCode or country only when the page itself labels that part ("Ville : Sousse"). Entry locations (work, education) are copied whole as written: "ISSAT, SOUSSE" stays "ISSAT, SOUSSE".
11. Ratings shown only as graphics go to skills[].level or languages[].fluency: dots, stars or circles -> "filled/total" ("4/5", "3/7"); bars or sliders -> the filled share estimated in fifths with "~" ("~3/5"). A level written in words or numbers ("courant", "B2", "80%") is copied as written instead.
12. Text lists (highlights, keywords, technologies, courses, roles): one item per bullet or comma-separated item, as written.
13. basics.hasPhoto: true if this page shows a photo of the person, otherwise false.
14. Anything on the page that fits no key goes to extra as {{"section": the heading it is under, "label": its label, "value": the text}}. Do not drop information. Section headings, page numbers and decorative text are not information.
15. corrections: one entry per place you fixed under rule 1: {{"original": the word or words as written on the page, "corrected": what you wrote instead, "kind": "spelling" or "grammar"}}. Changed letters or accents of a word are "spelling"; changed gender, plural or verb form agreement is "grammar". Every word whose letters or accents differ from the page must be listed, one entry per place. corrections is [] when nothing was fixed."""

FIX_PROMPT = (
    "Your JSON failed validation against the template:\n{errors}\nReturn the corrected, complete JSON object only."
)

MAX_RATE_LIMIT_WAITS = 30
SMALL_PAGE_PX = 1500  # longest side below this -> upscale 2x (1,130 of 2,451 pages)

_EMPTY = json.dumps({**template(), "basics": {**template()["basics"], "hasPhoto": None}})


def _extract_json(raw: str) -> dict | None:
    """Parse the outermost {...} out of a reply that may have code fences or prose."""
    s = (raw or "").strip()
    if "{" not in s or "}" not in s:
        return None
    try:
        obj = json.loads(s[s.index("{") : s.rindex("}") + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _is_empty(record: dict) -> bool:
    """Valid JSON with nothing extracted: in practice this means the endpoint dropped the image."""
    rec = {**record, "basics": {**record["basics"], "hasPhoto": None}}
    return json.dumps(rec) == _EMPTY


def _image_part(path: Path) -> tuple[dict, int]:
    """Return (content part, upscale factor). Small pages are sent upscaled 2x: on a
    648x912 page the teacher read "Decathlon" as "Debenhams" at 1x and correctly at 2x."""
    with Image.open(path) as im:
        if max(im.size) >= SMALL_PAGE_PX:
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            data, factor = path.read_bytes(), 1
        else:
            buf = io.BytesIO()
            im.convert("RGB").resize((im.width * 2, im.height * 2), Image.LANCZOS).save(buf, "PNG")
            mime, data, factor = "image/png", buf.getvalue(), 2
    b64 = base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}, factor


STRIPS_NOTE = (
    "The first image is the whole page. The next {n} images are the same page cut into {n} "
    "horizontal strips from top to bottom, enlarged for reading small text. Neighbouring strips "
    "overlap slightly, so text in an overlap appears in two strips: record it once. Use the whole "
    "page for the layout and the strips to read every letter.\n\n"
)


def _strip_parts(path: Path, n: int, factor: int) -> list[dict]:
    """The page cut into n horizontal strips, each overlapping its neighbours by 5% of the page
    height, scaled by `factor`. The teacher's image encoder shrinks a whole page (4 images cost
    ~2.7K prompt tokens), so strips give it more pixels per letter."""
    parts = []
    with Image.open(path) as im:
        w, h = im.size
        pad = int(h * 0.05)
        for i in range(n):
            top, bottom = max(0, h * i // n - pad), min(h, h * (i + 1) // n + pad)
            strip = im.convert("RGB").crop((0, top, w, bottom))
            if factor != 1:
                strip = strip.resize((w * factor, (bottom - top) * factor), Image.LANCZOS)
            buf = io.BytesIO()
            strip.save(buf, "PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return parts


def label_page(
    sess: requests.Session,
    img_path: Path,
    *,
    model: str,
    base_url: str,
    key: str,
    max_tokens: int,
    max_retries: int,
    retry_sleep: float,
    strips: int = 0,
) -> tuple[dict | None, list[dict], int]:
    """Return (record or None, attempts, upscale factor). Each attempt: {status, raw, errors, usage, seconds}."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    # Image FIRST, then text: the proxy drops an image that comes after the text.
    image, factor = _image_part(img_path)
    if strips:
        content = [
            image,
            *_strip_parts(img_path, strips, factor),
            {"type": "text", "text": STRIPS_NOTE.format(n=strips) + PROMPT},
        ]
    else:
        content = [image, {"type": "text", "text": PROMPT}]
    first = [{"role": "user", "content": content}]
    messages = first
    attempts: list[dict] = []
    tries = rate_limited = 0
    while tries < max_retries:
        t0 = time.time()
        att = {"status": None, "raw": None, "errors": None, "usage": None}
        try:
            r = sess.post(
                url,
                headers=headers,
                timeout=600,
                json={"model": model, "max_tokens": max_tokens, "temperature": 0, "messages": messages},
            )
            data = r.json()
        except (requests.RequestException, ValueError) as e:
            att.update(status="transport_error", errors=f"{type(e).__name__}: {e}"[:500])
            data = None
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict) and str(err.get("code")) == "429":
            # Proxy rate limit: wait and resend; does not use up the retry budget.
            rate_limited += 1
            attempts.append({**att, "status": "rate_limited", "seconds": round(time.time() - t0, 1)})
            if rate_limited > MAX_RATE_LIMIT_WAITS:
                return None, attempts, factor
            time.sleep(min(60, 15 * 2 ** (rate_limited - 1)))
            continue
        tries += 1
        if data is not None and ("error" in data or "choices" not in data):
            att.update(status="api_error", errors=json.dumps(data)[:500])
        elif data is not None:
            choice = data["choices"][0]
            raw = (choice.get("message", {}).get("content") or "").strip()
            att.update(raw=raw, usage=data.get("usage"), finish_reason=choice.get("finish_reason"))
            parsed = _extract_json(raw)
            if not raw or (parsed is None and any(m in raw.lower() for m in _REFUSAL_MARKERS)):
                att["status"] = "empty_or_refusal"
            elif parsed is None:
                att["status"] = "not_json"
            else:
                try:
                    record = to_record(parsed)
                except ValidationError as e:
                    errs = [f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()]
                    att.update(status="invalid", errors=errs)
                    # Same conversation, so the teacher corrects its own answer.
                    messages = first + [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": FIX_PROMPT.format(errors="\n".join(errs))},
                    ]
                else:
                    if _is_empty(record):
                        att["status"] = "all_empty"
                    else:
                        att.update(status="ok", seconds=round(time.time() - t0, 1))
                        attempts.append(att)
                        return record, attempts, factor
        att["seconds"] = round(time.time() - t0, 1)
        attempts.append(att)
        if att["status"] != "invalid":
            messages = first
        time.sleep(retry_sleep)
    return None, attempts, factor


def label_task(task: dict, **kw) -> dict:
    img = DATA / task["image"]
    out = {**task, "model": kw["model"], "ts": time.time()}
    if not img.exists():
        return {**out, "status": "missing_image", "record": None, "attempts": []}
    t0 = time.time()
    record, attempts, factor = label_page(requests.Session(), img, **kw)
    return {
        **out,
        "status": "ok" if record else "failed",
        "record": record,
        "upscale": factor,
        "strips": kw.get("strips", 0),
        "attempts": attempts,
        "latency_s": round(time.time() - t0, 1),
    }


def _read_ids(path: Path) -> list[str]:
    text = path.read_text()
    try:
        ids = json.loads(text)
    except json.JSONDecodeError:
        ids = text.replace(",", "\n").split()
    return [i.strip() for i in ids if i.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Label CV pages with a teacher VLM (one call per page).")
    ap.add_argument("--documents", type=Path, default=DOCUMENTS_PATH)
    ap.add_argument("--output", type=Path, required=True, help="JSONL, appended to")
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--ids", type=Path, help="JSON list or newline/comma file of cv_ids")
    sel.add_argument("--sample", type=int, help="random sample of N CVs (seed 42)")
    sel.add_argument("--n", type=int, help="first N CVs")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--retry-sleep", type=float, default=3.0)
    ap.add_argument(
        "--workers", type=int, default=3, help="the proxy returns 429 for glm-5.3-flash at 8 parallel requests"
    )
    ap.add_argument(
        "--strips",
        type=int,
        default=0,
        help="also send the page cut into N enlarged horizontal strips (0 = whole page only)",
    )
    args = ap.parse_args(argv)

    key = os.environ.get("LABEL_API_KEY")
    if not key or not args.base_url:
        ap.error("set LABEL_BASE_URL and LABEL_API_KEY in the environment or in .env at the repository root")

    docs = [json.loads(line) for line in args.documents.read_text().splitlines()]
    if args.ids:
        wanted = _read_ids(args.ids)
        by_id = {d["cv_id"]: d for d in docs}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            ap.error(f"{len(missing)} ids not in {args.documents}: {missing[:5]}")
        docs = [by_id[i] for i in wanted]
    elif args.sample:
        docs = random.Random(42).sample(docs, min(args.sample, len(docs)))
    elif args.n:
        docs = docs[: args.n]

    done = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            row = json.loads(line)
            if row["status"] == "ok":
                done.add((row["cv_id"], row["page"]))
    tasks = [
        {"cv_id": d["cv_id"], "page": p["page"], "num_pages": d["num_pages"], "image": p["image_path"]}
        for d in docs
        for p in d["pages"]
        if (d["cv_id"], p["page"]) not in done
    ]
    print(
        f"{len(docs)} CVs, {len(tasks)} pages to label ({len(done)} already ok) "
        f"with {args.model}, {args.workers} workers",
        flush=True,
    )

    kw = dict(
        model=args.model,
        base_url=args.base_url,
        key=key,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        strips=args.strips,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0
    t_start = time.time()
    with args.output.open("a") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(label_task, t, **kw) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            n_ok += row["status"] == "ok"
            n_fail += row["status"] != "ok"
            statuses = ",".join(a["status"] for a in row["attempts"])
            print(
                f"[{i}/{len(tasks)}] {row['cv_id'][:40]:40s} p{row['page']} {row['status']:7s} "
                f"{row.get('latency_s', 0):5.0f}s [{statuses}] | ok={n_ok} fail={n_fail} "
                f"| {time.time() - t_start:.0f}s",
                flush=True,
            )

    print(f"Done: {n_ok} ok, {n_fail} not ok -> {args.output}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
