"""VLM auto-labeling: extract flat-JSON CV fields via a teacher VLM.

Milestone M3.5 (pilot, 50 CVs) and M4 (full, 2180 CVs) use this same harness.
Default teacher: ``gemini-3.5-flash-newapi`` through the LiteLLM proxy; swap the
model/base-url/key via CLI flags or env vars.

Key transport findings baked in (verified empirically against the proxy):
  - The image part MUST come BEFORE the text part in the content array. If it
    comes after, the proxy's translation layer drops it and the model sees text
    only (returns "no image provided"). Verified: image-first yields correct
    answers ("4" shapes, "blue circle", real CV field extraction).
  - The proxy route is ~85% reliable per call (occasional dropouts even with
    correct ordering). So we retry on empty/refusal responses up to N times.
  - ``max_tokens`` must be >= ~1500 for extraction (reasoning tokens count).

Outputs one JSONL line per CV:
  {cv_id, fields: {...normalized...}, raw_responses: [...], attempts, model, ts, status}
Audited ``raw_responses`` kept so the M5 spot-check can verify the teacher.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import FIELDS, SCALAR_FIELDS, LIST_FIELDS, normalize  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"

DEFAULT_BASE_URL = os.environ.get(
    "LITELLM_BASE_URL", "https://litellm.furkankucuk.net/v1"
)
DEFAULT_KEY = os.environ["LITELLM_API_KEY"]  # key moved to .env (2026-10-09)
DEFAULT_MODEL = os.environ.get("VLABEL_MODEL", "gemini-3.5-flash-newapi")

# Refusal/empty markers we treat as "image didn't arrive" -> retry.
_REFUSAL_MARKERS = (
    "no image", "cannot see", "forgot to attach", "forgot to include",
    "please provide", "please upload", "provide the text", "provide the res",
    "upload the image", "i don't see an image", "didn't receive",
)

PROMPT = (
    "Extract these fields from the resume image. "
    "Return ONLY a JSON object — no prose, no markdown code fences.\n"
    f"String fields: {', '.join(f for f in FIELDS if f in SCALAR_FIELDS or f in ('photo_present','num_pages'))}.\n"
    f"Array-of-strings fields: {', '.join(f for f in FIELDS if f in LIST_FIELDS)}.\n"
    "For photo_present use true/false. For num_pages use an integer. "
    "If a field is absent, use empty string or empty array."
)


def _looks_like_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in _REFUSAL_MARKERS)


def _is_empty_extraction(fields: dict) -> bool:
    """Detect 'soft refusal': valid JSON returned but every field empty.

    Observed in the M3.5 pilot (12/50): the image was dropped on the attempt
    that parsed, so the model emitted a valid but all-empty object. Treat as a
    transport dropout and retry.
    """
    return not any(v not in ("", [], False, 0) for v in fields.values())


def _extract_json(raw: str) -> dict | None:
    """Parse JSON out of a model response that may have code fences or prose."""
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        s = s.rsplit("```", 1)[0].strip()
    # Find the outermost {...}
    if "{" in s:
        s = s[s.index("{"):]
    if "}" in s:
        s = s[: s.rindex("}") + 1]
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def label_one_image(
    client_session: requests.Session,
    b64: str,
    *,
    model: str,
    max_tokens: int,
    base_url: str,
    key: str,
    max_retries: int,
    retry_sleep: float,
) -> tuple[dict | None, list[str], str]:
    """Send one image, return (parsed_fields_or_None, raw_responses, status).

    Retries on transport dropouts (refusal markers) and JSON parse failures.
    Retains every raw response for audit.
    """
    raw_responses: list[str] = []
    url = f"{base_url.rstrip('/')}/chat/completions"
    # IMAGE-FIRST ordering — critical for the proxy to forward the image.
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    for attempt in range(1, max_retries + 1):
        try:
            r = client_session.post(url, headers=headers, json=payload, timeout=300)
            data = r.json()
        except Exception as e:
            raw_responses.append(f"<exception:{type(e).__name__}:{e}>")
            time.sleep(retry_sleep)
            continue
        if "error" in data or "choices" not in data:
            raw_responses.append(json.dumps(data)[:500])
            time.sleep(retry_sleep)
            continue
        raw = (data["choices"][0].get("message", {}).get("content") or "").strip()
        raw_responses.append(raw)
        if not raw or _looks_like_refusal(raw):
            time.sleep(retry_sleep)
            continue
        parsed = _extract_json(raw)
        if parsed is None:
            time.sleep(retry_sleep)
            continue
        normalized = normalize(parsed)
        # Soft-refusal check: valid JSON but all fields empty -> image dropped.
        if _is_empty_extraction(normalized):
            time.sleep(retry_sleep)
            continue
        return normalized, raw_responses, "ok"
    return None, raw_responses, "failed"


def label_document(doc: dict, *, data_root: Path, **kw) -> dict:
    """Label one CV record (multi-page: label page 1; the bulk of fields live there).

    Multi-page merge (per PLAN M4) will extend this to iterate pages and merge
    field lists with dedup; for the M3.5 pilot we label page 1 only — enough to
    measure per-field teacher accuracy on single-page CVs, which is the gate.
    """
    cv_id = doc["cv_id"]
    page1 = doc["pages"][0]
    # documents.jsonl image paths are relative to data/.
    img_path = data_root / page1["image_path"]
    if not img_path.exists():
        return {"cv_id": cv_id, "status": "missing_image", "fields": {},
                "raw_responses": [], "attempts": 0, "model": kw.get("model", "")}
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    sess = requests.Session()
    t0 = time.time()
    fields, raws, status = label_one_image(sess, b64, **kw)
    return {
        "cv_id": cv_id,
        "status": status,
        "fields": fields or {},
        "attempts": len(raws),
        "raw_responses": raws,
        "model": kw.get("model", ""),
        "ts": time.time(),
        "latency_s": round(time.time() - t0, 1),
        "num_pages": doc["num_pages"],
        "image": page1["image_path"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="VLM auto-labeling harness.")
    ap.add_argument("--documents", type=Path, default=DOCUMENTS_PATH)
    ap.add_argument("--output", type=Path, required=True, help="JSONL output path")
    ap.add_argument("--n", type=int, default=0, help="label first N (0 = all)")
    ap.add_argument("--sample", type=int, default=0,
                    help="random sample N (0 = use --n sequential)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--key", default=DEFAULT_KEY)
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="dense CVs can reach ~4k tokens; 8192 leaves headroom")
    ap.add_argument("--max-retries", type=int, default=8,
                    help="image-transport dropout is the dominant failure mode (~15 "
                         "percent per attempt); 8 retries drives failure well under 1 percent")
    ap.add_argument("--retry-sleep", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=6,
                    help="concurrent labeling threads. Dropouts are random per-call, "
                         "so concurrency decouples them and cuts wall-time N-fold.")
    ap.add_argument("--id-filter", default=None,
                    help="only label CVs whose id is in this newline/comma file")
    args = ap.parse_args()

    docs = [json.loads(l) for l in args.documents.read_text().splitlines()]
    if args.id_filter:
        wanted = set()
        for tok in Path(args.id_filter).read_text().replace(",", "\n").split():
            wanted.add(tok.strip())
        docs = [d for d in docs if d["cv_id"] in wanted]
    elif args.sample:
        import random
        random.seed(42)
        docs = random.sample(docs, min(args.sample, len(docs)))
    elif args.n:
        docs = docs[: args.n]
    print(f"Labeling {len(docs)} CVs with {args.model} ({args.workers} workers)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(model=args.model, base_url=args.base_url, key=args.key,
              max_tokens=args.max_tokens, max_retries=args.max_retries,
              retry_sleep=args.retry_sleep)
    data_root = PROJECT_ROOT / "data"

    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_ok = n_fail = 0
    t_start = time.time()
    out = args.output.open("w")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(label_document, doc, data_root=data_root, **kw): doc
                       for doc in docs}
            for i, fut in enumerate(as_completed(futures)):
                rec = fut.result()
                out.write(json.dumps(rec) + "\n"); out.flush()
                if rec["status"] == "ok": n_ok += 1
                else: n_fail += 1
                ne = sum(1 for v in rec["fields"].values() if v not in ("", [], False, 0))
                elapsed = time.time() - t_start
                rate = (i + 1) / max(elapsed, 1)
                eta = (len(docs) - i - 1) / rate if rate > 0 else 0
                print(f"[{i+1}/{len(docs)}] {rec['cv_id'][:40]:40s} {rec['status']:7s} "
                      f"{rec['attempts']}att {rec.get('latency_s',0):.0f}s {ne}ne | "
                      f"ok={n_ok} fail={n_fail} | {rate:.1f}/s ETA={eta:.0f}s",
                      flush=True)
    finally:
        out.close()

    print(f"\nDone: {n_ok} ok, {n_fail} failed ({100*n_fail/len(docs):.1f}% fail). -> {args.output}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
