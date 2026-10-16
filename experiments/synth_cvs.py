"""Synthetic CV pages with exact labels: real training records remixed and rendered in new layouts.

    python experiments/synth_cvs.py --n 500 --seed 0 --out data/synth_v1_s0   # needs headless Chromium / Chrome

Each page starts from one training record (never a validation or reviewed gold page), gets a new name and new contact
details, sometimes takes one or two sections from another CV in the same language, and is rendered as HTML in one of
several layouts, fonts, colours and date styles, printed to PDF by headless Chrome and rasterised at 120 dpi (the
resolution of the real pages). Every value of the label is printed on the page, so the label is exact for the image;
dates stay ISO in the label and are written in words or numbers on the page, as on real CVs. Pages that do not fit
on one A4 page after trimming are skipped. Writes OUT/images/*.jpg and OUT/labels.jsonl in the label-file format
(image paths relative to the data folder).
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pypdfium2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tiny_resume_parser.data import DATA, LABELS, is_validation  # noqa: E402
from tiny_resume_parser.models import STUDENT_SKIP  # noqa: E402
from tiny_resume_parser.schema import to_record  # noqa: E402

# Chromium headless shell (e.g. installed by `playwright install chromium-headless-shell`); full Chrome also works
CHROME = os.environ.get("CHROME") or next(
    iter(
        sorted(
            glob.glob(
                str(Path.home() / "Library/Caches/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell")
            )
        )
    ),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)
GOLD = DATA / "processed/labels_gold_v6_verified.jsonl"
MONTHS = {
    "fr": "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split(),
    "en": "January February March April May June July August September October November December".split(),
}
HEAD = {
    "fr": dict(
        work="Expérience professionnelle|Expériences|Parcours professionnel|Stages et expériences",
        education="Formation|Éducation|Parcours académique|Diplômes",
        skills="Compétences|Compétences techniques|Savoir-faire",
        languages="Langues|Compétences linguistiques",
        projects="Projets|Projets académiques|Réalisations",
        interests="Centres d'intérêt|Loisirs|Activités",
        certificates="Certifications|Certificats|Formations complémentaires",
        volunteer="Vie associative|Bénévolat|Activités associatives",
        awards="Distinctions|Prix et distinctions",
        references="Références",
        publications="Publications",
        profile="Profil|À propos|Résumé",
        contact="Contact|Coordonnées",
        present="Présent|Aujourd'hui|En cours|à ce jour",
        tech="Technologies|Outils|Environnement",
        role="Rôle|Rôles",
        score="Moyenne|Note",
        honors="Mention",
    ),
    "en": dict(
        work="Work Experience|Experience|Professional Experience",
        education="Education|Academic Background",
        skills="Skills|Technical Skills",
        languages="Languages",
        projects="Projects|Academic Projects",
        interests="Interests|Hobbies",
        certificates="Certifications|Certificates",
        volunteer="Volunteering|Associative Life",
        awards="Awards|Honours",
        references="References",
        publications="Publications",
        profile="Profile|About Me|Summary",
        contact="Contact",
        present="Present|Now|Ongoing",
        tech="Technologies|Tools",
        role="Role|Roles",
        score="GPA|Grade",
        honors="Honours",
    ),
}
PERSONAL = {
    "fr": dict(
        birthDate="Date de naissance",
        placeOfBirth="Lieu de naissance",
        age="Âge",
        gender="Sexe",
        maritalStatus="Situation familiale",
        nationality="Nationalité",
        drivingLicense="Permis",
        availability="Disponibilité",
    ),
    "en": dict(
        birthDate="Date of birth",
        placeOfBirth="Place of birth",
        age="Age",
        gender="Gender",
        maritalStatus="Marital status",
        nationality="Nationality",
        drivingLicense="Driving licence",
        availability="Availability",
    ),
}
TYPE_TAG = {
    "fr": {"internship": "Stage", "freelance": "Freelance", "job": "Emploi"},
    "en": {"internship": "Internship", "freelance": "Freelance", "job": "Job"},
}
PROJECT_TAG = {
    "fr": {
        "final_year": "Projet de fin d'études",
        "academic": "Projet académique",
        "personal": "Projet personnel",
        "professional": "Projet professionnel",
    },
    "en": {
        "final_year": "Final-year project",
        "academic": "Academic project",
        "personal": "Personal project",
        "professional": "Professional project",
    },
}
FONTS = [
    "Helvetica",
    "Arial",
    "Georgia",
    "Times New Roman",
    "Verdana",
    "Avenir Next",
    "Futura",
    "Palatino",
    "Gill Sans",
]
ACCENTS = ["#1f4e79", "#2e7d32", "#8e24aa", "#c62828", "#00695c", "#37474f", "#ef6c00", "#000000", "#283593"]
FR_WORDS = set("le la les des du de et en pour avec sur dans une un au aux stage formation compétences".split())
EN_WORDS = set("the and of in for with on to at skills experience education languages degree".split())


def esc(x) -> str:
    return html.escape(str(x))


def language(rec: dict) -> str:
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿœ]+", json.dumps(rec, ensure_ascii=False).lower())
    return "en" if sum(w in EN_WORDS for w in words) > 1.5 * sum(w in FR_WORDS for w in words) else "fr"


def pick(s: str) -> str:
    return random.choice(s.split("|"))


def date_text(d: str | None, lang: str, style: int) -> str:
    """ISO label date -> how a CV writes it (style fixed per page)."""
    if not d:
        return ""
    if d == "present":
        return pick(HEAD[lang]["present"])
    p = d.split("-")
    if len(p) == 1:
        return p[0]
    y, m = p[0], int(p[1])
    if len(p) == 3:
        return f"{p[2]}/{p[1]}/{y}" if style != 2 else f"{int(p[2])} {MONTHS[lang][m - 1]} {y}"
    month = MONTHS[lang][m - 1]
    return [f"{month.capitalize()} {y}", f"{p[1]}/{y}", f"{month[:3].capitalize()}. {y}", f"{month} {y}"][style]


def dates(a, b, lang, style) -> str:
    s, e = date_text(a, lang, style), date_text(b, lang, style)
    return f"{s} – {e}" if s and e else s or e


def new_identity(rec: dict, pools: dict) -> dict:
    """Swap name, emails and phone digits; the rest of basics stays as written in the source."""
    b = dict(rec.get("basics") or {})
    first, last = random.choice(pools["first"]), random.choice(pools["last"])
    if b.get("name"):
        b["name"] = random.choice([f"{first} {last}", f"{first} {last.upper()}", f"{last.upper()} {first}"])
    if b.get("emails"):
        user = random.choice([f"{first}.{last}", f"{first}{last}", f"{first[0]}{last}", f"{last}.{first}"])
        dom = random.choice(["gmail.com", "yahoo.fr", "outlook.com", "hotmail.com", "gmail.com"])
        b["emails"] = [re.sub(r"[^a-z0-9.]", "", user.lower()) + str(random.randint(1, 99)) * random.randint(0, 1)]
        b["emails"][0] += "@" + dom
    if b.get("phones"):
        # new digits, same format; a leading +country code is kept
        b["phones"] = [
            (m := re.match(r"\+\d{1,3}", p))
            and m.group() + re.sub(r"\d", lambda _: str(random.randint(0, 9)), p[m.end() :])
            or re.sub(r"\d", lambda _: str(random.randint(0, 9)), p)
            for p in b["phones"]
        ]
    return {**rec, "basics": b}


def remix(rec: dict, same_lang: list[dict]) -> dict:
    out = dict(rec)
    for sec in random.sample(["education", "skills", "languages", "interests", "certificates", "projects"], 2):
        if random.random() < 0.35:
            donor = random.choice(same_lang)
            if donor.get(sec):
                out[sec] = donor[sec]
    return out


def bold(x, after: str = "") -> str:
    return f"<b>{esc(x)}</b>{after}" if x else ""


def li(items) -> str:
    return "<ul>" + "".join(f"<li>{esc(x)}</li>" for x in items) + "</ul>" if items else ""


def entry(title, sub, when, body="") -> str:
    head = f"<div class=et><b>{title}</b>" + (f"<span class=when>{esc(when)}</span>" if when else "") + "</div>"
    return f"<div class=entry>{head}" + (f"<div class=sub>{sub}</div>" if sub else "") + body + "</div>"


def section(key, lang, body) -> str:
    return f"<section><h2>{esc(pick(HEAD[lang][key]))}</h2>{body}</section>" if body else ""


def render_sections(r: dict, lang: str, ds: int) -> dict[str, str]:
    h = HEAD[lang]
    s = {}
    s["work"] = "".join(
        entry(
            esc(w.get("position") or w.get("name") or ""),
            " · ".join(
                esc(x)
                for x in [w.get("name") if w.get("position") else None, w.get("location"), w.get("url")]
                + [TYPE_TAG[lang].get(w.get("type"))]
                if x
            ),
            dates(w.get("startDate"), w.get("endDate"), lang, ds),
            (f"<p>{esc(w['summary'])}</p>" if w.get("summary") else "")
            + li(w.get("highlights"))
            + (
                f"<p><i>{esc(pick(h['tech']))} :</i> {esc(', '.join(w['technologies']))}</p>"
                if w.get("technologies")
                else ""
            ),
        )
        for w in r.get("work") or []
    )
    s["volunteer"] = "".join(
        entry(
            esc(v.get("position") or v.get("organization") or ""),
            " · ".join(esc(x) for x in [v.get("organization") if v.get("position") else None, v.get("url")] if x),
            dates(v.get("startDate"), v.get("endDate"), lang, ds),
            (f"<p>{esc(v['summary'])}</p>" if v.get("summary") else "") + li(v.get("highlights")),
        )
        for v in r.get("volunteer") or []
    )
    s["education"] = "".join(
        entry(
            " — ".join(esc(x) for x in [e.get("studyType"), e.get("area")] if x) or esc(e.get("institution") or ""),
            " · ".join(
                esc(x)
                for x in [
                    e.get("institution") if (e.get("studyType") or e.get("area")) else None,
                    e.get("location"),
                    e.get("url"),
                ]
                if x
            ),
            dates(e.get("startDate"), e.get("endDate"), lang, ds),
            (f"<p>{esc(pick(h['score']))} : {esc(e['score'])}</p>" if e.get("score") else "")
            + (f"<p>{esc(h['honors'])} : {esc(e['honors'])}</p>" if e.get("honors") else "")
            + (f"<p>{esc(e['summary'])}</p>" if e.get("summary") else "")
            + li(e.get("courses")),
        )
        for e in r.get("education") or []
    )
    s["projects"] = "".join(
        entry(
            esc(p.get("name") or ""),
            " · ".join(esc(x) for x in [PROJECT_TAG[lang].get(p.get("type")), p.get("entity"), p.get("url")] if x),
            dates(p.get("startDate"), p.get("endDate"), lang, ds),
            (f"<p>{esc(p['description'])}</p>" if p.get("description") else "")
            + li(p.get("highlights"))
            + (
                f"<p><i>{esc(pick(h['tech']))} :</i> {esc(', '.join(p['technologies']))}</p>"
                if p.get("technologies")
                else ""
            )
            + (f"<p><i>{esc(pick(h['role']))} :</i> {esc(', '.join(p['roles']))}</p>" if p.get("roles") else ""),
        )
        for p in r.get("projects") or []
    )
    s["skills"] = "".join(
        f"<p>{bold(k.get('name'), ' : ')}{esc(', '.join(k.get('keywords') or []))}"
        + (f" ({esc(k['level'])})" if k.get("level") else "")
        + "</p>"
        for k in r.get("skills") or []
    )
    s["languages"] = "".join(
        f"<p><b>{esc(x.get('language') or '')}</b>"
        + (f" : {esc(x['fluency'])}" if x.get("fluency") else "")
        + (f" ({esc(x['certificate'])})" if x.get("certificate") else "")
        + "</p>"
        for x in r.get("languages") or []
    )
    s["interests"] = "".join(
        f"<p>{bold(i.get('name'))}"
        + (f"{' : ' if i.get('name') else ''}{esc(', '.join(i['keywords']))}" if i.get("keywords") else "")
        + "</p>"
        for i in r.get("interests") or []
    )
    s["certificates"] = "".join(
        entry(
            esc(c.get("name") or ""),
            " · ".join(esc(x) for x in [c.get("issuer"), c.get("url")] if x),
            date_text(c.get("date"), lang, ds),
            f"<p>{esc(c['summary'])}</p>" if c.get("summary") else "",
        )
        for c in r.get("certificates") or []
    )
    s["awards"] = "".join(
        entry(
            esc(a.get("title") or ""),
            esc(a.get("awarder") or ""),
            date_text(a.get("date"), lang, ds),
            f"<p>{esc(a['summary'])}</p>" if a.get("summary") else "",
        )
        for a in r.get("awards") or []
    )
    s["publications"] = "".join(
        entry(
            esc(p.get("name") or ""),
            " · ".join(esc(x) for x in [p.get("publisher"), p.get("url")] if x),
            date_text(p.get("releaseDate"), lang, ds),
            f"<p>{esc(p['summary'])}</p>" if p.get("summary") else "",
        )
        for p in r.get("publications") or []
    )
    s["references"] = "".join(
        f"<p><b>{esc(x.get('name') or '')}</b>{(' — ' + esc(x['reference'])) if x.get('reference') else ''}</p>"
        for x in r.get("references") or []
    )
    extra = {}
    for x in r.get("extra") or []:
        extra.setdefault(x.get("section") or "", []).append(
            f"<p>{bold(x.get('label'), ' : ')}{esc(x.get('value') or '')}</p>"
        )
    s["extra"] = "".join(
        f"<section><h2>{esc(sec)}</h2>{''.join(v)}</section>" if sec else "".join(v) for sec, v in extra.items()
    )
    return s


def contact_block(b: dict, lang: str, ds: int) -> str:
    loc = b.get("location") or {}
    city = " ".join(x for x in [loc.get("postalCode"), loc.get("city")] if x)
    place = ", ".join(x for x in [loc.get("address"), city, loc.get("region"), loc.get("country")] if x)
    lines = [*b.get("emails", []), *b.get("phones", []), place, b.get("url")]
    for p in b.get("profiles") or []:
        val = p.get("url") or p.get("username") or ""
        lines.append(f"{p['network']} : {val}" if p.get("network") and val else p.get("network") or val)
    for k, label in PERSONAL[lang].items():
        if b.get(k):
            v = date_text(b[k], lang, ds) if k == "birthDate" else b[k]
            lines.append(f"{label} : {v}")
    return "".join(f"<p>{esc(x)}</p>" for x in lines if x)


AVATAR = (
    "<svg class=photo viewBox='0 0 100 100'><rect width=100 height=100 fill='#c9ccd1'/>"
    "<circle cx=50 cy=38 r=20 fill='#8d939b'/><ellipse cx=50 cy=92 rx=34 ry=26 fill='#8d939b'/></svg>"
)


def page_html(r: dict, lang: str) -> str:
    ds = random.randrange(4)
    layout = random.choice(["single", "left", "right", "band"])
    font, accent, size = random.choice(FONTS), random.choice(ACCENTS), random.choice([9, 9.5, 10, 10.5, 11])
    b = r.get("basics") or {}
    s = render_sections(r, lang, ds)
    side_keys = ["skills", "languages", "interests", "certificates", "awards", "references"]
    main_keys = ["work", "education", "projects", "volunteer", "publications"]
    random.shuffle(main_keys)
    random.shuffle(side_keys)
    photo = AVATAR if b.get("hasPhoto") else ""
    head = (
        f"<header>{photo}<div><h1>{esc(b.get('name') or '')}</h1>"
        + (f"<div class=label>{esc(b['label'])}</div>" if b.get("label") else "")
        + "</div></header>"
    )
    profile = section("profile", lang, f"<p>{esc(b['summary'])}</p>" if b.get("summary") else "")
    contact = section("contact", lang, contact_block(b, lang, ds))
    main = "".join(section(k, lang, s[k]) for k in main_keys) + s["extra"]
    side = "".join(section(k, lang, s[k]) for k in side_keys)
    if layout == "single":
        body = head + contact + profile + main + side
    else:
        cols = f"<div class=side>{contact}{side}</div><div class=main>{profile}{main}</div>"
        if layout == "right":
            cols = f"<div class=main>{profile}{main}</div><div class=side>{contact}{side}</div>"
        body = head + f"<div class=cols>{cols}</div>"
    return f"""<!doctype html><html><head><meta charset=utf-8><style>
@page {{ size: A4; margin: {random.choice([10, 12, 14, 16])}mm; }}
body {{ font-family: '{font}', sans-serif; font-size: {size}pt; color: #222; margin: 0; }}
h1 {{ font-size: {size * 2.1:.1f}pt; margin: 0; color: {accent if layout != "band" else "#fff"}; }}
header {{ display: flex; gap: 12px; align-items: center; margin-bottom: 8px;
  {f"background:{accent};color:#fff;padding:10px;" if layout == "band" else ""} }}
.label {{ font-size: {size * 1.2:.1f}pt; }}
.photo {{ width: 70px; height: 70px; border-radius: {random.choice(["50%", "4px"])}; }}
h2 {{ font-size: {size * 1.25:.1f}pt; color: {accent}; border-bottom: 1px solid {accent}; margin: 9px 0 4px;
  text-transform: {random.choice(["uppercase", "none"])}; }}
p {{ margin: 2px 0; }} ul {{ margin: 2px 0 2px 16px; padding: 0; }}
.cols {{ display: flex; gap: 14px; }} .side {{ width: 34%; }} .main {{ flex: 1; }}
.et {{ display: flex; justify-content: space-between; gap: 8px; }} .when {{ white-space: nowrap; color: #555; }}
.sub {{ color: #444; font-style: italic; }} .entry {{ margin-bottom: 5px; }}
</style></head><body>{body}</body></html>"""


def trim(r: dict) -> dict:
    """Shorten the record so the page fits: drop the last highlight / course / item of the longest list."""
    r = json.loads(json.dumps(r))
    cands = []
    for sec in ("work", "projects", "education", "volunteer"):
        for item in r.get(sec) or []:
            for k in ("highlights", "courses"):
                if len(item.get(k) or []) > 1:
                    cands.append((len(item[k]), item, k))
    if cands:
        _, item, k = max(cands, key=lambda c: c[0])
        item[k] = item[k][:-1]
        return r
    longest = max(
        (s for s in ("work", "projects", "certificates", "volunteer") if len(r.get(s) or []) > 1),
        key=lambda s: len(r[s]),
        default=None,
    )
    if longest:
        r[longest] = r[longest][:-1]
    return r


def to_pdf(html_text: str, tmp: Path) -> pypdfium2.PdfDocument:
    src, pdf = tmp / "p.html", tmp / "p.pdf"
    src.write_text(html_text)
    subprocess.run(
        [
            CHROME,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={tmp / 'chrome'}",
        ]
        + [f"--print-to-pdf={pdf}", src.as_uri()],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return pypdfium2.PdfDocument(str(pdf))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, default=DATA / "synth_v1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    gold = {json.loads(line)["cv_id"] for line in GOLD.open()}
    src = [json.loads(line) for line in LABELS.open()]
    src = [r for r in src if not is_validation(r["cv_id"]) and r["cv_id"] not in gold]
    recs = [{k: v for k, v in r["record"].items() if k not in STUDENT_SKIP} for r in src]
    by_lang = {"fr": [], "en": []}
    for rec in recs:
        by_lang[language(rec)].append(rec)
    names = [(r.get("basics") or {}).get("name") or "" for r in recs]
    toks = [n.split() for n in names if 2 <= len(n.split()) <= 3]
    pools = {"first": [t[0].capitalize() for t in toks], "last": [t[-1].capitalize() for t in toks]}
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    with tempfile.TemporaryDirectory() as tmp, (args.out / "labels.jsonl").open("w") as f:
        while done < args.n:
            lang = "fr" if random.random() < 0.8 else "en"
            rec = to_record(new_identity(remix(random.choice(by_lang[lang]), by_lang[lang]), pools))  # rendered = label
            for _ in range(12):
                doc = to_pdf(page_html(rec, lang), Path(tmp))
                if len(doc) == 1:
                    break
                rec = trim(rec)
            else:
                skipped += 1
                continue
            cv_id = f"synth-{args.seed}-{done:05d}"
            img = doc[0].render(scale=120 / 72).to_pil().convert("RGB")
            img.save(args.out / "images" / f"{cv_id}.jpg", quality=90)
            label = to_record(rec)
            row = {"cv_id": cv_id, "page": 1, "num_pages": 1, "image": f"{args.out.name}/images/{cv_id}.jpg"}
            f.write(json.dumps(row | {"record": label}, ensure_ascii=False) + "\n")
            done += 1
            if done % 50 == 0:
                print(f"{done} pages, {skipped} skipped", flush=True)
    print(f"wrote {done} pages to {args.out} ({skipped} skipped: did not fit on one page)")


if __name__ == "__main__":
    main()
