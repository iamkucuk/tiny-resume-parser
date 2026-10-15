"""Deterministic checks of teacher page labels (src/auto_label.py output) against tesseract OCR.

Per page:
  - schema: the record still validates against src/schema.py
  - email: emails in the OCR text vs basics.emails, both directions
  - phone: the digits of each basics.phones entry appear in the OCR digits
  - ungrounded values: text values whose words are mostly absent from the OCR
    (likely invented); fuzzy word match, since OCR misreads too
  - coverage: share of OCR words (>= 4 letters) found anywhere in the record
    (low = information dropped)

OCR is tesseract 5 (fra+eng, tessdata_fast) of every page, one text file per image in
data/processed/ocr_tesseract/<image file name>.txt. It has its own errors, so a flag is a
reason to look, not a verdict; the Haiku page check and human review decide.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import to_record  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OCR_DIR = PROJECT_ROOT / "data" / "processed" / "ocr_tesseract"

# Values the teacher converts or classifies rather than copies.
SKIP_KEYS = {"type", "corrected", "kind", "hasPhoto", "network", "startDate", "endDate", "date", "releaseDate", "birthDate"}
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def norm_tokens(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", t)


def leaves(obj, path=""):
    """Yield (path, string value) for every non-null string in the record."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in SKIP_KEYS:
                yield from leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def grounding(value: str, ocr_set: set[str], ocr_list: list[str]) -> float:
    toks = [t for t in norm_tokens(value) if len(t) >= 2]
    if not toks:
        return 1.0
    hit = sum(t in ocr_set or (len(t) >= 4 and bool(difflib.get_close_matches(t, ocr_list, n=1, cutoff=0.8)))
              for t in toks)
    return hit / len(toks)


def check_row(row: dict, threshold: float) -> dict:
    out = {"cv_id": row["cv_id"], "page": row["page"], "flags": []}
    if row["status"] != "ok":
        out["flags"].append(f"label_status:{row['status']}")
        return out
    try:
        rec = to_record(row["record"])
    except ValidationError as e:
        out["flags"].append(f"schema_invalid:{len(e.errors())}")
        return out
    ocr_path = OCR_DIR / f"{Path(row['image']).name}.txt"
    if not ocr_path.exists():
        out["flags"].append("no_ocr")
        return out
    ocr = ocr_path.read_text()
    ocr_list = sorted(set(norm_tokens(ocr)))
    ocr_set = set(ocr_list)

    ocr_emails = {e.lower().rstrip(".") for e in EMAIL_RE.findall(ocr)}
    emails = {e.lower() for e in rec["basics"]["emails"]}
    if ocr_emails - emails:
        out["flags"].append(f"email_missing:{sorted(ocr_emails - emails)}")
    if ocr_emails and emails - ocr_emails:
        out["flags"].append(f"email_not_in_ocr:{sorted(emails - ocr_emails)}")

    ocr_digit_lines = [re.sub(r"\D", "", line) for line in ocr.splitlines()]
    for phone in rec["basics"]["phones"]:
        digits = re.sub(r"\D", "", phone)
        if digits and not any(digits[-8:] in d for d in ocr_digit_lines):
            out["flags"].append(f"phone_not_in_ocr:{phone}")

    ungrounded = []
    rec_tokens: set[str] = set()
    n_values = 0
    for path, value in leaves(rec):
        n_values += 1
        rec_tokens.update(norm_tokens(value))
        g = grounding(value, ocr_set, ocr_list)
        if g < threshold:
            ungrounded.append({"path": path, "value": value, "grounding": round(g, 2)})
    words = {t for t in ocr_set if len(t) >= 4 and t.isalpha()}
    out.update(n_values=n_values, ungrounded=ungrounded,
               coverage=round(len(words & rec_tokens) / max(len(words), 1), 3))
    if ungrounded:
        out["flags"].append(f"ungrounded_values:{len(ungrounded)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Check teacher page labels against OCR.")
    ap.add_argument("labels", type=Path, help="JSONL from src/auto_label.py")
    ap.add_argument("--output", type=Path, required=True, help="JSONL report, one line per page")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="flag a value when fewer than this share of its words are in the OCR")
    ap.add_argument("--min-coverage", type=float, default=0.7)
    args = ap.parse_args()

    latest = {}
    for line in args.labels.read_text().splitlines():
        row = json.loads(line)
        latest[(row["cv_id"], row["page"])] = row  # last line per page wins (reruns append)
    report = []
    for row in latest.values():
        r = check_row(row, args.threshold)
        if r.get("coverage") is not None and r["coverage"] < args.min_coverage:
            r["flags"].append(f"low_coverage:{r['coverage']}")
        report.append(r)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in report))

    flagged = [r for r in report if r["flags"]]
    covs = sorted(r["coverage"] for r in report if "coverage" in r)
    print(f"{len(report)} pages, {len(flagged)} flagged; coverage median "
          f"{covs[len(covs) // 2] if covs else None}, min {covs[0] if covs else None}")
    for r in flagged:
        print(f"  {r['cv_id'][:40]:40s} p{r['page']} {'; '.join(r['flags'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
