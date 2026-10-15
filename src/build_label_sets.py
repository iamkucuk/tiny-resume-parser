"""Split the teacher labels into a training set and the gold evaluation set, and filter `corrections`.

    .venv/bin/python src/build_label_sets.py

Reads data/processed/labels_full_v6.jsonl (GLM teacher labels, one row per page) and keeps the
last ok row per page for the CVs in cv_ids_full_size.json (every page longer than 640 px).
CVs in train_pool_ids.json go to labels_train_v6.jsonl; CVs in gold_200/gold_ids.json go to
labels_gold_v6.jsonl. A CV in neither list is an error.

A corrections entry is dropped when:
  - same:     original == corrected
  - not_in_label: the corrected text appears in no label value (e.g. a fixed heading that is not stored)
  - ocr:      the page's Tesseract text contains the corrected word but not the original, so the
              page most likely already had the corrected form (the teacher invented the "original")
The label values are not changed. Each output row gets `corrections_dropped` with the reasons.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
LABELS = PROCESSED / "labels_full_v6.jsonl"
OCR_DIR = PROCESSED / "ocr_tesseract"


def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)


def _has_word(word: str, text: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(word.lower()) + r"(?!\w)", text) is not None


def _drop_reason(entry: dict, label_text: str, ocr_text: str) -> str | None:
    if entry["original"] == entry["corrected"]:
        return "same"
    if entry["corrected"] not in label_text:
        return "not_in_label"
    if _has_word(entry["corrected"], ocr_text) and not _has_word(entry["original"], ocr_text):
        return "ocr"
    return None


def main() -> int:
    full = set(json.loads((PROCESSED / "cv_ids_full_size.json").read_text()))
    pool = set(json.loads((PROCESSED / "train_pool_ids.json").read_text()))
    gold = set(json.loads((PROCESSED / "gold_200" / "gold_ids.json").read_text()))

    latest = {}
    for line in LABELS.read_text().splitlines():
        r = json.loads(line)
        if r["cv_id"] in full:
            latest[(r["cv_id"], r["page"])] = r  # reruns append; last line per page wins
    not_ok = [k for k, r in latest.items() if r["status"] != "ok"]
    if not_ok:
        raise SystemExit(f"{len(not_ok)} pages are not ok, e.g. {not_ok[:3]}")
    stray = {cv for cv, _ in latest} - pool - gold
    if stray:
        raise SystemExit(f"{len(stray)} CVs are in neither the training pool nor the gold set, e.g. {sorted(stray)[:3]}")

    reasons = Counter()
    kept_total = 0
    out = {"train": [], "gold": []}
    for (cv_id, page), r in sorted(latest.items()):
        rec = r["record"]
        label_text = "\n".join(_strings({k: v for k, v in rec.items() if k != "corrections"}))
        ocr_path = OCR_DIR / (Path(r["image"]).name + ".txt")
        if not ocr_path.exists():
            raise SystemExit(f"no OCR text for {cv_id} p{page}: {ocr_path}")
        ocr_text = ocr_path.read_text().lower()
        kept, dropped = [], []
        for entry in rec["corrections"]:
            reason = _drop_reason(entry, label_text, ocr_text)
            if reason:
                dropped.append({**entry, "reason": reason})
                reasons[reason] += 1
            else:
                kept.append(entry)
        kept_total += len(kept)
        row = {k: r[k] for k in ("cv_id", "page", "num_pages", "image")}
        row["record"] = {**rec, "corrections": kept}
        row["corrections_dropped"] = dropped
        out["train" if cv_id in pool else "gold"].append(row)

    for split, rows in out.items():
        path = PROCESSED / f"labels_{split}_v6.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        print(f"{path.name}: {len({r['cv_id'] for r in rows})} CVs, {len(rows)} pages")
    print(f"corrections kept {kept_total}, dropped {sum(reasons.values())} {dict(reasons)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
