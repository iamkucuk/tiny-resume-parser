"""Extra training pages from the hadikp resume PDFs (data/raw/Resumes PDF/<category>/*.pdf, one image per PDF).

    .venv/bin/python src/prepare_hadikp.py --dims DIMS.txt --n 20 --out data/processed/documents_hadikp_pilot.jsonl

DIMS.txt: one line per PDF, "<width> <height> <category>/<file>.pdf" (from `pdfimages -list`).
Takes a seeded random sample of PDFs whose embedded image is at least --min-width px wide, copies the
embedded JPEG unchanged (`pdfimages -j`) to data/hadikp/<cv_id>.jpg and writes documents.jsonl-style rows
for `auto_label.py --documents`. cv_ids that would fall in the validation split (train_lora.is_validation)
are skipped, so the dev pages stay the same.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import tempfile
from pathlib import Path

from train_lora import is_validation

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "Resumes PDF"
OUT_DIR = ROOT / "data" / "hadikp"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=Path, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--min-width", type=int, default=900)
    ap.add_argument("--skip", type=Path, nargs="*", default=[], help="documents files whose cv_ids are already taken")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    taken = {json.loads(l)["cv_id"] for f in args.skip for l in f.open()}
    rows = []
    for line in args.dims.open():
        w, h, rel = line.rstrip("\n").split(" ", 2)
        if int(w) >= args.min_width:
            rows.append((int(w), int(h), rel))
    random.Random(42).shuffle(rows)
    OUT_DIR.mkdir(exist_ok=True)
    docs = []
    for w, h, rel in rows:
        if len(docs) == args.n:
            break
        cv_id = "hk-" + hashlib.sha1(rel.encode()).hexdigest()[:12]
        if cv_id in taken or is_validation(cv_id):
            continue
        dest = OUT_DIR / f"{cv_id}.jpg"
        if not dest.exists():
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run(["pdfimages", "-j", "-f", "1", "-l", "1", str(RAW / rel), f"{tmp}/img"], check=True)
                found = sorted(Path(tmp).glob("img-*"))
                if len(found) != 1 or found[0].suffix != ".jpg":
                    print(f"skip {rel}: {[p.name for p in found]}")
                    continue
                found[0].rename(dest)
        docs.append({"cv_id": cv_id, "num_pages": 1, "multipage": False, "source": "hadikp", "pdf": rel,
                     "pages": [{"page": 1, "image_path": f"hadikp/{dest.name}", "width": w, "height": h,
                                "split": "train", "full_size": True}]})
    args.out.write_text("".join(json.dumps(d) + "\n" for d in docs))
    print(f"{len(docs)} pages -> {args.out}")


if __name__ == "__main__":
    main()
