"""Where the data lives and how it is split.

Label files are JSONL, one page per line: {"cv_id", "page", "image" (path under data/), "record" (a CVRecord dict)}.
The validation split is 5% of CVs chosen by a hash of the cv_id, so every page of a CV falls on the same side.
The reviewed gold file is never trained on: pages 1-50 are a development set, pages 51-254 the test set.
The data folder is <repository>/data, or the folder named by the TINY_RESUME_PARSER_DATA environment variable.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("TINY_RESUME_PARSER_DATA", ROOT / "data"))
LABELS = DATA / "processed" / "labels_train_v6.jsonl"
GOLD = DATA / "processed" / "labels_gold_v6_verified.jsonl"


def is_validation(cv_id: str) -> bool:
    """5% of CVs, fixed by a hash of the cv_id, so all pages of a CV fall on the same side."""
    return int(hashlib.sha1(cv_id.encode()).hexdigest(), 16) % 20 == 0


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def load_rows(name: str, limit: int | None = None, train_file: Path = LABELS) -> list[dict]:
    """name: gold (reviewed pages), val (validation split of the training labels), train (the rest)."""
    rows = read_jsonl(GOLD if name == "gold" else train_file)
    if name == "val":
        rows = [r for r in rows if is_validation(r["cv_id"])]
    elif name == "train":
        rows = [r for r in rows if not is_validation(r["cv_id"])]
    return rows[:limit] if limit else rows


def label_index() -> dict[tuple[str, int], dict]:
    """(cv_id, page) -> record for every labelled page; a page in both files is scored against the reviewed gold."""
    out = {}
    for path in (LABELS, GOLD):
        for r in read_jsonl(path):
            out[(r["cv_id"], r["page"])] = r["record"]
    return out


def page_image(row: dict) -> Image.Image:
    return Image.open(DATA / row["image"]).convert("RGB")
