"""Reconstruct per-document records from the un-augmented Roboflow source export.

Milestone M2. Reads the COCO export in `data/roboflow_source/` (fork
`vizioai/cv-parsing-ksgbu-qiqkv`, version 1: auto-orient only, no resize, no
augmentation) and emits `data/processed/documents.jsonl` — one record per CV,
with its page images grouped in order and each page's bounding boxes attached.

The output is the canonical inventory consumed by every downstream stage:
  - M3/M3.5  sample from it (gold set, pilot)
  - M4       VLM-labels each CV (reads image_path per page)
  - M6       builds DocTags targets via `coco_annotations_to_field_bboxes` + the VLM labels

Grouping: each image's original upload name (COCO `extra.name`) carries the page
marker. Three styles occur: `X_page1.png` (1-based), `X_page_0.jpg` (0-based),
`X_page-0001.jpg` (1-based). No marker → single-page CV.

Duplicates (verified 2026-10-08): the source holds the same page uploaded more
than once — under the same name, under different names, and sometimes once at
full size and once already shrunk to 640×640. Pages are compared with a 96×96
grayscale fingerprint; Pearson correlation ≥ DUP_CORR means "same page" (all 486
same-name duplicate pairs scored > 0.99; distinct same-template CVs scored
0.97–0.99). One copy is kept per duplicate cluster, preferring the largest image.

Coordinate space: COCO bboxes are absolute pixels on each image's own canvas
(`width` × `height`, which varies per page). Passed through unchanged;
`schema.record_to_doctags(page_size=(width, height))` converts to loc tokens.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DIR = DATA_DIR / "roboflow_source"
OUTPUT_PATH = DATA_DIR / "processed" / "documents.jsonl"

SPLITS = ("train", "valid", "test")
DUP_CORR = 0.99
FP_SIZE = 96
SHRUNK_SIZE = (640, 640)  # pages the dataset owner uploaded already resized

# (regex, offset added to the captured number to make it 1-based)
_PAGE_MARKERS = (
    (re.compile(r"^(?P<cv>.+?)_page_(?P<n>\d+)\.[^.]+$", re.I), 1),
    (re.compile(r"^(?P<cv>.+?)_page-(?P<n>\d+)\.[^.]+$", re.I), 0),
    (re.compile(r"^(?P<cv>.+?)_page(?P<n>\d+)\.[^.]+$", re.I), 0),
)


def parse_upload_name(name: str) -> tuple[str, int]:
    """Upload name -> (cv_key, 1-based page number)."""
    for pattern, offset in _PAGE_MARKERS:
        m = pattern.match(name)
        if m:
            return m.group("cv"), int(m.group("n")) + offset
    return name.rsplit(".", 1)[0], 1


def load_images() -> list[dict]:
    """All images across splits, with boxes as (class_name, [x,y,w,h]) in pixels.

    Junk classes (`words-pPx7`, `Resume`) are KEPT; filtering happens in
    `schema.coco_annotations_to_field_bboxes`.
    """
    images = []
    for split in SPLITS:
        coco = json.loads((SOURCE_DIR / split / "_annotations.coco.json").read_text())
        cat_map = {c["id"]: c["name"] for c in coco["categories"]}
        boxes_by_image: dict[int, list] = defaultdict(list)
        for ann in coco["annotations"]:
            boxes_by_image[ann["image_id"]].append((cat_map[ann["category_id"]], ann["bbox"]))
        for im in coco["images"]:
            cv_key, page_no = parse_upload_name(im["extra"]["name"])
            images.append({
                "upload_name": im["extra"]["name"],
                "cv_key": cv_key,
                "page": page_no,
                "image_path": f"roboflow_source/{split}/{im['file_name']}",
                "split": split,
                "image_id": im["id"],
                "width": im["width"],
                "height": im["height"],
                "full_size": (im["width"], im["height"]) != SHRUNK_SIZE,
                "boxes": boxes_by_image.get(im["id"], []),
            })
    return images


def duplicate_clusters(images: list[dict]) -> list[int]:
    """Cluster id per image; images in one cluster are the same page."""
    fps = np.stack([
        np.asarray(Image.open(DATA_DIR / im["image_path"]).convert("L")
                   .resize((FP_SIZE, FP_SIZE), Image.BOX), np.float32).ravel()
        for im in images
    ])
    fps -= fps.mean(1, keepdims=True)
    fps /= np.linalg.norm(fps, axis=1, keepdims=True) + 1e-6
    corr = fps @ fps.T

    parent = list(range(len(images)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(*np.where(np.triu(corr, 1) >= DUP_CORR)):
        parent[find(a)] = find(b)
    return [find(i) for i in range(len(images))]


def _best(copies: list[dict]) -> dict:
    """Largest image wins; ties broken by path for determinism."""
    return max(copies, key=lambda im: (im["width"] * im["height"], im["image_path"]))


def reconstruct(images: list[dict], clusters: list[int]) -> tuple[list[dict], Counter]:
    stats: Counter = Counter()
    for im, c in zip(images, clusters):
        im["cluster"] = c

    # Group by cv_key, then by page number. Same page number with different
    # content = two different CVs that share an upload name.
    by_cv: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for im in images:
        by_cv[im["cv_key"]][im["page"]].append(im)

    docs = []
    for cv_key, pages in sorted(by_cv.items()):
        variants = {p: {} for p in pages}  # page -> cluster -> copies
        for p, ims in pages.items():
            for im in ims:
                variants[p].setdefault(im["cluster"], []).append(im)
        n_variants = max(len(v) for v in variants.values())
        if n_variants > 1 and len(pages) > 1:
            # Multi-page CV whose name is shared by different content: page order
            # cannot be recovered reliably. Drop rather than guess.
            stats["dropped_ambiguous_multipage_name"] += 1
            continue
        for k in range(n_variants):
            chosen = []
            for p in sorted(pages):
                copies = list(variants[p].values())[k] if n_variants > 1 else next(iter(variants[p].values()))
                chosen.append((p, copies))
            cv_id = cv_key if n_variants == 1 else f"{cv_key}__v{k + 1}"
            pages_out, prev = [], 0
            for p, copies in chosen:
                if p != prev + 1:
                    stats["missing_interior_pages"] += 1
                prev = p
                best = _best(copies)
                stats["dropped_same_name_copies"] += len(copies) - 1
                pages_out.append({
                    "page": p,
                    **{k2: best[k2] for k2 in ("image_path", "upload_name", "split", "image_id",
                                               "width", "height", "full_size", "boxes", "cluster")},
                })
            docs.append({"cv_id": cv_id, "num_pages": len(pages_out),
                         "multipage": len(pages_out) > 1, "pages": pages_out})

    # Cross-name duplicates: a CV whose every page is a duplicate of a page in an
    # earlier-kept CV of the same length is the same CV under another name.
    docs.sort(key=lambda d: (-sum(p["width"] * p["height"] for p in d["pages"]), d["cv_id"]))
    kept, seen = [], {}
    for d in docs:
        sig = tuple(p["cluster"] for p in d["pages"])
        if sig in seen:
            stats["dropped_cross_name_duplicate_cvs"] += 1
            continue
        seen[sig] = d["cv_id"]
        kept.append(d)

    # Partial overlaps (a page shared between two kept CVs) are a train/eval leak
    # risk: mark them so the gold set can exclude them.
    owners: dict[int, set] = defaultdict(set)
    for d in kept:
        for p in d["pages"]:
            owners[p["cluster"]].add(d["cv_id"])
    for d in kept:
        d["shares_page_with"] = sorted(
            {o for p in d["pages"] for o in owners[p["cluster"]]} - {d["cv_id"]})
        d["all_full_size"] = all(p["full_size"] for p in d["pages"])
        for p in d["pages"]:
            del p["cluster"]
    kept.sort(key=lambda d: d["cv_id"])
    return kept, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconstruct per-CV document records.")
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()

    images = load_images()
    print(f"Loaded {len(images)} images from {SOURCE_DIR}")
    print("Fingerprinting pages for duplicate detection...")
    clusters = duplicate_clusters(images)
    print(f"  {len(set(clusters))} distinct pages")

    records, stats = reconstruct(images, clusters)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    pages = [p for r in records for p in r["pages"]]
    print(f"\nWrote {len(records)} CV records to {args.output}")
    print(f"  pages: {len(pages)}  (full-size {sum(p['full_size'] for p in pages)}, "
          f"640x640 {sum(not p['full_size'] for p in pages)})")
    print(f"  CVs with every page full-size: {sum(r['all_full_size'] for r in records)}")
    print(f"  multi-page CVs: {sum(r['multipage'] for r in records)}")
    print(f"  pages with >=1 box: {sum(1 for p in pages if p['boxes'])}")
    print(f"  CVs sharing a page with another CV: {sum(1 for r in records if r['shares_page_with'])}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
