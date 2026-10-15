"""M3: Stratified gold-set sampling + train/gold split.

Produces a held-out gold-200 set (150 single-page + 50 multi-page CVs) reserved
from training, and writes:
  - data/processed/gold_200/gold_ids.json    — the 200 cv_ids (the split contract)
  - data/processed/gold_200/manifest.json     — per-CV page paths for the reviewer
  - data/processed/train_pool_ids.json        — the ~1,980 cv_ids NOT in gold

The gold labels themselves are produced in M4 (VLM auto-labeling) and verified by
the user (the "VLM-label then I correct" choice): every gold CV is labeled by
Gemini Pro, then the user corrects the labels via a review doc. So this script
only fixes the *membership* of the gold set, not its labels.

Eligibility (2026-10-08): gold CVs must look like real-use input, so only CVs
with every page at full size (not the 640×640 pages the dataset owner uploaded
pre-shrunk), no missing pages, and no page shared with another CV (leak risk)
are sampled. Every other CV goes to the train pool.

Sampling is seeded for reproducibility.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"
GOLD_DIR = PROJECT_ROOT / "data" / "processed" / "gold_200"
TRAIN_POOL_PATH = PROJECT_ROOT / "data" / "processed" / "train_pool_ids.json"

N_SINGLE = 150
N_MULTI = 50
SEED = 42


def is_gold_eligible(doc: dict) -> bool:
    complete = [p["page"] for p in doc["pages"]] == list(range(1, doc["num_pages"] + 1))
    return doc["all_full_size"] and complete and not doc["shares_page_with"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Sample the gold-200 evaluation set.")
    ap.add_argument("--documents", type=Path, default=DOCUMENTS_PATH)
    ap.add_argument("--gold-dir", type=Path, default=GOLD_DIR)
    ap.add_argument("--n-single", type=int, default=N_SINGLE)
    ap.add_argument("--n-multi", type=int, default=N_MULTI)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    docs = [json.loads(line) for line in args.documents.read_text().splitlines()]
    eligible = [d for d in docs if is_gold_eligible(d)]
    single = [d for d in eligible if not d["multipage"]]
    multi = [d for d in eligible if d["multipage"]]
    print(f"Gold-eligible pool: {len(single)} single-page, {len(multi)} multi-page CVs "
          f"(of {len(docs)} total)")

    if args.n_single > len(single):
        print(f"⚠ requested {args.n_single} single but only {len(single)} available")
    if args.n_multi > len(multi):
        print(f"⚠ requested {args.n_multi} multi but only {len(multi)} available "
              f"(multi-page coverage in gold will be lower)")

    rng = random.Random(args.seed)
    gold_single = rng.sample(single, min(args.n_single, len(single)))
    gold_multi = rng.sample(multi, min(args.n_multi, len(multi)))
    gold = gold_single + gold_multi

    gold_ids = {d["cv_id"] for d in gold}
    train_pool_ids = [d["cv_id"] for d in docs if d["cv_id"] not in gold_ids]

    # Sanity: no overlap
    assert not (gold_ids & set(train_pool_ids)), "gold/train overlap!"

    args.gold_dir.mkdir(parents=True, exist_ok=True)

    # 1) gold_ids.json — the membership contract
    (args.gold_dir / "gold_ids.json").write_text(
        json.dumps(sorted(gold_ids), indent=2)
    )

    # 2) gold manifest — page paths + box summary for the reviewer / M4
    gold_manifest = {
        "n_single": len(gold_single),
        "n_multi": len(gold_multi),
        "n_total": len(gold),
        "cvs": [
            {
                "cv_id": d["cv_id"],
                "num_pages": d["num_pages"],
                "multipage": d["multipage"],
                "page_count_present": len(d["pages"]),
                "pages": [
                    {
                        "page": p["page"],
                        "image_path": p["image_path"],
                        "split": p["split"],
                        "n_boxes": len(p["boxes"]),
                    }
                    for p in d["pages"]
                ],
            }
            for d in gold
        ],
    }
    (args.gold_dir / "manifest.json").write_text(json.dumps(gold_manifest, indent=2))

    # 3) train pool
    args.gold_dir.parent.joinpath("train_pool_ids.json").write_text(
        json.dumps(train_pool_ids, indent=2)
    )

    print(f"\nGold set: {len(gold)} CVs ({len(gold_single)} single + {len(gold_multi)} multi)")
    print(f"  -> {args.gold_dir / 'gold_ids.json'}")
    print(f"  -> {args.gold_dir / 'manifest.json'}")
    print(f"Train pool: {len(train_pool_ids)} CVs")
    print(f"  -> {args.gold_dir.parent / 'train_pool_ids.json'}")

    # Multi-page page-count distribution in gold (for eval-strata awareness)
    from collections import Counter
    multi_dist = Counter(d["num_pages"] for d in gold_multi)
    print(f"\nGold multi-page distribution: {dict(sorted(multi_dist.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
