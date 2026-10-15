"""Generate answers for labelled pages and score them (field precision / recall / F1, strict).

    python -m tiny_resume_parser.evaluate --model runs/NAME/model --set gold --limit 50 --max-patches 12 --out runs/NAME/eval_gold
    python -m tiny_resume_parser.evaluate --model runs/NAME/model --set val --device cpu --batch 1 --out runs/NAME/eval_val

Sets: gold = reviewed pages (1-50 development, --start 50 for the 204 test pages); val = validation split of the
training labels; train = training pages; --labels FILE scores the pages of any label file instead. Writes out/pages.jsonl (one line per page: status, counts, answer) and
out/summary.json. Lenient, content and repaired scores: run rescore.py on the same folder.
Status per page: valid (passes the schema), schema_error (JSON but not the schema; still scored), unparsed (scored 0).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from .data import load_rows, page_image, read_jsonl
from .generate import generate_pages, load_model
from .metrics import score
from .models import MAX_PATCHES, MODEL_ID


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=MODEL_ID, help="base model id or a fine-tuned model / checkpoint folder")
    ap.add_argument("--set", choices=["gold", "val", "train"], default="gold")
    ap.add_argument("--labels", type=Path, help="score the pages of this label file instead of --set")
    ap.add_argument("--start", type=int, default=0, help="skip the first N pages of the set (test pages: --start 50)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-patches", type=int, default=MAX_PATCHES, help="image tiles per page, as in training")
    ap.add_argument("--max-new-tokens", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=1, help="pages generated together (left-padded)")
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"], help="default: cuda, else mps, else cpu")
    args = ap.parse_args(argv)

    rows = (read_jsonl(args.labels) if args.labels else load_rows(args.set))[args.start :]
    rows = rows[: args.limit] if args.limit else rows
    model, processor, device = load_model(args.model, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{device}, {args.labels or args.set}: {len(rows)} pages")

    total = Counter()
    with (args.out / "pages.jsonl").open("w") as f:
        for start in range(0, len(rows), args.batch):
            batch = rows[start : start + args.batch]
            t0 = time.time()
            done = generate_pages(
                model, processor, [page_image(r) for r in batch], args.max_patches, args.max_new_tokens
            )
            seconds = round((time.time() - t0) / len(batch))
            for r, (answer, n_new) in zip(batch, done):
                s = score(answer, r["record"]) | {
                    "cv_id": r["cv_id"],
                    "page": r["page"],
                    "tokens": n_new,
                    "hit_limit": n_new >= args.max_new_tokens,
                    "seconds": seconds,
                }
                total.update(
                    {k: s[k] for k in ("pred", "gold", "hit")} | {s["status"]: 1, "hit_limit": int(s["hit_limit"])}
                )
                f.write(json.dumps(s | {"answer": answer}, ensure_ascii=False) + "\n")
                f.flush()
                f1 = 2 * s["hit"] / max(s["pred"] + s["gold"], 1)
                print(
                    f"  {r['cv_id'][:30]} p{r['page']}: {s['status']}, F1 {f1:.2f}, {n_new} tokens, {seconds} s/page",
                    flush=True,
                )

    p, rc = total["hit"] / max(total["pred"], 1), total["hit"] / max(total["gold"], 1)
    summary = {
        "set": str(args.labels) if args.labels else args.set,
        "start": args.start,
        "pages": len(rows),
        "model": args.model,
        "batch": args.batch,
        "max_patches": args.max_patches,
        "valid": total["valid"],
        "schema_error": total["schema_error"],
        "unparsed": total["unparsed"],
        "hit_token_limit": total["hit_limit"],
        "field_precision": round(p, 4),
        "field_recall": round(rc, 4),
        "field_f1": round(2 * p * rc / max(p + rc, 1e-9), 4),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
