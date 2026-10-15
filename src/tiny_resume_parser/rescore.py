"""Score saved answers again (pages.jsonl written by evaluate.py) with every metric in metrics.py.

    python -m tiny_resume_parser.rescore runs/NAME/eval_ep5_gold [more eval dirs ...]
    python -m tiny_resume_parser.rescore --repair runs/NAME/eval_ep5_gold   # repaired answers -> rescore_repair.json

Writes rescore[_repair].json into each eval dir and prints strict / lenient / content F1.
Runs are compared raw against raw; repaired scores are reported next to them, never instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import label_index, read_jsonl
from .metrics import rescore


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("eval_dirs", type=Path, nargs="+", help="folders holding pages.jsonl from evaluate.py")
    ap.add_argument(
        "--labels", type=Path, help="label file the answers were scored against (default: all project labels)"
    )
    ap.add_argument("--repair", action="store_true", help="repair broken answers before scoring")
    args = ap.parse_args(argv)
    labels = {(r["cv_id"], r["page"]): r["record"] for r in read_jsonl(args.labels)} if args.labels else label_index()
    suffix = "_repair" * args.repair
    for d in args.eval_dirs:
        res = rescore(d, labels, args.repair)
        (d / f"rescore{suffix}.json").write_text(json.dumps(res, indent=1))
        print(
            f"{d}: "
            + " | ".join(
                f"{m} F1 {res[m]['F1']:.3f} (P {res[m]['P']:.3f} R {res[m]['R']:.3f})"
                for m in ("strict", "lenient", "content")
            )
        )


if __name__ == "__main__":
    main()
