"""Label voting pilot: do labels chosen from several teacher runs agree better with the reviewed gold than one run?

    python experiments/label_vote.py runs/label_ceiling/glm_rerun.jsonl runs/label_vote/run3.jsonl ... [--pages 50]

Each input is one full teacher run (label.py output) over the CVs behind the first --pages reviewed gold pages; the
original labelling run is not an input, because the reviewed gold was made from it. Per page:
  single   each run on its own (mean over runs): how well one teacher draw agrees with the reviewed gold
  medoid   the run with the highest mean lenient field F1 against the other runs: the most typical draw, a whole
           valid record that can be used as a training label
  pairs    (path, value) pairs found in at least half of the runs: a diagnostic upper reference, not a record
All scores are lenient field F1 pooled over pages (metrics.py), against labels_gold_v6_verified.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tiny_resume_parser.data import load_rows, read_jsonl  # noqa: E402
from tiny_resume_parser.metrics import f1, lenient, pairs  # noqa: E402
from tiny_resume_parser.schema import to_record  # noqa: E402


def lpairs(record: dict) -> Counter:
    return lenient(pairs(to_record(record)))


def page_f1(a: Counter, b: Counter) -> float:
    return f1(sum((a & b).values()), sum(a.values()), sum(b.values()))[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", type=Path, nargs="+", help="label.py outputs, one per teacher run")
    ap.add_argument("--pages", type=int, default=50, help="first N reviewed gold pages (the development pages)")
    ap.add_argument("--out", type=Path, help="write per-page choices and the summary as JSON")
    args = ap.parse_args()
    if len(args.runs) < 3:
        ap.error("give at least 3 runs: with 2, both runs agree with each other equally and no medoid exists")

    gold = load_rows("gold")[: args.pages]
    runs = []
    for path in args.runs:
        last = {(r["cv_id"], r["page"]): r for r in read_jsonl(path)}  # readers take the last line per page
        runs.append({k: r["record"] for k, r in last.items() if r["status"] == "ok"})
    keep = [g for g in gold if all((g["cv_id"], g["page"]) in run for run in runs)]
    print(f"{len(runs)} runs; {len(keep)} of {len(gold)} gold pages labelled ok in every run")

    tot = {name: Counter() for name in ("single", "medoid", "pairs")}
    choices = []
    for g in keep:
        key = (g["cv_id"], g["page"])
        gp = lpairs(g["record"])
        rp = [lpairs(run[key]) for run in runs]
        agree = [sum(page_f1(rp[i], rp[j]) for j in range(len(rp)) if j != i) / (len(rp) - 1) for i in range(len(rp))]
        best = max(range(len(rp)), key=agree.__getitem__)
        count = Counter()
        for p in rp:
            count.update(set(p))  # in how many runs each pair occurs (once per run)
        voted = Counter(
            {pair: max(p[pair] for p in rp) for pair, n in count.items() if n * 2 >= len(rp)}
        )  # half or more
        for name, preds in (("single", rp), ("medoid", [rp[best]]), ("pairs", [voted])):
            for p in preds:
                tot[name].update(
                    hit=sum((p & gp).values()) / len(preds),
                    pred=sum(p.values()) / len(preds),
                    gold=sum(gp.values()) / len(preds),
                )
        choices.append(dict(page=f"{key[0]} p{key[1]}", medoid=best, agreement=[round(a, 3) for a in agree]))
    summary = {
        name: dict(zip(("P", "R", "F1"), (round(x, 4) for x in f1(c["hit"], c["pred"], c["gold"]))))
        for name, c in tot.items()
    }
    summary = dict(runs=[str(p) for p in args.runs], pages=len(keep)) | summary
    print(json.dumps(summary))
    if args.out:
        args.out.write_text(json.dumps(dict(summary=summary, pages=choices), indent=1))


if __name__ == "__main__":
    main()
