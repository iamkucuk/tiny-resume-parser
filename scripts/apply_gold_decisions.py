"""Apply the reviewed gold-label fixes and write a separate verified gold file.

    python scripts/apply_gold_decisions.py

Reads data/processed/labels_gold_v6.jsonl (left unchanged) and
data/processed/gold_v6_check/decisions.json (one entry per accepted Haiku finding, each
checked on the page image or its Tesseract text), writes labels_gold_v6_verified.jsonl.
Each decision names a page id prefix (`<cv_id>_p<page>`), a path like "work[0].name" and one of:
  set: [old, new]        the value must equal old
  sub: [old, new]        old must occur exactly once in the value
  set_list: [n, new]     the list must have n items
  append: item           added to the list
Any mismatch stops the run: a decision written against a different label is an error.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tiny_resume_parser.schema import to_record

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "processed" / "labels_gold_v6.jsonl"
DECISIONS = ROOT / "data" / "processed" / "gold_v6_check" / "decisions.json"
OUT = ROOT / "data" / "processed" / "labels_gold_v6_verified.jsonl"


def split_path(path: str) -> list:
    return [int(k) if k.isdigit() else k for k in re.findall(r"[^.\[\]]+", path)]


def apply(record: dict, d: dict) -> None:
    *parents, last = split_path(d["path"])
    node = record
    for k in parents:
        node = node[k]
    value = node[last]
    if "set" in d:
        old, new = d["set"]
        if value != old:
            raise ValueError(f"{d['page']} {d['path']}: expected {old!r}, found {value!r}")
        node[last] = new
    elif "sub" in d:
        old, new = d["sub"]
        if value.count(old) != 1:
            raise ValueError(f"{d['page']} {d['path']}: {old!r} occurs {value.count(old)} times in {value!r}")
        node[last] = value.replace(old, new)
    elif "set_list" in d:
        n, new = d["set_list"]
        if len(value) != n:
            raise ValueError(f"{d['page']} {d['path']}: expected {n} items, found {len(value)}")
        node[last] = new
    elif "append" in d:
        value.append(d["append"])
    else:
        raise ValueError(f"unknown decision {d}")


def main() -> int:
    rows = [json.loads(line) for line in GOLD.read_text().splitlines()]
    by_id = {f"{r['cv_id']}_p{r['page']}": r for r in rows}
    decisions = json.loads(DECISIONS.read_text())
    changed, errors = set(), []
    for d in decisions:
        ids = [k for k in by_id if k.startswith(d["page"])]
        try:
            if len(ids) != 1:
                raise ValueError(f"page prefix {d['page']!r} matches {ids}")
            apply(by_id[ids[0]]["record"], d)
            changed.add(ids[0])
        except (ValueError, KeyError, IndexError) as e:  # list them all, then stop
            errors.append(f"{type(e).__name__}: {e}")
    if errors:
        raise SystemExit("decisions that do not fit the label (nothing written):\n" + "\n".join(errors))
    for r in rows:
        r["record"] = to_record(r["record"])
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(decisions)} fixes on {len(changed)} of {len(rows)} pages -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
