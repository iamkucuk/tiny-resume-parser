"""One fixed form for label choices the teacher makes inconsistently (docs/RESEARCH_JOURNAL.md, label ceiling).

Rules (no page text is dropped):
- list fields that hold several short items (skills.keywords, work/projects.technologies, education.courses,
  interests.keywords) are split on , ; | • · into one item each;
- interests become one entry per interest: names and keywords merged, each as {"name": ...};
- a skill group with no keywords is a list of skills, not a group name: its name moves to keywords.
Used to build a training label file (python -m tiny_resume_parser.canon IN OUT) and by rescore.py --canon on answers and gold alike.
"""

from __future__ import annotations

import copy
import json
import re
import sys

SPLIT = re.compile(r"\s*[,;|•·]\s*")


def split_list(xs) -> list[str]:
    return [p.strip() for x in xs or [] if isinstance(x, str) for p in SPLIT.split(x) if p.strip()]


def canon(record: dict) -> dict:
    r = copy.deepcopy(record)
    for s in r.get("skills") or []:
        s["keywords"] = split_list(s.get("keywords"))
        if not s["keywords"] and s.get("name"):
            s["keywords"], s["name"] = split_list([s["name"]]), None
    for sec in ("work", "projects"):
        for e in r.get(sec) or []:
            e["technologies"] = split_list(e.get("technologies"))
    for e in r.get("education") or []:
        e["courses"] = split_list(e.get("courses"))
    if r.get("interests"):
        names = []
        for e in r["interests"]:
            names += ([e["name"]] if e.get("name") else []) + list(e.get("keywords") or [])
        r["interests"] = [{"name": n} for n in split_list(names)]
    return r


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    rows = [json.loads(line) for line in open(src)]
    with open(dst, "w") as f:
        for row in rows:
            row["record"] = canon(row["record"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows -> {dst}")
