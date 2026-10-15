"""Classify every unmatched (path, value) pair of one scored run into error types. Usage: analyze.py <run eval dir> <out.json>"""

import collections
import json
import re
import sys
from difflib import SequenceMatcher

from tiny_resume_parser import data as ev
from tiny_resume_parser.metrics import pairs
from tiny_resume_parser.schema import to_record

run, out = sys.argv[1], sys.argv[2]
gold = {(r["cv_id"], r["page"]): r for r in ev.load_rows("gold", None)}
pages = [json.loads(l) for l in open(f"{run}/pages.jsonl")]
sim = lambda a, b: SequenceMatcher(None, a.lower(), b.lower()).ratio()
norm = lambda s: re.sub(r"\W+", "", s.lower())

res = []
for p in pages:
    g = gold[(p["cv_id"], p["page"])]
    gp = pairs(to_record(g["record"]))
    err = None
    try:
        raw = json.loads(p["answer"])
        try:
            pred = to_record(raw)
        except Exception as e:
            pred, err = raw, str(e).splitlines()[0:3]
    except json.JSONDecodeError as e:
        pred, err = {}, f"JSON error: {e}"
    pp = pairs(pred) if isinstance(pred, dict) else collections.Counter()
    hit = pp & gp
    miss_g = list((gp - hit).elements())
    miss_p = list((pp - hit).elements())
    items = []
    # 1 format only: same path, same text after removing punctuation/case
    for path, v in list(miss_p):
        m = [(gpth, gv) for (gpth, gv) in miss_g if gpth == path and norm(gv) == norm(v)]
        if m:
            items.append({"type": "format", "path": path, "pred": v, "gold": m[0][1]})
            miss_p.remove((path, v))
            miss_g.remove(m[0])
    # 1 near miss: same path, similar text
    for path, v in list(miss_p):
        cands = [(sim(v, gv), (gpth, gv)) for (gpth, gv) in miss_g if gpth == path]
        if cands:
            s, best = max(cands)
            if s >= 0.8:
                items.append({"type": "misread", "path": path, "pred": v, "gold": best[1], "sim": round(s, 2)})
                miss_p.remove((path, v))
                miss_g.remove(best)
    # 2 wrong place: same text (normalized) under a different path
    for path, v in list(miss_p):
        m = [(gpth, gv) for (gpth, gv) in miss_g if norm(gv) == norm(v) and norm(v)]
        if m:
            items.append({"type": "wrong_key", "path": path, "pred": v, "gold_path": m[0][0]})
            miss_p.remove((path, v))
            miss_g.remove(m[0])
    gold_text = " ".join(norm(gv) for _, gv in gp.elements())
    for path, v in miss_p:
        if (path, v) in gp:
            items.append({"type": "duplicate", "path": path, "pred": v})
            continue
        items.append(
            {"type": "extra_on_page" if norm(v) and norm(v) in gold_text else "extra", "path": path, "pred": v}
        )
    for path, v in miss_g:
        items.append({"type": "missing", "path": path, "gold": v})
    res.append(
        {
            "cv_id": p["cv_id"],
            "page": p["page"],
            "image": g["image"],
            "status": p["status"],
            "tokens": p["tokens"],
            "hit_limit": p["hit_limit"],
            "err": err,
            "pred_n": sum(pp.values()),
            "gold_n": sum(gp.values()),
            "hit_n": sum(hit.values()),
            "items": items,
            "answer_tail": p["answer"][-600:],
            "answer": p["answer"],
            "gold_record": to_record(g["record"]),
            "entries": {
                k: [
                    len(g["record"].get(k) or []),
                    len(pred.get(k) or []) if isinstance(pred, dict) and isinstance(pred.get(k), list) else 0,
                ]
                for k in ("work", "education", "projects", "certificates", "volunteer", "awards")
            },
            "hits": [list(k) for k in hit.elements()],
        }
    )
json.dump(res, open(out, "w"), ensure_ascii=False)
c = collections.Counter(i["type"] for r in res for i in r["items"])
print(run, "pages", len(res), dict(c), "hits", sum(r["hit_n"] for r in res), "gold", sum(r["gold_n"] for r in res))
