"""Field-level scores of a model answer against a label; every number in the README comes from here.

A record is flattened into (path, value) pairs, list positions ignored (Donut-style field F1, arXiv 2111.15664):
    {"skills": [{"name": "SQL"}]}  ->  ("skills.name", "SQL")
strict   the pair must match exactly.
lenient  pairs compared after casefolding and dropping everything that is not a letter or digit, so "NVIDIA" =
         "Nvidia" and "Sfax, Tunisie" = "Sfax Tunisie", but "09.25" != "09.20". Accents are kept (e != é).
content  the words of all values in one top-level section, compared as a multiset: how text is split into values
         and which field of the section holds it do not matter; a word with any letter or digit difference is a miss.
repair   (optional, before scoring) a broken answer keeps its complete values up to the break and duplicate list
         items are dropped; valid answers are not changed.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from .canon import canon
from .models import STUDENT_SKIP
from .schema import _drop_empty, to_record


def pairs(record: dict) -> Counter:
    """Multiset of (path, value) leaves, list positions ignored, empty values and STUDENT_SKIP keys left out."""
    out = Counter()

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(x, list):
            for v in x:
                walk(v, path)
        else:
            out[(path, str(x).strip())] += 1

    walk(_drop_empty({k: v for k, v in record.items() if k not in STUDENT_SKIP}), "")
    return out


def score(answer: str, gold: dict) -> dict:
    """status: valid (passes the schema), schema_error (JSON but not the schema; still scored), unparsed (scored as 0)."""
    try:
        pred = json.loads(answer)
    except json.JSONDecodeError:
        pred, status = {}, "unparsed"
    else:
        try:
            pred, status = to_record(pred), "valid"
        except Exception:
            status = "schema_error"
        if not isinstance(pred, dict):
            pred, status = {}, "unparsed"
    p, g = pairs(pred), pairs(to_record(gold))
    hit = sum((p & g).values())
    return {"status": status, "pred": sum(p.values()), "gold": sum(g.values()), "hit": hit}


def lenient(c: Counter) -> Counter:
    """Pairs with values casefolded and reduced to letters and digits."""
    out = Counter()
    for (path, value), n in c.items():
        v = "".join(ch for ch in unicodedata.normalize("NFKC", value).casefold() if ch.isalnum())
        out[(path, v)] += n
    return out


def dedup(x):
    """Drop exact duplicate items from every list (a repetition loop writes the same entry again and again)."""
    if isinstance(x, dict):
        return {k: dedup(v) for k, v in x.items()}
    if isinstance(x, list):
        seen, out = set(), []
        for v in x:
            k = json.dumps(v, sort_keys=True)
            if k not in seen:
                seen.add(k)
                out.append(dedup(v))
        return out
    return x


def repair(answer: str):
    """Best-effort JSON from a broken answer: fix invalid escapes; else keep the complete values before the cut-off end
    or, failing that, before the first syntax error, and close the brackets. Duplicate list items are dropped."""
    a = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", answer)
    try:
        return dedup(json.loads(a))
    except json.JSONDecodeError as e:
        pos = e.pos
    p = salvage(a)
    return dedup(p) if p is not None else (dedup(q) if (q := salvage(a[:pos])) is not None else None)


def salvage(a: str):
    """Keep the values that are complete before the end of a, close the open brackets, parse."""
    stack, in_str, esc, last_ok = [], False, False, None  # last_ok: (cut index after a complete value, open brackets)
    for i, ch in enumerate(a):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            last_ok = (i + 1, list(stack))
        elif ch == ",":
            last_ok = (i, list(stack))
    if not last_ok:
        return None
    cut, open_ = last_ok
    try:
        return json.loads(a[:cut].rstrip().rstrip(",") + "".join("}" if c == "{" else "]" for c in reversed(open_)))
    except json.JSONDecodeError:
        return None


def content(c: Counter) -> Counter:
    """(top-level section, word) multiset: the words a section contains, wherever they were placed."""
    out = Counter()
    for (path, value), n in c.items():
        for w in re.split(r"[^\w]|_", unicodedata.normalize("NFKC", value).casefold()):
            if w:
                out[(path.split(".")[0], w)] += n
    return out


def parse(answer: str, fix: bool = False) -> dict:
    """Answer text -> record dict ({} if it does not parse); with fix, broken answers are repaired first."""
    try:
        pred = json.loads(answer)
    except json.JSONDecodeError:
        pred = repair(answer) if fix else None
    if not isinstance(pred, dict):
        return {}
    try:
        return to_record(pred)
    except Exception:
        return pred  # schema error: scored on its raw JSON, as evaluate.score does


def f1(h: int, p: int, g: int) -> tuple[float, float, float]:
    """Precision, recall, F1 from hits, predicted and gold counts."""
    P, R = h / max(p, 1), h / max(g, 1)
    return P, R, 2 * P * R / max(P + R, 1e-9)


def rescore(eval_dir: Path, labels: dict, use_canon: bool = False, fix: bool = False) -> dict:
    """Strict, lenient and content P/R/F1 of every answer in eval_dir/pages.jsonl, overall and per section.
    labels: (cv_id, page) -> record (data.label_index). use_canon: apply canon.py to both sides first."""
    tot = {"strict": Counter(), "lenient": Counter(), "content": Counter()}
    sec = {"strict": Counter(), "lenient": Counter(), "content": Counter()}
    for line in (eval_dir / "pages.jsonl").read_text().splitlines():
        r = json.loads(line)
        g, p = to_record(labels[(r["cv_id"], r["page"])]), parse(r["answer"], fix)
        if use_canon:
            g = canon(g)
            try:
                p = canon(p)
            except (AttributeError, TypeError):  # schema-error answer with odd structure: scored as written
                pass
        g, p = pairs(g), pairs(p)
        for name, (pp, gg) in {
            "strict": (p, g),
            "lenient": (lenient(p), lenient(g)),
            "content": (content(p), content(g)),
        }.items():
            hit = pp & gg
            tot[name].update({"hit": sum(hit.values()), "pred": sum(pp.values()), "gold": sum(gg.values())})
            for c, key in ((hit, "hit"), (pp, "pred"), (gg, "gold")):
                for (path, _), n in c.items():
                    sec[name][(path.split(".")[0], key)] += n
    out = {}
    for name in tot:
        P, R, F = f1(tot[name]["hit"], tot[name]["pred"], tot[name]["gold"])
        out[name] = {"P": round(P, 4), "R": round(R, 4), "F1": round(F, 4)}
    sections = sorted({s for s, _ in sec["strict"]})
    out["sections"] = {
        s: {
            name: round(f1(sec[name][(s, "hit")], sec[name][(s, "pred")], sec[name][(s, "gold")])[2], 3) for name in sec
        }
        | {"gold": sec["strict"][(s, "gold")]}
        for s in sections
    }
    return out
