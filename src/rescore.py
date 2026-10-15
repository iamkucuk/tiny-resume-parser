"""Rescore saved answers (pages.jsonl from evaluate.py) with strict and lenient field F1, overall and per section.

    .venv/bin/python src/rescore.py [--canon] [--repair] runs/lora_v6/eval_gold_free [more eval dirs ...]

strict  = evaluate.pairs: (path, value) must match exactly.
lenient = same pairs after casefolding and removing everything that is not a letter or digit, so
          "Nvidia"/"NVIDIA" or "Sfax, Tunisie"/"Sfax Tunisie" match, but "09.25"/"09.20" or "Tunisie"/"Tunasie" do not.
          Accents are kept: "Électrique" vs "Electrique" still counts as an error.
content = the words of all values in one top-level section, compared per page as a multiset (casefolded, split
          on everything that is not a letter or digit). How text is split into values, and which field of the same
          section holds it, does not matter; a word with any letter or digit difference is a miss (user, 2026-10-14).
The gold record for each page is looked up by (cv_id, page) in the gold, then the training label file.
--repair: answers that are not valid JSON are repaired before scoring (invalid backslash escapes doubled; an
          answer cut off at the token limit, or broken by a syntax error, keeps its complete values up to the break
          and gets its brackets closed; then exact duplicate list items are dropped, which removes repetition loops).
          Valid answers are not touched. Written to rescore_repair.json, so the default scores stay comparable.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from evaluate import GOLD, pairs
from canon import canon
from schema import to_record
from train_lora import LABELS


def lenient(c: Counter) -> Counter:
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
    a = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', answer)
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
    out = Counter()
    for (path, value), n in c.items():
        for w in re.split(r"[^\w]|_", unicodedata.normalize("NFKC", value).casefold()):
            if w:
                out[(path.split(".")[0], w)] += n
    return out


def parse(answer: str, fix: bool = False) -> dict:
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
    P, R = h / max(p, 1), h / max(g, 1)
    return P, R, 2 * P * R / max(P + R, 1e-9)


def rescore(eval_dir: Path, labels: dict, use_canon: bool = False, fix: bool = False) -> dict:
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
        for name, (pp, gg) in {"strict": (p, g), "lenient": (lenient(p), lenient(g)),
                               "content": (content(p), content(g))}.items():
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
    out["sections"] = {s: {name: round(f1(sec[name][(s, "hit")], sec[name][(s, "pred")], sec[name][(s, "gold")])[2], 3)
                           for name in sec} | {"gold": sec["strict"][(s, "gold")]} for s in sections}
    return out


def main(argv: list[str]) -> None:
    labels = {}
    for path in (LABELS, GOLD):  # gold last: a page in both is scored against the reviewed label
        for line in path.read_text().splitlines():
            r = json.loads(line)
            labels[(r["cv_id"], r["page"])] = r["record"]
    use_canon = "--canon" in argv  # same canon.py rules on answers and gold; written to rescore_canon.json
    fix = "--repair" in argv
    suffix = "_canon" * use_canon + "_repair" * fix
    for d in [a for a in argv if a not in ("--canon", "--repair")]:
        res = rescore(Path(d), labels, use_canon, fix)
        (Path(d) / f"rescore{suffix}.json").write_text(json.dumps(res, indent=1))
        print(f"{d}: strict F1 {res['strict']['F1']:.3f} (P {res['strict']['P']:.3f} R {res['strict']['R']:.3f}) | "
              f"lenient F1 {res['lenient']['F1']:.3f} (P {res['lenient']['P']:.3f} R {res['lenient']['R']:.3f}) | "
              f"content F1 {res['content']['F1']:.3f} (P {res['content']['P']:.3f} R {res['content']['R']:.3f})")


if __name__ == "__main__":
    main(sys.argv[1:])
