"""Key-by-key mode (--keywise): our code writes the JSON keys, the model writes one key's whole value per call.

29 keys in template() order (corrections skipped): the 17 keys of basics, then the 12 other sections. A value is written
in full by the model, quotes and brackets included, or `null` when the page has none. The prefix of key i holds every
earlier key with its value (empty ones as null), so the model sees what it already wrote.
"""

from __future__ import annotations

import json

from .models import STUDENT_SKIP as SKIP
from .schema import _drop_empty, template, to_record

KEYS = [("basics", k) for k in template()["basics"]] + [(k,) for k in template() if k not in ("basics", *SKIP)]


def dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def value_form(v) -> str:
    """The value text the model learns: today's compact form (empty inner keys dropped), or null when empty."""
    v = _drop_empty(v)
    return "null" if v in (None, [], {}, "") else dumps(v)


def values(record: dict) -> list[str]:
    """The 29 true value texts of a labelled page, in KEYS order."""
    r = to_record(record)
    return [value_form(r[key[0]] if len(key) == 1 else r[key[0]][key[1]]) for key in KEYS]


def prefix(vals: list[str], i: int) -> str:
    """JSON text from `{` up to and including key i's `"key":`, holding the values of keys 0..i-1."""
    parts, in_basics = ["{"], False
    for j, key in enumerate(KEYS[: i + 1]):
        if key[0] == "basics" and not in_basics:
            parts.append('"basics":{')
            in_basics = True
        elif key[0] == "basics":
            parts.append(",")
        elif in_basics:  # first section after basics closes it
            parts.append("},")
            in_basics = False
        else:
            parts.append(",")
        parts.append(dumps(key[-1]) + ":")
        if j < i:
            parts.append(vals[j])
    return "".join(parts)


def assemble(vals: list[str]) -> str:
    """The full JSON text from all 29 value texts (null values included, as the model saw them)."""
    return prefix(vals, len(KEYS) - 1) + vals[-1] + "}"


def answer(vals: list[str]) -> str:
    """The stored answer: today's whole-page form, null keys dropped (the schema rejects null for list sections)."""
    return dumps(_drop_empty(json.loads(assemble(vals))))
