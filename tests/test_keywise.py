"""Key-by-key mode (tiny_resume_parser/keywise.py) must lose nothing: for every labelled page, the 29 value texts re-assemble into an
answer with the same scored fields as the label, and every prefix is an exact text prefix of the assembled JSON
(so training prompts equal the prompts generation builds). Run: pytest, or python tests/test_keywise.py
The full check needs data/processed/ (not in the repository); without it only the hand-made record is checked."""

import json

import pytest

from tiny_resume_parser import keywise
from tiny_resume_parser.data import GOLD, LABELS
from tiny_resume_parser.metrics import pairs
from tiny_resume_parser.schema import template, to_record

SAMPLE = {
    "basics": {"name": "Ada Lovelace", "emails": ["ada@example.com"]},
    "work": [
        {
            "name": "Analytical Engines Ltd",
            "position": "Programmer",
            "startDate": "1843-01",
            "highlights": ["Wrote the first program"],
        }
    ],
    "skills": [{"name": "Mathematics", "keywords": ["Calculus", "Algebra"]}],
    "languages": [{"language": "English", "fluency": "Native"}],
}
LABEL_FILES = [LABELS, GOLD]


def check(record: dict) -> int:
    """Assert the round trip for one record; return its number of null values."""
    vals = keywise.values(record)
    full = keywise.assemble(vals)
    for i, v in enumerate(vals):
        assert full.startswith(keywise.prefix(vals, i) + v), i
    assert pairs(to_record(json.loads(keywise.answer(vals)))) == pairs(to_record(record))
    return vals.count("null")


def test_key_order():
    assert len(keywise.KEYS) == 29 and keywise.KEYS[0] == ("basics", "name") and ("corrections",) not in keywise.KEYS
    assert [k[0] for k in keywise.KEYS[17:]] == [k for k in template() if k not in ("basics", "corrections")]


def test_round_trip_sample():
    check(to_record(SAMPLE))


def test_round_trip_all_labels():
    if not all(f.exists() for f in LABEL_FILES):
        pytest.skip("data/processed/ label files not present (not in the repository)")
    pages = nulls = 0
    for f in LABEL_FILES:
        for line in f.read_text().splitlines():
            nulls += check(json.loads(line)["record"])
            pages += 1
    print(f"ok: {pages} pages, {29 * pages} values, {nulls / (29 * pages):.1%} null")


if __name__ == "__main__":
    test_key_order()
    test_round_trip_sample()
    test_round_trip_all_labels()
