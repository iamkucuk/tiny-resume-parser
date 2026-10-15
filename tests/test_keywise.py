"""Key-by-key mode (src/keywise.py) must lose nothing: for every labelled page, the 29 value texts re-assemble into an
answer with the same scored fields as the label, and every prefix is an exact text prefix of the assembled JSON
(so training prompts equal the prompts generation builds). Run: .venv/bin/python tests/test_keywise.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import keywise  # noqa: E402
from evaluate import pairs  # noqa: E402
from schema import template, to_record  # noqa: E402

assert len(keywise.KEYS) == 29 and keywise.KEYS[0] == ("basics", "name") and ("corrections",) not in keywise.KEYS
assert [k[0] for k in keywise.KEYS[17:]] == [k for k in template() if k not in ("basics", "corrections")]
pages = nulls = 0
for name in ("labels_train_v6.jsonl", "labels_gold_v6_verified.jsonl"):
    for line in (ROOT / "data" / "processed" / name).read_text().splitlines():
        record = json.loads(line)["record"]
        vals = keywise.values(record)
        full = keywise.assemble(vals)
        for i, v in enumerate(vals):
            assert full.startswith(keywise.prefix(vals, i) + v), (name, i)
        assert pairs(to_record(json.loads(keywise.answer(vals)))) == pairs(to_record(record)), name
        pages, nulls = pages + 1, nulls + vals.count("null")
print(f"ok: {pages} pages, {29 * pages} values, {nulls / (29 * pages):.1%} null")
