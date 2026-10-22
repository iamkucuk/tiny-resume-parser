"""Does a dedicated page reader (nvidia/Nemotron-3.5-Parse, 0.63B, page -> Markdown) read CV pages better than ResuVL-L?

Field F1 cannot be computed for Nemotron Parse, because it writes Markdown, not our JSON. This script measures a
weaker thing that both models can be scored on: how much of the label's text appears anywhere in the model's
output, ignoring which field it was placed in. It is an upper bound on what a second step (Markdown -> JSON) could
recover. Fields the labelling model rewrites instead of copying (ISO dates, true/false, fixed categories) are left
out, because a page reader cannot write them and ResuVL-L was trained to. Two counts:
  values  a label value counts as found if it appears in the output, compared as in lenient field F1 (casefolded,
          letters and digits only, spaces ignored, so letter-spaced headings still match);
  words   label words matched against output words as a multiset, so one misread character costs one word, not a
          whole paragraph.
    # on a GPU node, with transformers >= 5.5 on PYTHONPATH:
    python experiments/nemotron_parse.py read --out runs/NP_dev
    # anywhere:
    python experiments/nemotron_parse.py coverage runs/NP_dev/pages.jsonl \
        runs/Q35_soup3/eval_final_gold/pages.jsonl runs/Q35_soup3/eval_final_val/pages.jsonl

Reads the development pages (gold 1-50 and the validation split); writes out/pages.jsonl with one line per page:
cv_id, page, set, the raw output, tokens and seconds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

from tiny_resume_parser.data import label_index, load_rows, page_image, read_jsonl
from tiny_resume_parser.metrics import pairs, read_answer
from tiny_resume_parser.schema import to_record

MODEL = "nvidia/Nemotron-3.5-Parse"
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"  # the model card's default


def norm(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", text).casefold() if ch.isalnum())


def read(out: Path, limit: int | None) -> None:
    import torch
    from transformers import AutoModel, AutoProcessor, GenerationConfig

    rows = [r | {"set": "gold"} for r in load_rows("gold", 50)] + [r | {"set": "val"} for r in load_rows("val")]
    rows = rows[:limit] if limit else rows
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    config = GenerationConfig.from_pretrained(MODEL)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "pages.jsonl").open("w") as f:
        for r in rows:
            t0 = time.time()
            inputs = processor(images=[page_image(r)], text=PROMPT, return_tensors="pt", add_special_tokens=False)
            with torch.inference_mode():
                ids = model.generate(**inputs.to("cuda"), generation_config=config)
            text = processor.batch_decode(ids, skip_special_tokens=False)[0]
            line = {"cv_id": r["cv_id"], "page": r["page"], "set": r["set"], "tokens": ids.shape[1]}
            f.write(json.dumps(line | {"seconds": round(time.time() - t0, 1), "text": text}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{r['set']} {r['cv_id'][:30]} p{r['page']}: {line['tokens']} tokens, {time.time() - t0:.1f} s")


def output_text(line: dict) -> str:
    """The text a page's output contains: Nemotron Parse without its box and class tags, ResuVL-L's JSON values."""
    if "text" in line:
        return re.sub(r"<[^>]*>", " ", line["text"])
    return " ".join(value for (_, value) in pairs(read_answer(line["answer"], fix=True)[0]))


REWRITTEN = {"birthDate", "startDate", "endDate", "date", "releaseDate", "hasPhoto", "type", "kind"}


def words(text: str) -> Counter:
    return Counter(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def coverage(files: list[Path]) -> None:
    labels = label_index()
    print("output file | set | pages | label values found | label words found")
    for path in files:
        by_set: dict[str, Counter] = {}
        for line in read_jsonl(path):
            text = output_text(line)
            joined = norm(text)
            record = to_record(labels[(line["cv_id"], line["page"])])
            values = [v for (key, v) in pairs(record) if key.rsplit(".", 1)[-1] not in REWRITTEN and norm(v)]
            label_words = words(" ".join(values))
            s = line.get("set") or ("val" if path.parent.name.endswith("val") else "gold")  # eval_*_gold or eval_*_val
            n = by_set.setdefault(s, Counter())
            n.update(
                pages=1,
                hit=sum(norm(v) in joined for v in values),
                total=len(values),
                whit=sum((label_words & words(text)).values()),
                wtotal=sum(label_words.values()),
            )
        for s, n in sorted(by_set.items()):
            print(
                f"{path} | {s} | {n['pages']} | {n['hit'] / n['total']:.3f} ({n['hit']}/{n['total']}) | "
                f"{n['whit'] / n['wtotal']:.3f} ({n['whit']}/{n['wtotal']})"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--limit", type=int)
    c = sub.add_parser("coverage")
    c.add_argument("files", type=Path, nargs="+")
    args = ap.parse_args(sys.argv[1:])
    read(args.out, args.limit) if args.cmd == "read" else coverage(args.files)
