"""Plain-text transcripts of training pages with granite-docling-258M, for a reading stage before JSON training.

    python src/transcribe.py --train-file data/processed/labels_train_v6_hk1k.jsonl --out data/processed/transcripts.jsonl \
        [--shard 0 --shards 4] [--limit 16]

granite-docling writes DocTags (text elements with <loc_N> boxes); to_text() keeps only the text, one element per
line. Validation-split pages are skipped (they are scored), and the script stops if any page belongs to a gold CV.
Output rows: cv_id, page, num_pages, image, transcript, tokens, hit_limit. Already written pages are skipped on rerun.
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from train_lora import ROOT, is_validation

GRANITE_ID = "ibm-granite/granite-docling-258M"
GOLD = ROOT / "data" / "processed" / "labels_gold_v6_verified.jsonl"


def to_text(doctags: str) -> str:
    """DocTags -> plain text: drop location tokens, end each element at its closing tag, drop the other tags."""
    s = re.sub(r"<loc_\d+>", "", doctags)
    s = re.sub(r"</[^>]+>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in s.splitlines())
    return "\n".join(line for line in lines if line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=6000)
    args = ap.parse_args()

    gold_cvs = {json.loads(line)["cv_id"] for line in GOLD.read_text().splitlines()}
    rows = [json.loads(line) for line in args.train_file.read_text().splitlines()]
    rows = [r for r in rows if not is_validation(r["cv_id"])][args.shard::args.shards][:args.limit]
    if any(r["cv_id"] in gold_cvs for r in rows):
        raise SystemExit("a training page belongs to a gold CV: stop")
    done = set()
    if args.out.exists():
        done = {(r["cv_id"], r["page"]) for r in map(json.loads, args.out.read_text().splitlines())}
    rows = [r for r in rows if (r["cv_id"], r["page"]) not in done]
    print(f"{len(rows)} pages to transcribe ({len(done)} already done)")

    processor = AutoProcessor.from_pretrained(GRANITE_ID)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(GRANITE_ID, dtype=torch.bfloat16).cuda().eval()
    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Convert this page to docling."}]}]
    prompt = processor.apply_chat_template(msg, add_generation_prompt=True)
    end_id = processor.tokenizer.convert_tokens_to_ids("<|end_of_text|>")
    t0 = time.time()
    with args.out.open("a") as f:
        for i in range(0, len(rows), args.batch):
            b = rows[i:i + args.batch]
            images = [Image.open(ROOT / "data" / r["image"]).convert("RGB") for r in b]
            inputs = processor(text=[prompt] * len(b), images=[[im] for im in images], padding=True,
                               return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
            for r, ids in zip(b, out[:, inputs["input_ids"].shape[1]:]):
                ids = ids.tolist()
                n = ids.index(end_id) if end_id in ids else len(ids)
                doctags = processor.tokenizer.decode(ids[:n], skip_special_tokens=False)
                f.write(json.dumps({k: r[k] for k in ("cv_id", "page", "num_pages", "image")} |
                                   {"transcript": to_text(doctags), "tokens": n, "hit_limit": n >= args.max_new_tokens},
                                   ensure_ascii=False) + "\n")
            f.flush()
            print(f"{i + len(b)}/{len(rows)} pages, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
