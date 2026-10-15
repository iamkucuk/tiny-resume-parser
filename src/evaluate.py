"""Score a saved LoRA adapter (or the base model) on held-out pages with field-level precision/recall/F1.

    .venv/bin/python src/evaluate.py --adapter runs/lora_v2/adapter --set gold --limit 50 --out runs/lora_v2/eval_gold50
    .venv/bin/python src/evaluate.py --adapter runs/lora_v2/adapter --set gold --limit 50 --constrained --out runs/lora_v2/eval_gold50_c
    In a notebook: import evaluate; summary = evaluate.main([...])

Sets: gold = data/processed/labels_gold_v6_verified.jsonl (reviewed, CVs never trained on);
val = the 5% validation split of the training labels; train = the first training pages.
Field score (Donut-style, arXiv 2111.15664): every leaf value of the record becomes a (path, value) pair, list
positions ignored; a pair counts only if both path and value match exactly. Pages whose answer does not parse
score 0 predicted pairs. `corrections` is not scored (left out of the student target, train_lora.STUDENT_SKIP).
--constrained: lm-format-enforcer allows only tokens that keep the answer valid JSON for schema.CVRecord.
--keywise: key-by-key mode (src/keywise.py): one generation per key (29 per page), --max-new-tokens caps each value;
a value that is not valid JSON becomes null and is counted in value_errors.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import keywise
from schema import CVRecord, _drop_empty, to_record
from train_lora import LABELS, MAX_PATCHES, MODEL_ID, ROOT, STUDENT_SKIP, family, is_validation, load_processor, messages

GOLD = ROOT / "data" / "processed" / "labels_gold_v6_verified.jsonl"


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


def generate_keywise(model, processor, text: str, images: list, args) -> list[tuple[str, int, int, int]]:
    """All pages of a batch step through the 29 keys together. Prompt = chat template + generation prompt + the page's
    prefix, tokenized by the processor exactly as train_lora.Collator.keywise_batch does. Returns per page:
    (answer, tokens generated, values that were not valid JSON (key -> text as written), values that hit --max-new-tokens)."""
    vals = [[] for _ in images]
    bad = [{} for _ in images]
    tokens, limits = [0] * len(images), [0] * len(images)
    for i in range(len(keywise.KEYS)):
        prompts = [text + keywise.prefix(v, i) for v in vals]  # v holds the values of keys 0..i-1
        inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt",
                           **family(processor)["size"](args.max_patches)).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
                                 repetition_penalty=args.repetition_penalty, pad_token_id=processor.tokenizer.pad_token_id)
        for j, seq in enumerate(out[:, inputs["input_ids"].shape[1]:]):
            n_new = int((seq != processor.tokenizer.pad_token_id).sum())
            value = processor.tokenizer.decode(seq, skip_special_tokens=True).strip()
            tokens[j] += n_new
            limits[j] += n_new >= args.max_new_tokens
            try:
                json.loads(value)
            except json.JSONDecodeError:
                bad[j][".".join(keywise.KEYS[i])] = value[:300]
                value = "null"
            vals[j].append(value)
    return [(keywise.answer(v), t, b, l) for v, t, b, l in zip(vals, tokens, bad, limits)]


def load_rows(name: str, limit: int | None) -> list[dict]:
    path = GOLD if name == "gold" else LABELS
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if name == "val":
        rows = [r for r in rows if is_validation(r["cv_id"])]
    elif name == "train":
        rows = [r for r in rows if not is_validation(r["cv_id"])]
    return rows[:limit] if limit else rows


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", type=Path, help="LoRA adapter directory; omit to score the base model")
    ap.add_argument("--set", choices=["gold", "val", "train"], default="gold")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=2500)
    ap.add_argument("--constrained", action="store_true")
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"], help="default: cuda, else mps, else cpu")
    ap.add_argument("--max-patches", type=int, default=MAX_PATCHES, help="tile cap per page; use the value the adapter was trained with")
    ap.add_argument("--model", default=MODEL_ID, help="base model id or path (a full fine-tune's out/model dir)")
    ap.add_argument("--start", type=int, default=0, help="skip the first N rows of the set (e.g. gold test pages = --start 50)")
    ap.add_argument("--batch", type=int, default=1, help="pages generated together (left-padded); 1 = one at a time")
    ap.add_argument("--repetition-penalty", type=float, default=1.0, help=">1 lowers the odds of tokens already generated (loop control)")
    ap.add_argument("--keywise", action="store_true", help="key-by-key generation (model trained with train_lora.py --keywise)")
    args = ap.parse_args(argv)

    rows = load_rows(args.set, None)[args.start:]
    rows = rows[:args.limit] if args.limit else rows
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16  # bf16 is ~80x slower on CPUs without bf16 hardware
    processor = load_processor(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()  # merged: faster generation
    model = model.to(device).eval()
    extra = {}
    if args.constrained:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{device}, {args.set} {len(rows)} pages, adapter {args.adapter}, constrained {args.constrained}")

    processor.tokenizer.padding_side = "left"  # generation continues from the right end of every row
    text = processor.apply_chat_template(messages(None), tokenize=False, add_generation_prompt=True)
    total = Counter()
    with (args.out / "pages.jsonl").open("w") as f:
        for start in range(0, len(rows), args.batch):
            batch = rows[start:start + args.batch]
            images = [Image.open(ROOT / "data" / r["image"]).convert("RGB") for r in batch]
            inputs = processor(text=[text] * len(batch), images=images, padding=True, return_tensors="pt",
                               **family(processor)["size"](args.max_patches)).to(device)
            if args.constrained:  # a fresh enforcer per batch: it keeps state per prompt
                extra = {"prefix_allowed_tokens_fn": build_transformers_prefix_allowed_tokens_fn(
                    processor.tokenizer, JsonSchemaParser(CVRecord.model_json_schema()))}
            t0 = time.time()
            if args.keywise:
                done = generate_keywise(model, processor, text, images, args)
                seconds = round((time.time() - t0) / len(batch))
                for r, (answer, n_new, bad, limits) in zip(batch, done):
                    errors = len(bad)
                    s = score(answer, r["record"]) | {"cv_id": r["cv_id"], "page": r["page"], "tokens": n_new, "hit_limit": limits > 0,
                                                      "value_errors": errors, "bad_values": bad, "value_hit_limit": limits, "seconds": seconds}
                    total.update({k: s[k] for k in ("pred", "gold", "hit", "value_errors", "value_hit_limit")} | {s["status"]: 1, "hit_limit": int(s["hit_limit"])})
                    f.write(json.dumps(s | {"answer": answer}, ensure_ascii=False) + "\n")
                    f.flush()
                    f1 = 2 * s["hit"] / max(s["pred"] + s["gold"], 1)
                    print(f"  {r['cv_id'][:30]} p{r['page']}: {s['status']}, F1 {f1:.2f}, {n_new} tokens, {errors} bad values, {seconds} s/page", flush=True)
                continue
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
                                     repetition_penalty=args.repetition_penalty, pad_token_id=processor.tokenizer.pad_token_id, **extra)
            seconds = round((time.time() - t0) / len(batch))
            prompt_len = inputs["input_ids"].shape[1]
            for r, seq in zip(batch, out[:, prompt_len:]):
                n_new = int((seq != processor.tokenizer.pad_token_id).sum())  # rows that stopped early are padded
                answer = processor.tokenizer.decode(seq, skip_special_tokens=True)
                s = score(answer, r["record"]) | {"cv_id": r["cv_id"], "page": r["page"], "tokens": n_new,
                                                  "hit_limit": n_new >= args.max_new_tokens, "seconds": seconds}
                total.update({k: s[k] for k in ("pred", "gold", "hit")} | {s["status"]: 1, "hit_limit": int(s["hit_limit"])})
                f.write(json.dumps(s | {"answer": answer}, ensure_ascii=False) + "\n")
                f.flush()
                f1 = 2 * s["hit"] / max(s["pred"] + s["gold"], 1)
                print(f"  {r['cv_id'][:30]} p{r['page']}: {s['status']}, F1 {f1:.2f}, {n_new} tokens, {seconds} s/page", flush=True)

    p, rc = total["hit"] / max(total["pred"], 1), total["hit"] / max(total["gold"], 1)
    summary = {"set": args.set, "start": args.start, "pages": len(rows), "model": args.model, "adapter": str(args.adapter), "constrained": args.constrained, "batch": args.batch, "max_patches": args.max_patches,
               "valid": total["valid"], "schema_error": total["schema_error"], "unparsed": total["unparsed"],
               "hit_token_limit": total["hit_limit"], "field_precision": round(p, 4), "field_recall": round(rc, 4),
               "field_f1": round(2 * p * rc / max(p + rc, 1e-9), 4)}
    if args.keywise:
        summary |= {"keywise": True, "value_errors": total["value_errors"], "value_hit_limit": total["value_hit_limit"]}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
