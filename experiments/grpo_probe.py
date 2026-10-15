"""Why GRPO did not raise dev F1: sample answers the way GRPO training did and measure what its reward could teach.

    python experiments/grpo_probe.py --model runs/Z_aug_vision/checkpoint-528 --out runs/grpo_probe/samples.jsonl
    python experiments/grpo_probe.py --analyze runs/grpo_probe/samples.jsonl

Generation (GPU): for 40 training pages (the pages GRPO trained on) and gold pages 1-50, one greedy answer and
8 answers sampled exactly as in train_grpo.py (temperature 1, top_p 1, top_k 0, 2,500 tokens), from the model GRPO
started from. Writes one JSONL row per page. --analyze (CPU) prints, per set:
  1. where the within-group reward spread comes from (answers that are not JSON vs. answers that differ in content);
  2. greedy reward vs the mean and the best of the 8 samples (what GRPO could push toward);
  3. of the gold values the greedy answer misses, the share found in at least one sample;
  4. pairs of JSON samples: on gold pages, how often the teacher-label reward (the GRPO reward) and the reviewed-label reward order
     two samples differently, and how often lenient and content scores do.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_grpo import answer_reward  # noqa: E402  the GRPO reward itself

from tiny_resume_parser.data import DATA, load_rows, page_image, read_jsonl  # noqa: E402
from tiny_resume_parser.generate import generate_pages, load_model, prompt_text  # noqa: E402
from tiny_resume_parser.metrics import content, lenient, pairs, parse  # noqa: E402
from tiny_resume_parser.models import family  # noqa: E402
from tiny_resume_parser.schema import to_record  # noqa: E402

TEACHER_GOLD = DATA / "processed/labels_gold_v6.jsonl"  # teacher labels of the gold pages, before review


def sample(model, processor, image, n: int, max_patches: int, max_new_tokens: int) -> list[str]:
    # The page is repeated n times in the batch. num_return_sequences would copy pixel_values tile by tile
    # (repeat_interleave on the tile axis), so each sample would see the page's tiles in the wrong order.
    inputs = processor(
        text=[prompt_text(processor)] * n,
        images=[image] * n,
        padding=True,
        return_tensors="pt",
        **family(processor)["size"](max_patches),
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    return processor.tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def generate(args) -> None:
    model, processor, _ = load_model(args.model)
    train = load_rows("train")
    random.Random(0).shuffle(train)
    rows = [("train", r) for r in train[: args.train_pages]] + [("gold", r) for r in load_rows("gold", 50)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {(j["set"], j["cv_id"], j["page"]) for j in read_jsonl(args.out)} if args.out.exists() else set()
    with args.out.open("a") as f:
        for i, (name, r) in enumerate(rows):
            if (name, r["cv_id"], r["page"]) in done:
                continue
            image = page_image(r)
            greedy = generate_pages(model, processor, [image], args.max_patches, args.max_new_tokens)[0][0]
            samples = sample(model, processor, image, args.n, args.max_patches, args.max_new_tokens)
            f.write(
                json.dumps({"set": name, "cv_id": r["cv_id"], "page": r["page"], "greedy": greedy, "samples": samples})
                + "\n"
            )
            f.flush()
            print(f"{i + 1}/{len(rows)} {name} {r['cv_id']} p{r['page']}", flush=True)


def gold_values(answer: str, gold: Counter) -> Counter:
    return lenient(pairs(parse(answer))) & gold


def analyze(path: Path) -> None:
    labels = {(r["cv_id"], r["page"]): r["record"] for r in read_jsonl(DATA / "processed/labels_train_v6.jsonl")}
    reviewed = {(r["cv_id"], r["page"]): r["record"] for r in load_rows("gold")}
    teacher = {(r["cv_id"], r["page"]): r["record"] for r in read_jsonl(TEACHER_GOLD)}
    rows = read_jsonl(path)
    for name in ("train", "gold"):
        part = [r for r in rows if r["set"] == name]
        if not part:
            continue
        stds, stds_json, broken, greedy_r, mean_r, best_r = [], [], 0, [], [], []
        missed = found = 0
        order = {"teacher_vs_reviewed": [0, 0], "lenient_vs_content": [0, 0]}  # [disagree, compared]
        for r in part:
            key = (r["cv_id"], r["page"])
            label = teacher[key] if name == "gold" else labels[key]  # the label GRPO's reward used
            rew = [answer_reward(s, label) for s in r["samples"]]
            ok = [x for s, x in zip(r["samples"], rew) if parse(s)]
            broken += len(rew) - len(ok)
            stds.append(st.pstdev(rew))
            stds_json.append(st.pstdev(ok) if len(ok) > 1 else 0.0)
            greedy_r.append(answer_reward(r["greedy"], label))
            mean_r.append(st.mean(rew))
            best_r.append(max(rew))
            truth = pairs(to_record(reviewed[key] if name == "gold" else label))
            gold = lenient(truth)
            miss = gold - gold_values(r["greedy"], gold)
            union = Counter()
            for s in r["samples"]:
                union |= gold_values(s, gold)
            missed += sum(miss.values())
            found += sum((miss & union).values())
            for a, b in combinations(range(len(rew)), 2):
                if abs(rew[a] - rew[b]) < 0.01 or not (parse(r["samples"][a]) and parse(r["samples"][b])):
                    continue  # only pairs of JSON answers: a broken answer is last under every score
                if name == "gold":
                    ra, rb = (answer_reward(r["samples"][i], reviewed[key]) for i in (a, b))
                    order["teacher_vs_reviewed"][1] += 1
                    order["teacher_vs_reviewed"][0] += (rew[a] - rew[b]) * (ra - rb) < 0
                ca, cb = (content_f1(r["samples"][i], content(truth)) for i in (a, b))
                order["lenient_vs_content"][1] += 1
                order["lenient_vs_content"][0] += (rew[a] - rew[b]) * (ca - cb) < 0
        n = len(part)
        print(f"== {name}: {n} pages x {len(part[0]['samples'])} samples")
        print(f"  reward std within a page: {st.mean(stds):.3f}; among JSON answers only: {st.mean(stds_json):.3f}")
        print(f"  samples that are not JSON: {broken} of {n * len(part[0]['samples'])}")
        print(
            f"  reward greedy {st.mean(greedy_r):.3f} | sample mean {st.mean(mean_r):.3f} | best of {len(part[0]['samples'])} {st.mean(best_r):.3f}"
        )
        print(
            f"  pages where the best sample beats greedy by > 0.02: {sum(b > g + 0.02 for b, g in zip(best_r, greedy_r))} of {n}"
        )
        print(f"  gold values greedy misses: {missed}; found in >= 1 sample: {found} ({found / max(missed, 1):.1%})")
        for k, (d, c) in order.items():
            if c:
                print(f"  sample pairs ordered differently, {k.replace('_', ' ')}: {d} of {c} ({d / c:.1%})")


def content_f1(answer: str, gold: Counter) -> float:
    """Content F1 of one answer (correct words in the right section count, wherever placed); gold = content(pairs)."""
    p = content(pairs(parse(answer)))
    n = sum(p.values()) + sum(gold.values())
    return 2 * sum((p & gold).values()) / n if n else 1.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--train-pages", type=int, default=40)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-patches", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=2500)
    a = ap.parse_args()
    analyze(a.analyze) if a.analyze else generate(a)
