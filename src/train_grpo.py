"""GRPO (TRL) after supervised fine-tuning: reward = lenient field F1 of the sampled answer against the page label.

    PYTHONPATH=$D/venv/trl_pkgs python src/train_grpo.py --model runs/Z_aug_vision/checkpoint-N --out runs/G1_grpo

TRL lives in a separate folder ($D/venv/trl_pkgs, pinned to the venv's torch/transformers) so the shared venv is unchanged.
Reward (same pairing as rescore.py, so it is the metric we report): lenient F1 of evaluate.pairs over the parsed
answer vs the label record; an answer that is not JSON gets 0; a schema error is scored on its raw JSON minus 0.1.
Validation-split CVs are never used (train_lora.is_validation), so the dev scores stay held out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, Image as ImageFeature
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import GRPOConfig, GRPOTrainer

from evaluate import pairs
from rescore import lenient
from schema import to_record
from train_lora import LABELS, MODEL_ID, PROMPT, ROOT, is_validation


def answer_reward(answer: str, label: dict) -> float:
    try:
        pred = json.loads(answer)
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(pred, dict):
        return 0.0
    penalty = 0.0
    try:
        pred = to_record(pred)
    except Exception:
        penalty = 0.1  # scored on its raw JSON, as rescore.parse does, but valid output is preferred
    p, g = lenient(pairs(pred)), lenient(pairs(to_record(label)))
    hit, n_p, n_g = sum((p & g).values()), sum(p.values()), sum(g.values())
    f1 = 2 * hit / (n_p + n_g) if n_p + n_g else 1.0
    return f1 - penalty


def reward_f1(prompts, completions, label, **kwargs) -> list[float]:
    return [answer_reward(c[0]["content"] if isinstance(c, list) else c, json.loads(lab)) for c, lab in zip(completions, label)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="supervised checkpoint to start from")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-file", type=Path, default=LABELS)
    ap.add_argument("--limit", type=int, default=0, help="first N training pages (smoke test)")
    ap.add_argument("--max-patches", type=int, default=12)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=2500)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0, help="KL weight to the starting model (0 = no reference model)")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--prompts-per-step", type=int, default=4, help="distinct pages per optimizer step")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--train-vision", action="store_true", help="also update the vision encoder")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in args.train_file.read_text().splitlines()]
    rows = [r for r in rows if not is_validation(r["cv_id"])]
    if args.limit:
        rows = rows[:args.limit]
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    ds = Dataset.from_list([{"prompt": messages, "image": str(ROOT / "data" / r["image"]),
                             "label": json.dumps(r["record"], ensure_ascii=False)} for r in rows])
    ds = ds.cast_column("image", ImageFeature())
    print(f"GRPO on {len(ds)} training pages, {args.num_generations} samples each, from {args.model}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)  # checkpoints hold weights only
    processor.image_processor.max_patches = args.max_patches  # TRL calls the processor without max_patches; same cap as SFT/eval
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
    for name, p in model.named_parameters():
        p.requires_grad = args.train_vision or not name.startswith("model.vision_tower")

    cfg = GRPOConfig(
        output_dir=str(args.out), seed=args.seed, bf16=True,
        learning_rate=args.lr, num_train_epochs=args.epochs, lr_scheduler_type="constant_with_warmup", warmup_steps=5,
        per_device_train_batch_size=args.num_generations, gradient_accumulation_steps=args.prompts_per_step,
        num_generations=args.num_generations, max_completion_length=args.max_new_tokens,
        temperature=args.temperature, beta=args.beta,
        gradient_checkpointing=True, logging_steps=1, save_steps=args.save_steps, save_only_model=True,
        report_to="none", log_completions=False, remove_unused_columns=False,
    )
    trainer = GRPOTrainer(model=model, reward_funcs=[reward_f1], args=cfg, train_dataset=ds, processing_class=processor)
    trainer.train()
    trainer.save_model(str(args.out / "model"))
    processor.save_pretrained(str(args.out / "model"))
    (args.out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))


if __name__ == "__main__":
    main()
