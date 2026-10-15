"""DPO + SFT term: continue a LoRA adapter so that it stops writing broken or wrong JSON. No TRL; a plain PyTorch loop.

    .venv/bin/python src/train_dpo.py --adapter runs/lora_v6/adapter --pages runs/lora_v6/eval_train_all_free/pages.jsonl \\
        --max-patches 12 --out runs/dpo_v1
    .venv/bin/python src/train_dpo.py --adapter ... --pages ... --limit 20 --epochs 1 --out runs/dpo_smoke     # smoke test

--pages is the pages.jsonl that evaluate.py --set train wrote for the starting adapter (its own greedy answers).
A page becomes a preference pair when its answer is not valid, or its page F1 is below --f1-threshold:
chosen = the SFT target of the label (train_lora.student_target), rejected = the model's own answer.
Loss per pair = -logsigmoid(beta * ((pc - refc) - (pr - refr))) + sft_weight * (chosen NLL per answer token),
p = sum of answer-token log-probs (the Collator's mask: after "<|im_start|>assistant\\n" up to the closing <|im_end|>).
ref = the starting adapter, evaluated once before training and kept as numbers, so only one model is in memory.
A rejected answer that hit the token limit never wrote <|im_end|>, so its mask stops before the one the chat template adds.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

from train_lora import LABELS, MAX_PATCHES, MODEL_ID, Collator, is_validation, student_target


def build_pairs(pages_file: Path, labels_file: Path, f1_threshold: float) -> tuple[list[dict], dict]:
    labels = {}
    for line in labels_file.read_text().splitlines():
        r = json.loads(line)
        labels[(r["cv_id"], r["page"])] = r
    pairs, kept, skipped = [], Counter(), Counter()
    buckets = Counter()  # page F1 of the kept pairs, in tenths
    for line in pages_file.read_text().splitlines():
        p = json.loads(line)
        key = (p["cv_id"], p["page"])
        if key not in labels:
            raise SystemExit(f"{key} is in {pages_file} but not in {labels_file}")
        row = labels[key]
        if is_validation(p["cv_id"]):
            skipped["validation"] += 1
            continue
        f1 = 2 * p["hit"] / max(p["pred"] + p["gold"], 1)
        if p["status"] == "valid" and f1 >= f1_threshold:
            skipped["good"] += 1
            continue
        chosen = student_target(row["record"])
        if p["answer"] == chosen:
            skipped["answer equals chosen"] += 1
            continue
        pairs.append({"cv_id": p["cv_id"], "page": p["page"], "image": row["image"], "record": row["record"],
                      "rejected": p["answer"], "truncated": bool(p["hit_limit"]), "status": p["status"], "f1": f1})
        kept[p["status"]] += 1
        buckets[f"{min(int(f1 * 10), 9) / 10:.1f}-{min(int(f1 * 10), 9) / 10 + 0.1:.1f}"] += 1
    summary = {"pairs": len(pairs), "by_status": dict(kept), "truncated_rejected": sum(p["truncated"] for p in pairs),
               "f1_histogram": dict(sorted(buckets.items())), "skipped": dict(skipped), "f1_threshold": f1_threshold}
    return pairs, summary


def seq_logp(model, batch: dict, device, dtype) -> tuple[torch.Tensor, int]:
    """Sum of log-probs of the answer tokens (labels != -100) and how many there are."""
    labels = batch.pop("labels")[0, 1:].to(device)
    batch = {k: v.to(device) for k, v in batch.items()}
    batch["pixel_values"] = batch["pixel_values"].to(dtype)
    logits = model(**batch).logits[0, :-1]
    mask = labels != -100
    n = int(mask.sum())
    if n == 0:
        raise ValueError("no answer tokens in the labels")
    logp = F.log_softmax(logits[mask].float(), dim=-1).gather(1, labels[mask].unsqueeze(1))
    return logp.sum(), n


def dpo_loss(pc, pr, ref_c, ref_r, n_chosen, beta: float, sft_weight: float) -> dict:
    """pc, pr: policy log-probs of chosen / rejected; ref_c, ref_r: reference. n_chosen: chosen answer tokens."""
    reward_c, reward_r = beta * (pc - ref_c), beta * (pr - ref_r)
    dpo = -F.logsigmoid(reward_c - reward_r)
    sft = -pc / n_chosen
    return {"loss": dpo + sft_weight * sft, "dpo": dpo, "sft": sft, "margin": reward_c - reward_r,
            "acc": (reward_c > reward_r).float()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", type=Path, required=True, help="starting LoRA adapter directory (saved by train_lora.py)")
    ap.add_argument("--pages", type=Path, required=True, help="pages.jsonl of evaluate.py --set train for that adapter")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-file", type=Path, default=LABELS, help="label JSONL the pages are looked up in, by (cv_id, page)")
    ap.add_argument("--model", default=MODEL_ID, help="base model id or path")
    ap.add_argument("--max-patches", type=int, default=MAX_PATCHES, help="tile cap per page; use the value the adapter was trained with")
    ap.add_argument("--f1-threshold", type=float, default=0.9, help="valid answers with a page F1 below this are rejected answers too")
    ap.add_argument("--limit", type=int, help="use the first N pairs only (smoke test)")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft-weight", type=float, default=1.0, help="weight of the chosen answer's NLL per token")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8, help="pairs per optimizer step")
    ap.add_argument("--save-every", type=int, help="save the adapter to out/checkpoint-<step> every N optimizer steps")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="turn off gradient checkpointing: faster, but two ~5k-token graphs per pair need far more GPU memory")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    pairs, summary = build_pairs(args.pages, args.train_file, args.f1_threshold)
    if args.limit:
        pairs = pairs[:args.limit]
        summary["used"] = len(pairs)
    print(json.dumps(summary))
    if not pairs:
        raise SystemExit("no preference pairs")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pairs_summary.json").write_text(json.dumps(summary, indent=1))
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.model)
    collate = Collator(processor, max_patches=args.max_patches)  # perturb off: log-probs must belong to the exact text
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type != "cpu" else torch.float32  # as train_lora: float32 on CPU
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype)
    base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    for m in model.modules():
        if isinstance(m, LoraLayer):  # the reference pass runs in eval mode (no dropout); the policy must match it, or the first margin is noise
            for name in m.lora_dropout:
                m.lora_dropout[name] = torch.nn.Identity()
    if cuda:
        if not torch.cuda.is_bf16_supported():
            raise SystemExit(f"{torch.cuda.get_device_name()} has no bf16")
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()  # as train_lora: float32 trainable weights, bf16 autocast matmuls
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise SystemExit("adapter has no trainable parameters")
    print(f"{device}; {sum(p.numel() for p in params) / 1e6:.1f} M trainable parameters in {len(params)} tensors")

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda)

    def rows_of(pair: dict) -> tuple[dict, dict]:
        chosen = {k: pair[k] for k in ("cv_id", "page", "image", "record")}
        return chosen, chosen | {"answer": pair["rejected"], "truncated": pair["truncated"]}

    # Reference log-probs: the starting adapter, once, before any update.
    model.eval()
    ref = []
    t0 = time.time()
    with torch.no_grad(), autocast():
        for i, pair in enumerate(pairs):
            chosen, rejected = rows_of(pair)
            ref.append((seq_logp(model, collate([chosen]), device, dtype)[0].item(),
                        seq_logp(model, collate([rejected]), device, dtype)[0].item()))
            if (i + 1) % 20 == 0 or i + 1 == len(pairs):
                print(f"  reference {i + 1}/{len(pairs)} ({time.time() - t0:.0f} s)", flush=True)
    model.train()

    steps_per_epoch = math.ceil(len(pairs) / args.accum)
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(0.05 * total), total)
    history, step = [], 0
    for epoch in range(args.epochs):
        order = list(range(len(pairs)))
        random.shuffle(order)
        for g in range(0, len(order), args.accum):
            group = order[g:g + args.accum]
            agg = Counter()
            t0 = time.time()
            for i in group:
                chosen, rejected = rows_of(pairs[i])
                with autocast():
                    pc, n_chosen = seq_logp(model, collate([chosen]), device, dtype)
                    pr, _ = seq_logp(model, collate([rejected]), device, dtype)
                out = dpo_loss(pc, pr, ref[i][0], ref[i][1], n_chosen, args.beta, args.sft_weight)
                (out["loss"] / len(group)).backward()
                agg.update({k: v.item() / len(group) for k, v in out.items()})
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()  # returns the norm before clipping
            if not math.isfinite(grad_norm):
                raise SystemExit(f"step {step + 1}: gradient norm {grad_norm}")
            lr = sched.get_last_lr()[0]
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "epoch": epoch + g / len(order), "loss": agg["loss"], "dpo": agg["dpo"], "sft": agg["sft"],
                   "reward_margin": agg["margin"], "reward_accuracy": agg["acc"], "grad_norm": grad_norm, "learning_rate": lr}
            history.append(rec)
            print(f"step {step}/{total}: " + ", ".join(f"{k} {v:.4g}" for k, v in rec.items() if k not in ("step", "epoch"))
                  + f", {time.time() - t0:.0f} s", flush=True)
            if args.save_every and step % args.save_every == 0:
                model.save_pretrained(args.out / f"checkpoint-{step}")

    model.save_pretrained(args.out / "adapter")
    (args.out / "log_history.json").write_text(json.dumps(history, indent=1))
    print(f"saved {args.out / 'adapter'}: {step} steps, final reward accuracy {history[-1]['reward_accuracy']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
