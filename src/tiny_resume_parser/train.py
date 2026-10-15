"""Fine-tune a student model on teacher-labelled CV pages: page image -> JSON.

    python -m tiny_resume_parser.train --config configs/internvl3-1b.toml --out runs/NAME   # adopted recipe
    python -m tiny_resume_parser.train --config configs/granite-docling-258m.toml --seed 1 --out runs/NAME
    python -m tiny_resume_parser.train --limit 20 --epochs 1 --out runs/smoke                     # quick check

Every weight is trained, the vision encoder included. Reads the training labels (data.LABELS or --train-file) minus
the 5% validation split; writes out/model, out/checkpoint-<step> per epoch with --save-every-epoch, and
out/log_history.json. A config file sets the defaults; any flag on the command line overrides it.
Loss is on the answer tokens only: image, prompt and the assistant header are masked (-100), and a check on the first
batch stops the run if any image token is learnt.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import tomllib
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageFilter
from transformers import AutoModelForImageTextToText, Trainer, TrainingArguments

from .data import LABELS, is_validation, page_image
from .models import MAX_PATCHES, MODEL_ID, family, load_processor, messages, student_target


def augment_page(im: Image.Image) -> Image.Image:
    """Photometric/geometric noise that keeps every character readable, so the label stays true:
    lower resolution (resize down and back), small rotation, JPEG artefacts, contrast/brightness, slight blur."""
    w, h = im.size
    s = random.uniform(0.55, 1.0)
    im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC).resize((w, h), Image.BICUBIC)
    im = im.rotate(random.uniform(-1.5, 1.5), resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    im = ImageEnhance.Contrast(im).enhance(random.uniform(0.7, 1.2))
    im = ImageEnhance.Brightness(im).enhance(random.uniform(0.85, 1.1))
    if random.random() < 0.3:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.8)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=random.randint(45, 90))
    return Image.open(buf).convert("RGB")


class Collator:
    """perturb: Nougat's anti-repetition augmentation (arXiv 2308.13418; nougat/utils/dataset.py):
    while random() < perturb, one random answer token in the INPUT is replaced by a random vocabulary token
    (~0.11 tokens per page at 0.1). Unlike Nougat, the label keeps the correct token, so the model learns to
    continue correctly after a wrong token instead of learning to write noise."""

    def __init__(
        self,
        processor,
        perturb: float = 0.0,
        max_patches: int = MAX_PATCHES,
        augment: float = 0.0,
    ):
        self.processor = processor
        self.perturb = perturb
        self.augment = augment
        self.size = family(processor)["size"](max_patches)
        tok = processor.tokenizer
        self.n_text_tokens = tok.vocab_size  # perturbation draws ids below it
        # granite-docling counts its special tokens (<image>, end) inside vocab_size: a perturbation must never insert one
        self.special_ids = set(tok.added_tokens_decoder)
        self.header = tok(family(processor)["header"], add_special_tokens=False).input_ids
        self.end_id = tok.convert_tokens_to_ids(family(processor)["end"])

    def __call__(self, rows: list[dict]) -> dict:
        texts = [
            self.processor.apply_chat_template(messages(student_target(r["record"])), tokenize=False) for r in rows
        ]
        images = [page_image(r) for r in rows]
        images = [augment_page(im) if random.random() < self.augment else im for im in images]
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt", **self.size)
        labels = torch.full_like(batch["input_ids"], -100)
        h = len(self.header)
        for i, ids in enumerate(batch["input_ids"].tolist()):
            starts = [j for j in range(len(ids) - h + 1) if ids[j : j + h] == self.header]
            if len(starts) != 1:
                raise ValueError(
                    f"expected one assistant header, found {len(starts)} ({rows[i]['cv_id']} p{rows[i]['page']})"
                )
            first = starts[0] + h
            last = ids.index(self.end_id, first)  # the answer's closing <|im_end|> is learnt, so the model stops
            labels[i, first : last + 1] = batch["input_ids"][i, first : last + 1]
            while random.random() < self.perturb:  # after labels are copied, so labels stay correct
                new_id = random.randint(0, self.n_text_tokens - 1)
                while new_id in self.special_ids:
                    new_id = random.randint(0, self.n_text_tokens - 1)
                batch["input_ids"][i, random.randint(first, last - 1)] = new_id
        batch["labels"] = labels
        return batch


def build_parser() -> argparse.ArgumentParser:
    """Defaults are the InternVL3-1B recipe; configs/*.toml set each model's recipe."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--config", type=Path, help="TOML file of defaults (configs/*.toml); command-line flags override it"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=MODEL_ID, help="base model id or path")
    ap.add_argument("--train-file", type=Path, default=LABELS, help="label JSONL; the validation split is removed")
    ap.add_argument("--limit", type=int, help="train on the first N training pages only (quick check)")
    ap.add_argument("--epochs", type=float, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=1, help="pages per device per step")
    ap.add_argument(
        "--accum", type=int, default=16, help="gradient accumulation steps (effective batch = batch x accum)"
    )
    ap.add_argument(
        "--max-patches", type=int, default=MAX_PATCHES, help="image tiles per page; score with the same value"
    )
    ap.add_argument("--augment", type=float, default=0.5, help="share of pages that get image noise (0 = off)")
    ap.add_argument("--perturb", type=float, default=0.1, help="token perturbation probability (0 = off)")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="no gradient checkpointing: faster, more GPU memory")
    ap.add_argument("--save-every-epoch", action="store_true", help="save out/checkpoint-<step> after every epoch")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    first, _ = ap.parse_known_args(argv)
    if first.config:
        cfg = tomllib.loads(first.config.read_text())
        unknown = set(cfg) - {a.dest for a in ap._actions}
        if unknown:
            raise SystemExit(f"{first.config}: unknown keys {sorted(unknown)}")
        ap.set_defaults(**cfg)
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in Path(args.train_file).read_text().splitlines()]
    train = [r for r in rows if not is_validation(r["cv_id"])]
    val = [r for r in rows if is_validation(r["cv_id"])]
    if args.limit:
        train = train[: args.limit]
    print(f"train {len(train)} pages, validation {len(val)} pages ({len({r['cv_id'] for r in val})} CVs)")

    processor = load_processor(args.model)
    collator = Collator(processor, args.perturb, args.max_patches, args.augment)
    probe = Collator(processor, max_patches=args.max_patches)(train[:1])  # gate check: loss only on the answer
    is_img = probe["input_ids"] == processor.image_token_id
    n_img, n_tiles = int(is_img.sum()), probe["pixel_values"].shape[:-3].numel()
    per_tile = getattr(processor, "image_seq_length", None) or processor.image_seq_len  # InternVL 256, SmolVLM 64
    if n_img != n_tiles * per_tile or (probe["labels"][is_img] != -100).any():
        raise SystemExit(f"label mask check failed: {n_img} image tokens for {n_tiles} tiles")
    print(f"mask check ok: {n_img} image tokens masked, {int((probe['labels'] != -100).sum())} answer tokens learnt")

    # Weights always in float32: in pure bf16 many small updates round away (and gave grad-norm spikes). On a GPU
    # with bf16 (A100, L4, H100) the matmuls run in bf16 autocast; on a T4, Apple MPS or a CPU, in float32.
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32)
    model.enable_input_require_grads()
    print(f"full fine-tune: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters, bf16 {bf16}")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out),
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=0.01,
            bf16=bf16,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.accum,
            gradient_checkpointing=not args.no_grad_ckpt,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=args.logging_steps,
            save_strategy="epoch" if args.save_every_epoch else "no",
            save_only_model=True,
            remove_unused_columns=False,
            dataloader_num_workers=args.workers,
            seed=args.seed,
            ddp_find_unused_parameters=False,
            report_to=[],
        ),
        train_dataset=train,
        data_collator=collator,
    )
    trainer.train()
    if trainer.is_world_process_zero():  # torchrun: only rank 0 saves
        model.save_pretrained(args.out / "model")
        processor.save_pretrained(args.out / "model")
        (args.out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
