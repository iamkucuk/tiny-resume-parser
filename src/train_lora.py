"""LoRA fine-tune of InternVL3-1B: CV page image -> ATS JSON (recipe: docs/FINETUNING_RECIPE.md).

    .venv/bin/python src/train_lora.py --limit 50 --epochs 2 --out runs/smoke     # smoke test (gate)
    .venv/bin/python src/train_lora.py --out runs/lora_v1                          # full run
    In a notebook: sys.path.insert(0, "src"); import train_lora; train_lora.main(["--out", ...], callbacks=[colab_utils.TrainingDashboard()])

Trains on data/processed/labels_train_v6.jsonl minus a 5% validation split chosen by CV.
Loss only on the answer tokens: image, prompt and the "<|im_start|>assistant\\n" header are -100.
After training, generates answers for --check-pages training pages and reports whether each
parses and passes to_record, and how many distinct answers there are.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

import keywise
from schema import to_record, to_target

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data" / "processed" / "labels_train_v6.jsonl"
MODEL_ID = "OpenGVLab/InternVL3-1B-hf"
PROMPT = "Extract this CV page as JSON."
MAX_PATCHES = 6  # default tile cap: every page <= 7 tiles = 1,792 image tokens (docs/FINETUNING_RECIPE.md); --max-patches changes it
SMOLVLM_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
GRANITE_ID = "ibm-granite/granite-docling-258M"  # also Idefics3; told apart from SmolVLM by its chat template
READ_PROMPT = "Write out all text on this page."  # reading stage: rows with a "transcript" (src/transcribe.py)
# Per processor class: the text that opens the answer, the token that ends it, the vision encoder's parameter prefix,
# and the processor kwargs for --max-patches. SmolVLM (Idefics3) cuts the page into 512 px tiles with the longest side
# at most longest_edge; its default 2048 gives 12 tiles + 1 thumbnail on an A4 page, the same count as InternVL at 12;
# --max-patches 20 / 30 / 48 select 2560 / 3072 / 4096 px.
SMOLVLM_EDGE = {12: 2048, 20: 2560, 30: 3072, 48: 4096}  # tiles on an A4 page (3x4, 4x5, 5x6, 6x8) -> longest_edge in px
FAMILY = {
    "InternVLProcessor": dict(header="<|im_start|>assistant\n", end="<|im_end|>", vision="model.vision_tower",
                              size=lambda n: {"max_patches": n}),
    "Idefics3Processor": dict(header="\nAssistant:", end="<end_of_utterance>", vision="model.vision_model",
                              size=lambda n: {"size": {"longest_edge": SMOLVLM_EDGE[n]}} if n in SMOLVLM_EDGE else
                              sys.exit(f"SmolVLM: --max-patches must be one of {sorted(SMOLVLM_EDGE)}, got {n}")),
}
FAMILY["GraniteDocling"] = FAMILY["Idefics3Processor"] | dict(header="<|start_of_role|>assistant<|end_of_role|>",
                                                              end="<|end_of_text|>")


def family(processor) -> dict:
    name = type(processor).__name__
    if name == "Idefics3Processor" and "<|start_of_role|>" in (processor.chat_template or ""):
        name = "GraniteDocling"
    return FAMILY[name]
# Left out of the student target. corrections records the teacher's spelling fixes; the student sees only
# the page, and lora_v1 (with it) repeated correction entries until the token limit on 8 of 10 test pages.
STUDENT_SKIP = ("corrections",)
LM_LINEAR = r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


class VisionLrTrainer(Trainer):
    """Trainer whose optimizer gives the vision encoder its own learning rate (set vision_lr, vision_prefix)."""

    def create_optimizer(self):
        opt = super().create_optimizer()
        vision = {id(p) for n, p in self.model.named_parameters() if n.startswith(self.vision_prefix)}
        groups = []
        for g in opt.param_groups:  # Trainer's groups split by weight decay; split each again by vision / rest
            for is_vision in (False, True):
                ps = [p for p in g["params"] if (id(p) in vision) == is_vision]
                if ps:
                    groups.append({**g, "params": ps, "lr": self.vision_lr if is_vision else g["lr"]})
        if not any(g["lr"] == self.vision_lr for g in groups):
            raise SystemExit(f"--vision-lr: no trainable parameter starts with {self.vision_prefix!r}")
        opt.param_groups[:] = groups
        print("optimizer groups (params, lr):", [(sum(p.numel() for p in g["params"]), g["lr"]) for g in groups])
        return opt


def load_processor(model: str):
    """Checkpoints and weight averages of a full fine-tune hold no processor: take the base model's."""
    if not Path(model).exists() or (Path(model) / "preprocessor_config.json").exists():
        return AutoProcessor.from_pretrained(model)
    cfg = json.loads((Path(model) / "config.json").read_text())
    # both Idefics3 models: granite-docling's vocabulary has 100,352 tokens, SmolVLM's 49,280
    base = MODEL_ID if cfg["model_type"] == "internvl" else {100352: GRANITE_ID, 49280: SMOLVLM_ID}[cfg["text_config"]["vocab_size"]]
    return AutoProcessor.from_pretrained(base)


def is_validation(cv_id: str) -> bool:
    """5% of CVs, fixed by a hash of the cv_id, so all pages of a CV fall on the same side."""
    return int(hashlib.sha1(cv_id.encode()).hexdigest(), 16) % 20 == 0


def student_target(record: dict) -> str:
    return to_target({k: v for k, v in record.items() if k not in STUDENT_SKIP})


def messages(answer: str | None, prompt: str = PROMPT) -> list[dict]:
    m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        m.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return m


def augment_page(im: Image.Image) -> Image.Image:
    """Photometric/geometric noise that keeps every character readable, so the label stays true:
    lower resolution (resize down and back), small rotation, JPEG artefacts, contrast/brightness, slight blur."""
    from PIL import ImageEnhance, ImageFilter
    import io
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

    def __init__(self, processor, perturb: float = 0.0, max_patches: int = MAX_PATCHES, augment: float = 0.0,
                 keywise: bool = False):
        self.processor = processor
        self.keywise = keywise
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
        if self.keywise:
            return self.keywise_batch(rows)
        # train_dpo.py sets row["answer"] (a model-written answer, scored as a DPO rejected) and row["truncated"]
        # a row with "transcript" is a reading-stage example: prompt READ_PROMPT, answer the page's plain text
        texts = [self.processor.apply_chat_template(
                     messages(r["transcript"], READ_PROMPT) if "transcript" in r else
                     messages(r["answer"] if "answer" in r else student_target(r["record"])), tokenize=False)
                 for r in rows]
        images = [Image.open(ROOT / "data" / r["image"]).convert("RGB") for r in rows]
        images = [augment_page(im) if random.random() < self.augment else im for im in images]
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt", **self.size)
        labels = torch.full_like(batch["input_ids"], -100)
        h = len(self.header)
        for i, ids in enumerate(batch["input_ids"].tolist()):
            starts = [j for j in range(len(ids) - h + 1) if ids[j:j + h] == self.header]
            if len(starts) != 1:
                raise ValueError(f"expected one assistant header, found {len(starts)} ({rows[i]['cv_id']} p{rows[i]['page']})")
            first = starts[0] + h
            last = ids.index(self.end_id, first)  # the answer's closing <|im_end|> is learnt, so the model stops
            if rows[i].get("truncated"):  # the model never wrote this <|im_end|> (the chat template added it): not part of the answer
                last -= 1
            labels[i, first:last + 1] = batch["input_ids"][i, first:last + 1]
            while random.random() < self.perturb:  # after labels are copied, so labels stay correct
                new_id = random.randint(0, self.n_text_tokens - 1)
                while new_id in self.special_ids:
                    new_id = random.randint(0, self.n_text_tokens - 1)
                batch["input_ids"][i, random.randint(first, last - 1)] = new_id
        batch["labels"] = labels
        return batch

    def keywise_batch(self, rows: list[dict]) -> dict:
        """One random key per row (src/keywise.py). The prompt part is tokenized exactly as evaluate.py --keywise
        tokenizes it (chat template + generation prompt + prefix, through the processor); the value's ids and the end
        token are appended after, so a BPE merge across the prefix/value boundary (e.g. ':"') never occurs in
        training when it cannot occur at inference. Sets self.last_values (the values learnt) for the mask check."""
        if len(rows) != 1:
            raise SystemExit("--keywise trains with --batch 1 (the value ids are appended after the processor's padding)")
        r = rows[0]
        vals = keywise.values(r["record"])
        i = random.randrange(len(keywise.KEYS))
        text = self.processor.apply_chat_template(messages(None), tokenize=False, add_generation_prompt=True)
        image = Image.open(ROOT / "data" / r["image"]).convert("RGB")
        if random.random() < self.augment:
            image = augment_page(image)
        batch = self.processor(text=[text + keywise.prefix(vals, i)], images=[image], return_tensors="pt", **self.size)
        value = self.processor.tokenizer(vals[i], add_special_tokens=False).input_ids + [self.end_id]
        first = batch["input_ids"].shape[1]
        batch["input_ids"] = torch.cat([batch["input_ids"], torch.tensor([value])], 1)
        batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones(1, len(value), dtype=batch["attention_mask"].dtype)], 1)
        labels = torch.full_like(batch["input_ids"], -100)
        labels[0, first:] = batch["input_ids"][0, first:]
        while random.random() < self.perturb:  # inside the value only, end token excluded; labels stay correct
            if len(value) < 2:
                break
            new_id = random.randint(0, self.n_text_tokens - 1)
            while new_id in self.special_ids:
                new_id = random.randint(0, self.n_text_tokens - 1)
            batch["input_ids"][0, random.randint(first, first + len(value) - 2)] = new_id
        batch["labels"] = labels
        self.last_values = [vals[i]]
        return batch


def check_generation(model, processor, rows: list[dict], out: Path, max_new_tokens: int = 2500,
                     max_patches: int = MAX_PATCHES, **gen_kwargs) -> None:
    """Generate for each row, report validity, and save every answer to out/generations.jsonl.
    gen_kwargs go to model.generate (e.g. no_repeat_ngram_size)."""
    model.eval()
    answers, valid = [], 0
    saved = (out / "generations.jsonl").open("w")
    for r in rows:
        text = processor.apply_chat_template(messages(None), tokenize=False, add_generation_prompt=True)
        image = Image.open(ROOT / "data" / r["image"]).convert("RGB")
        inputs = processor(text=[text], images=[image], return_tensors="pt", **family(processor)["size"](max_patches)).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, **gen_kwargs)
        answer = processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answers.append(answer)
        try:
            to_record(json.loads(answer))
            ok = "valid"
            valid += 1
        except Exception as e:  # report every failure; nothing is repaired
            ok = f"INVALID ({type(e).__name__}: {str(e)[:80]}; starts {answer[:60]!r})"
        print(f"  {r['cv_id']} p{r['page']}: {ok}, {len(answer)} chars, {time.time() - t0:.0f} s")
        saved.write(json.dumps({"cv_id": r["cv_id"], "page": r["page"], "result": ok, "answer": answer}, ensure_ascii=False) + "\n")
    saved.close()
    print(f"generation check: {valid}/{len(rows)} valid, {len(set(answers))} distinct answers")


def main(argv: list[str] | None = None, callbacks: list | None = None) -> int:
    """callbacks: extra TrainerCallbacks; if given, they replace the default progress bar."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, help="train on the first N training pages only (smoke test)")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1, help="per-device batch size")
    ap.add_argument("--accum", type=int, default=16, help="gradient accumulation steps (effective batch = batch x accum)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-patches", type=int, default=MAX_PATCHES,
                    help="tile cap per page (+1 thumbnail); 12 lets tall pages use up to 13 tiles. Score with the same value")
    ap.add_argument("--check-pages", type=int, default=10)
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--perturb", type=float, default=0.1, help="Nougat token-perturbation probability (0 = off)")
    ap.add_argument("--augment", type=float, default=0.0, help="probability that a training page image gets random resolution/rotation/JPEG/contrast/blur noise (0 = off)")
    ap.add_argument("--train-projector", action="store_true",
                    help="also fully train the image-to-text projector (4.5 M params; saved in the adapter)")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="turn off gradient checkpointing: ~1/3 less compute per step, more GPU memory")
    ap.add_argument("--save-every-epoch", action="store_true",
                    help="save the adapter (no optimizer state) to out/checkpoint-<step> after every epoch")
    ap.add_argument("--max-new-tokens", type=int, default=2500, help="generation limit for the check (small = quick CPU smoke)")
    ap.add_argument("--model", default=MODEL_ID, help="base model id or path (default InternVL3-1B-hf)")
    ap.add_argument("--full-ft", action="store_true", help="train all language-model and projector weights instead of LoRA; vision encoder frozen")
    ap.add_argument("--train-vision", action="store_true", help="with --full-ft: train the vision encoder too")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vision-lr", type=float, default=0,
                    help="learning rate of the vision encoder with --train-vision (discriminative fine-tuning, Howard & Ruder 2018; 0 = same as --lr)")
    ap.add_argument("--neftune", type=float, default=0, help="NEFTune noise alpha on input embeddings during training (Jain et al. 2023; 0 = off)")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (image loading + processing)")
    ap.add_argument("--train-file", type=Path, default=LABELS, help="training label JSONL (validation split by cv_id is still removed)")
    ap.add_argument("--keywise", type=int, default=0, help="key-by-key mode (src/keywise.py): N samples per page per "
                    "epoch, each one random key of 29; the model learns only that key's value (0 = whole-page mode)")
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in args.train_file.read_text().splitlines()]
    train = [r for r in rows if not is_validation(r["cv_id"])]
    val = [r for r in rows if is_validation(r["cv_id"])]
    if args.limit:
        train = train[:args.limit]
    print(f"train {len(train)} pages, validation {len(val)} pages ({len({r['cv_id'] for r in val})} CVs)")

    processor = load_processor(args.model)
    collator = Collator(processor, args.perturb, args.max_patches, args.augment, keywise=bool(args.keywise))
    probe_collator = Collator(processor, max_patches=args.max_patches, keywise=bool(args.keywise))
    probe = probe_collator(train[:1])  # gate check: loss only on the answer
    is_img = probe["input_ids"] == processor.image_token_id
    n_img, n_tiles = int(is_img.sum()), probe["pixel_values"].shape[:-3].numel()
    per_tile = getattr(processor, "image_seq_length", None) or processor.image_seq_len  # InternVL 256, SmolVLM 64
    if n_img != n_tiles * per_tile or (probe["labels"][is_img] != -100).any():
        raise SystemExit(f"label mask check failed: {n_img} image tokens for {n_tiles} tiles")
    if args.keywise:  # the learnt tokens must be exactly the value + end token
        learnt = processor.tokenizer.decode(probe["labels"][probe["labels"] != -100])
        want = probe_collator.last_values[0] + family(processor)["end"]
        if learnt != want:
            raise SystemExit(f"keywise mask check failed: learnt {learnt[:200]!r}, expected {want[:200]!r}")
        print(f"keywise check ok: learnt {learnt[:120]!r}; {len(train)} pages x {args.keywise} samples per epoch")
        train = train * args.keywise  # Collator draws the key; Trainer shuffles
    print(f"mask check ok: {n_img} image tokens masked, {int((probe['labels'] != -100).sum())} answer tokens learnt")

    # bf16 on CUDA/MPS; float32 on CPU, where bf16 math without hardware support stalled a Colab
    # CPU smoke run at step 0 for 30+ min.
    accel = torch.cuda.is_available() or torch.backends.mps.is_available()
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16 if accel else torch.float32)
    model.enable_input_require_grads()
    if args.full_ft:
        for name, p in model.named_parameters():
            p.requires_grad = args.train_vision or not name.startswith(family(processor)["vision"])
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"full fine-tune: {n / 1e6:.1f} M trainable parameters (vision encoder {'trained' if args.train_vision else 'frozen'})")
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=LM_LINEAR,
                                             modules_to_save=["multi_modal_projector"] if args.train_projector else None))
        model.print_trainable_parameters()
    cuda = torch.cuda.is_available()
    if cuda:
        # Mixed precision: LoRA weights in float32, matmuls in bf16 autocast. The pure-bf16 MPS
        # smoke runs had grad-norm spikes up to ~1e14 (clipped). T4 has no bf16: stop, pick A100/L4.
        if not torch.cuda.is_bf16_supported():
            raise SystemExit(f"{torch.cuda.get_device_name()} has no bf16; use an A100 or L4 runtime")
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    trainer = (VisionLrTrainer if args.vision_lr else Trainer)(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out),
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=0.01,
            bf16=cuda,
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
            neftune_noise_alpha=args.neftune or None,
            ddp_find_unused_parameters=False,
            report_to=[],
        ),
        train_dataset=train,
        data_collator=collator,
        callbacks=callbacks,
    )
    if args.vision_lr:
        trainer.vision_lr, trainer.vision_prefix = args.vision_lr, family(processor)["vision"]
    if callbacks:
        for cb in list(trainer.callback_handler.callbacks):  # ProgressCallback or NotebookProgressCallback
            if type(cb).__name__.endswith("ProgressCallback"):
                trainer.remove_callback(cb)
    trainer.train()
    if not trainer.is_world_process_zero():  # torchrun: only rank 0 saves and checks
        return 0
    model.save_pretrained(args.out / ("model" if args.full_ft else "adapter"))
    if args.full_ft:
        processor.save_pretrained(args.out / "model")
    (args.out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))

    if args.keywise:
        print("generation check skipped: whole-page generation does not apply to --keywise (score with evaluate.py --keywise)")
    elif args.check_pages:
        check_generation(model, processor, train[:args.check_pages], args.out, args.max_new_tokens, args.max_patches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
