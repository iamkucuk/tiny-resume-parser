"""Load a trained student and generate answers for page images (used by evaluate.py and parse.py).

Generation is greedy. Pages in one batch are left-padded, so every answer continues from the right end of its row.
On CPU the model runs in float32: bf16 is ~80x slower on CPUs without bf16 hardware.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText

from . import keywise
from .models import MAX_PATCHES, family, load_processor, messages


def pick_device(device: str | None = None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_model(model: str, adapter: Path | None = None, device: str | None = None):
    """model: Hugging Face id or a fine-tune's model/checkpoint dir; adapter: LoRA dir merged into it.
    Returns (model, processor, device)."""
    device = pick_device(device)
    processor = load_processor(model)
    processor.tokenizer.padding_side = "left"
    m = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.float32 if device == "cpu" else torch.bfloat16)
    if adapter:
        from peft import PeftModel

        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()  # merged: faster generation
    return m.to(device).eval(), processor, device


def prompt_text(processor) -> str:
    return processor.apply_chat_template(messages(None), tokenize=False, add_generation_prompt=True)


def generate_pages(
    model,
    processor,
    images: list[Image.Image],
    max_patches: int = MAX_PATCHES,
    max_new_tokens: int = 3000,
    repetition_penalty: float = 1.0,
    **extra,
) -> list[tuple[str, int]]:
    """Whole-page answers for a batch of page images: [(answer text, tokens generated)].
    extra goes to model.generate (e.g. prefix_allowed_tokens_fn for constrained decoding)."""
    inputs = processor(
        text=[prompt_text(processor)] * len(images),
        images=images,
        padding=True,
        return_tensors="pt",
        **family(processor)["size"](max_patches),
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=processor.tokenizer.pad_token_id,
            **extra,
        )
    pad = processor.tokenizer.pad_token_id
    return [
        (processor.tokenizer.decode(seq, skip_special_tokens=True), int((seq != pad).sum()))
        for seq in out[:, inputs["input_ids"].shape[1] :]
    ]  # rows that stopped early are padded


def generate_keywise(
    model, processor, images: list, max_patches: int, max_new_tokens: int, repetition_penalty: float = 1.0
) -> list[tuple[str, int, dict, int]]:
    """All pages of a batch step through the 29 keys together. Prompt = chat template + generation prompt + the page's
    prefix, tokenized by the processor exactly as train.Collator.keywise_batch does. Returns per page:
    (answer, tokens generated, values that were not valid JSON (key -> text as written), values that hit --max-new-tokens)."""
    text = prompt_text(processor)
    vals = [[] for _ in images]
    bad = [{} for _ in images]
    tokens, limits = [0] * len(images), [0] * len(images)
    for i in range(len(keywise.KEYS)):
        prompts = [text + keywise.prefix(v, i) for v in vals]  # v holds the values of keys 0..i-1
        inputs = processor(
            text=prompts, images=images, padding=True, return_tensors="pt", **family(processor)["size"](max_patches)
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                repetition_penalty=repetition_penalty,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        for j, seq in enumerate(out[:, inputs["input_ids"].shape[1] :]):
            n_new = int((seq != processor.tokenizer.pad_token_id).sum())
            value = processor.tokenizer.decode(seq, skip_special_tokens=True).strip()
            tokens[j] += n_new
            limits[j] += n_new >= max_new_tokens
            try:
                json.loads(value)
            except json.JSONDecodeError:
                bad[j][".".join(keywise.KEYS[i])] = value[:300]
                value = "null"
            vals[j].append(value)
    return [(keywise.answer(v), t, b, n) for v, t, b, n in zip(vals, tokens, bad, limits)]
