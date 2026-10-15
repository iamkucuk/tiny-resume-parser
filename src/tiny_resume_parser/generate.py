"""Load a trained student and generate answers for page images (used by evaluate.py and parse.py).

Generation is greedy. Pages in one batch are left-padded, so every answer continues from the right end of its row.
The model runs in bf16 where the hardware supports it, else in float32 (bf16 is ~80x slower on CPUs without bf16
hardware, and T4 GPUs have none).
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText

from .models import MAX_PATCHES, family, load_processor, messages


def pick_device(device: str | None = None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_model(model: str, device: str | None = None):
    """model: Hugging Face id or a fine-tuned model / checkpoint folder. Returns (model, processor, device)."""
    device = pick_device(device)
    processor = load_processor(model)
    processor.tokenizer.padding_side = "left"
    bf16 = device == "mps" or (device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False))
    m = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.bfloat16 if bf16 else torch.float32)
    return m.to(device).eval(), processor, device


def prompt_text(processor) -> str:
    return processor.apply_chat_template(messages(None), tokenize=False, add_generation_prompt=True)


def generate_pages(
    model,
    processor,
    images: list[Image.Image],
    max_patches: int = MAX_PATCHES,
    max_new_tokens: int = 3000,
) -> list[tuple[str, int]]:
    """Whole-page answers for a batch of page images: [(answer text, tokens generated)]."""
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
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    pad = processor.tokenizer.pad_token_id
    return [
        (processor.tokenizer.decode(seq, skip_special_tokens=True), int((seq != pad).sum()))
        for seq in out[:, inputs["input_ids"].shape[1] :]
    ]  # rows that stopped early are padded
