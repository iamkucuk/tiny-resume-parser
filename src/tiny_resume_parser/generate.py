"""Load a trained student and generate answers for page images (used by evaluate.py and parse.py).

Generation is greedy. Pages in one batch are left-padded, so every answer continues from the right end of its row.
The model runs in bf16 where the hardware supports it, else in float32 (bf16 is ~80x slower on CPUs without bf16
hardware, and T4 GPUs have none). Without the causal-conv1d kernel (CPU, Apple GPU), the short depthwise
convolutions of Qwen3.5's linear-attention layers run as shifted multiply-adds: PyTorch's CPU convolution runs them
one channel at a time, about 220 ms per call, which made a page take over an hour.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
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
    for layer in m.modules():  # Qwen3.5 linear-attention layers without the causal-conv1d kernel
        if hasattr(layer, "causal_conv1d_update") and getattr(layer, "causal_conv1d_fn", True) is None:
            conv = layer.conv1d
            conv.forward = lambda x, c=conv: depthwise_conv1d(
                F.pad(x, (c.padding[0],) * 2), c.weight.squeeze(1), c.bias
            )
            layer.causal_conv1d_update = conv1d_update
    return m.to(device).eval(), processor, device


def depthwise_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """x (batch, channels, length), weight (channels, K): F.conv1d(x, weight[:, None], bias, groups=channels)."""
    n = x.shape[-1] - weight.shape[-1] + 1
    out = sum(weight[:, k, None] * x[:, :, k : k + n] for k in range(weight.shape[-1]))
    return out if bias is None else out + bias[:, None]


def conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    """transformers' torch_causal_conv1d_update (one generation step), with depthwise_conv1d."""
    seq_len, state_len = hidden_states.shape[-1], conv_state.shape[-1]
    x = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(x[:, :, -state_len:])
    return F.silu(depthwise_conv1d(x, weight, bias)[:, :, -seq_len:]).to(hidden_states.dtype)


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
