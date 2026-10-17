"""Load a trained student and generate answers for page images (used by evaluate.py and parse.py).

Generation is greedy. Pages in one batch are left-padded, so every answer continues from the right end of its row.
The model runs in bf16 where the hardware supports it, else in float32 (bf16 is ~80x slower on CPUs without bf16
hardware, and T4 GPUs have none). Without the causal-conv1d kernel (CPU, Apple GPU), the short depthwise
convolutions of Qwen3.5's linear-attention layers run as shifted multiply-adds: PyTorch's CPU convolution runs them
one channel at a time, about 220 ms per call, which made a page take over an hour. On a CPU with bf16 hardware
(Apple M2 and later, AVX512-BF16, AMX, Arm bf16), Qwen3.5's language model runs in bf16 and its vision encoder in
float32: generation reads half the bytes per token, and pages took 55 s instead of 128 s on an M2 Max with the same
F1 (0.811 vs 0.816 on 10 validation pages).
"""

from __future__ import annotations

import platform
import subprocess
import sys

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
    if device == "cpu" and m.config.model_type == "qwen3_5" and cpu_has_bf16():
        m.model.language_model.to(torch.bfloat16)
        m.lm_head.to(torch.bfloat16)
    return m.to(device).eval(), processor, device


def cpu_has_bf16() -> bool:
    """Whether this CPU has bf16 instructions (without them bf16 matrix products are emulated and slow)."""
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "hw.optional.arm.FEAT_BF16"], capture_output=True, text=True)
        return out.stdout.strip() == "1"
    if platform.machine() in ("x86_64", "AMD64"):
        return torch.cpu._is_avx512_bf16_supported() or torch.cpu._is_amx_tile_supported()
    try:
        return " bf16" in open("/proc/cpuinfo").read()
    except OSError:
        return False


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
    streamer=None,
) -> list[tuple[str, int]]:
    """Whole-page answers for a batch of page images: [(answer text, tokens generated)].

    streamer: an optional transformers streamer (e.g. TextIteratorStreamer) that receives the tokens as they are made.
    """
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
            streamer=streamer,
        )
    pad = processor.tokenizer.pad_token_id
    return [
        (processor.tokenizer.decode(seq, skip_special_tokens=True), int((seq != pad).sum()))
        for seq in out[:, inputs["input_ids"].shape[1] :]
    ]  # rows that stopped early are padded
