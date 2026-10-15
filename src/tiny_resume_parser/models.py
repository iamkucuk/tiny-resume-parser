"""The three student model families and how each one is prompted.

Every family needs three facts: the text that opens the assistant's answer, the token that ends it, and how
--max-patches becomes processor settings. InternVL caps the number of 448 px
tiles directly; SmolVLM and granite-docling (both Idefics3) cut the page into 512 px tiles with the longest side at
most longest_edge, so --max-patches 12 / 20 / 30 / 48 select 2048 / 2560 / 3072 / 4096 px (3x4 to 6x8 tiles on A4).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoProcessor

from .schema import to_target

INTERNVL_ID = "OpenGVLab/InternVL3-1B-hf"
SMOLVLM_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
GRANITE_ID = "ibm-granite/granite-docling-258M"  # also Idefics3; told apart from SmolVLM by its chat template
MODEL_ID = INTERNVL_ID
PROMPT = "Extract this CV page as JSON."
MAX_PATCHES = 12  # InternVL3-1B recipe; SmolVLM uses 30 (configs/*.toml)
# Tile cap each released recipe was trained with; inference must use the same value (configs/*.toml).
TRAINED_PATCHES = {"InternVLProcessor": 12, "Idefics3Processor": 30, "GraniteDocling": 12}
IDEFICS3_EDGE = {12: 2048, 20: 2560, 30: 3072, 48: 4096}  # tiles on an A4 page -> longest_edge in px

FAMILY = {
    "InternVLProcessor": dict(
        header="<|im_start|>assistant\n",
        end="<|im_end|>",
        size=lambda n: {"max_patches": n},
    ),
    "Idefics3Processor": dict(
        header="\nAssistant:",
        end="<end_of_utterance>",
        size=lambda n: (
            {"size": {"longest_edge": IDEFICS3_EDGE[n]}}
            if n in IDEFICS3_EDGE
            else sys.exit(f"SmolVLM / granite-docling: --max-patches must be one of {sorted(IDEFICS3_EDGE)}, got {n}")
        ),
    ),
}
FAMILY["GraniteDocling"] = FAMILY["Idefics3Processor"] | dict(
    header="<|start_of_role|>assistant<|end_of_role|>", end="<|end_of_text|>"
)

# Left out of the student target. corrections records the teacher's spelling fixes, which the student cannot see on
# the page; a first model trained with it repeated correction entries until the token limit on 8 of 10 pages.
STUDENT_SKIP = ("corrections",)


def family_name(processor) -> str:
    name = type(processor).__name__
    if name == "Idefics3Processor" and "<|start_of_role|>" in (processor.chat_template or ""):
        name = "GraniteDocling"
    return name


def family(processor) -> dict:
    return FAMILY[family_name(processor)]


def load_processor(model: str):
    """Checkpoints and weight averages of a full fine-tune hold no processor: take the base model's."""
    if not Path(model).exists() or (Path(model) / "preprocessor_config.json").exists():
        return AutoProcessor.from_pretrained(model)
    cfg = json.loads((Path(model) / "config.json").read_text())
    # both Idefics3 models: granite-docling's vocabulary has 100,352 tokens, SmolVLM's 49,280
    base = (
        INTERNVL_ID
        if cfg["model_type"] == "internvl"
        else {100352: GRANITE_ID, 49280: SMOLVLM_ID}[cfg["text_config"]["vocab_size"]]
    )
    return AutoProcessor.from_pretrained(base)


def student_target(record: dict) -> str:
    """The exact text the student learns to write for one page."""
    return to_target({k: v for k, v in record.items() if k not in STUDENT_SKIP})


def messages(answer: str | None, prompt: str = PROMPT) -> list[dict]:
    """Chat turns: the page image and the prompt; with answer, also the assistant turn (training)."""
    m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        m.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return m
