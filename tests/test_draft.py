"""Drafting with the MTP head must not change the answer: every committed token is the model's own greedy choice.

A tiny random Qwen3.5 model in float32 (where a pass over several tokens rounds like one-token passes) is decoded
plainly and with 1-3 drafts per pass; the answers must be identical. The head is set to always draft one token that
the model also writes often, so drafts are both accepted and rejected and both paths of draft_decode run.
"""

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForImageTextToText

from tiny_resume_parser.draft import draft_decode, load_drafter
from tiny_resume_parser.generate import generate_pages
from tiny_resume_parser.models import QWEN35_ID, load_processor
from tiny_resume_parser.mtp import Capture, MTPHead


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch.manual_seed(0)
    processor = load_processor(QWEN35_ID)
    cfg = AutoConfig.from_pretrained(QWEN35_ID)
    t, v = cfg.text_config, cfg.vision_config
    t.hidden_size, t.num_hidden_layers, t.intermediate_size = 64, 4, 128
    t.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 32
    t.linear_num_key_heads = t.linear_num_value_heads = 2
    t.linear_key_head_dim = t.linear_value_head_dim = 16
    v.hidden_size, v.depth, v.num_heads, v.intermediate_size, v.out_hidden_size = 32, 1, 2, 64, 64
    model = AutoModelForImageTextToText.from_config(cfg).float().eval()
    image = Image.new("RGB", (224, 280), "white")

    # make token 0 the model's choice at some answer positions: its output row points along the mean final state
    cap = Capture(model).data
    generate_pages(model, processor, [image], 1, 30)
    h = cap["hidden"][0]
    with torch.no_grad():
        model.lm_head.weight[0] = h.mean(0) / h.mean(0).norm() ** 2 * model.lm_head(h).max(-1).values.mean()

    # a head whose output is zero (Qwen3.5's norm scales by 1 + weight), so it always drafts token 0
    state = {f"mtp.{k}": torch.randn_like(p) * 0.5 for k, p in MTPHead(t).state_dict().items()}
    state["mtp.norm.weight"] = -torch.ones(t.hidden_size)
    path = tmp_path_factory.mktemp("head") / "mtp.safetensors"
    save_file(state, path)
    return model, processor, image, str(path)


@pytest.mark.parametrize("drafts", [1, 2, 3])
def test_drafted_answer_is_the_greedy_answer(tiny, drafts):
    model, processor, image, head_file = tiny
    greedy, n = generate_pages(model, processor, [image], 1, 60)[0]
    head, cap = load_drafter(model, head_file)
    answer, n_answer, passes, accepted = draft_decode(model, processor, head, cap, image, 1, 60, drafts)
    assert answer == greedy and n_answer == n
    assert 0 < accepted < n_answer  # some drafts kept, some rejected: both paths ran
    # each pass commits its accepted drafts and one token of its own (the last pass may pass the token limit)
    assert passes + accepted >= n_answer - 1 and passes < n_answer - 1
