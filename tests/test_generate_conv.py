"""The CPU depthwise convolution used for Qwen3.5 must give what PyTorch's convolution gives."""

import torch
import torch.nn.functional as F

from tiny_resume_parser.generate import conv1d_update, depthwise_conv1d


def test_depthwise_conv1d_matches_grouped_conv1d():
    x, w, b = torch.randn(2, 8, 30), torch.randn(8, 4), torch.randn(8)
    assert torch.allclose(depthwise_conv1d(x, w, b), F.conv1d(x, w[:, None], b, groups=8), atol=1e-5)


def test_conv1d_update_matches_transformers_reference():
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_causal_conv1d_update

    x, w = torch.randn(1, 8, 1), torch.randn(8, 4)
    state_a, state_b = torch.randn(1, 8, 4), None
    state_b = state_a.clone()
    out_a = conv1d_update(x, state_a, w)
    out_b = torch_causal_conv1d_update(x, state_b, w)
    assert torch.allclose(out_a, out_b, atol=1e-5) and torch.equal(state_a, state_b)
