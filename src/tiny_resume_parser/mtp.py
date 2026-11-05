"""Qwen3.5's multi-token prediction (MTP) head: from the main model's hidden state at position i and the embedding
of token i+1, it predicts token i+2. Live experiment (2026-11-05), used two ways:

  - extra training loss: train.py --mtp-weight W adds W x the head's cross-entropy on the answer tokens, so the
    language model also learns to look two tokens ahead (Gloeckle et al. 2024, arXiv 2404.19737; DeepSeek-V3 uses
    weight 0.3 then 0.1). The head is attached as model.mtp and saved with the model under Qwen's own tensor names
    (mtp.*); transformers skips those names when loading, so evaluation and parsing read the plain model.
  - drafting for faster decoding: experiments/mtp_decode.py.

Wiring follows vLLM's Qwen3.5 MTP (vllm/model_executor/models/qwen3_5_mtp.py): RMSNorm the next token's embedding
and the main model's final (post-norm) hidden state, concatenate them (embedding first), project to the hidden size,
one full-attention decoder layer, RMSNorm, then the main model's output layer. Checked on ResuVL-L: the pretrained
head attached unchanged predicts token i+2 of the model's own answers 85.9 % of the time with the post-norm state,
68.2 % with the pre-norm one (5 validation pages, 4,517 tokens).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm

from .models import QWEN35_ID


class MTPHead(torch.nn.Module):
    def __init__(self, text_config):
        super().__init__()
        c = copy.deepcopy(text_config)
        c.layer_types, c.num_hidden_layers = ["full_attention"], 1
        h = c.hidden_size
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(h, c.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(h, c.rms_norm_eps)
        self.fc = torch.nn.Linear(2 * h, h, bias=False)
        self.layers = torch.nn.ModuleList([Qwen3_5DecoderLayer(c, 0)])
        self.norm = Qwen3_5RMSNorm(h, c.rms_norm_eps)

    def forward(self, next_embeds, hidden, position_embeddings, attention_mask=None, past_key_values=None):
        """next_embeds[:, i] = embedding of token i+1, hidden[:, i] = main post-norm state at i -> state predicting i+2."""
        x = torch.cat([self.pre_fc_norm_embedding(next_embeds), self.pre_fc_norm_hidden(hidden)], -1)
        x = self.layers[0](
            self.fc(x),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        return self.norm(x)


def pretrained_state(source: str = QWEN35_ID) -> dict[str, torch.Tensor]:
    """The mtp.* tensors, keys without the 'mtp.' prefix, from a .safetensors file, a model folder (e.g. a model
    trained with --mtp-weight) or a Hub id (the local cache is used)."""
    path = Path(source)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.safetensors"))
    else:
        from huggingface_hub import hf_hub_download

        index = hf_hub_download(source, "model.safetensors.index.json")
        names = sorted({f for k, f in json.load(open(index))["weight_map"].items() if k.startswith("mtp.")})
        files = [Path(hf_hub_download(source, f)) for f in names]
    out = {}
    for f in files:
        with safe_open(f, "pt") as st:
            out |= {k.removeprefix("mtp."): st.get_tensor(k) for k in st.keys() if k.startswith("mtp.")}
    if not out:
        raise SystemExit(f"{source} has no mtp.* tensors")
    return out


class Capture:
    """Forward hooks that keep what the head needs from one forward pass of the main model: the input embeddings
    (image features included), the post-norm hidden states and the rotary position embeddings."""

    def __init__(self, model):
        lm = model.model.language_model
        self.data = {}
        lm.register_forward_pre_hook(lambda m, a, kw: self.data.update(embeds=kw["inputs_embeds"]), with_kwargs=True)
        lm.norm.register_forward_hook(lambda m, i, o: self.data.update(hidden=o))
        full = next(layer for layer in lm.layers if layer.layer_type == "full_attention")
        full.register_forward_pre_hook(
            lambda m, a, kw: self.data.update(pe=kw["position_embeddings"]), with_kwargs=True
        )


def mtp_loss(head: MTPHead, lm_head, cap: dict, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the head's prediction of token i+2 at every position where token i+2 is a learnt label.
    The output layer runs only at those positions (all positions would need ~4 GB of logits per page)."""
    e, h = cap["embeds"], cap["hidden"]
    cos, sin = cap["pe"]
    x = head(e[:, 1:], h[:, :-1], (cos[:, :-1], sin[:, :-1]))  # position i: predicts token i+2
    target = labels[:, 2:]
    keep = target != -100
    logits = lm_head(x[:, :-1][keep])
    return torch.nn.functional.cross_entropy(logits.float(), target[keep])
