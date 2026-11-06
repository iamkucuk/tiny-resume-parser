"""Faster greedy decoding for ResuVL-L: the MTP head (mtp.py) drafts the next tokens and the model checks them.

    tiny-resume-parser parse cv.pdf --model furkankucuk/ResuVL-L --drafts 3

Each main pass feeds the tokens committed since the last pass plus up to `drafts` tokens proposed by the head. The
model's own greedy choice at every fed position decides: drafts are kept while they match it, and the first mismatch
is replaced by the model's choice. The committed tokens are therefore the model's greedy answer; in float32 the
answer is identical to plain decoding. In bf16 a pass over several tokens rounds differently from a one-token pass,
so near-tied tokens can change: on 30 validation pages about one answer in four differed somewhere, with the same
field F1 (docs/findings.md). Speed: about 2.3-2.5x on an H100 with 2-3 drafts; none on a CPU, where a pass over
several tokens costs nearly as much as that many one-token passes.

Two changes to transformers 5.5.4 (pinned in pyproject.toml) are applied: Qwen3.5's linear-attention (Gated
DeltaNet) layers use their cached state only for one new token, so the cached branch is run for several tokens too
(continue_linear_attn); and a rejected draft is undone by restoring those layers' saved states and cropping the
attention cache. One page at a time (batch size 1).
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from .generate import prompt_text
from .models import family
from .mtp import Capture, MTPHead, pretrained_state


def continue_linear_attn(self, hidden_states, cache_params):
    """Qwen3_5GatedDeltaNet.forward's cached branch (transformers 5.5.4), for any number of new tokens."""
    bsz, seq_len, _ = hidden_states.shape
    layer = cache_params.layers[self.layer_idx]
    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(bsz, seq_len, -1, self.head_v_dim)
    b, a = self.in_proj_b(hidden_states), self.in_proj_a(hidden_states)
    mixed_qkv = self.causal_conv1d_update(
        mixed_qkv, layer.conv_states, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation
    ).transpose(1, 2)
    q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    q = q.reshape(bsz, seq_len, -1, self.head_k_dim)
    k = k.reshape(bsz, seq_len, -1, self.head_k_dim)
    v = v.reshape(bsz, seq_len, -1, self.head_v_dim)
    g = -self.A_log.float().exp() * torch.nn.functional.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        q = q.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        k = k.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    out, state = self.recurrent_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=b.sigmoid(),
        initial_state=layer.recurrent_states,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    cache_params.update_recurrent_state(state, self.layer_idx)
    out = self.norm(out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)).reshape(bsz, seq_len, -1)
    return self.out_proj(out)


def patch_linear_layers(model) -> None:
    for m in model.modules():
        if type(m).__name__ == "Qwen3_5GatedDeltaNet":

            def fwd(hidden_states, cache_params=None, attention_mask=None, _m=m, _orig=m.forward):
                cached = cache_params is not None and cache_params.has_previous_state(_m.layer_idx)
                if cached and hidden_states.shape[1] > 1:
                    return continue_linear_attn(_m, hidden_states, cache_params)
                return _orig(hidden_states, cache_params=cache_params, attention_mask=attention_mask)

            m.forward = fwd


def save_linear_states(cache):
    return [
        (i, layer.conv_states.clone(), layer.recurrent_states.clone())
        for i, layer in enumerate(cache.layers)
        if getattr(layer, "is_recurrent_states_initialized", False)
    ]


def restore(cache, saved, length: int) -> None:
    for i, c, r in saved:
        cache.layers[i].conv_states.copy_(c)
        cache.layers[i].recurrent_states.copy_(r)
    cache.crop(length)


def load_drafter(model, source: str):
    """Prepare `model` for draft_decode with the mtp.* tensors of `source` (model folder, .safetensors file or Hub
    id): returns (head, capture). The head runs in the language model's dtype."""
    if model.config.model_type != "qwen3_5":
        raise SystemExit(f"drafting needs a Qwen3.5 model (ResuVL-L); this model is {model.config.model_type}")
    patch_linear_layers(model)
    head = MTPHead(model.config.text_config)
    head.load_state_dict(pretrained_state(source))
    lm = model.model.language_model
    return head.to(device=model.device, dtype=lm.dtype).eval(), Capture(model).data


@torch.no_grad()
def draft_decode(model, processor, head, cap, image, max_patches: int, max_new_tokens: int = 3000, drafts: int = 3):
    """Greedy answer for one page with up to `drafts` drafts per main pass: (answer text, answer tokens, main passes,
    drafts accepted). With drafts > 1 the head runs again on its own output state and draft, as vLLM does for
    Qwen3.5; those chained drafts sit at speculative positions and are removed from the head's cache after each pass."""
    lm = model.model.language_model
    embed = lm.embed_tokens
    end_id = processor.tokenizer.convert_tokens_to_ids(family(processor)["end"])
    pos_hook = lm.rotary_emb.register_forward_pre_hook(lambda m, a: cap.update(pos=a[1]))
    try:
        inputs = processor(
            text=[prompt_text(processor)], images=[image], return_tensors="pt", **family(processor)["size"](max_patches)
        ).to(model.device)
        out = model(**inputs, use_cache=True)
        cache = out.past_key_values
        toks = inputs["input_ids"][0].tolist()
        n_prompt = C = len(toks)  # C: positions in the main cache
        E = cap["embeds"][0]  # prompt input embeddings, image features included
        H, COS, SIN = cap["hidden"][0], cap["pe"][0][0], cap["pe"][1][0]  # per position with a correct prefix
        POS = cap["pos"][:, 0]  # (3, positions): rotary position ids, for the positions of chained drafts
        toks.append(int(out.logits[0, -1].argmax()))
        pending = [toks[-1]]  # committed tokens not yet in the main cache
        mcache, M = DynamicCache(), 0  # the head's cache and the positions it has processed
        passes = accepted = 0
        while toks[-1] != end_id and len(toks) - n_prompt < max_new_tokens:
            # head over its new positions i: (H_i, embedding of token i+1); the last one drafts the token after toks[-1]
            last = H.shape[0] - 1
            nxt = torch.stack([E[j] if j < n_prompt else embed.weight[toks[j]] for j in range(M + 1, last + 2)])[None]
            q = last + 1 - M
            mask = torch.ones(q, M + q, dtype=torch.bool, device=model.device).tril(M)[None, None]
            x = head(nxt, H[M:][None], (COS[M:][None], SIN[M:][None]), attention_mask=mask, past_key_values=mcache)
            x = x[:, -1:]
            M = last + 1
            draft = [int(model.lm_head(x[:, -1])[0].argmax())]
            for k in range(1, drafts):
                pe = lm.rotary_emb(x, POS[:, -1:, None] + k)
                x = head(embed.weight[draft[-1]][None, None], x, pe, past_key_values=mcache)
                draft.append(int(model.lm_head(x[:, -1])[0].argmax()))
            mcache.crop(M)
            saved = save_linear_states(cache)
            ids = torch.tensor([pending + draft], device=model.device)
            pred = model(input_ids=ids, past_key_values=cache, use_cache=True).logits[0].argmax(-1).tolist()
            passes += 1
            p = len(pending)
            n_ok = 0
            while n_ok < len(draft) and pred[p - 1 + n_ok] == draft[n_ok]:
                n_ok += 1
            accepted += n_ok
            new = slice(H.shape[0] - C, p + n_ok)  # fed positions with a correct prefix, not stored yet
            H = torch.cat([H, cap["hidden"][0][new]])
            COS, SIN = torch.cat([COS, cap["pe"][0][0][new]]), torch.cat([SIN, cap["pe"][1][0][new]])
            POS = torch.cat([POS, cap["pos"][:, 0][:, new]], 1)
            committed = draft[:n_ok] + [pred[p - 1 + n_ok]]  # accepted drafts, then the model's own next token
            toks += committed
            if n_ok == len(draft):  # everything fed was right: the cache keeps it
                C += p + n_ok
                pending = [committed[-1]]
            else:  # undo the fed tokens; the committed ones are fed again next pass
                restore(cache, saved, C)
                pending = pending + committed
            if end_id in committed:
                toks = toks[: len(toks) - len(committed) + committed.index(end_id) + 1]
    finally:
        pos_hook.remove()
    answer = toks[n_prompt:]
    answer = answer[: answer.index(end_id) + 1] if end_id in answer else answer
    answer = answer[:max_new_tokens]
    return processor.tokenizer.decode(answer, skip_special_tokens=True), len(answer), passes, accepted
