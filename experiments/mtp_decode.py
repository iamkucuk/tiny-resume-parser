"""Qwen3.5's MTP head as a drafter for ResuVL-L: does it make greedy decoding faster without changing the answer?

    python experiments/mtp_decode.py accept --model MODEL --mtp SOURCE --pages runs/X/eval_final_val/pages.jsonl --n 5
    python experiments/mtp_decode.py decode --model MODEL --mtp SOURCE --pages runs/X/eval_final_val/pages.jsonl --n 10

accept  teacher forcing on saved answers: share of answer tokens i+2 the head predicts from the main model's state at
        i and token i+1 (an estimate of how often a draft is accepted).
decode  for each page, plain greedy decoding (generate.generate_pages) and draft-and-check decoding: each step the head
        proposes token n+2 and the main model checks it while writing token n+1. Committed tokens are always the
        main model's own greedy choice, so both answers must be identical; reports times and tokens per main pass.
--mtp: a .safetensors file or model folder holding mtp.* tensors, or a Hub id (default Qwen/Qwen3.5-0.8B: its
pretrained head, which ResuVL-L's own file does not keep). Writes --out (JSON) if given.

Two changes to transformers 5.5.4 are needed and applied here: the linear-attention (Gated DeltaNet) layers take
their cached state only for one new token, so the cached branch is run for 2-3 tokens too (continue_linear_attn);
and a rejected draft is undone by restoring those layers' saved states and cropping the attention cache.
Measured on an M2 Max CPU (2026-11-05): a main pass costs 51 / 79 / 109 ms for 1 / 2 / 3 tokens, so checking a draft
costs ~1.5 passes and the CPU gains nothing; on a GPU a 2-token pass should cost about the same as 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tiny_resume_parser.data import GOLD, LABELS, page_image, read_jsonl  # noqa: E402
from tiny_resume_parser.generate import generate_pages, load_model, prompt_text  # noqa: E402
from tiny_resume_parser.models import QWEN35_ID, family, messages  # noqa: E402
from tiny_resume_parser.mtp import Capture, MTPHead, pretrained_state  # noqa: E402

assert transformers.__version__ == "5.5.4", "continue_linear_attn follows Qwen3_5GatedDeltaNet.forward of 5.5.4"


def continue_linear_attn(self, hidden_states, cache_params):
    """Qwen3_5GatedDeltaNet.forward's cached branch, for any number of new tokens."""
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


def patch_linear_layers(model):
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


def restore(cache, saved, length):
    for i, c, r in saved:
        cache.layers[i].conv_states.copy_(c)
        cache.layers[i].recurrent_states.copy_(r)
    cache.crop(length)


@torch.no_grad()
def draft_decode(model, processor, head, cap, image, max_patches=12, max_new_tokens=3000):
    """Greedy answer with one MTP draft per main pass: (answer text, answer tokens, main passes, drafts accepted)."""
    embed = model.model.language_model.embed_tokens
    end_id = processor.tokenizer.convert_tokens_to_ids(family(processor)["end"])
    inputs = processor(
        text=[prompt_text(processor)], images=[image], return_tensors="pt", **family(processor)["size"](max_patches)
    ).to(model.device)
    out = model(**inputs, use_cache=True)
    cache = out.past_key_values
    toks = inputs["input_ids"][0].tolist()
    n_prompt = C = len(toks)  # C: positions in the main cache
    E = cap["embeds"][0]  # prompt input embeddings, image features included
    H, COS, SIN = cap["hidden"][0], cap["pe"][0][0], cap["pe"][1][0]  # per position with a correct prefix
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
        draft = int(model.lm_head(x[:, -1])[0].argmax())
        M = last + 1
        saved = save_linear_states(cache)
        o = model(
            input_ids=torch.tensor([pending + [draft]], device=model.device), past_key_values=cache, use_cache=True
        )
        pred = o.logits[0].argmax(-1).tolist()
        passes += 1
        p = len(pending)
        ok = pred[p - 1] == draft
        new = slice(H.shape[0] - C, p + 1 if ok else p)  # skip positions stored after an earlier rejected draft
        H = torch.cat([H, cap["hidden"][0][new]])
        COS, SIN = torch.cat([COS, cap["pe"][0][0][new]]), torch.cat([SIN, cap["pe"][1][0][new]])
        if ok:
            accepted += 1
            toks += [draft, pred[p]]
            C += p + 1
            pending = [pred[p]]
            if draft == end_id:
                toks.pop()
        else:  # commit the main model's own token and undo the draft in the main cache
            toks.append(pred[p - 1])
            restore(cache, saved, C)
            pending = pending + [pred[p - 1]]
    answer = toks[n_prompt:]
    answer = answer[: answer.index(end_id) + 1] if end_id in answer else answer
    answer = answer[:max_new_tokens]
    return processor.tokenizer.decode(answer, skip_special_tokens=True), len(answer), passes, accepted


@torch.no_grad()
def accept_rate(model, processor, head, cap, image, answer, max_patches=12):
    """Teacher forcing on a saved answer: (tokens i+2 the head predicts, answer tokens)."""
    fam = family(processor)
    text = processor.apply_chat_template(messages(answer), tokenize=False)
    batch = processor(text=[text], images=[image], return_tensors="pt", **fam["size"](max_patches)).to(model.device)
    ids = batch["input_ids"][0].tolist()
    header = processor.tokenizer(fam["header"], add_special_tokens=False).input_ids
    a = next(j for j in range(len(ids)) if ids[j : j + len(header)] == header) + len(header)
    b = ids.index(processor.tokenizer.convert_tokens_to_ids(fam["end"]), a)
    model(**batch)
    e, h, (cos, sin) = cap["embeds"], cap["hidden"], cap["pe"]
    pred = model.lm_head(head(e[:, 1:], h[:, :-1], (cos[:, :-1], sin[:, :-1])))[0].argmax(-1).tolist()
    return sum(pred[i] == ids[i + 2] for i in range(a - 2, b - 1)), b - a + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["accept", "decode"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp", default=QWEN35_ID, help=".safetensors file, model folder or Hub id holding mtp.*")
    ap.add_argument("--pages", type=Path, required=True, help="pages.jsonl written by evaluate.py")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"])
    ap.add_argument("--max-new-tokens", type=int, default=3000)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--fp32", action="store_true", help="run the model in float32 (tests whether bf16 rounding differs)"
    )
    ap.add_argument(
        "--answers", type=Path, help="decode: write DIR/greedy/pages.jsonl and DIR/mtp/pages.jsonl for rescore"
    )
    args = ap.parse_args()

    model, processor, device = load_model(args.model, args.device)
    if args.fp32:
        model.float()
    if args.answers:
        for sub in ("greedy", "mtp"):
            (args.answers / sub).mkdir(parents=True, exist_ok=True)
        files = {sub: (args.answers / sub / "pages.jsonl").open("w") for sub in ("greedy", "mtp")}
    if args.mode == "decode":
        patch_linear_layers(model)
    head = MTPHead(model.config.text_config)
    head.load_state_dict(pretrained_state(args.mtp))
    head = head.to(device=device, dtype=model.model.language_model.dtype).eval()
    cap = Capture(model).data
    images = {(r["cv_id"], r["page"]): r["image"] for p in (LABELS, GOLD) for r in read_jsonl(p)}
    rows = read_jsonl(args.pages)[: args.n]
    res = []
    for r in rows:
        image = page_image({"image": images[(r["cv_id"], r["page"])]})
        page = f"{r['cv_id']} p{r['page']}"
        if args.mode == "accept":
            right, n = accept_rate(model, processor, head, cap, image, r["answer"])
            res.append(dict(page=page, right=right, tokens=n))
        else:
            t0 = time.time()
            ref, n_ref = generate_pages(model, processor, [image], 12, args.max_new_tokens)[0]
            t1 = time.time()
            ans, n_ans, passes, acc = draft_decode(model, processor, head, cap, image, 12, args.max_new_tokens)
            t2 = time.time()
            res.append(
                dict(
                    page=page,
                    tokens=n_ref,
                    same=ans == ref,
                    greedy_s=round(t1 - t0, 2),
                    mtp_s=round(t2 - t1, 2),
                    passes=passes,
                    accepted=acc,
                )
            )
            if args.answers:
                for sub, text in (("greedy", ref), ("mtp", ans)):
                    files[sub].write(json.dumps(dict(cv_id=r["cv_id"], page=r["page"], answer=text)) + "\n")
                    files[sub].flush()
        print(json.dumps(res[-1]), flush=True)
    if args.mode == "accept":
        summary = dict(right=sum(x["right"] for x in res), tokens=sum(x["tokens"] for x in res))
        summary["share"] = round(summary["right"] / summary["tokens"], 4)
    else:
        summary = dict(
            same=sum(x["same"] for x in res),
            greedy_s=round(sum(x["greedy_s"] for x in res), 1),
            mtp_s=round(sum(x["mtp_s"] for x in res), 1),
            passes=sum(x["passes"] for x in res),
            accepted=sum(x["accepted"] for x in res),
        )
    summary = dict(mode=args.mode, model=args.model, mtp=args.mtp, device=device, pages=len(res)) | summary
    print(json.dumps(summary))
    if args.out:
        args.out.write_text(json.dumps(dict(summary=summary, pages=res), indent=1))


if __name__ == "__main__":
    main()
