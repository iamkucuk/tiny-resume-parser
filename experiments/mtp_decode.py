"""Qwen3.5's MTP head as a drafter for ResuVL-L: does it make greedy decoding faster without changing the answer?

    python experiments/mtp_decode.py accept --model MODEL --mtp SOURCE --pages runs/X/eval_final_val/pages.jsonl --n 5
    python experiments/mtp_decode.py decode --model MODEL --mtp SOURCE --pages runs/X/eval_final_val/pages.jsonl --n 10

accept  teacher forcing on saved answers: share of answer tokens i+2 the head predicts from the main model's state at
        i and token i+1 (an estimate of how often a draft is accepted).
decode  for each page, plain greedy decoding (generate.generate_pages) and draft-and-check decoding (--drafts K
        tokens drafted per main pass). Committed tokens are always the main model's own greedy choice, so in float32
        (--fp32) both answers are identical; in bf16 some differ by rounding. Reports times, main passes and
        accepted drafts; --answers writes both answer sets for rescore.
--mtp: a .safetensors file or model folder holding mtp.* tensors, or a Hub id. Default Qwen/Qwen3.5-0.8B: the base
model's pretrained head. ResuVL-L's weight file holds a head trained for it (release 0.3.0). Writes --out (JSON).

The drafting itself is the package's (src/tiny_resume_parser/draft.py, parse --drafts); this script times it
against plain decoding and scores both answers. Measured on an M2 Max CPU (2026-11-05): a main pass costs 51 / 79 /
109 ms for 1 / 2 / 3 tokens, so the CPU gains nothing; results on an H100: docs/findings.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tiny_resume_parser.data import GOLD, LABELS, page_image, read_jsonl  # noqa: E402
from tiny_resume_parser.draft import draft_decode, patch_linear_layers  # noqa: E402
from tiny_resume_parser.generate import generate_pages, load_model  # noqa: E402
from tiny_resume_parser.models import QWEN35_ID, family, messages  # noqa: E402
from tiny_resume_parser.mtp import Capture, MTPHead, pretrained_state  # noqa: E402


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
    ap.add_argument("--drafts", type=int, default=1, help="decode: MTP drafts per main pass (the head is chained)")
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
            ans, n_ans, passes, acc = draft_decode(
                model, processor, head, cap, image, 12, args.max_new_tokens, args.drafts
            )
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
    summary = (
        dict(mode=args.mode, model=args.model, mtp=args.mtp, drafts=args.drafts, device=device, pages=len(res))
        | summary
    )
    print(json.dumps(summary))
    if args.out:
        args.out.write_text(json.dumps(dict(summary=summary, pages=res), indent=1))


if __name__ == "__main__":
    main()
