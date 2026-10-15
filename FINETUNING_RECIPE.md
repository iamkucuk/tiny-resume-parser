# Fine-Tuning Recipe: InternVL3-1B for CV page → ATS JSON

> Rewritten 2026-10-11 (the granite-docling recipe it replaces is gone). Facts below were
> researched by GLM (Z Code session `sess_e829c33d`) and the load-bearing ones re-checked
> on the installed libraries (transformers 4.57.6, torch 2.14.1). Unchecked claims are
> marked *(unverified)*. Update this file from real runs.

## Summary
LoRA on the language model of `OpenGVLab/InternVL3-1B-hf`, vision encoder and projector
frozen, loss only on the answer tokens. Target = one-line JSON with empty keys left out;
code restores the full template with `to_record`. Plain HF `Trainer` + our own collator
(not TRL, see "Why not TRL"). Compare the model with the adapter on and off to see what
changed. Escalate to a full language-model fine-tune only if LoRA underfits.

## Model facts (checked locally)
| Item | Value |
|---|---|
| Classes | `AutoModelForImageTextToText` → `InternVLForConditionalGeneration`; `AutoProcessor` → `InternVLProcessor` |
| Parts | `model.vision_tower` (InternViT, 24 layers), `model.multi_modal_projector` (~4.5 M params), `model.language_model` (Qwen2.5-0.5B, 24 layers), tied `lm_head` |
| Chat template | `<\|im_start\|>user\n<IMG_CONTEXT>\n{prompt}<\|im_end\|>\n<\|im_start\|>assistant\n{answer}<\|im_end\|>\n` — no system message |
| Special ids | `<IMG_CONTEXT>` 151667, `<img>` 151665, `</img>` 151666, `<\|im_end\|>` (eos) 151645 |
| Tiling | The processor tiles by default (`crop_to_patches=True` set by `InternVLProcessor`, although the image processor's own attribute is False). Tile 448², 256 tokens per tile, +1 thumbnail when >1 tile. Grid depends only on aspect ratio. |
| Patches per training page (max_patches=12) | 7 patches: 1,290 pages · 13: 162 · 10: 11 · 9: 5 · 4: 3 · 3: 1 |

So a typical page is 7 × 256 = 1,792 image tokens; US-letter-shaped pages get 13 × 256 =
3,328. **Decision: `max_patches=6` for training and inference** — every page then gets at
most 7 patches (1,792 tokens), the same resolution 88% of pages already get, and the
official InternVL 1B LoRA script uses 6. Revisit only if small text on letter pages is misread.

## Data format
- One sample = one page image + fixed instruction → answer.
- **Answer** = minified JSON of the record with every `null`, `[]` and `{}` key removed
  (median 887 tokens, p90 1,287, max 2,167 on the 1,472 training pages). A helper
  `to_sparse_target` is still to be written next to `to_target` in `src/schema.py`;
  after generation, `to_record` refills the removed keys and validates.
- `corrections` is **left out of the answer** (`STUDENT_SKIP` in `src/train_lora.py`; decision 2026-10-11).
  It stays in the teacher labels for auditing. Reason: run `lora_v1` (with it) repeated one correction entry
  until the token limit on 8 of 10 test pages; `to_record` refills it as `[]`.
- Instruction: short and fixed (not the teacher's long rules prompt); the schema is learnt from data.
- Validation split: 5% of training CVs, split by CV, never by page.
- Sequence length: ≤ ~1,800 image + ~60 prompt + ≤ 2,170 answer ≈ 4,050 tokens. No truncation.

## Collator and label masking
Our collator must:
1. Render prompt + answer with `processor.apply_chat_template`, run the processor with the image.
2. `labels = input_ids.clone()`, then set −100 on everything up to and including
   `<|im_start|>assistant\n` (this covers `<img>`, all `<IMG_CONTEXT>`, `</img>`, the prompt)
   and on padding. Keep the closing `<|im_end|>` as a label so the model learns to stop.
3. Pass `pixel_values`, `attention_mask` through.
Check per batch: the number of `<IMG_CONTEXT>` tokens = patches × 256, or the model raises
"Image features and image tokens do not match".

## Why not TRL
TRL and PEFT are not installed. TRL's VLM collator puts loss on the whole sequence by
default; its `assistant_only_loss` needs `{% generation %}` markers that this chat template
lacks, and for vision models needs a newer transformers than 4.57.6 *(TRL docs, unverified
locally)*. Our own collator is ~30 lines and makes the masking explicit.

## LoRA config (PEFT)
- Install: `peft` (needs operator approval; not installed).
- Targets: the 7 linear layers of every LM block, **LM only**. The vision tower also has
  `q_proj/k_proj/v_proj`, so a plain name list would adapt it too. Use a regex:
  `target_modules=r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"`.
- r = 16, alpha = 32, dropout = 0.05 (official InternVL2.5-1B LoRA script uses r 16; alpha = 2r
  and dropout 0.05 are its code defaults *(unverified)*).
- Projector frozen, as in the official scripts. Option if loss stalls: `modules_to_save=["multi_modal_projector"]`.
- Gradient checkpointing with a frozen tower: call `model.enable_input_require_grads()`
  before `get_peft_model()`, else "element 0 of tensors does not require grad".

## Hyperparameters (starting point)
| Setting | Value | Reference |
|---|---|---|
| lr | 1e-4, cosine, warmup 3% | official LoRA script: 4e-5 at batch 16; full FT: 1e-5 at batch 128 |
| Effective batch | 16 (per-device 1–2 × grad-accum) | official LoRA script: 16 |
| Epochs | 2 (1,400 pages ≈ 175 steps) | official: 1 epoch on far larger sets |
| Weight decay | 0.01 | official LoRA script |
| Precision | bf16 | |
| Attention | SDPA (flash-attention 2 needs CUDA) | |

Memory estimate *(arithmetic, not measured)*: LoRA at ~4,000 tokens, batch 1, bf16,
checkpointing ≈ 8 GB, peak 10–11 GB, mostly the 4,096 × 151,674 logits. Full LM fine-tune
≈ 10–14 GB.

## Hardware
- Apple MPS: bf16 should work on macOS 26; no report of InternVL-hf training on MPS exists.
  Use it for the smoke test only if a step takes < ~10 s; otherwise a CUDA GPU.
- Full run: one CUDA GPU with ≥ 24 GB (not yet chosen).

## Smoke test (gate before the full run)
~50 training pages, 2 epochs, the config above. Pass when all hold:
1. Training loss falls steadily; no NaN.
2. In one batch, labels are −100 on every image and prompt token and real on the answer (assert in code).
3. Generated answers for 10 training pages parse and pass `to_record`.
4. Answers differ across pages (not one memorised record).

## Observing the behaviour change (LoRA on vs off)
Same weights, `with model.disable_adapter():` for the base model. On 20 gold pages compare:
JSON validity, per-field scores (the evaluation metrics), output length, generation time.
Also run ~10 general prompts (describe an image, answer a question about a non-CV image)
to see what the adapter changed outside the task.

## Inference and deployment
- Same processor settings (`max_patches=6`), greedy decoding, `use_cache=True`,
  `max_new_tokens` ≈ 2,500, stop at `<|im_end|>`.
- Then `to_record(json.loads(output))`; a parse failure is reported, not repaired silently.
- Deploy merged: `model = model.merge_and_unload()` (returns a new model; assign it).
  Loading an un-merged adapter directory gave NaN sampling for InternVL2-4B (OpenGVLab/InternVL#621).

## When LoRA underfits
Full LM fine-tune: vision tower frozen, LM + projector trained, lr 1e-5, cosine, warmup 3%,
weight decay 0.05 (official InternVL3-1B full fine-tune script).

## Sources
- Official scripts: `internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_1b_dynamic_res_2nd_finetune_full.sh`,
  `internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_lora.sh` (github.com/OpenGVLab/InternVL)
- Installed source: `transformers/models/internvl/{modeling,processing}_internvl.py`, `models/got_ocr2/image_processing_got_ocr2.py`
- TRL SFT docs (huggingface.co/docs/trl/main/en/sft_trainer); PEFT LoRA docs; peft#137 (input require grads)
- LoRA vs full: QLoRA (arXiv 2305.14314), "LoRA Learns Less and Forgets Less" (arXiv 2405.09673)
