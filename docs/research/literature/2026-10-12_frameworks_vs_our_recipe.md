<!-- Research by GLM-5.3-Flash (plan mode, 2026-10-12), brief: what ready-made fine-tuning frameworks do that our script does not.
Checked on 2026-10-12 against the installed code and data:
- HF processor default max_patches = 12 (confirmed). At max 12, training pages get 7 tiles (1,290), 13 tiles (162), 9-10 (16), 3-4 (4):
  raising our cap 6 -> 12 changes only the 178 pages (12%) that want more than 7 tiles.
- Gradient-accumulation loss normalisation: trainer.py:636-643 and 4061 present as quoted (transformers 4.57.6) -> not a bug in our stack.
- Framework URLs (Axolotl, ms-swift, LLaMA-Factory, TRL) were read by GLM sub-agents, not re-opened afterwards. -->
# Ready-made fine-tuning frameworks vs. our InternVL3-1B CV→JSON recipe (2026-10-12)

**TL;DR:** Stay on our script. No framework both supports `InternVL3-1B-hf` and beats our recipe on the things that touch our measured errors; the one accuracy-relevant default we lack everywhere (per-epoch eval + best-checkpoint selection) is ~30 lines to add. The suspected gradient-accumulation loss-normalization bug is **not present** in our stack — verified end-to-end in the installed transformers 4.57.6 / peft 0.21.2 (details in §4, "cleared").

Provenance note: framework claims below were verified by three web-research sub-agents against GitHub code/docs/papers (URLs inline); I independently verified only our side (repo files + installed libraries).

## 1. Framework support: InternVL3-1B and SmolVLM

| Framework | InternVL3-1B | Weights used | SmolVLM | LoRA / full | Maturity signals |
|---|---|---|---|---|---|
| **LLaMA-Factory** | **Yes** — `InternVL3-1B-hf` registered in `src/llamafactory/extras/constants.py`; support PR [#7258](https://github.com/hiyouga/LLaMA-Factory/pull/7258) (Apr 2025) | HF-converted (`-hf`) | **No** (only text smollm2 templates) | Both | 75.4k★, pushed 2026-10-10 |
| **ms-swift** | **Yes** — [examples/models/internvl3/train.sh](https://github.com/modelscope/ms-swift/blob/main/examples/models/internvl3/train.sh) | **Original trust-remote-code** (`InternVL3-1B-Instruct`, template `internvl2_5`) — not our `-hf` port | No (code search 0) | Both + DoRA/rsLoRA/PiSSA/LoRA-GA | 15.8k★, v4.5.3 (2026-09), pushed 2026-10-11 |
| **XTuner** | **No** — legacy configs only InternVL v1.5/v2; new v1 pipeline only InternVL3.**5**-1B/8B ([internvl_config.py](https://github.com/InternLM/XTuner/blob/main/xtuner/v1/model/compose/internvl/internvl_config.py#L122-L147)) | 3.5 = HF weights | No | Both | 5.2k★, active but mid-refactor to v1 |
| **InternVL repo (own scripts)** | **Yes** — [`shell/internvl3.0/2nd_finetune/internvl3_1b_dynamic_res_2nd_finetune_full.sh`](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_1b_dynamic_res_2nd_finetune_full.sh) | Original | n/a | **Full only** for 3.0 (1B LoRA script exists only for 2.5) | **Stalled: last push 2025-09-22** |
| **Unsloth** | **No** — InternVL fails to load, [issue #2996](https://github.com/unslothai/unsloth/issues/2996); no notebook | — | No | Both (Qwen2.5/3-VL, Gemma3, Llama-3.2-Vision, Pixtral) | 77.8k★, near-daily pushes |
| **Axolotl** | **No** (only an InternVL3.**5**-8B QLoRA [example](https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/internvl3_5/internvl3_5-8b-qlora.yml); InternVL3 tracked in [#3105](https://github.com/axolotl-ai-cloud/axolotl/issues/3105)) | 3.5 = HF | **Yes** ([smolvlm2-2B-lora.yaml](https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/smolvlm2/smolvlm2-2B-lora.yaml)) | LoRA/QLoRA/full | 12.6k★, daily pushes |
| **HF TRL SFTTrainer** | **Works** — model-agnostic for transformers-native VLMs; InternVL in transformers since **4.52.0** (ours: 4.57.6); no SFT-specific InternVL blocker found (InternVL+TRL traffic is GRPO-focused: [#4061](https://github.com/huggingface/trl/issues/4061)) | HF (ours exactly) | **Yes** — official [SmolVLM cookbook](https://github.com/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_smol_vlm_sft_trl.ipynb) + docs VLM guide | Both | 19.5k★, daily pushes |
| **torchtune** | No (vision recipes = Llama-3.2-Vision only) | — | No | Both | 5.8k★, cadence slowed (last push 2026-09-09) |
| Document-specific (olmOCR training, granite-docling) | **No** — olmOCR trains Qwen2.5-VL-7B ([olmocr/train](https://github.com/allenai/olmocr/tree/main/olmocr/train)); granite-docling's training code is not open (nanoVLM at IBM) | — | — | — | — |

Only LLaMA-Factory and TRL can train **our exact artifact** (`InternVL3-1B-hf`); ms-swift would force the original weights; everything else doesn't support the model family.

## 2. Framework defaults / recommended VLM-SFT settings (quoted from code & configs)

| Setting | LLaMA-Factory | ms-swift | XTuner (v1) | InternVL official 3.0-1B | Unsloth | Axolotl (VLM examples) | TRL | torchtune (Llama-V vision) |
|---|---|---|---|---|---|---|---|---|
| LR (LoRA SFT) | 1e-4 (example yaml) | 1e-4 [default] | 1e-4 constant | — (full only) | 2e-4 [recommended] | 2e-4 | 2e-5 default / **1e-4 LoRA tip** | 1e-4 |
| Separate ViT / projector LR | **No** on main (code search 0) | **Yes**: `vit_lr`/`aligner_lr` param groups ([optimizers/multimodal.py](https://github.com/modelscope/ms-swift/blob/main/swift/optimizers/multimodal.py#L59-L62)) | no (all-trainable default) | — | no | no | no | no |
| Frozen by default | ViT + projector | ViT + aligner (`freeze_vit/freeze_aligner=True`) | **nothing** (all trainable) | ViT only (`freeze_backbone True`); LM+projector trained | nothing (LoRA on vision **and** LM) | LoRA regex = LM only | your LoraConfig decides | **decoder frozen**, encoder+fusion LoRA'd |
| LoRA rank/α defaults | rank 8, α=2·rank | rank 8, **α=32** (scale 4) | legacy: r128/α256, LM only | r16 (2.5-1B script, lr **4e-5**, 1 epoch) | r16, α16, dropout 0, all-linear | r32, α16, dropout 0.05, attn+MLP | your LoraConfig (cookbook: r8/α8) | r8, α16, q/v/o+MLP |
| DoRA / rsLoRA / LoRA+ / PiSSA | all flag-gated, off | all supported, off | some | no | DoRA only | DoRA/rsLoRA flags | PEFT pass-through | rsLoRA flag |
| Epochs / schedule (examples) | 3, cosine, warmup 0.1 | 1, warmup 0.05 | 1, constant, warmup 0 | 1, cosine, warmup 0.03, wd 0.05, batch 128 | 1–3 | 1, cosine, warmup 0.1, wd 0 | 3, linear | 1, cosine, warmup 100 steps |
| Packing / bucketing | `packing=None`, `neat_packing` opt-in | binpack available, off | `pack_max_length` (packed by default in v1) | `group_by_length True` | none for vision | **`sample_packing: false` in every VLM example** (open PR #3817) | **hard-blocked for VLMs** ([ValueError, sft_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py)) | none |
| Dynamic res / tiles | pixel caps 768² default | pixel caps / `MAX_PIXELS` | **max_dynamic_patch 12** | **max_dynamic_patch 12**, 448px, thumbnail | processor default | processor default | `max_length=None` required (else image tokens truncate → error) | — |
| Loss masking | `train_on_prompt=False` [default] | response-only | response-only | response-only | `train_on_responses_only=True` in vision collator | **historic collator bug [#3617](https://github.com/axolotl-ai-cloud/axolotl/issues/3617): most MM models trained on full conversation** | `completion_only_loss` auto for prompt-completion; `assistant_only_loss` needs `{% generation %}` | response-only |
| Label smoothing / NEFTune | NEFTune exposed | NEFTune exposed | — | — | — | NEFTune flag | `neftune_noise_alpha`, `label_smoothing_factor` (both off) | — |
| Liger kernels | flag, off | flag, off | — | — | internal | flag | flag, off | — |
| Eval / checkpointing | per-epoch optional, eval commented out in example; early-stop callback | eval tied to save, `metric_for_best_model='loss'`, `early_stop_interval` | step-based | save_steps 200, eval "no" | — | **`evals_per_epoch: 1`, `saves_per_epoch: 1`** in all three VLM examples | **eval off by default**, save_steps 500 | `run_val_every_n_steps: null` |
| Chat template | **its own `intern_vl` template**, not the checkpoint's | own `internvl2_5` | own | own | patched processors | own | **the model's own processor/template** | own |

Cross-framework reading: LoRA-SFT LR consensus is **1e-4–2e-4** (InternVL-official's 4e-5 is the outlier, prior research rec #4); LoRA α between 1r and 2r is normal (ms-swift's α=32 at rank 8 = scale 4 is the aggressive end); response-only loss is a first-class default everywhere; **nobody enables early stopping by default**, and per-epoch eval is the exception (Axolotl examples, ms-swift), not the rule.

## 3. Document/OCR training practices relevant to our three error classes

**Wrong-section placement** — honest finding: **no published fine-tuning study isolates wrong-field-placement errors** ([agent search](https://tianpan.co)). Nearest evidence:
- **SmolDocling/DocTags** [arXiv 2503.11576]: explicit markup "explicitly separate[s] textual content from document structure … minimizing confusions" — but that is about a general markup *format* for pretraining-scale data; prior research rec #12 (keep minified JSON for a fixed fine-tuned schema) still stands, and constrained decoding already caps format damage.
- **InternVL2.5 report** [arXiv 2412.05271] data filtering: "even a few thousand noisy samples can lead to aberrant model behavior during inference" — LLM-based quality scoring + repetition detection on training data. Our teacher labels were reviewed only on the 254-page gold set, not the 1,393 training pages.
- **olmOCR-2** [arXiv 2510.19817]: generates **unit tests** per document and uses them as RL rewards (30,381 tests, reward = fraction passing). The same invariants (e.g., "an entry with `startDate`+`position` belongs in `work`, not `education`") are usable as *data checks* — directly aimed at placement.

**Misread characters** (~0.08 F1 of our gap is sim≥0.8 near-misses):
- **Resolution budget**: InternVL's own recipe sets n_max=12 uniformly, and "12 or 24 for high-resolution documents" vs "6 or 12 for low-res" [InternVL2.5 report]; the HF port's preprocessor default is `max_patches: 12`; the official 3.0-1B finetune script uses `max_dynamic_patch 12`. **Our cap of 6 is a deliberate downsize** — its own recipe note said "revisit only if small text on letter pages is misread", and misreads are now measured. 162/1,472 training pages (11%) want 12+1 tiles and get downgraded to 6+1 (`train_lora.py:32`).
- **Tiny-LR vision tuning**: SmolDocling unfroze the ViT at 2e-6 vs text 2e-4 (100×); ms-swift ships `vit_lr`; Qwen2.5-VL official has `vision_tower_lr`. But InternVL's own 1B scripts freeze the ViT everywhere — mixed signals, so this is a later lever, not a default.
- **olmOCR "document anchoring"** (injecting the PDF text layer) cuts hallucinated characters [arXiv 2502.18443 §2.2] — **not applicable** to our image-only inputs without changing the task.

**Repetition loops** (already mostly solved: 2/50 pages hit the token limit under constrained decoding, DESIGN_NOTES Part 3):
- **Nougat perturbation** [arXiv 2308.13418]: ours (0.1, input-only, labels correct) is a deliberate variant of Nougat's; paper value matches.
- **olmOCR-2** found output format changes loop rates ("with JSON there were more incidences of repetition loops"; YAML "reduced the retry rate dramatically") and retries at dynamic temperature 0.1→0.8 — inference-side; our constrained decoding occupies this slot.
- **MPO** (InternVL2.5, [arXiv 2411.10442]): DPO alone produced "16.4% … gibberish or repetitive outputs" vs **0.3% for MPO**, and "the SFT loss is a key component" — relevant when we do the preference stage already planned (DESIGN_NOTES Part 1 #3): keep an NLL term in the loss.
- **GOT-OCR2** [arXiv 2409.01704]: 3-stage curriculum (encoder pretrain → joint → decoder post-train with frozen encoder) — data-scale-dependent, not transferable at 1.4k pages.

## 4. Gap list (ranked by expected effect on our measured errors)

| # | Practice | Who does it (source) | Our status (file:line) | Expected effect | Cost | Evidence strength |
|---|---|---|---|---|---|---|
| 1 | **Per-epoch eval + best-checkpoint selection on held-out field F1** | Axolotl VLM examples `evals_per_epoch: 1` + `saves_per_epoch: 1` ([yaml](https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/smolvlm2/smolvlm2-2B-lora.yaml)); ms-swift `metric_for_best_model` + `early_stop_interval` ([sft_args.py L406-434](https://github.com/modelscope/ms-swift/blob/main/swift/arguments/sft_args.py#L406-L434)) | `train_lora.py:184` — `save_strategy="no"` by default, no `eval_strategy`, no `load_best_model_at_end` anywhere; gold eval happens once per finished run (DESIGN_NOTES Part 3) | Model selection directly on placement + misread errors; empirically settles "2 vs 4 epochs" (loss 0.11 still falling, F1 0.58 — epoch 3 vs 4 untested) | Low — `evaluate.py` exists; save adapter per epoch, score each, or a TrainerCallback | **Strong** (protocol; our own prior rec #2, half-adopted) |
| 2 | **Raise tile budget 6 → 12 for tall pages** | InternVL2.5 report §2.1 (n_max 12; 12–24 for documents); HF preprocessor default `max_patches: 12`; official 3.0-1B script `max_dynamic_patch 12` | `train_lora.py:32` (`MAX_PATCHES = 6`); recipe's own revisit trigger ("small text misread") has now fired (sim≥0.8 gap = 0.08 F1) | Misreads, esp. on the 11% of pages (162/1,472) downgraded from 13 to 7 tiles; secondarily layout→section binding | Medium — seq to ~5.6k tokens on those pages; memory + time | **Medium-strong** (vendor recipe + default vs our deliberate downsize) |
| 3 | **Teacher-label placement verification (code-checkable invariants)** | InternVL2.5 data-quality filtering quote (arXiv 2412.05271); olmOCR-2 unit-test generation (arXiv 2510.19817) | Data side — training labels unreviewed for section placement (gold only; DESIGN_NOTES Part 1 #5); model writes extra `work` entries on 8–10/44–48 gold pages | **Wrong-section** — if the teacher systematically misplaces entries, the student learns it; invariants catch it for free where code can check | Low for heuristics; teacher re-run costs money (operator call) | Medium (no direct study of placement noise; strong analogies) |
| 4 | **Separate LR per component** (projector below LoRA LR; ViT ≈1e-6 as a later lever) | ms-swift `vit_lr`/`aligner_lr` ([optimizers/multimodal.py](https://github.com/modelscope/ms-swift/blob/main/swift/optimizers/multimodal.py#L59-L62)); Qwen2.5-VL official `mm_projector_lr`/`vision_tower_lr` ([argument.py](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune)); SmolDocling text 2e-4 vs ViT 2e-6 (arXiv 2503.11576) | `train_lora.py:156-157,174` — one LR (1e-4) for both the LoRA and the fully-trained projector (13.3M params) | Uncertain sign; may stabilize the projector that binds layout→sections (run v3 changed two things at once, gain unattributed) | Low — param groups in the optimizer | Medium (framework + paper practice; no controlled result at our scale) |
| 5 | **Inference repetition guard** (Nougat logit-variance stop B=15/6.75; olmOCR-2 temp-escalation retry) | Nougat §5.4 (arXiv 2308.13418); olmOCR-2 §4 (arXiv 2510.19817) | Absent; constrained decoding already cut loops to 2/50 hitting the limit (`train_lora.py:89-113` has no detector, only `max_new_tokens`) | Small residual cleanup; also catches *free*-decoding loops | ~20 lines | Medium |
| 6 | **Preference stage with an SFT-loss term (MPO-style)** | InternVL MPO paper (arXiv 2411.10442): DPO 16.4% repetitive vs MPO 0.3%; InternVL3-1B-hf card tags MMPR-v1.2 | Planned, not started (DESIGN_NOTES Part 1 #3) — MPO's result says: don't do plain DPO, keep the NLL term | Repetition residue + placement (pairs = right-vs-wrong section) | High (needs pairs + stage 2 training) | **Strong paper, wrong scale** (shown at 8B+, untested at 1B/1.4k pages) |
| 7 | **NEFTune** (`neftune_noise_alpha`) | Plain HF TrainingArguments (installed trainer.py:648 wires it), TRL/LF expose it | Absent | Unknown for extraction; cheap A/B under gap-1's metric | One flag | Low-medium (chat evals only; prior research rec #11 judged unproven — unchanged) |

**Checked and cleared — NOT gaps** (each verified against our code or the installed libraries):
- **Grad-accum loss normalization**: correct in our stack, see below.
- **Answer-only loss incl. closing `<|im_end|>`** (`train_lora.py:74-85`) — equals TRL `completion_only_loss`, Unsloth `train_on_responses_only`.
- **LoRA targets** = LM attention+MLP only (`train_lora.py:37`) — matches Unsloth/Axolotl practice and avoids ms-swift's `all-linear` (which would hit the ViT).
- **α=2r** — in-family (LLaMA-Factory α=2r; Unsloth α=r; ms-swift α=4r at rank 8).
- **No packing** — TRL hard-blocks packing for VLMs; at batch size 1 padding waste is zero.
- **No truncation** — matches TRL's VLM advice (`max_length=None`).
- **No early stopping** — frameworks don't enable it by default either (gap 1 is about eval+selection, not ES).
- **lr 1e-4, cosine, warmup 3%, wd 0.01, 4 epochs** — inside the framework LoRA-SFT envelope (1e-4–2e-4, 1–3 epochs); the 2.5×-higher-than-official concern from prior research rec #4 is best settled empirically by gap 1, not by another guess.
- **Chat template** — we use the model's own processor template for train and inference; LLaMA-Factory/ms-swift substitute their own (a switch risk, not a gap).

**The specific check the brief asked for — gradient accumulation under transformers 4.57.6, verified in the installed libraries:**
- `trainer.py:636-643`: `model_accepts_loss_kwargs` = forward has `**kwargs` → `InternVLForConditionalGeneration.forward` has `**kwargs: Unpack[TransformersKwargs]` (`modeling_internvl.py:830`); PEFT wrapper also has `**kwargs` (`peft_model.py:1903`, and Trainer unwraps PEFT anyway).
- `trainer.py:4060-4064`: the mean-of-means fallback (`loss / current_gradient_accumulation_steps`) applies only when `not model_accepts_loss_kwargs` — not our case. `num_items_in_batch` is injected over the whole accumulation window.
- `modeling_internvl.py:899-902` passes kwargs to the loss; `loss_utils.py:35-37` uses `reduction="sum"` then divides by `num_items_in_batch` → **token-weighted normalization across all 16 micro-batches is correct.** It would break only if we added a `compute_loss_func` or overrode `compute_loss` — we don't.
- Fix first shipped in transformers 4.46 ("Gradient accumulation bugfix": [PR #34191](https://github.com/huggingface/transformers/pull/34191), [PR #34283](https://github.com/huggingface/transformers/pull/34283) in the [v4.46.0 release notes](https://github.com/huggingface/transformers/releases/tag/v4.46.0)).

## 5. Recommendation

**Stay on our hand-written script. Adopt gaps 1, 2, 3 (and cheaply 4); skip a framework migration.** One line each:
1. **Eval + checkpoint selection** — save the adapter per epoch, run `evaluate.py --set val` on each, pick by field F1 (gold stays untouched as final judge). Biggest expected effect on our measured errors per unit of work; also makes every later A/B (tiles, LRs, NEFTune) decidable.
2. **`MAX_PATCHES` 6 → 12 A/B** — the recipe's own revisit trigger fired; ~11% of pages get real resolution back.
3. **Training-label placement checks** — free, code-checkable invariants over the 1,393 teacher labels; fix or down-weight violations before the next GPU run.
4. **Split projector LR** (~1e-5) from LoRA LR — small param-group change; attribute v3-style gains properly by changing one thing per run.

**Honest downside of staying:** we keep re-implementing plumbing frameworks ship (eval loops, param groups, two-stage recipes) and carry the maintenance/testing burden ourselves; if this later needs multi-GPU, Megatron-style sequence parallel, or a second model family, ms-swift/LLaMA-Factory become the cheaper home. **If we ever switch:** TRL SFTTrainer is the only migration that keeps `InternVL3-1B-hf` and our own template (use prompt-completion datasets for `completion_only_loss`; keep the perturbation collator custom; expect zero accuracy delta — the loss math is identical, verified above). LLaMA-Factory supports our exact checkpoint but substitutes its own template and has no SmolVLM path; ms-swift requires the original weights; Unsloth cannot load InternVL; XTuner targets 3.5 only.

