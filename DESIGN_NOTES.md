# Design notes: what to improve, and why the project is built this way

> Written 2026-10-11, during the smoke-test stage of the InternVL3-1B LoRA run. Part 1 lists
> improvements not yet done; Part 2 explains the current design (training stage, choices, reasons).

## Part 1 — What could be done better

| # | Improvement | Why | Status |
|---|---|---|---|
| 1 | **Smoke-test gate = overfit test.** Train on a few pages many times and require the model to reproduce them. | The first gate (valid JSON after 26 steps on 50 pages) tested learning, not the pipeline: 0/10 valid, but no bug found — the model kept generic JSON-Resume-like keys (`"type":"person"`, `"fluency"`) and repeated entries. | Running: 50 pages × 8 epochs (`runs/smoke_e8/`) |
| 2 | **Constrained decoding** at inference with a JSON-schema grammar (e.g. outlines, xgrammar). | Only tokens that keep the output valid against `src/schema.py` are allowed, so invalid JSON cannot occur. This is the failure seen in the first smoke test. | Not started; needs a new package (ask first) |
| 3 | **Task-level RL or preference stage after SFT.** GRPO with rewards computed by code: passes `to_record`, field-level F1 vs label, no repeated entries. Cheaper option: DPO on (valid, invalid) answer pairs from the SFT model. | The task has rewards a program can compute; SFT alone does not penalise invalid or repeated output. | Not started |
| 4 | **Synthetic CV pages with exact labels.** Generate records, render them as pages in many layouts, mix with real pages. | Labels are correct by construction, so no teacher errors. Synthetic layouts differ from real ones, so mix, don't replace. | Not started |
| 5 | **Less noise in teacher labels.** Several teacher runs (or two teachers), keep values they agree on. | The teacher repeats the same mistakes, and the student will learn them: grammar-fixing school names (National→Nationale), dropping doubled letters in emails, translating names (Tunisie→Tunisia), 6-dot ratings read as /5. Measured: ~0.4% wrong values on full-size pages; 96 fixes on 61 of 254 gold pages. | Gold set fixed by review; training labels not |
| 6 | **Zero-shot JSON baseline on the gold set**, with the same metrics as the fine-tuned model. | Only zero-shot page *reading* was measured (granite-docling 0.96, InternVL3-1B 0.87, SmolVLM-500M 0.16). Without the JSON baseline, the gain from fine-tuning cannot be stated exactly. | Not started |
| 7 | **Multi-page CVs.** Design how per-page records merge into one record per CV. | The model works one page at a time. | Not designed |

## Part 2 — Current design

### Training stage
Supervised fine-tuning (SFT) by **knowledge distillation**: a large teacher (GLM-5.3-Flash) labeled
the pages; the small student (InternVL3-1B) learns the teacher's text answers. Sequence-level, not
logit-level: the teacher API returns no token probabilities, and the tokenizers differ. No RL yet.

| Stage | What happens | Who did it |
|---|---|---|
| 1. Pretraining | Learn language and images from very large data | OpenGVLab (language part: Qwen2.5-0.5B) |
| 2. General instruction tuning (SFT) + preference tuning | Follow instructions | OpenGVLab *(from the InternVL3 paper; not checked in detail)* |
| 3. **Task-specific SFT** | CV page image → our JSON format | **This project, now** |
| 4. Task RL / preference tuning | Improve measurable task scores | Not done (Part 1, item 3) |

### Choices and reasons
1. **Teacher labels** — the CVs had no labels; GLM labeled all 1,726 full-size pages.
2. **Fixed format** (`src/schema.py`, based on JSON Resume plus fields from a survey of this data).
   `to_record` rejects unknown keys and bad dates, so wrong output fails loudly.
3. **Label quality checked before training** — Haiku sub-agents compared labels with page images;
   claims were checked by hand (about half were false). Gold set (254 pages) fully reviewed:
   `data/processed/labels_gold_v6_verified.jsonl`. Train/gold split by CV, never by page.
4. **Model** — InternVL3-1B first; SmolVLM-500M later as the smaller option.
5. **LoRA, language model only** — r 16, 8.8 M trainable parameters (0.93%); vision encoder and
   projector frozen. Cheap, fits a laptop, changes less of what the model knows; the adapter can be
   switched off to compare. Full fine-tune only if LoRA underfits.
6. **Loss only on the answer** — image tokens, prompt and chat header get label −100. Own collator,
   because TRL's assistant-only loss needs `{% generation %}` markers this chat template lacks.
7. **Shorter target** — null/empty keys dropped (26% fewer tokens); `to_record` adds them back.
8. **Fixed resolution** — at most 6 tiles of 448 px + thumbnail (≤ 1,792 image tokens); 88% of
   pages already had this.
9. **Smoke test before the full run** — the full run takes ~4 h on this Mac; a short run checks the
   pipeline first. Details: `FINETUNING_RECIPE.md`.

## Part 3 — Run log

| Run | Setup | Result on 10 training pages |
|---|---|---|
| `lora_v1` (Colab L4, 2026-10-11) | 1,393 pages, 2 epochs, 176 steps, 40 min; loss 0.80 → 0.22 | 0/10 valid; all 10 hit the 2,500-token limit; 8/10 repeat one `corrections` entry (`original` = `corrected`, which never occurs in the 4,618 label entries) |
| `lora_v1` + `no_repeat_ngram_size=20` (`src/regenerate.py`, Mac) | same adapter | 3/10 valid, all stop; the ban turns loops into syntax slips (dropped quote, `correcteds`) |
| `lora_v1` + `no_repeat_ngram_size=10` | same adapter | 0/10 valid; the ban also blocks normal JSON |
| `lora_v2` (Colab L4, 2026-10-11) | Same as v1 plus: `corrections` left out of the student target (`train_lora.STUDENT_SKIP`; mean target 900 → 843 tokens); Nougat token perturbation `--perturb 0.1` (input only, labels correct; ~0.13 tokens/page). 176 steps, 40 min; loss 0.73 → 0.19 | See the gold-page table below |

`lora_v2` on 50 held-out gold pages (`labels_gold_v6_verified.jsonl`), `src/evaluate.py`, outputs in `runs/lora_v2/eval_gold_*`:

| Decoding | Valid | Schema error | Not JSON | Hit 2,500 tokens | Field P / R / F1 (all pages) | F1 on parsed pages |
|---|---|---|---|---|---|---|
| free | 21 | 15 | 14 | 12 | 0.47 / 0.40 / 0.43 | 0.51 (36 pages) |
| `--constrained` (lm-format-enforcer) | 44 | 0 | 6 | 6 | 0.43 / 0.42 / 0.42 | 0.46 (44 pages) |

Loops are no longer in one field: free-decoding loops end in `technologies` (4), `summary` (2), `name` (2) and 4 others; e.g. `"C","PHP"` repeated. Constrained decoding removes every schema error and halves the loops, but field F1 does not rise: the extra pages it rescues are scored with low F1. Main remaining problem: wrong or missing values (F1 ~0.5 even on parsed pages), then repetition.

Per-field breakdown (free decoding, 36 parsed pages):
- Entries in the wrong section: the model writes more `work` entries than gold on 16/36 pages (projects 12, education 7). Inspected pages show education, club and project entries placed in `work`. Precision of `work`/`projects`/`education` is ~0.4.
- Misread characters: `OACA` → `OCAAC`, `SAFELEC` → `SAFEELEC`, `technicien` → `technique`. Counting values with string similarity ≥ 0.8 as hits raises F1 0.51 → 0.59 (constrained 0.46 → 0.54), so ~0.08 of the gap is near misses.
- Teacher spelling fixes (`corrections`) explain only 14 of 285 near-miss values; the rest are the model's own misreads.
- Weakest fields: `projects.highlights` (F1 ~0.08), `awards` (~0.07), `extra` (0), `basics.summary` (~0.25), `work.technologies` (~0.30). Strongest: `basics.name`, `phones`, `label`, `languages` (0.7–0.85).

Research behind v2 (loops, hyperparameters, constrained decoding, evaluation, format, data), with sources and what was verified: `research/2026-10-11_training_best_practice.md`.

Lessons: (1) loss 0.22 did not show the loops; only free generation did. (2) The 8-epoch Mac test already looped in `corrections` on 5/10 pages; it was read as underfitting without checking where answers broke. Before a GPU run, locate every failure, don't only count them. (3) Score on held-out pages with field F1, not validity on 10 training pages.

`lora_v3` (Colab L4, 2026-10-12): `lora_v2` + 4 epochs (352 steps, 81 min) + image-to-text projector fully trained (`--train-projector`, 13.3 M trainable). Loss 0.74 → 0.11. Same 50 gold pages:

| Run, decoding | Valid | Schema error | Not JSON | Hit 2,500 tokens | Field F1 (all pages) | F1 on parsed pages (exact / similarity ≥ 0.8) | Pages with extra `work` entries |
|---|---|---|---|---|---|---|---|
| v2 free | 21 | 15 | 14 | 12 | 0.43 | 0.51 / 0.59 (36) | 16 / 36 |
| v2 constrained | 44 | 0 | 6 | 6 | 0.42 | 0.46 / 0.54 (44) | 21 / 44 |
| v3 free | 35 | 9 | 6 | 3 | 0.58 | 0.61 / 0.70 (44) | 8 / 44 |
| v3 constrained | 48 | 0 | 2 | 2 | 0.58 | 0.59 / 0.67 (48) | 10 / 48 |

Every section improved (work 0.47 → 0.62, education 0.47 → 0.54, projects 0.46 → 0.57, awards 0.07 → 0.30); `extra` is still 0. One run changed two things, so the gain is not split between epochs and projector.

`lora_v4` (Colab L4, 2026-10-12): `lora_v3` setup for 8 epochs (704 steps, 2 h 19 min, gradient checkpointing off), adapter saved per epoch. Scored with `evaluate.py --batch 8`. Same 50 gold pages:

| Adapter, decoding | Valid | Schema error | Not JSON | Hit 2,500 tokens | Field P / R / F1 |
|---|---|---|---|---|---|
| ep4 free | 40 | 4 | 6 | 5 | — / — / 0.57 |
| ep6 free | 34 | 12 | 4 | 3 | 0.66 / 0.62 / **0.64** |
| ep6 constrained | **50** | 0 | 0 | 0 | 0.61 / 0.65 / 0.63 |
| ep8 free | 37 | 8 | 5 | 3 | — / — / 0.60 |
| ep8 constrained | 49 | 0 | 1 | 1 | 0.59 / 0.61 / 0.60 |

Paired bootstrap (2,000 resamples of the 50 pages, free F1): ep6 − ep8 +0.040 (ep6 higher in 99%), ep6 − v3 +0.059 (100%), ep8 − v3 +0.018 (77%), ep4 − v3 −0.011 (28%). Best is epoch 6; later epochs overfit. Batched scoring: 8.6 s/page vs 27.7 s (3.2×).

`lora_v5` (Colab L4, 2026-10-12): `lora_v4` with the projector frozen (only change; same seed). 2 h 14 min, loss 0.35 → 0.08 per epoch. Same 50 gold pages; Δ = frozen − trained (v4), paired bootstrap 2,000 resamples:

| Adapter, decoding | Valid | Schema error | Not JSON | Hit 2,500 tokens | Field P / R / F1 | Δ F1 vs v4 (P frozen higher) |
|---|---|---|---|---|---|---|
| ep6 free | 41 | 5 | 4 | 4 | 0.64 / 0.59 / 0.61 | −0.027 (10%) |
| ep6 constrained | 49 | 0 | 1 | 1 | 0.62 / 0.61 / 0.62 | −0.010 (31%) |
| ep8 free | 40 | 5 | 5 | 3 | 0.63 / 0.57 / 0.60 | −0.001 (48%) |
| ep8 constrained | 47 | 0 | 3 | 3 | 0.60 / 0.55 / 0.58 | −0.027 (9%) |

Training the projector adds 0–0.03 F1, consistent in direction but within page-to-page noise; kept on.

`lora_v6` (Colab L4, 2026-10-12): `lora_v4` + tile cap 12 (`--max-patches 12`, scored with the same cap) + LoRA rank 32. 2 h 31 min, loss 0.30 → 0.036 per epoch. Same 50 gold pages; Δ = v6 − v4, paired bootstrap 2,000 resamples:

| Adapter, decoding | Valid | Schema error | Not JSON | Hit 2,500 tokens | Field P / R / F1 | Δ F1 vs v4 (P v6 higher) |
|---|---|---|---|---|---|---|
| ep6 free | 32 | 10 | 8 | 4 | 0.68 / 0.55 / 0.61 | −0.03 |
| ep6 constrained | 47 | 0 | 3 | 3 | 0.65 / 0.60 / 0.62 | −0.004 (45%) |
| ep8 free | 37 | 8 | 5 | 1 | 0.67 / 0.60 / 0.63 | +0.032 (84%) |
| ep8 constrained | 49 | 1 | 0 | 0 | 0.64 / 0.64 / **0.64** | +0.039 (98%) |

Best epoch moves to 8. Free ep6 on the 6 pages that get more tiles: +0.094 (96%); other 44 pages −0.048.
