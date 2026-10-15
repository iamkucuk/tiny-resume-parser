# Research record

This folder records the research as it happened: every hypothesis, the reason it was tried, the run, the result,
and what came next — including the ideas that did not work. The record was written during the work, not after it.
File paths and script names inside entries are the ones used at the time; on 2026-10-15 the code moved into the
`tiny_resume_parser` package (`src/train_lora.py` became `tiny_resume_parser.train`, `src/auto_label.py`
`tiny_resume_parser.label`, other scripts kept their names).

## Where to look

| File | What it holds | Read it to |
|---|---|---|
| [RESEARCH_JOURNAL.md](RESEARCH_JOURNAL.md) | One entry per hypothesis: why → what ran → result → what went right / wrong → next. Starts with the current best per model and the fixed evaluation protocol. | follow the reasoning |
| [DEVLOG.md](DEVLOG.md) | One entry per training run: settings, job, per-epoch scores, what was kept or deleted. | find the numbers of one run |
| [DESIGN_NOTES.md](DESIGN_NOTES.md) | Result tables and the design choices behind the schema, labels and training. | compare runs side by side |
| [PLAN.md](PLAN.md) | The original plan (2026-10-11) and its decisions: schema, teacher, data split. | see where the project started |
| [DATASET_RESEARCH.md](DATASET_RESEARCH.md) | The page data: sources, licences, how pages were grouped into CVs. | check what the models were trained on |
| [FINETUNING_RECIPE.md](FINETUNING_RECIPE.md) | The first LoRA recipe for InternVL3-1B (later replaced by full fine-tuning). | see the starting recipe |
| [literature/](literature/) | Three literature reviews (2026-10-11, -12, -14) with sources and what was checked. | see which published work each idea came from |
| [results_2026-10-15.html](results_2026-10-15.html) | Results page: best model per student, what changed, where errors remain. | read a summary with charts |

## How a change is accepted

1. State the hypothesis and the published work or observation it comes from (journal entry, written before the run).
2. Change one thing against a named baseline run.
3. Choose the epoch on the validation pages, report the reviewed gold pages at that epoch.
4. Adopt only if the gain beats the run-to-run spread (~0.01–0.02 F1, measured with seed repeats) on both sets.
5. Record the result either way; delete weights that are no longer used.

The 204-page test set is scored only at milestones, so the final number is not chosen on the pages it reports.

## Findings

Scores are lenient field F1, gold development pages / validation pages.

### What worked

Each change was tested against a named baseline over several training seeds (seed-to-seed spread ≈ 0.012 F1).

| Change | Effect (lenient gold / val) |
|---|---|
| Full fine-tune instead of LoRA | baseline for everything below |
| Image noise during training (resize, small rotation, contrast, blur, JPEG; p = 0.5) | +0.035 / +0.007, 3 seeds |
| Training the vision encoder too | +0.011 / +0.022, 3 seeds |
| Averaging the weights of 10 fine-tuned seeds (model soup) | +0.024 / +0.024 over the 4-seed mean |
| Answer repair: fix invalid escapes, keep the valid part of a cut-off answer, drop duplicate list items | up to +0.018, never below −0.001 |
| SmolVLM: lr 6e-5 and 10 epochs (vs the InternVL recipe) | +0.22 / +0.19 over the InternVL recipe (0.374 / 0.437) |
| SmolVLM: 30 image tiles (3,072 px longest edge) instead of 12 | +0.043 / +0.032, 2-seed means (20 tiles scored the same) |
| SmolVLM: averaging the weights of 3 seeds | 0.644 / 0.676, +0.025 / +0.022 over the 3-seed mean |
| granite-docling: lr 1e-4 instead of 6e-5 | +0.024 / +0.031, 2-seed means |
| granite-docling: averaging the weights of 4 seeds | 0.660 / 0.649, +0.044 / +0.033 over the seed mean (6 seeds: lower val) |

### What did not work

| Tried | Result |
|---|---|
| Preference training (DPO) on the model's own answers | lower F1 |
| Reinforcement learning (GRPO) with field F1 as reward, 4 variants | no measurable change |
| Repetition penalty 1.02 / 1.05 against looping answers | no gain |
| Schema-constrained decoding | 0.694 vs 0.693 free — no gain |
| 24 image tiles instead of 12 | −0.05 gold |
| Training on canonicalised labels | no gain over raw labels |
| Batch size > 1, batch-32 scoring | no gain in F1 or speed |
| SmolVLM: 48 image tiles; granite-docling: 30 or 48 tiles | no gain (SmolVLM) / lower (granite-docling) |
| SmolVLM: a first training stage on page transcripts (Pix2Struct-style) | +0.005 / −0.023 — rejected |
| SmolVLM: 993 extra training pages (2,386 instead of 1,393) | +0.006 / −0.011 (1 seed) — rejected |
| NEFTune noise on text embeddings (arXiv 2310.05914) | SmolVLM −0.003 / −0.005 (1 seed); granite +0.003 / +0.007 (3 paired seeds) |
| 10× lower learning rate for the vision encoder (discriminative fine-tuning) | granite −0.062 / −0.019 and −0.025 / −0.038 (2 seeds); InternVL +0.026 / −0.038 |
| Averaging 5 SmolVLM seeds instead of 3 | 0.657 / 0.673 vs 0.644 / 0.676 — not better on val |

### Key-by-key generation (tested 2026-10-15, not adopted)

Instead of writing the whole JSON in one answer, the model is called once per key (29 keys): the code writes the JSON
up to `"key":` (with the values already generated for earlier keys) and the model writes only that key's value,
quotes and brackets included, or `null`. Training uses the same teacher labels, 8 random keys per page per epoch.

| Run (lenient gold / val) | Key-by-key | Whole-page reference |
|---|---|---|
| InternVL, from the base model, 3 epochs | 0.647 / 0.650 | 0.705 / 0.708 (9-seed mean) |
| InternVL, from the base model, 6 epochs | 0.707 / 0.685 | 0.705 / 0.708 |
| InternVL, from the best weights, 3 epochs | 0.711 / 0.704 | 0.728 / 0.732 (start weights) |
| InternVL, from the best weights, 6 epochs | 0.728 / 0.710 | 0.728 / 0.732 (start weights) |
| SmolVLM, 3 epochs | 0.530 / 0.526 | 0.619 / 0.654 (3-seed mean) |
| granite-docling, 3 epochs | 0.474 / 0.463 | 0.616 / 0.616 (seed mean) |

With enough training it matches whole-page accuracy but has not beaten it, and it needs 29 generation calls per page
instead of 1. The 6-epoch run from the best InternVL weights equalled them on gold and lost 0.022 on val, so it is not adopted.

## Timeline

| Date | Step | Outcome |
|---|---|---|
| 2026-10-09 | Schema fixed (JSON Resume layout) after reading all 2,451 pages; GLM-5.3-Flash chosen as teacher | `src/schema.py`, `src/auto_label.py` |
| 2026-10-11 | First LoRA fine-tunes of InternVL3-1B on Colab | answers repeat entries until the token limit; fixed by dropping the `corrections` field |
| 2026-10-12 | LoRA v4–v6 (more epochs, 12 image tiles, rank 32) | LoRA plateaus; full fine-tuning next |
| 2026-10-13 | Full fine-tuning on the TRUBA cluster; image noise and vision-encoder training; DPO; GRPO; weight averaging | DPO lowers F1, GRPO no change; weight average of 10 models: 0.728 / 0.732, still the InternVL best |
| 2026-10-13 | SmolVLM-500M added as a second student | own recipe found on 2026-10-14: lr 6e-5, 10 epochs |
| 2026-10-14 | Literature review; granite-docling-258M added as a third student; content score added | SmolVLM 30 image tiles adopted |
| 2026-10-15 | Weight averages for SmolVLM (0.644 / 0.676) and granite-docling (0.660 / 0.649) | rejected: key-by-key generation, NEFTune, extra pages for SmolVLM; a 10× lower vision-encoder learning rate (both models) |

Scores are lenient field F1 on gold pages 1–50 / the 79 validation pages.
