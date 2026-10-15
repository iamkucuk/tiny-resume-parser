# Research record

This folder records the research as it happened: every hypothesis, the reason it was tried, the run, the result,
and what came next — including the ideas that did not work. The record was written during the work, not after it.

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

## Timeline

| Date | Step | Outcome |
|---|---|---|
| 2026-10-09 | Schema fixed (JSON Resume layout) after reading all 2,451 pages; GLM-5.3-Flash chosen as teacher | `src/schema.py`, `src/auto_label.py` |
| 2026-10-11 | First LoRA fine-tunes of InternVL3-1B on Colab | answers repeat entries until the token limit; fixed by dropping the `corrections` field |
| 2026-10-12 | LoRA v4–v6 (more epochs, 12 image tiles, rank 32) | LoRA plateaus; full fine-tuning next |
| 2026-10-13 | Full fine-tuning on the TRUBA cluster; image noise and vision-encoder training; DPO; GRPO; weight averaging | DPO lowers F1, GRPO no change; weight average of 10 models: 0.728 / 0.732, still the InternVL best |
| 2026-10-13 | SmolVLM-500M added as a second student | own recipe found on 2026-10-14: lr 6e-5, 10 epochs |
| 2026-10-14 | Literature review; granite-docling-258M added as a third student; content score added | SmolVLM 30 image tiles adopted |
| 2026-10-15 | Weight averages for SmolVLM (0.644 / 0.676) and granite-docling (0.660 / 0.649) | rejected: key-by-key generation, NEFTune, extra pages for SmolVLM; a lower vision-encoder learning rate is worse on the first granite seed (more seeds running) |

Scores are lenient field F1 on gold pages 1–50 / the 79 validation pages.
