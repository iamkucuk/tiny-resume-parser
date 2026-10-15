# Research log

This folder is the unedited record of the work, written while it happened: every hypothesis, why it was tried,
the run, the result and what came next. Run names and hypothesis numbers (H15, `S_soup10_vision`) are the ones used
at the time. The curated summary is [../findings.md](../findings.md). File paths inside older entries predate the
2026-10-15 restructure: `src/train_lora.py` became `tiny_resume_parser.train`, `src/auto_label.py` became
`tiny_resume_parser.label`, and `truba/` became `slurm/`.

## Where to look

| File | What it holds | Read it to |
|---|---|---|
| [RESEARCH_JOURNAL.md](RESEARCH_JOURNAL.md) | One entry per hypothesis: why → what ran → result → what went right / wrong → next. Starts with the current best per model and the fixed evaluation protocol. | follow the reasoning |
| [DEVLOG.md](DEVLOG.md) | One entry per training run: settings, job, per-epoch scores, what was kept or deleted. | find the numbers of one run |
| [DESIGN_NOTES.md](DESIGN_NOTES.md) | Result tables and the design choices behind the schema, labels and training. | compare runs side by side |
| [PLAN.md](PLAN.md) | The original plan (2026-10-11) and its decisions: schema, teacher, data split. | see where the project started |
| [DATASET_RESEARCH.md](DATASET_RESEARCH.md) | The page data: sources, licences, how pages were grouped into CVs. | check what the models were trained on |
| [FINETUNING_RECIPE.md](FINETUNING_RECIPE.md) | The first LoRA recipe for InternVL3-1B, replaced by full fine-tuning on 2026-10-13. | see the starting recipe |
| [notebooks/](notebooks/) | The Colab notebooks of the first LoRA run and the speed test, as they ran. | see the early runs |
| [literature/](literature/) | Three literature reviews (2026-10-11, -12, -14) with sources and what was checked. | see which published work each idea came from |
| [results_2026-10-15.html](results_2026-10-15.html) | Results page of 2026-10-15 (before the test-set scores): best model per student, what changed, where errors remain. | see the state on that date |

## How a change is accepted

1. State the hypothesis and the published work or observation it comes from (journal entry, written before the run).
2. Change one thing against a named baseline run.
3. Choose the epoch on the validation pages, report the reviewed gold pages at that epoch.
4. Adopt only if the gain beats the run-to-run spread (~0.01–0.02 F1, measured with seed repeats) on both sets.
5. Record the result either way; delete weights that are no longer used.

The 204-page test set is scored only at milestones, so the final number is not chosen on the pages it reports.

## Experiments by phase

An index into the log, with the numbers of each step. Scores are lenient field F1, gold development pages /
validation pages, at the validation-chosen epoch. "Confounded" means two things were changed in the same run, so
the gain cannot be split between them. Scores of the first LoRA runs (phase 2) used the exact-match F1 of that time.

**1. Data and labels (2026-10-09 to 10-11)**

| Decision | Evidence | Verdict |
|---|---|---|
| Drop the 725 pages stored at 640 × 640 px | teacher labels on them had 6.8 % wrong values vs ~1 % on larger pages | adopted |
| Leave the teacher's `corrections` list (spelling fixes) out of the target | ~1 in 5 entries named an "original" that is not on the page | adopted |
| Second model pass that checks and fixes labels (Haiku, 30 pages) | fixed 14 real errors but made ~8 correct values wrong | rejected |
| Compact JSON target: empty keys left out | median 887 tokens vs 1,198 with all keys (YAML: 1,008) | adopted |
| 254 gold pages: every page checked and reviewed on the image | 96 fixes on 61 pages | adopted |
| Teacher run twice on the same pages | 0.912 lenient between runs (`work` 0.94, `interests` 0.73, `extra` 0.32): the ceiling for students | measured |

**2. First LoRA fine-tunes of InternVL3-1B on Colab (2026-10-11 to 10-12)** — scores on the 50 gold pages

| Run | Change | Result | Verdict |
|---|---|---|---|
| lora_v1 | LoRA rank 16, 2 epochs | 0/10 answers valid; all hit the 2,500-token limit; 8/10 repeat one `corrections` entry | baseline |
| lora_v1 + n-gram ban | `no_repeat_ngram_size` 20 / 10 | 3/10 valid with syntax errors / 0/10 | rejected |
| lora_v2 | `corrections` removed **and** token perturbation (Nougat, arXiv 2308.13418: one random input answer token swapped on ~10 % of pages, ~0.11 tokens per page on average; labels unchanged) | 21/50 valid, F1 0.43; the `corrections` loops are gone | adopted, confounded |
| lora_v2, constrained decoding | output forced to follow the schema | 44/50 valid, F1 0.42 | valid JSON, no F1 gain |
| lora_v3 | 4 epochs **and** the image-to-text projector trained | 35/50 valid, F1 0.58 | adopted, confounded |
| lora_v4 | 8 epochs, one adapter saved per epoch | epoch 6 best (0.64); epoch 8 lower (0.60) | best epoch 6 |
| lora_v5 | projector frozen again | 0.61 vs 0.64 (−0.027, within noise) | projector kept trainable |
| lora_v6 | 12 image tiles **and** rank 32 | 0.63 free / 0.64 constrained, 49/50 valid; +0.094 on the 6 tall pages, −0.048 on the other 44 | 12 tiles adopted, confounded |
| — | gradient checkpointing off; scoring 8 pages per batch | 1.27× faster training; 3.2× faster scoring with identical answers | adopted |

Token perturbation was never tested on its own: it entered in the same run that removed `corrections`, and it has
been on (0.1) in every run since.

**3. Full fine-tuning on the TRUBA cluster (2026-10-13)**

| Change | Result | Verdict |
|---|---|---|
| Full fine-tune instead of LoRA (lr 1e-5, 6 epochs) | 0.693 / 0.669 vs LoRA 0.650–0.662: +0.03 | adopted |
| Image noise on half of the pages (resize, rotation ≤ 1.5°, contrast, blur, JPEG) | +0.035 / +0.007, 3 seeds | adopted |
| Vision encoder trained too | +0.011 / +0.022, 3 seeds | adopted |
| Both together, 4 seeds (the reference for later runs) | 0.704 / 0.708, seed-to-seed SD ≈ 0.012 | adopted |
| Half of the training pages | −0.05 to −0.08: more distinct pages help | measured |
| lr 2e-5; lr 5e-6; 8 epochs; image noise on every page | tie or lower | rejected |
| 24 image tiles instead of 12 | −0.05 gold | rejected |
| Training on canonical labels (`canon.py`) | −0.01 | rejected |
| 2 pages per step instead of 1 | ~−0.02 | rejected |
| Constrained decoding on the full fine-tune | 0.694 vs 0.693 free; 49/50 valid; 5.4× slower | for deployment only |
| 2B and 8B InternVL with LoRA (outside the < 1B limit; not candidates) | 0.754 / 0.716; 8B with 24 tiles 0.824 / 0.812 | out of scope |

**4. Training after supervised fine-tuning (2026-10-13)**

| Method | Start | Result | Verdict |
|---|---|---|---|
| Preference training (DPO) on the model's own wrong answers | lora_v6, 0.650 gold | 0.583 → 0.556 → 0.539 gold as training goes on | rejected |
| Reinforcement learning (GRPO, TRL), reward = lenient field F1, lr 1e-6 | full fine-tune, 0.705 / 0.719 | 0.711 / 0.718; gold unparsed pages 3 → 1, limit hits 2 → 0; val unparsed 2 → 3 | no change |
| GRPO lr 3e-6 / temperature 0.7 / KL penalty 0.04 | same | 0.718 / 0.704; 0.705 / 0.714; 0.707 / 0.715 | no change |
| Repetition penalty 1.05 / 1.02 at generation | full fine-tune | 0.712 / 0.705 (no limit hits); 0.706 / 0.711 | no gain |

GRPO raised the training reward from 0.73 to 0.78–0.85, but greedy answers on new pages did not change beyond the
seed spread: in ~350 steps it fitted the training pages, not reading in general.

**5. Weight averaging and answer repair (2026-10-13 to 10-14)**

| Change | Result | Verdict |
|---|---|---|
| Average of 4 seeds | 0.721 / 0.711 vs their mean 0.707 / 0.706 | helps |
| Average of 10 vision-trained seeds | 0.728 / 0.732 (+0.024 / +0.024) | adopted: InternVL best |
| Averages of 13–19 models | up to 0.745 / 0.724 (better gold, lower val) | not chosen |
| Answer repair (fix escapes, keep values up to a break, drop duplicate list items) | 0.728 → 0.746 gold; never lowers a page by more than 0.001 | adopted as output step |
| 993 extra pages from a second dataset (hadikp, licence unknown) | 6-seed mean +0.011 / +0.013 | small and uncertain; cannot be released |

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
| 2026-10-15 | Milestone: the three best models scored once on the 204 held-out test pages | lenient F1 0.745 (InternVL), 0.677 (SmolVLM), 0.633 (granite-docling) |

Scores are lenient field F1 on gold pages 1–50 / the 79 validation pages.
