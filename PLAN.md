# Plan: Fine-tune a sub-1B vision-language model to turn CV page images into ATS JSON

> Living document, authoritative for decisions and sequencing. Rewritten 2026-10-11 to
> replace the granite-docling / DocTags / Gemini plan; the old schema and labeler are kept
> in `src/_legacy/`.

## Status (2026-10-11)
- Done: document reconstruction, gold-set selection, JSON schema, teacher labeling of all
  1,726 full-size pages, label quality checks, training/gold label files, gold labels verified
  (Haiku check of all 254 pages + review on the images: 96 fixes on 61 pages), sparse JSON
  training target, LoRA training script (`src/train_lora.py`).
- Next: smoke-test fine-tune (gate) → full run → evaluation on `labels_gold_v6_verified.jsonl`.

## Goal
One model under 1B parameters that reads a CV page image and writes one JSON record with a
fixed template (ATS use). No text locations or layout in the output.

## Decisions
| Topic | Decision | Evidence / date |
|---|---|---|
| Output schema | **JSON Resume** base + fields found in this data + `extra[]` of `{section,label,value}` for anything unmapped. `src/schema.py` (pydantic, unknown keys rejected). Every key always present: missing = `null` / `[]`, never "N/A". Personal attributes included (birthDate, gender, maritalStatus, nationality, drivingLicense, hasPhoto…). `basics.emails[]` / `basics.phones[]` are lists. Dates ISO (`YYYY[-MM[-DD]]` or `present`). work/project `type` only when the page states it. Ratings: dots `"4/5"`, bars `"~3/5"`. Locations kept whole in `address`. | User, 2026-10-09; 20-CV pilot |
| Spelling and grammar | The teacher **fixes** spelling and grammar of ordinary words and lists each fix in `corrections`; names, organisations, brands, abbreviations, technology names, emails, URLs, phones and codes are copied exactly. | User, 2026-10-09 |
| Base model | **InternVL3-1B** (`OpenGVLab/InternVL3-1B-hf`, 0.94B) first; **SmolVLM-500M-Instruct** later. granite-docling dropped (layout output). | Zero-shot, 15 pages, share of page words read: granite 0.96, InternVL3-1B 0.87, SmolVLM-500M 0.16; valid JSON: InternVL3-1B 15/20, SmolVLM 0/20 |
| InternVL3-1B settings | `max_patches=6` (`InternVLProcessor` tiles by default): every page ≤ 7 tiles = ≤ 1,792 image tokens; with 12, US-letter pages got 13 tiles (3,328 tokens). Needs `torchvision`. License metadata conflicts (Qwen license vs Apache-2.0) — check before commercial use. | HF configs, 2026-10-09 |
| Teacher | **GLM-5.3-Flash** via the LiteLLM proxy (`src/auto_label.py`), one call per page. A small model is used only for checks, never as labeler. | User, 2026-10-09 |
| Pages used | **Only pages with longest side > 640 px** (1,605 CVs, 1,726 pages). The 725 pages the owner uploaded at 640×640 are dropped: labels on them had 6.8% wrong values vs ~1% on larger pages, and no higher-resolution originals exist on Roboflow. | User, 2026-10-10 |
| Multi-page | Teacher: one call per page. Student: one page per sample vs all pages in one request, decided by comparing both in the first training run. | Zero-shot multi-image test, 2026-10-08 |
| `corrections` filter | Entries dropped by code before training (`src/build_label_sets.py`): original == corrected; corrected text in no label value; page OCR has the corrected word but not the original. 715 of 6,124 dropped. Since 2026-10-11 the whole list is also left out of the student's training answer (`STUDENT_SKIP` in `src/train_lora.py`) after `lora_v1` looped in it. | User, 2026-10-11 |
| Label fix pass | **No** Haiku check-and-fix pass. Tested on 30 pages: 14 real errors fixed, ~8 correct values made wrong (undid valid grammar fixes, removed spaces). | 2026-10-11 |
| Training target | Minified JSON with `null`/`[]`/`{}` keys left out (median 887 tokens vs 1,198 full minified); `to_record` refills them after generation. YAML measured longer (1,008); TOON not adopted (nested lists, unfamiliar to the model). | Token counts on 1,472 training pages, 2026-10-11 |
| Fine-tune method | LoRA r 16 on the language model only, vision tower and projector frozen; adapter on/off comparison; full LM fine-tune only if LoRA underfits. Recipe: `FINETUNING_RECIPE.md`. | User, 2026-10-11 |
| Latency | <500 ms/CV is not reachable when generating ~1,000 JSON tokens; target to be re-set from measured GPU throughput. | InternVL3-1B: median 22 s/page on Apple MPS |

## Data
| File | What it holds |
|---|---|
| `data/roboflow_source/` | 2,914 original uploads of `invoicemodelisation/cv-parsing` (Roboflow fork `vizioai/cv-parsing-ksgbu-qiqkv` v1), no augmentation |
| `data/processed/documents.jsonl` | 2,162 CVs / 2,451 pages, grouped by `_page_N` markers, duplicates removed, per-page width/height (`src/reconstruct.py`) |
| `data/processed/gold_200/` | 200 evaluation CVs (150 single-page, 50 multi-page; all pages full-size, none missing or shared) — never trained on (`scripts/build_gold_set.py`) |
| `data/processed/train_pool_ids.json` | the other 1,962 CVs, for training |
| `data/processed/cv_ids_full_size.json` | the 1,605 CVs whose every page is longer than 640 px |
| `data/processed/labels_full_v6.jsonl` | raw teacher output for all larger pages (also 47 rows of dropped small pages from before the restart) |
| `data/processed/labels_train_v6.jsonl` | **training labels**: 1,406 CVs / 1,472 pages, `corrections` filtered |
| `data/processed/labels_gold_v6.jsonl` | gold teacher labels before review (kept unchanged) |
| `data/processed/labels_gold_v6_verified.jsonl` | **gold labels for evaluation**: 199 CVs / 254 pages, reviewed fixes applied (`src/apply_gold_decisions.py`) |
| `data/processed/gold_v6_check/` | Haiku findings per gold page, `triage.json`, `decisions.json` (the accepted fixes) |
| `data/processed/ocr_tesseract/` | Tesseract text per page, used only for checks and the `corrections` filter |
| `data/processed/final_v6_check/`, `final_v6_fix/` | 30-page Haiku check findings; 30-page check-and-fix test |
| `data/raw/` | hadikp, 8,905 unlabeled single-page images (optional later self-distillation) |

## Label quality (larger pages)
- 30 random pages, Haiku-checked against the image, 2,612 values: ~0.4% wrong letters or
  values, ~0.5% rule slips (technology-name casing changed, organisation names
  grammar-fixed, unlisted fixes, wrong key). Haiku misses some errors, so the true rate is
  probably 1–2%.
- `corrections` before filtering: about 1 entry in 5 named an "original" not on the page.

## Project structure
```
src/
  schema.py            # pydantic record template + prompt template + validation (to_record)
  reconstruct.py       # roboflow_source → documents.jsonl
  auto_label.py        # GLM teacher labeling, resumable, --ids to limit CVs
  label_progress.py    # progress of a running auto_label.py job
  check_labels.py      # code checks of labels vs Tesseract OCR
  build_label_sets.py  # labels_full_v6 → labels_train_v6 / labels_gold_v6, filters corrections
  apply_gold_decisions.py  # gold_v6_check/decisions.json → labels_gold_v6_verified
  train_lora.py        # InternVL3-1B LoRA fine-tune + generation check
  evaluate.py          # field-level P/R/F1 of an adapter on held-out pages; --constrained = lm-format-enforcer
  colab_utils.py       # inline HTML dashboard / result tables for the Colab notebook
  _legacy/             # old flat-17-field DocTags schema and labeler
scripts/build_gold_set.py
```

## Commands
```bash
.venv/bin/python src/auto_label.py --ids data/processed/cv_ids_full_size.json --output data/processed/labels_full_v6.jsonl --workers 3
.venv/bin/python src/label_progress.py data/processed/labels_full_v6.jsonl --ids data/processed/cv_ids_full_size.json
.venv/bin/python src/build_label_sets.py
```

## Next milestones
1. **Training data format** — prompt-completion rows: page image + fixed instruction →
   label JSON. Hold out a validation split from the training CVs (by CV, not by page).
2. **Smoke-test fine-tune (gate)** — ~50 samples, 2 epochs, LoRA, vision encoder frozen.
   Pass: loss falls; outputs are valid JSON under `to_record`; outputs differ across pages;
   labels are -100 on prompt and image tokens.
3. **Full run** — `FINETUNING_RECIPE.md` (InternVL3-1B LoRA). Needs `peft` installed and a CUDA GPU chosen.
4. **Evaluation** — metrics to be designed for the JSON schema (per-field exact match for
   single values, F1 for lists, JSON validity), on the gold set, multi-page CVs reported
   separately. Baselines: zero-shot InternVL3-1B, the teacher itself.
5. SmolVLM-500M run, then optional self-distillation on hadikp.

## Open questions
- **Improvements not yet done** (constrained decoding, task RL/DPO, synthetic pages, teacher-label
  noise, zero-shot JSON baseline, multi-page merge): `DESIGN_NOTES.md` Part 1.
- **Full-run hardware**: this Mac (MPS, ~4 h) or a CUDA GPU.

## Gotchas
- GLM sometimes spends all output tokens on reasoning (`finish_reason: length`);
  `--max-tokens 32768` fixes it. The proxy returns 429 at 8 workers; use 3.
- `caffeinate -i` stops only idle sleep: keep the Mac plugged in with the lid open during
  long labeling runs (a 9 h stall on 2026-10-10 came from low battery).
- Haiku checkers count spacing/punctuation differences and allowed accent fixes as errors;
  confirm key findings on the image.
- Rerunning `auto_label.py` appends rows; the last row per page wins.
