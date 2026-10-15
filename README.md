# tiny-resume-parser — read CVs the way a person does, with a small vision-language model

tiny-resume-parser fine-tunes small vision-language models (VLMs, under 1B parameters) to read a CV page **as an image**
and write its content as JSON (name, contact details, work history, education, skills, languages, projects, …) in a
fixed schema that an applicant-tracking system (ATS) can import. The models run on a laptop CPU.

## Why read the page image instead of parsing the PDF

Most resume parsers extract the PDF's text layer and pass it to rules or a language model. That text layer is not what
a recruiter sees:

| Problem with the PDF text layer | What happens | Reading the rendered page |
|---|---|---|
| Hidden text | White, tiny or off-page text (keyword stuffing, or instructions aimed at an AI screener — prompt injection) is extracted like any other text | Text that is not visible on the page is not in the image |
| Reading order | Two-column layouts, sidebars and tables come out interleaved or out of order | The model sees the layout and reads it as one page |
| Scans and photos | No text layer at all; OCR is needed first and adds its own errors | Same input as a born-digital PDF: one image |
| Broken or unusual PDFs | Fonts without Unicode mapping, text drawn as curves, designer templates | Whatever renders correctly can be read |

So one model handles every CV the same way, whether it started as a born-digital PDF, a scan or a phone photo, and
whether or not current PDF parsers can handle it. It sees what a human reviewer sees.

What this does not solve: an instruction written in **visible** text on the page is still seen by the model, as it
is by a human reader. The training target is fixed-schema JSON (no free-text answers), which limits what such text can
change, but robustness to injected text has **not been tested yet**.

## Why small

The training labels come from GLM-5.3-Flash, a ~300B-parameter model that needs data-center hardware. The fine-tuned
students are 0.26B–0.94B parameters and read a page in 30–60 s on a laptop CPU, so a CV never has to leave the machine
that reads it.

> **Status: research in progress.** Every score below is on the development sets. The held-out test set
> (204 reviewed pages) has not been scored yet; it is scored once, at the end.

## Results

Field F1 on the two development sets (gold = 50 human-reviewed pages, val = 79 teacher-labelled pages).
*Lenient* ignores case, spacing and punctuation; any letter or digit difference is still an error.
*Content* also accepts correct text that was split, merged or placed under a nearby field.

| Student | Params | Lenient F1 gold / val | Content F1 gold / val |
|---|---|---|---|
| InternVL3-1B (average of 10 fine-tuned models) | 0.94B | **0.728 / 0.732** (0.746 gold with answer repair) | **0.839 / 0.852** |
| SmolVLM-500M (average of 3 fine-tuned models, 30 image tiles) | 0.51B | **0.644 / 0.676** | 0.813 / 0.797 |
| granite-docling-258M (average of 4 fine-tuned models) | 0.26B | **0.660 / 0.649** | 0.798 / 0.780 |
| *Teacher (GLM-5.3-Flash) vs reviewed gold — the ceiling* | ~300B | *0.912* | *0.967* |

The teacher itself agrees with the human-reviewed pages at only 0.912 lenient F1, so a student trained on its labels
is measured against a ceiling of ~0.91, not 1.0. On content F1 the best student reaches 0.84 of a 0.97 ceiling.

Scores updated 2026-10-15. Since 2026-10-14, SmolVLM rose from 0.596 / 0.627 and granite-docling from 0.542 / 0.546,
through more image tiles (SmolVLM), a higher learning rate (granite-docling) and weight averaging (both).

### Speed on a laptop CPU

Apple M2 Max, CPU only, float32, 10 validation pages, one page at a time (`evaluate.py --device cpu --batch 1`).

| Student | Seconds per page, median (range) | Tokens/s | Peak RAM | Valid JSON |
|---|---|---|---|---|
| granite-docling-258M | 30 (19–45) | 34.7 | 2.9 GB | 6/10 |
| SmolVLM-500M | 59 (36–77) | 18.3 | 3.9 GB | 10/10 |
| InternVL3-1B (average) | 62 (35–80) | 15.6 | 4.9 GB | 9/10 |

Model loading takes ~5 s. No quantisation has been tried yet.

## How it works

1. **Pages.** CV page images from a public Roboflow Universe dataset (CC BY 4.0, see [Data and licence](#data-and-licence)),
   one image per page.
2. **Teacher labels.** GLM-5.3-Flash reads each page and writes JSON in the schema of [src/schema.py](src/schema.py)
   (`src/auto_label.py`). Pydantic validates every answer; the teacher is told to copy text as written, not to fix spelling.
3. **Gold set.** 254 pages were reviewed by hand (`scripts/build_gold_set.py`, `src/apply_gold_decisions.py`).
   Pages 1–50 are a development set; pages 51–254 are the test set.
4. **Training.** Full fine-tune of each student on 1,393 teacher-labelled pages ([src/train_lora.py](src/train_lora.py)),
   on single H100/H200 GPUs of the TRUBA cluster ([truba/exp.slurm](truba/exp.slurm)).
5. **Scoring.** [src/evaluate.py](src/evaluate.py) generates answers, [src/rescore.py](src/rescore.py) scores them
   field by field (strict, lenient, canonical, content, repaired).

### The schema

One `CVRecord` per page: `basics` (name, label, emails, phones, location, profiles, …), `work`, `volunteer`,
`education`, `awards`, `certificates`, `publications`, `skills`, `languages`, `interests`, `references`, `projects`
and `extras`. It follows the [JSON Resume](https://jsonresume.org/schema) layout with a few additions for the CVs
in the data (e.g. `work.type`, `education.honors`). Missing fields are `null` or `[]`, never invented.

## What worked

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

## What did not work

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

The full record — every hypothesis, why it was tried, what ran and what came out — is in
[docs/RESEARCH_JOURNAL.md](docs/RESEARCH_JOURNAL.md); one entry per training run is in [docs/DEVLOG.md](docs/DEVLOG.md).
[docs/README.md](docs/README.md) explains how a change is accepted and gives a dated timeline.

## Where the remaining errors are

For the best InternVL model on the 50 gold pages: 6 % of pages (3) produce no parseable JSON and carry 6.7 % of all
gold values. On the pages that parse, most lost values are items the model left out or worded differently, spread
over all sections; misread characters are the smallest error type. Answers that repeat until the token limit are now rare for all three
students (0–2 of 79 val pages). The gap between lenient and content F1 (0.11–0.17) shows that most remaining lost
points are correct text placed under a different field or split differently, not misread text.

## Evaluation protocol

- Every decision is made on the two development sets; the test set is scored only at milestones.
- A run's epoch is chosen on val, then gold is reported at that epoch (choosing on gold would overstate noisy runs).
- A difference counts only if it beats the seed-to-seed spread (~0.01–0.02) on **both** development sets.
  Two adopted changes (image noise, vision encoder) passed on one set and were positive but smaller on the other;
  the journal states this.
- Runs are compared raw against raw; repaired and content scores are reported next to them, never instead.

## Limitations

- Test-set scores are not reported yet; all numbers above are development scores.
- Labels come from one teacher model; the 0.912 teacher–gold agreement bounds what the students can learn.
- Most pages are French and English CVs from one dataset; other languages and layouts are untested.
- Some training runs used an extra set of pages (`hadikp/resume-data-pdf`) whose licence could not be checked; a
  public release will be trained without them.
- 2B, 8B and 14B models were tried early on; they are outside this project's sub-1B limit and are not reported here.
- Input is one page image. PDF pages must be rendered to images first (for example with `pdftoppm`); a built-in PDF
  step and the merging of multi-page CVs into one record are not implemented yet.
- Robustness to prompt injection (hidden or visible instructions on the page) has not been measured.

## Repository layout

```
src/                 the pipeline, run as scripts (python src/NAME.py)
  schema.py          output schema (pydantic) and the JSON form the model is trained on
  auto_label.py      teacher labelling with GLM-5.3-Flash
  train_lora.py      full fine-tune or LoRA (--full-ft, --train-vision, --augment, --model, --max-patches, --keywise)
  soup.py            weight averaging of fine-tuned models
  evaluate.py        generate answers on gold / val / train pages; field F1
  rescore.py         strict / lenient / canonical / content / repaired scoring
  canon.py           canonical label form (list splitting, normalisation)
  keywise.py         key-by-key generation (tested, not adopted)
  train_dpo.py, train_grpo.py, transcribe.py, ...   experiments recorded in docs/ (not adopted)
scripts/             one-off data and analysis scripts (gold-set building, error analysis)
truba/               SLURM jobs for the TRUBA cluster: train → score each epoch → keep the val-chosen checkpoint
tests/               scoring rules and key-by-key round trip (pytest)
notebooks/           Colab notebooks: first LoRA run, CPU/GPU speed test
docs/                research record: journal, run log, design notes, literature reviews, results page
```

Data (`data/`) and model weights (`runs/`) are not in the repository.

## Commands

```bash
pip install -r requirements.txt
```

```bash
python src/evaluate.py --model PATH_TO_MODEL --set val --device cpu --batch 1 --max-patches 12 --out runs/NAME/eval_val
```

```bash
python src/rescore.py --repair runs/NAME/eval_val
```

```bash
python src/train_lora.py --full-ft --train-vision --augment 0.5 --max-patches 12
```

```bash
pytest
```

Tested with transformers 4.57.6, torch 2.6.0 (CUDA) and 2.14.1 (CPU), peft 0.21.2, pydantic 2.13.

## Data and licence

- CV pages: Roboflow Universe CV dataset, CC BY 4.0. **Source link and author credit: to be added before release.**
- Base models: [InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf),
  [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct),
  [granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M) — each under its own licence.
- Code: [PolyForm Noncommercial 1.0.0](LICENSE.md) — free to use, change and share for any noncommercial purpose
  (personal, research, education, nonprofit); commercial use is not permitted. This is a source-available licence,
  not an OSI open-source licence, because those must allow commercial use.
- Model weights and documentation (when released): CC BY-NC 4.0.

## Next steps

1. Score the test set (gold pages 51–254) once for the best model of each track.
2. Train a release model without the unverified extra pages; publish weights, a model card and a demo.
3. SmolVLM: two more seeds for a 5-model weight average (running); a published method aimed at field-placement errors.
4. granite-docling: a 10× lower learning rate for the vision encoder (discriminative fine-tuning, Howard & Ruder 2018), 3 seeds (running).
5. InternVL: the same 10× lower vision-encoder learning rate, 3 seeds (queued; key-by-key generation was not adopted).
