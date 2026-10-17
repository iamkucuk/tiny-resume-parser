# Code tour

The package is about 1,500 lines in 13 modules under [`src/tiny_resume_parser/`](../src/tiny_resume_parser). This
page walks through them in the order a page travels: from the label format, through training, to scoring and the
`parse` command. Each section names the idea the module implements and shows the few lines that carry it.

## 1. What the model must write: `schema.py`

[`schema.py`](../src/tiny_resume_parser/schema.py) defines `CVRecord`, a Pydantic model with the sections of a CV
(`basics`, `work`, `education`, `skills`, `languages`, `projects` and others). Every label and every model answer is
checked against it. `to_target` turns a record into the exact string the student learns to write: empty values are
dropped and the JSON is compact, so the answer spends no tokens on `null` fields or indentation.

## 2. Where pages come from: `data.py`

A label file is JSONL, one page per line, with the page image path and the record. The validation split is chosen
by a hash of the CV id, so every page of one CV falls on the same side and no CV is seen in training and validation:

```python
def is_validation(cv_id: str) -> bool:
    return int(hashlib.sha1(cv_id.encode()).hexdigest(), 16) % 20 == 0
```

The data folder is `data/` at the repository root, or the folder named by `TINY_RESUME_PARSER_DATA`. The
[`examples/`](../examples) folder uses the same format.

## 3. Four model families, one prompt: `models.py`

[`models.py`](../src/tiny_resume_parser/models.py) holds what differs between the base models of ResuVL-L, ResuVL-L v0.1, ResuVL-M and
ResuVL-S, keyed by the name of their processor class: the text that opens the assistant's answer, the token that ends
it, and how the page is cut into image tiles. ResuVL-L v0.1 takes a tile count; ResuVL-M and ResuVL-S take a longest
edge in pixels, so `--max-patches 12` becomes 2,048 px; ResuVL-L takes a pixel budget, 12 × 448 × 448. ResuVL-L's
answer opens after an empty thinking block, which belongs to the header, so the model neither learns nor writes it.
Each tile becomes 64 (ResuVL-M, ResuVL-S) or 256 (ResuVL-L v0.1) image tokens, which is most of
the input: the first example page, read by ResuVL-S at 12 tiles, is 832 image tokens and 489 answer
tokens.

`STUDENT_SKIP` removes the `corrections` list from the student's target. The teacher fixes spelling mistakes and
lists them there; the student cannot see a correction on the page, and the first model trained with it wrote
correction entries until it hit the token limit.

## 4. Training: `train.py`

Training is ordinary supervised fine-tuning with the Hugging Face `Trainer`. Three pieces are specific to this task.

**The loss mask.** The model sees the image, the prompt and the answer, but learns only the answer. The collator
finds the assistant header in each tokenised example and copies only the tokens after it into `labels`; everything
else stays `-100`, which the loss ignores. The closing token is learnt too, so the model learns to stop:

```python
first = starts[0] + h
last = ids.index(self.end_id, first)
labels[i, first : last + 1] = batch["input_ids"][i, first : last + 1]
```

Before training, a check on the first batch counts the image tokens and stops the run if any of them is learnt.

**Token perturbation.** Taken from Nougat (arXiv 2308.13418): a few answer tokens in the input are replaced by random
tokens while the labels keep the correct ones, so the model practises continuing correctly after its own mistake.
At `--perturb 0.1` this is about 0.11 tokens per page. Its measured effect was small; see
[findings.md](findings.md).

**Image noise.** `augment_page` lowers the resolution, rotates by up to 1.5°, changes contrast and adds JPEG
artefacts on half of the pages. Every change keeps each character readable, so the label stays true.

Precision: on a GPU with bf16 the weights stay in float32 and the matrix products run in bf16; on a GPU without bf16
(such as a T4) or a CPU, training runs in float32.

## 5. Reading a page: `generate.py`

`generate_pages` builds the prompt with the processor's chat template, cuts the image into the same tiles as in
training and decodes greedily. The tile setting at inference must match the one the model was trained with.

## 6. Scoring: `metrics.py`, `evaluate.py`, `rescore.py`

Field F1 (as in Donut, arXiv 2111.15664) flattens a record into `(path, value)` pairs and ignores list order:

```python
{"skills": [{"name": "SQL"}, {"name": "Python"}]}  ->  {("skills.name", "SQL"), ("skills.name", "Python")}
```

Precision is the share of predicted pairs found in the label, recall the share of label pairs found in the answer.
`metrics.py` has four variants. Strict compares values exactly. Lenient casefolds and keeps only letters and digits,
so `"Sfax, Tunisie"` equals `"Sfax Tunisie"` but `"09.25"` does not equal `"09.20"`. Content compares the words of a
whole section, so splitting one value into two is not an error. Repair cuts a broken answer at the break and keeps
the complete values before it. [evaluation.md](evaluation.md) explains when each is used.

`evaluate.py` generates and writes one line per page with the strict counts and the answer; `rescore.py` reads that
file again and computes the other variants without running the model.

## 7. Averaging models: `soup.py`

Several fine-tunes of the same base model, trained with different seeds, are averaged weight by weight (a uniform
model soup, arXiv 2203.05482). The soup paper's explanation is that fine-tunes from one starting point stay in one low-loss region,
so their average lands inside it. Here the average of 10 ResuVL-L v0.1 models beat the mean single model by about 0.02 F1
([findings.md](findings.md)), at no extra inference cost.

## 8. Using a model: `parse.py` and `cli.py`

`parse` renders each PDF page at 120 dpi (the resolution of the training pages), reads it with the model and repairs
the answer. `cli.py` maps the subcommands (`train`, `evaluate`, `rescore`, `soup`, `parse`, `label`) to these modules.

## 9. Making labels: `label.py`

The labels come from a large vision model, the teacher, through any OpenAI-compatible endpoint. The prompt asks it to
copy text as written, to fix spelling mistakes in ordinary words and list them in `corrections`, and to answer in the
`CVRecord` schema. A reviewed subset of 254 pages measures how good the teacher is; see [evaluation.md](evaluation.md).
