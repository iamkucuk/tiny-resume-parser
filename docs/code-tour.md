# Code tour

The package contains about 1,500 lines across 13 modules under [`src/tiny_resume_parser/`](../src/tiny_resume_parser).
This page walks through them in the order a page travels.
Each section names the idea the module implements and shows the few lines that carry it.

## 1. What the model must write: `schema.py`

[`schema.py`](../src/tiny_resume_parser/schema.py) defines `CVRecord`, a Pydantic model with the sections of a CV.
Every label and every model answer is checked against it.
Its sections include `basics`, `work`, `education`, `skills`, `languages`, `projects` and others.

`to_target` turns a record into the exact string the student learns to write.
Empty values are dropped and the JSON is compact.
Because of this format, the answer spends no tokens on `null` fields or indentation.

## 2. Where pages come from: `data.py`

A label file is JSONL with one page per line, holding the page image path and the record.
The validation split is chosen by a hash of the CV id.
Every page of one CV falls on the same side, so no CV is seen in training and validation:

```python
def is_validation(cv_id: str) -> bool:
    return int(hashlib.sha1(cv_id.encode()).hexdigest(), 16) % 20 == 0
```

The data folder is `data/` at the repository root, or the folder named by `TINY_RESUME_PARSER_DATA`.
The [`examples/`](../examples) folder uses the same format.

## 3. Four model families, one prompt: `models.py`

[`models.py`](../src/tiny_resume_parser/models.py) holds what differs between the base models of ResuVL-L, ResuVL-L v0.1, ResuVL-M and ResuVL-S, keyed by processor class name.
ResuVL-L's answer opens after an empty thinking block in the header, so the model neither learns nor writes it.

The differences cover three elements:
- The **text that opens the assistant's answer**
- The **token that ends it**
- How the **page is cut into image tiles**

The models cut pages into image tiles differently:
- **ResuVL-L v0.1**: takes a tile count.
- **ResuVL-M and ResuVL-S**: take a longest edge in pixels, so `--max-patches 12` becomes 2,048 px.
- **ResuVL-L**: takes a pixel budget of 12 × 448 × 448.

Each tile becomes 64 image tokens (ResuVL-M, ResuVL-S) or 256 image tokens (ResuVL-L v0.1).
Image tokens make up most of the input.
On the first example page read by ResuVL-S at 12 tiles, the input has 832 image tokens compared with 489 answer tokens.

`STUDENT_SKIP` removes the `corrections` list from the student's target.
The teacher fixes spelling mistakes and lists them there.
The student cannot see a correction on the page; the first model trained with it wrote correction entries until it hit the token limit.

## 4. Training: `train.py`

Training is ordinary supervised fine-tuning with the Hugging Face `Trainer`.
Three pieces are specific to this task.

**The loss mask.** The model sees the image, the prompt and the answer, but learns only the answer.
Before training, a check on the first batch counts the image tokens and stops the run if any of them is learnt.

The collator finds the assistant header in each tokenised example and copies only the tokens after it into `labels`.
Everything else stays `-100`, which the loss ignores.
The closing token is learnt too, so the model learns to stop:

```python
first = starts[0] + h
last = ids.index(self.end_id, first)
labels[i, first : last + 1] = batch["input_ids"][i, first : last + 1]
```

**Token perturbation.** Token perturbation is taken from Nougat (arXiv 2308.13418).
Its measured effect was small; see [findings.md](findings.md).
A few answer tokens in the input are replaced by random tokens while labels keep the correct ones, so the model practises continuing correctly after its own mistake.
At `--perturb 0.1`, this replaces about 0.11 tokens per page.

**Image noise.** `augment_page` adds noise to half of the pages:
- **Resolution**: lowers the resolution.
- **Rotation**: rotates by up to 1.5°.
- **Contrast**: changes contrast.
- **Artefacts**: adds JPEG artefacts.

Every change keeps each character readable, so the label stays true.

On a GPU with bf16, weights stay in float32 and matrix products run in bf16.
On a GPU without bf16 (such as a T4) or a CPU, training runs in float32.

## 5. Reading a page: `generate.py`

`generate_pages` reads a page by preparing the prompt, tiling the image and decoding greedily.
The tile setting at inference must match the one the model was trained with.

[`draft.py`](../src/tiny_resume_parser/draft.py) decodes one page faster on a GPU.
The multi-token prediction head of [`mtp.py`](../src/tiny_resume_parser/mtp.py) proposes the next tokens from the model's last state.
One pass of the model checks them all, keeping drafts up to the first one that differs from the model's own choice.

Qwen3.5's linear-attention layers only continue from their cached state one token at a time in transformers 5.5.4.
Therefore, `continue_linear_attn` runs that step for several tokens.
A rejected draft is undone by restoring the saved states and cropping the attention cache.

## 6. Scoring: `metrics.py`, `evaluate.py`, `rescore.py`

Field F1 (as in Donut, arXiv 2111.15664) flattens a record into `(path, value)` pairs and ignores list order:

```python
{"skills": [{"name": "SQL"}, {"name": "Python"}]}  ->  {("skills.name", "SQL"), ("skills.name", "Python")}
```

Precision is the share of predicted pairs found in the label.
Recall is the share of label pairs found in the answer.

`metrics.py` provides four variants:
- **Strict**: compares values exactly.
- **Lenient**: casefolds and keeps only letters and digits, so `"Sfax, Tunisie"` equals `"Sfax Tunisie"` but `"09.25"` does not equal `"09.20"`.
- **Content**: compares the words of a whole section, so splitting one value into two is not an error.
- **Repair**: cuts a broken answer at the break and keeps the complete values before it.

[evaluation.md](evaluation.md) explains when each is used.
`evaluate.py` generates and writes one line per page with the strict counts and the answer.
`rescore.py` reads that file again and computes the other variants without running the model.

## 7. Averaging models: `soup.py`

Several fine-tunes of the same base model, trained with different seeds, are averaged weight by weight as a uniform model soup (arXiv 2203.05482).
The soup paper's explanation is that fine-tunes from one starting point stay in one low-loss region, so their average lands inside it.
Here the average of 10 ResuVL-L v0.1 models beat the mean single model by about 0.02 F1 ([findings.md](findings.md)), at no extra inference cost.

## 8. Using a model: `parse.py` and `cli.py`

`parse` renders each PDF page at 120 dpi, reads it with the model and repairs the answer.
The 120 dpi resolution matches the resolution of the training pages.

`cli.py` maps the subcommands (`train`, `evaluate`, `rescore`, `soup`, `parse`, `label`) to these modules.

## 9. Making labels: `label.py`

The labels come from a large vision model, the teacher, through any OpenAI-compatible endpoint.
A reviewed subset of 254 pages measures how good the teacher is; see [evaluation.md](evaluation.md).

The prompt asks the teacher to:
- **Copy text** as written.
- **Fix spelling mistakes** in ordinary words and list them in `corrections`.
- **Answer** in the `CVRecord` schema.

