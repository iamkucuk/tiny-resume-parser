# Evaluation

This page defines the scores, the page sets they are computed on, and the results of the released models. Every
score comes from [`metrics.py`](../src/tiny_resume_parser/metrics.py), and each rule below has a test in
[`tests/test_scoring.py`](../tests/test_scoring.py).

## Field F1

A record is flattened into (path, value) pairs, ignoring list positions (field F1 as in Donut, arXiv 2111.15664):

```
{"work": [{"name": "BuzzCreators", "startDate": "2019-07"}]}  ->  ("work.name", "BuzzCreators"), ("work.startDate", "2019-07")
```

Precision is matched pairs over predicted pairs, recall is matched pairs over label pairs, both pooled over all pages.
A page whose answer is not valid JSON contributes no predicted pairs. Three matching rules are reported:

| Score | A pair matches when | Match | No match |
|---|---|---|---|
| strict | path and value are identical | `Sfax, Tunisie` = `Sfax, Tunisie` | `NVIDIA` ≠ `Nvidia` |
| lenient (headline) | the path is the same and the values are the same after lowercasing and removing everything that is not a letter or digit | `NVIDIA` = `Nvidia`; `+33 6 12-34` = `+33612 34` | `2021-04` ≠ `2021-04-21`; `Picard` ≠ `Picart`; `é` ≠ `e` |
| content | the same words appear in the same top-level section, under any field | `skills.name: "Python, SQL"` = `skills.name: "Python"` + `skills.keywords: ["SQL"]` | `Python` under `interests` ≠ under `skills` |

Lenient F1 is the headline score because case, spacing and punctuation do not matter when a CV is imported, while
any letter or digit difference does. Content F1 separates reading from field placement: the labelling model itself
splits text into fields inconsistently, so a small model trained on its labels is partly penalised for choices
that were arbitrary in the labels.

Repaired scores are reported next to raw ones, never instead of them. Repair keeps the complete values of a
broken answer up to the error and drops repeated list items; the `parse` command applies it.

## Page sets

The 50 reviewed gold pages and 79 validation pages (5 % of the training CVs, chosen by a hash of the CV id, with
model-written labels) are the development sets used for every decision. For each training run, the epoch is chosen
on the validation pages and the gold score is reported at that epoch. A change is kept only if it beats the
seed-to-seed spread (about 0.01 to 0.02 F1, larger for the two smaller models) on both sets. The remaining 204
reviewed pages are the test set, scored once per milestone so that no setting is chosen on them.

## Results

| Model | Development lenient, gold / val | Development content, gold / val | Test lenient (repaired) | Test strict | Test content | Test pages not parsed |
|---|---|---|---|---|---|---|
| InternVL3-1B, average of 10 runs | 0.728 / 0.732 | 0.839 / 0.852 | **0.745** (0.746) | 0.732 | 0.844 | 3 of 204 |
| SmolVLM-500M, average of 3 runs | 0.644 / 0.676 | 0.813 / 0.797 | 0.677 (0.683) | 0.667 | 0.788 | 4 of 204 |
| granite-docling-258M, average of 4 runs | 0.660 / 0.649 | 0.798 / 0.780 | 0.633 (0.647) | 0.623 | 0.752 | 16 of 204 |
| Labelling model, second run, against the corrected labels | 0.912 (50 gold pages) | 0.967 | | | | |

The test pages were scored on 2026-10-15, after every setting had been chosen. SmolVLM scores above granite-docling
on the test pages and below it on the 50 gold development pages; with 50 pages, a difference of 0.02 is within the
variation between pages.

## Reproducing a score

```bash
tiny-resume-parser evaluate --model runs/NAME/model --set gold --limit 50 --max-patches 12 --out runs/NAME/eval_gold
tiny-resume-parser rescore runs/NAME/eval_gold
```

Use `--set val` for the validation pages and `--set gold --start 50` for the test pages. SmolVLM needs
`--max-patches 30 --max-new-tokens 3000`. Add `--repair` to `rescore` for repaired scores.
