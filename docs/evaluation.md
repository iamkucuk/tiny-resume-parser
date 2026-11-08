# Evaluation

This page defines the scores, the page sets they are computed on, and the results of the released models. Every score comes from [`metrics.py`](../src/tiny_resume_parser/metrics.py). Each rule below has a test in [`tests/test_scoring.py`](../tests/test_scoring.py).

## Field F1

Field F1 scores evaluate records flattened into (path, value) pairs, ignoring list positions (field F1 as in Donut, arXiv 2111.15664). A page whose answer is not valid JSON contributes no predicted pairs. Precision is matched pairs over predicted pairs, while recall is matched pairs over label pairs, both pooled over all pages.

```
{"work": [{"name": "BuzzCreators", "startDate": "2019-07"}]}  ->  ("work.name", "BuzzCreators"), ("work.startDate", "2019-07")
```

Three matching rules are reported:

| Score | A pair matches when | Match | No match |
|---|---|---|---|
| strict | path and value are identical | `Sfax, Tunisie` = `Sfax, Tunisie` | `NVIDIA` ≠ `Nvidia` |
| lenient (headline) | the path is the same and the values are the same after lowercasing and removing everything that is not a letter or digit | `NVIDIA` = `Nvidia`; `+33 6 12-34` = `+33612 34` | `2021-04` ≠ `2021-04-21`; `Picard` ≠ `Picart`; `é` ≠ `e` |
| content | the same words appear in the same top-level section, under any field | `skills.name: "Python, SQL"` = `skills.name: "Python"` + `skills.keywords: ["SQL"]` | `Python` under `interests` ≠ under `skills` |

**Lenient F1** is the headline score because case, spacing, and punctuation do not matter when a CV is imported, while letter or digit differences do. **Content F1** separates reading from field placement. The labelling model splits text into fields inconsistently, which partly penalises a small model trained on its labels for arbitrary choices.

Repaired scores are reported next to raw ones, never instead of them. Repair keeps the complete values of a broken answer up to the error and drops repeated list items. The `parse` command applies it.

## Page sets

Two development sets guide every decision, while a test set is scored once per milestone. A change is kept only if it beats the seed-to-seed spread (about 0.01 to 0.02 F1, larger for the two smaller models) on both sets.

The page sets include:
- **50 reviewed gold pages**: development pages used for every decision.
- **79 validation pages**: 5 % of the training CVs, chosen by a hash of the CV id, with model-written labels.
- **204 test pages**: remaining reviewed pages, scored once per milestone so that no setting is chosen on them.

For each training run, the epoch is chosen on the validation pages, and the gold score is reported at that epoch.

## Labels and languages

The reviewed labels started as the labelling model's output, with review changing fewer than 1 % of values. Errors of the labelling model that the review did not catch remain in the reviewed labels and cannot be seen in any score. Compared with the reviewed labels, its first labels score 0.996 lenient field F1.

Its second run scores 0.912 on the 50 reviewed development pages. It scores 0.910 on the 38 French pages and 0.920 on the 12 English pages. This measures how consistent the labelling model is, not how correct.

Each page's language was estimated by counting common French and English words in its label:
- **Training pages**: 80 % French and 19 % English.
- **Test pages (204 pages)**: 149 French, 53 English, and 2 mixed.

The French training pages hold about 330,000 words, about 19,500 of them distinct (names and email parts included). Half of all label values are one word long. The README explains what this means for training.

## Results

Evaluation results compare the released models across development and test sets. ResuVL-M and ResuVL-S test scores differ across transformers versions, and the cause of the differences was not isolated further.

| Model | Development lenient, gold / val | Development content, gold / val | Test lenient (repaired) | Test strict | Test content | Test pages not parsed |
|---|---|---|---|---|---|---|
| ResuVL-L, average of 3 runs | 0.858 / 0.822 | 0.952 / 0.917 | **0.845** (0.845) | 0.835 | 0.916 | 2 of 204 |
| ResuVL-L v0.1, average of 10 runs | 0.732 / 0.724 | 0.847 / 0.855 | 0.735 (0.739) | 0.722 | 0.840 | 4 of 204 |
| ResuVL-M, average of 3 runs | 0.644 / 0.676 | 0.813 / 0.797 | 0.681 (0.682) | 0.670 | 0.791 | 4 of 204 |
| ResuVL-S, average of 4 runs | 0.660 / 0.649 | 0.798 / 0.780 | 0.652 (0.653) | 0.642 | 0.777 | 13 of 204 |
| Labelling model, second run, against the corrected labels | 0.912 (50 gold pages) | 0.967 | | | | |

ResuVL-L is released from version 0.2.0, trained with the ResuVL-L v0.1 recipe unchanged. Its test pages were scored once, on 2026-10-16, after the three runs and their average were chosen on the development pages. Its test scores are those of the released bfloat16 file; before rounding to bfloat16, the average scored 0.846.

The 204 test answers from ResuVL-L break down as follows:
- 185 **pass the schema**.
- 17 are **JSON that does not pass the schema**, and are still scored.
- 2 are **not valid JSON**, and are scored as empty.

Test scores of ResuVL-M and ResuVL-S use transformers 5.5.4, which the package uses from version 0.2.0. Under transformers 4.57.6, used for training and selection, the same weights scored 0.677 (0.683) and 0.633 (0.647). Their development scores are from 4.57.6.

Between the two library versions, 142 answers for ResuVL-M and 154 answers for ResuVL-S differ out of 204 answers. The image processor of these models was reimplemented between versions, and the cause was not isolated further. ResuVL-L v0.1 gives the same 204 answers under both versions.

The other test pages were scored on 2026-10-15, after every setting had been chosen. ResuVL-L v0.1 was rebuilt on 2026-10-16 from 10 runs trained only on the licensed pages and scored once more. An earlier average, three of whose members saw 241 pages of unknown licence, scored 0.745 and is not released.

ResuVL-M scores above ResuVL-S on the test pages and below it on the 50 gold development pages. With 50 pages, a difference of 0.02 is within the variation between pages.

## Reproducing a score

Scores are reproduced using the `evaluate` and `rescore` commands in sequence. ResuVL-L needs transformers 5.2 or later.

```bash
tiny-resume-parser evaluate --model runs/NAME/model --set gold --limit 50 --max-patches 12 --out runs/NAME/eval_gold
tiny-resume-parser rescore runs/NAME/eval_gold
```

Command arguments configure page sets and model requirements:
- **Validation pages**: use `--set val`.
- **Test pages**: use `--set gold --start 50`.
- **ResuVL-M**: needs `--max-patches 30 --max-new-tokens 3000`.
- **Repaired scores**: add `--repair` to `rescore`.

