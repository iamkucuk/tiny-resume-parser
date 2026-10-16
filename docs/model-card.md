# Model card

The three models are released as ResuVL-S, ResuVL-M and ResuVL-L. [ResuVL-S](https://huggingface.co/furkankucuk/ResuVL-S) and
[ResuVL-M](https://huggingface.co/furkankucuk/ResuVL-M) are on Hugging Face; ResuVL-L is being retrained without the pages of unknown licence
and will follow, with new scores.

| | ResuVL-L | ResuVL-M | ResuVL-S |
|---|---|---|---|
| Base model | OpenGVLab/InternVL3-1B-hf | HuggingFaceTB/SmolVLM-500M-Instruct | ibm-granite/granite-docling-258M |
| Parameters | 0.94B | 0.51B | 0.26B |
| Training | full fine-tune, average of 10 runs | full fine-tune, average of 3 runs | full fine-tune, average of 4 runs |
| Lenient F1, 204 test pages | 0.745 | 0.677 | 0.633 |
| Lenient F1 (gold dev / val) | 0.728 / 0.732 | 0.644 / 0.676 | 0.660 / 0.649 |
| CPU time per page (M2 Max, fp32) | ~62 s | ~68 s | ~30 s |

## Intended use

Reading one CV page image (or a rendered PDF page) and writing its content as JSON in a fixed schema, on the
machine that holds the CV. Output is a draft for a person or an applicant-tracking system to check.

## Not intended for

Ranking or rejecting candidates. The model extracts what is written; it does not judge it, and its errors (left-out
values, about 11 % of values on the dev pages) must not decide anything about a person without review.

## Training data

1,393 CV page images from the [Cv Parsing Dataset](https://universe.roboflow.com/invoicemodelisation/cv-parsing-ksgbu)
(invoiceModelisation, Roboflow Universe, CC BY 4.0), labelled by GLM-5.3-Flash. Mostly French
and English CVs, single pages. Some development runs added pages from `hadikp/resume-data-pdf`, whose licence could
not be verified; released weights will be trained without them.

## Evaluation

Field F1 on 204 held-out human-reviewed test pages (scored once), after choosing on 50 reviewed and 79 validation pages ([evaluation.md](evaluation.md)). The labelling model,
run a second time on 50 reviewed pages, scores 0.912 lenient F1 against them; this variation between labelling runs
limits what models trained on its labels can reach.

## Limitations

- One page at a time; multi-page CVs are returned page by page.
- Other languages, scripts and very unusual layouts are untested.
- Text written visibly on the page is read like any other text; robustness to injected instructions is not measured.
- Pages with no parseable answer on the test set: 3 of 204 (InternVL), 4 (SmolVLM), 16 (granite-docling, which also hits the token limit on 12 pages).

## Licence

Weights: CC BY-NC 4.0 (noncommercial). Each base model's own licence also applies.
