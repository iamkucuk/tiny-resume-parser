# Model card

The three models are released on Hugging Face as [ResuVL-L](https://huggingface.co/furkankucuk/ResuVL-L), [ResuVL-M](https://huggingface.co/furkankucuk/ResuVL-M) and
[ResuVL-S](https://huggingface.co/furkankucuk/ResuVL-S). From version 0.2.0, ResuVL-L is built on Qwen3.5-0.8B; the
earlier ResuVL-L, built on InternVL3-1B (0.94B, 0.735 on the test pages), stays available as revision `v0.1.0` of the
same repository. ResuVL-L needs transformers 5.2 or later.

| | ResuVL-L | ResuVL-M | ResuVL-S |
|---|---|---|---|
| Base model | Qwen/Qwen3.5-0.8B | HuggingFaceTB/SmolVLM-500M-Instruct | ibm-granite/granite-docling-258M |
| Parameters | 0.87B | 0.51B | 0.26B |
| Training | full fine-tune, average of 3 runs | full fine-tune, average of 3 runs | full fine-tune, average of 4 runs |
| Lenient F1, 204 test pages | 0.845 | 0.681 | 0.652 |
| Lenient F1 (gold dev / val) | 0.858 / 0.822 | 0.644 / 0.676 | 0.660 / 0.649 |
| CPU time per page (M2 Max, fp32) | ~128 s | ~68 s | ~30 s |

## Intended use

Reading one CV page image (or a rendered PDF page) and writing its content as JSON in a fixed schema, on the
machine that holds the CV. Output is a draft for a person or an applicant-tracking system to check.

## Not intended for

Ranking or rejecting candidates. The model extracts what is written; it does not judge it, and its errors (left-out
values and misread ones, about 17 % of label values on the test pages for ResuVL-L) must not decide anything about a person without review.

## Training data

1,393 CV page images from the [Cv Parsing Dataset](https://universe.roboflow.com/invoicemodelisation/cv-parsing-ksgbu)
(invoiceModelisation, Roboflow Universe, CC BY 4.0), labelled by GLM-5.3-Flash. 80 % French and
19 % English CVs, single pages. Some development runs added pages from `hadikp/resume-data-pdf`, whose licence could
not be verified; the released weights were trained without them.

## Limits of the training data

**The labels come from a model that was not made for French.** Every label was written by GLM-5.3-Flash, whose
model card lists English and Chinese as its languages, while 80 % of the CVs are French. Its errors on French text
become part of the training labels, and the students learn them. The 254 test and development pages were reviewed by
hand starting from those labels, and the review changed fewer than 1 % of values, so the test measures agreement with
reviewed model labels, not with labels written independently by people. Run a second time, the labelling model scores
0.912 against the reviewed labels, which limits what models trained on its labels can reach.

**English-trained base models, a small and narrow sample of French.** The base models learned to read mostly on
English text. SmolVLM-500M and granite-docling-258M list English as their only language; InternVL3-1B is listed as
multilingual and Qwen3.5-0.8B as covering 201 languages and dialects, but neither model card gives a share per
language, and French is likely a small part of their training data next to English. The training set here gives about 330,000 words of French from 1,116 CV pages, and it is narrow
French: names, job titles, school names, dates and short skill lists, with half of all values one word long and few
full sentences. Fine-tuning therefore asks models that read English well to learn to read French from a small and
narrow sample, which covers little of French spelling, accents and vocabulary, while long training on it may weaken
the English reading they started with. This makes training harder; how much of the remaining error it causes has not
been measured.

## Evaluation

Field F1 on 204 held-out human-reviewed test pages (scored once), after choosing on 50 reviewed and 79 validation pages ([evaluation.md](evaluation.md)). The labelling model,
run a second time on 50 reviewed pages, scores 0.912 lenient F1 against them; this variation between labelling runs
limits what models trained on its labels can reach.

## Limitations

- One page at a time; multi-page CVs are returned page by page.
- Other languages, scripts and very unusual layouts are untested.
- Text written visibly on the page is read like any other text; robustness to injected instructions is not measured.
- Pages with no parseable answer on the test set: 2 of 204 (ResuVL-L), 4 (ResuVL-M), 13 (ResuVL-S, which also hits the token limit on 7 pages).

## Licence

Weights: CC BY-NC 4.0 (noncommercial). Each base model's own licence also applies.
