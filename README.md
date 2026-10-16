<h1 align="center">tiny-resume-parser</h1>

<p align="center">Small vision-language models (0.26B–0.94B parameters) that read a CV page image and write its content as JSON. They run on a laptop CPU.</p>

<p align="center">
  <a href="LICENSE.md"><img alt="License: PolyForm Noncommercial" src="https://img.shields.io/badge/license-PolyForm%20Noncommercial-blue"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue">
  <a href="https://github.com/iamkucuk/tiny-resume-parser/actions/workflows/tests.yml"><img alt="tests" src="https://github.com/iamkucuk/tiny-resume-parser/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://colab.research.google.com/github/iamkucuk/tiny-resume-parser/blob/main/examples/quickstart.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

<p align="center"><img src="docs/assets/example.png" alt="Left: a two-column designer CV page with a photo. Right: the JSON the 0.94B model wrote for it, with name, contact details, work, education and skills in separate fields." width="900"></p>
<p align="center"><sub>A validation page the model never trained on, and part of the JSON the 0.94B model wrote for it.</sub></p>

Most resume parsers read the text layer of a PDF. That layer can differ from what a recruiter sees: hidden text is
extracted like visible text, two-column layouts come out interleaved, scanned pages have no text layer at all, and
some PDFs store text without a usable character mapping. These models read the rendered page instead, so every CV
is handled the same way, whether it is a born-digital PDF, a scan or a photo.

> Weights on Hugging Face: [ResuVL-S](https://huggingface.co/furkankucuk/ResuVL-S) (0.26B), [ResuVL-M](https://huggingface.co/furkankucuk/ResuVL-M) (0.51B) and
> [ResuVL-L](https://huggingface.co/furkankucuk/ResuVL-L) (0.94B).

## Results

<img alt="Bar charts of lenient field F1 and content F1 on the 204 held-out test pages: InternVL3-1B 0.735 and 0.840, SmolVLM-500M 0.677 and 0.788, granite-docling-258M 0.633 and 0.752; the dashed line is the labelling model run a second time on 50 reviewed development pages, 0.912 and 0.967." src="docs/assets/results.svg" width="900">

| Model | Parameters | Field F1 | Content F1 | CPU seconds per page |
|---|---|---|---|---|
| ResuVL-L (InternVL3-1B) | 0.94B | **0.735** | **0.840** | 62 |
| ResuVL-M (SmolVLM-500M) | 0.51B | 0.677 | 0.788 | 68 |
| ResuVL-S (granite-docling-258M) | 0.26B | 0.633 | 0.752 | 30 |

Scores are on 204 human-reviewed test pages that were not used for any decision. Field F1 counts a value as correct
when it matches the reviewed label apart from case, spacing and punctuation. Content F1 also accepts correct text
placed under another field of the same section. When the labelling model labels 50 reviewed pages a second time,
it scores 0.912 field F1 against them: its labels vary between runs, which limits what models trained on them can
reach. Definitions and development scores are in
[docs/evaluation.md](docs/evaluation.md); CPU times are medians over 10 validation pages on an Apple M2 Max in float32.

## Limits of the data

**The labels come from a model that was not made for French.** Every label was written by GLM-5.3-Flash, whose
model card lists English and Chinese as its languages, while 80 % of the CVs are French. Its errors on French text
become part of the training labels, and the students learn them. The 254 test and development pages were reviewed by
hand starting from those labels, and the review changed fewer than 1 % of values, so the test measures agreement with
reviewed model labels, not with labels written independently by people. Run a second time, the labelling model scores
0.912 against the reviewed labels, which limits what models trained on its labels can reach.

**English-trained base models, a small and narrow sample of French.** The base models learned to read mostly on
English text. SmolVLM-500M and granite-docling-258M list English as their only language; InternVL3-1B is listed as
multilingual, but its model card gives no share per language, and French is likely a small part of its training data
next to English. The training set here gives about 330,000 words of French from 1,116 CV pages, and it is narrow
French: names, job titles, school names, dates and short skill lists, with half of all values one word long and few
full sentences. Fine-tuning therefore asks models that read English well to learn to read French from a small and
narrow sample, which covers little of French spelling, accents and vocabulary, while long training on it may weaken
the English reading they started with. This makes training harder; how much of the remaining error it causes has not
been measured.

The numbers behind these two points are in [docs/evaluation.md](docs/evaluation.md#labels-and-languages).

## Quickstart

```bash
pip install git+https://github.com/iamkucuk/tiny-resume-parser
tiny-resume-parser parse cv.pdf --model furkankucuk/ResuVL-L
```

```json
[{"file": "cv.pdf", "page": 1, "valid": true, "record": {"basics": {"name": "Josephine Meyers", "emails": ["josephine.meyers@me.com"]}, "work": [...]}}]
```

PDF pages are rendered at 120 dpi; PNG and JPEG files are read as they are. Until weights are published, `--model`
takes a model trained with [docs/training.md](docs/training.md).

To see training and scoring end to end without the full dataset, [examples/quickstart.ipynb](examples/quickstart.ipynb)
([open in Colab](https://colab.research.google.com/github/iamkucuk/tiny-resume-parser/blob/main/examples/quickstart.ipynb))
fine-tunes granite-docling-258M on the eight pages in [examples/](examples) on Colab or a local machine.
A copy with all outputs from a finished run is
[examples/quickstart_executed.ipynb](examples/quickstart_executed.ipynb). [docs/code-tour.md](docs/code-tour.md) walks through the code in the order a page travels.

## How it works

```mermaid
flowchart LR
    A[CV page images] --> B[Large model writes<br/>a JSON label per page]
    B --> C[254 labels corrected<br/>by hand]
    B --> D[1,393 training pages]
    D --> E[Fine-tune a small model,<br/>several seeds]
    E --> F[Average the weights]
    C --> G[Score]
    F --> G
```

A large model (GLM-5.3-Flash) labels each page once. Each small model is then fully fine-tuned to write the same
JSON from the page image alone, several times with different seeds, and the fine-tuned weights are averaged into
one model of the same size. [docs/how-it-works.md](docs/how-it-works.md) describes each step.

## Findings

For the 0.94B model, three changes raised the score most: fine-tuning all weights instead of LoRA adapters, adding image noise while
also training the vision encoder, and averaging the weights of several fine-tuned models. Preference training
(DPO), reinforcement learning on the F1 score (GRPO), generating one JSON key at a time and constrained decoding
did not improve it; GRPO stayed within the seed spread both on training pages and on 974 pages the model had not seen. The largest group of remaining errors is values the model leaves out; misread characters are one of the smallest.
[docs/findings.md](docs/findings.md) gives the measured effect of each change.

## Repository

```
src/tiny_resume_parser/   the package: schema, data, models, train, generate, evaluate, metrics, soup, label, parse
configs/                  training settings for each model
examples/                 eight labelled pages and a notebook that trains on them (Colab or local)
slurm/                    cluster jobs: train and score, average, test
tests/                    scoring rules, PDF input, configs
docs/                     method, code tour, evaluation, training, findings, model card; research/ = the lab log
```

## Limitations

- Each page is returned separately; pages of one CV are not merged.
- The training CVs come from one dataset.
- Instructions written in visible text on the page are read like any other text; this has not been tested.

## Citation

```bibtex
@software{kucuk2026tinyresumeparser,
  author = {Kucuk, Furkan},
  title  = {tiny-resume-parser: small vision-language models that read CV page images into JSON},
  year   = {2026},
  url    = {https://github.com/iamkucuk/tiny-resume-parser}
}
```

The page images come from the [Cv Parsing Dataset](https://universe.roboflow.com/invoicemodelisation/cv-parsing-ksgbu)
(invoiceModelisation, Roboflow Universe, 2024, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). They
were deduplicated and grouped into CVs; the dataset's bounding boxes were not used.

## Licence

Code: [PolyForm Noncommercial 1.0.0](LICENSE.md), free for personal, research, educational and nonprofit use, not for
commercial use. Weights and documentation, once released: CC BY-NC 4.0. Each base model
([InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf),
[SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct),
[granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M)) keeps its own licence.
