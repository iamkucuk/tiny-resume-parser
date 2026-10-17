# Example pages

[![Try the models in Colab](https://img.shields.io/badge/Colab-try%20the%20models-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/iamkucuk/tiny-resume-parser/blob/main/examples/try_resuvl.ipynb)
[![Train on eight pages in Colab](https://img.shields.io/badge/Colab-train%20on%208%20pages-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/iamkucuk/tiny-resume-parser/blob/main/examples/train_on_eight_pages.ipynb)

Eight labelled CV pages for trying the code without the full dataset. The files use the same format as the full
label files ([docs/training.md](../docs/training.md#data)), so every command runs on them when
`TINY_RESUME_PARSER_DATA=examples` is set.

- `train.jsonl`: six pages from the training split.
- `eval.jsonl`: two pages from the validation split, which no released model was trained on.
- `pages/`: the page images, at the resolution used in training.

The labels were written by the teacher model and were not reviewed by hand, so they contain the teacher's errors.
Six pages are enough to see the whole pipeline run and the model memorise them (field F1 0.78 and 1.00 on those pages in
two test runs), but not to read new CVs (0.16 and 0.01 on the two unseen pages): [train_on_eight_pages.ipynb](train_on_eight_pages.ipynb) shows both.
[train_on_eight_pages_executed.ipynb](train_on_eight_pages_executed.ipynb) is the same notebook after a full run on an Apple M2 Max, with
all outputs saved ([open it in Colab](https://colab.research.google.com/github/iamkucuk/tiny-resume-parser/blob/main/examples/train_on_eight_pages_executed.ipynb)); its widgets show only after running it again.

## Quickstart

```bash
export TINY_RESUME_PARSER_DATA=examples
tiny-resume-parser train --model ibm-granite/granite-docling-258M --lr 1e-4 --max-patches 12 --augment 0.5 \
    --epochs 15 --accum 1 --train-file examples/train.jsonl --out runs/example
tiny-resume-parser evaluate --model runs/example/model --labels examples/eval.jsonl --max-patches 12 --out runs/example/eval
```

## Source

The page images come from the [Cv Parsing Dataset](https://universe.roboflow.com/invoicemodelisation/cv-parsing-ksgbu)
by invoiceModelisation on Roboflow Universe, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
