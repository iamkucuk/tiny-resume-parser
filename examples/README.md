# Example pages

Eight labelled CV pages for trying the code without the full dataset. The files use the same format as the full
label files ([docs/training.md](../docs/training.md#data)), so every command runs on them when
`TINY_RESUME_PARSER_DATA=examples` is set.

- `train.jsonl`: six pages from the training split.
- `eval.jsonl`: two pages from the validation split, which no released model was trained on.
- `pages/`: the page images, at the resolution used in training.

The labels were written by the teacher model and were not reviewed by hand, so they contain the teacher's errors.
Six pages are enough to see the whole pipeline run and the model memorise them (field F1 0.89 on those pages), but
not to read new CVs (0.00 on the two unseen pages): [quickstart.ipynb](quickstart.ipynb) shows both.

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
