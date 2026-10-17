# Changelog

## Unreleased

- Public docs and model cards name the models ResuVL-S, ResuVL-M, ResuVL-L and ResuVL-L v0.1 only; training configs
  are renamed to `configs/resuvl-s.toml`, `resuvl-m.toml`, `resuvl-l.toml` and `resuvl-l-v0.1.toml`.

## 0.2.2 — 2026-10-17

- `tiny-resume-parser parse` no longer reports a page as valid when the model's answer contains no JSON: such pages
  were returned as `"valid": true` with an empty record. Answers are now decoded in one function,
  `metrics.read_answer`, for scoring and for parsing; scores are unchanged.
- Tests for the training labels (only the answer and its end token are learnt; perturbation never changes labels)
  and for the parse command on stored answers.

## 0.2.1 — 2026-10-17

- On a CPU with bf16 instructions (Apple M2 and later, AVX512-BF16, AMX, Arm bf16), ResuVL-L's language model runs in
  bf16 and its vision encoder in float32: ResuVL-L reads a page in about 55 s instead of 128 s on an M2 Max, with
  the same field F1 on 10 validation pages (0.811 against 0.816).

## 0.2.0 — 2026-10-16

- ResuVL-L is now built on a newer base model (0.87B, average of 3 fine-tuned runs): lenient field F1 0.845 on the
  204 test pages, against 0.735 for ResuVL-L v0.1, which stays available as revision `v0.1.0` on Hugging Face.
- Training recipe `configs/resuvl-l.toml`; the package now needs transformers 5.5.4. Under it, ResuVL-S and
  ResuVL-M give different answers on most test pages and score 0.652 and 0.681 (0.633 and 0.677 under 4.57.6).
- Faster ResuVL-L inference without the causal-conv1d kernel (CPU, Apple GPU): the short depthwise convolutions run
  as shifted multiply-adds instead of PyTorch's per-channel CPU convolution; a CV page on an M2 Max CPU takes about
  2 minutes instead of over an hour.
- Checkpoints and weight averages keep the processor files, since bases with the same architecture can differ in
  tokenizer.
- `train` accepts `--lora` for larger teacher models and `--teacher` for logit distillation;
  `experiments/synth_cvs.py` renders synthetic CV pages from training labels.

## 0.1.0 — 2026-10-16

First public release.

- Package `tiny_resume_parser`: training, evaluation (strict, lenient, content and repaired field F1), weight
  averaging and a `tiny-resume-parser parse` command that reads PDFs and page images.
- Training recipes for ResuVL-S, ResuVL-M and ResuVL-L (`configs/`).
- Documentation: method, code tour, evaluation, training, findings and model card (`docs/`), and the unedited
  research log (`docs/research/`).
- Eight labelled example pages and a notebook that trains and scores a model on them, on Colab or locally, with an
  interactive page explorer (`examples/`).
- Weights on Hugging Face under CC BY-NC 4.0: ResuVL-S, ResuVL-M and ResuVL-L.
- Weights trained with float32 master weights on every device; bf16 is used only on GPUs that support it in hardware.
