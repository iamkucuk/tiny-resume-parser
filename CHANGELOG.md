# Changelog

## 0.2.0 — 2026-10-16

- ResuVL-L is now built on Qwen3.5-0.8B (0.87B, average of 3 fine-tuned runs): lenient field F1 0.845 on the 204
  test pages, against 0.735 for the InternVL3-1B version, which stays available as revision `v0.1.0` on Hugging Face.
- Training recipe `configs/qwen3.5-0.8b.toml`; the package now needs transformers 5.5.4. Under it, ResuVL-S and
  ResuVL-M give different answers on most test pages and score 0.652 and 0.681 (0.633 and 0.677 under 4.57.6).
- Faster Qwen3.5 inference without the causal-conv1d kernel (CPU, Apple GPU): the short depthwise convolutions run
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
- Training recipes for three students: InternVL3-1B, SmolVLM-500M and granite-docling-258M (`configs/`).
- Documentation: method, code tour, evaluation, training, findings and model card (`docs/`), and the unedited
  research log (`docs/research/`).
- Eight labelled example pages and a notebook that trains and scores a model on them, on Colab or locally, with an
  interactive page explorer (`examples/`).
- Weights on Hugging Face under CC BY-NC 4.0: ResuVL-S (from granite-docling-258M), ResuVL-M (from SmolVLM-500M) and
  ResuVL-L (from InternVL3-1B).
- Weights trained with float32 master weights on every device; bf16 is used only on GPUs that support it in hardware.
