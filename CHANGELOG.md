# Changelog

## 0.1.0 — 2026-10-16

First public release.

- Package `tiny_resume_parser`: training, evaluation (strict, lenient, content and repaired field F1), weight
  averaging and a `tiny-resume-parser parse` command that reads PDFs and page images.
- Training recipes for three students: InternVL3-1B, SmolVLM-500M and granite-docling-258M (`configs/`).
- Documentation: method, code tour, evaluation, training, findings and model card (`docs/`), and the unedited
  research log (`docs/research/`).
- Eight labelled example pages and a notebook that trains and scores a model on them, on Colab or locally, with an
  interactive page explorer (`examples/`).
- Weights on Hugging Face under CC BY-NC 4.0: ResuVL-S (from granite-docling-258M) and ResuVL-M (from SmolVLM-500M).
- Weights trained with float32 master weights on every device; bf16 is used only on GPUs that support it in hardware.
