# Changelog

## 0.3.1 — 2026-11-06

- **torchvision is now a dependency**. Since 0.2.0, ResuVL-L's processor (Qwen3.5) could not be loaded without it.
  Running `pip install` followed by `parse --model furkankucuk/ResuVL-L` failed where torchvision was not already installed, though it is on Colab.

  The new drafting test in CI found this failure.
  This test is the first test to load the processor.

## 0.3.0 — 2026-11-06

- With `tiny-resume-parser parse --drafts K`, ResuVL-L's multi-token prediction head **drafts up to K tokens per step** on a GPU (`draft.py`).
  The model checks them in one pass.
  There is no speed-up on a CPU, and without `--drafts`, nothing changes.

  With 3 drafts, decoding was 2.46 times faster on an H100 than plain decoding at the same field F1 on 30 validation pages.
  In bf16, about one answer in four differs somewhere from plain decoding, by rounding.
- **ResuVL-L's weight file on Hugging Face** now holds that head, where Qwen3.5's own files keep it.
  The head contains 20 million parameters in `mtp.*` tensors, trained for 16 minutes on the frozen model.
  Transformers ignores the head when loading, other weights are unchanged, and revision `v0.2.0` stays available without the head.
- The flags **`train --mtp-weight`, `--mtp-only` and `--mtp-init`** train with the head as an extra loss or train the head alone (`mtp.py`).
  As an extra loss, it did not change field F1 over five seeds.
- The script **`experiments/label_vote.py`** scores labels chosen from several labelling runs against the reviewed pages.
- **Public docs and model cards** name the models ResuVL-S, ResuVL-M, ResuVL-L and ResuVL-L v0.1 only.
  Training configs are renamed to match each model:

  | Model | Training config |
  | :--- | :--- |
  | ResuVL-S | `configs/resuvl-s.toml` |
  | ResuVL-M | `configs/resuvl-m.toml` |
  | ResuVL-L | `configs/resuvl-l.toml` |
  | ResuVL-L v0.1 | `configs/resuvl-l-v0.1.toml` |

## 0.2.2 — 2026-10-17

- The command `tiny-resume-parser parse` **no longer reports a page as valid** when the model's answer contains no JSON.
  Such pages were returned as `"valid": true` with an empty record.
  Answers are now decoded in one function, `metrics.read_answer`, for scoring and for parsing; scores are unchanged.
- **Tests** check training labels and the parse command on stored answers.
  Only the answer and its end token are learnt.
  Perturbation never changes labels.

## 0.2.1 — 2026-10-17

- **ResuVL-L reads a page in about 55 s instead of 128 s** on an M2 Max CPU.
  On a CPU with bf16 instructions (Apple M2 and later, AVX512-BF16, AMX, Arm bf16), its language model runs in bf16 and its vision encoder in float32, with the same field F1 on 10 validation pages (0.811 against 0.816).

## 0.2.0 — 2026-10-16

- **ResuVL-L is now built on a newer base model** of 0.87B parameters, averaged over 3 fine-tuned runs.
  On the 204 test pages, lenient field F1 reached 0.845, against 0.735 for ResuVL-L v0.1.
  ResuVL-L v0.1 stays available as revision `v0.1.0` on Hugging Face.
- The package now needs **transformers 5.5.4** and adds the training recipe `configs/resuvl-l.toml`.
  Under transformers 5.5.4, ResuVL-S and ResuVL-M give different answers on most test pages:

  | Model | Score under 5.5.4 | Score under 4.57.6 |
  | :--- | :--- | :--- |
  | ResuVL-S | 0.652 | 0.633 |
  | ResuVL-M | 0.681 | 0.677 |

- **ResuVL-L inference runs faster** without the causal-conv1d kernel on a CPU and an Apple GPU.
  Short depthwise convolutions run as shifted multiply-adds instead of PyTorch's per-channel CPU convolution.
  A CV page on an M2 Max CPU takes about 2 minutes instead of over an hour.
- **Checkpoints and weight averages** keep the processor files.
  Bases with the same architecture can differ in tokenizer.
- The `train` command accepts **`--lora` for larger teacher models** and `--teacher` for logit distillation.
  The script `experiments/synth_cvs.py` renders synthetic CV pages from training labels.

## 0.1.0 — 2026-10-16

First public release.

- **Package `tiny_resume_parser`**: training, evaluation (strict, lenient, content and repaired field F1), weight averaging, and a `tiny-resume-parser parse` command that reads PDFs and page images.
- **Training recipes** for ResuVL-S, ResuVL-M and ResuVL-L (`configs/`).
- **Documentation**: method, code tour, evaluation, training, findings and model card (`docs/`), and the unedited research log (`docs/research/`).
- **Eight labelled example pages** and a notebook that trains and scores a model on them, on Colab or locally, with an interactive page explorer (`examples/`).
- **Weights on Hugging Face** under CC BY-NC 4.0: ResuVL-S, ResuVL-M and ResuVL-L.
  Weights were trained with float32 master weights on every device; bf16 is used only on GPUs that support it in hardware.

