# Project context for the research (read fully before searching)

## Task
One vision-language model (VLM) reads one CV/resume page image and writes the page's content as JSON in a fixed
schema (basics: name, label, emails, phones, location, profiles; work, education, skills, languages, projects,
certificates, awards, interests, ...). Output is ~500-1,500 tokens of compact JSON per page. Code:
https://github.com/iamkucuk/tiny-resume-parser (public). Models: huggingface.co/furkankucuk/ResuVL-{S,M,L}.

## Hard constraints (from the project owner)
- The deployed model must have UNDER 1B parameters in total (vision encoder + projector + language model).
  Larger models may be used during training (teacher, source for pruning or distillation), but the shipped model
  must be <1B. Runs on a laptop CPU (~30-70 s per page today).
- One model, end to end, image -> JSON. PROHIBITED: a separate OCR engine (Tesseract, PaddleOCR text detector, etc.)
  whose text is fed to a model; any pipeline of OCR + LLM; a second LLM that rewrites the first one's output.
  ALLOWED: OCR-specialised VLMs as the base model; extra layers/heads/decoders inside the one model; any training
  method; synthetic data; larger models used only at training time.
- Licence of anything we train on must allow release (weights released CC BY-NC 4.0).

## Data
- 1,393 training pages + 79 validation pages from one public dataset (Roboflow "Cv Parsing Dataset", CC BY 4.0),
  80 % French CVs (mostly North African / French students and engineers), 19 % English. Single pages, 120 dpi.
- Labels written by a large VLM (GLM-5.3-Flash, whose model card lists only English and Chinese). 254 pages were
  reviewed by hand starting from those labels (review changed <1 % of values). Test = 204 of those, dev = 50.
- The French in the data is narrow: names, job titles, school names, dates, short skill lists; ~330k words in total,
  half of all values are one word.

## Metric and current results (lenient field F1 = (path, value) pairs, case/spacing/punctuation ignored)
| Model | Base (params) | Test F1 | Content F1 |
|---|---|---|---|
| ResuVL-L | InternVL3-1B (0.94B: InternViT-300M + Qwen2.5-0.5B) | 0.735 | 0.840 |
| ResuVL-M | SmolVLM-500M (SigLIP-base + SmolLM2-360M) | 0.677 | 0.788 |
| ResuVL-S | granite-docling-258M (SigLIP2-base + Granite-165M) | 0.633 | 0.752 |
- Goal: field F1 clearly above 0.9. The labelling model re-run on the same pages scores 0.912 against the reviewed
  labels (content F1 0.967), i.e. label noise is ~9 % at field level.
- Larger models trained the same way (LoRA, outside the size limit): InternVL3-2B 0.754, InternVL3-8B 0.824 (dev).
  Model size is the strongest lever measured so far.
- Generalisation gap: on TRAINING pages the 1B model scores 0.910; on unseen dev pages 0.703.
- Where the lost values go (dev, 1B model): value left out entirely 10.6 % of all gold values; page not parsed
  (invalid JSON / loop) 6.7 %; joined with other text 4.6 %; written in part 3.1 %; misread characters 2.9 %;
  put under another field 1.8 %. Misreading characters is NOT the main error; leaving values out is.

## Already tried (do not re-propose without a new reason)
Full fine-tune beats LoRA (+0.03). Image noise augmentation + training the vision encoder (+0.02). Uniform weight
averaging of 10 seeds ("model soup") (+0.02). More tiles (12 best for InternVL; 30 for SmolVLM). Nougat-style token
perturbation (no effect). DPO on own wrong answers (worse). GRPO with field-F1 reward after SFT, 4 settings (no gain:
little room on training pages, reward dominated by JSON validity). One-key-per-call generation (no gain). Schema-
constrained decoding with lm-format-enforcer (same F1, 5.4x slower). Repetition penalties / n-gram bans (no gain).
Canonicalised labels, NEFTune, lower vision-encoder LR, a first training stage on page transcripts, a second model
that checks labels (all no gain). Adding 993 pages from a second dataset: +0.011 (within noise; licence unknown).

## Compute
SLURM GPU cluster (A100/H100-class nodes, ~4 GPUs in daytime, more at night, 3-h jobs preferred); an Apple M2 Max.
A full fine-tune of the 1B model takes ~1 h on one GPU.
