# Experiments

Code for ideas that were tried and not adopted into the training recipe. Nothing in `src/` imports from here.

| Script | What it does | Outcome |
|---|---|---|
| `synth_cvs.py` | Renders synthetic CV pages with exact labels: a training record gets a new name and contact details, is laid out in one of several HTML templates, printed to PDF by headless Chrome and rasterised at 120 dpi. | Not adopted. For ResuVL-L v0.1, most of the gain came from the longer training it caused; for ResuVL-L, 1,000 pages gave no gain. Details: [findings](../docs/findings.md#a-newer-base-model-resuvl-l). |
| `nemotron_parse.py` | Runs nvidia/Nemotron-3.5-Parse (0.63B, page to Markdown) on the 129 development pages and measures how much of the label text appears anywhere in its output, next to ResuVL-L's answers. | Not adopted. It found 0.920 / 0.914 of label values (gold / validation) against 0.937 / 0.914 for ResuVL-L, in one pass: a dedicated page reader does not read more of these pages. |

Code for earlier experiments (LoRA, constrained decoding, DPO, GRPO and others) is kept in the git tag
`archive-2026-10-15`; restore a file with `git show archive-2026-10-15:PATH`.
