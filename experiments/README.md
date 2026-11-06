# Experiments

Code for ideas that were tried, and the measurement scripts behind adopted ones. Nothing in `src/` imports from here.

| Script | What it does | Outcome |
|---|---|---|
| `synth_cvs.py` | Renders synthetic CV pages with exact labels: a training record gets a new name and contact details, is laid out in one of several HTML templates, printed to PDF by headless Chrome and rasterised at 120 dpi. | Not adopted. For ResuVL-L v0.1, most of the gain came from the longer training it caused; for ResuVL-L, 1,000 pages gave no gain. Details: [findings](../docs/findings.md#a-newer-base-model-resuvl-l). |
| `nemotron_parse.py` | Runs nvidia/Nemotron-3.5-Parse (0.63B, page to Markdown) on the 129 development pages and measures how much of the label text appears anywhere in its output, next to ResuVL-L's answers. | Not adopted. It found 0.920 / 0.914 of label values (gold / validation) against 0.937 / 0.914 for ResuVL-L, in one pass: a dedicated page reader does not read more of these pages. |
| `mtp_decode.py` | Times decoding with ResuVL-L's multi-token prediction head drafting tokens (`--drafts K`) against plain decoding on saved development pages, and writes both answer sets for scoring; `accept` mode measures how often the head predicts the token after next. | Adopted into the package as `parse --drafts` (`src/tiny_resume_parser/draft.py`): 2.3 to 2.5 times faster on an H100 at the same F1. Details: [findings](../docs/findings.md#faster-decoding-with-the-base-models-drafting-head). |
| `label_vote.py` | Scores labels chosen from several runs of the labelling model against the reviewed gold pages: one run alone, the run that agrees most with the others (a whole record usable for training), and values found in at least half of the runs. | Pilot on the 50 gold pages: the most agreeing of 5 runs scores 0.934 against 0.915 for one run. Training on such labels has not been tried yet. |

Code for earlier experiments (LoRA, constrained decoding, DPO, GRPO and others) is kept in the git tag
`archive-2026-10-15`; restore a file with `git show archive-2026-10-15:PATH`.
