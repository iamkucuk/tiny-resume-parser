# Experiments (not adopted)

Code for methods that were tried and did not improve the recipe, with the cluster job that ran each one. Kept so
the results in [docs/findings.md](../docs/findings.md) can be reproduced. Run from the repository root with the package
installed (`pip install -e ".[train,experiments]"`).

| File | Method | Outcome |
|---|---|---|
| `train_dpo.py`, `dpo.slurm` | Preference training (DPO + SFT term) on the model's own wrong answers | every step lowered F1 |
| `train_grpo.py`, `grpo.slurm`, `grpo_progress.py` | GRPO with lenient field F1 as the reward (TRL) | no measurable change, 4 settings |
| `grpo_probe.py` | Samples answers as GRPO did, to measure what its reward could teach | analysis, 2026-10-15 |
| `transcribe.py`, `transcribe.slurm`, `reading_file.py` | A first training stage on page transcripts (Pix2Struct-style warm-up) | +0.005 / −0.023, rejected |

Other experiments live in the package behind a flag: key-by-key generation (`--keywise`, `keywise.py`), NEFTune
(`--neftune`), a separate vision-encoder learning rate (`--vision-lr`), canonical labels (`canon.py`).
