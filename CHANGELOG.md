# Changelog

This file lists changes to the code. Research results are in [docs/findings.md](docs/findings.md).

## Unreleased

- Package `tiny_resume_parser` with one CLI: `parse`, `train`, `evaluate`, `rescore`, `soup`, `label`.
- `parse`: CV images or PDFs (rendered at 120 dpi) to one JSON record per page.
- Training recipes as config files (`configs/*.toml`); `--vision-lr` option for a separate vision-encoder rate.
- Tests for the scoring rules, PDF rendering and config keys; ruff lint and format; CI on every push.
- `slurm/test.slurm`: one-time scoring of the 204 held-out test pages.
- Cluster jobs moved from `truba/` to `slurm/`; jobs of rejected experiments moved to `experiments/`.
