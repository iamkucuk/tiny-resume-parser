# Changelog

This file lists changes to the code. Research results are in [docs/findings.md](docs/findings.md).

## Unreleased

- Package `tiny_resume_parser` with one CLI: `parse`, `train`, `evaluate`, `rescore`, `soup`, `label`.
- `parse`: CV images or PDFs (rendered at 120 dpi) to one JSON record per page.
- Training recipes as config files (`configs/*.toml`); `--vision-lr` option for a separate vision-encoder rate.
- Tests for the scoring rules, PDF rendering and config keys; ruff lint and format; CI on every push.
- `slurm/test.slurm`: one-time scoring of the 204 held-out test pages.
- Cluster jobs in `slurm/` hold no cluster-specific settings: the environment goes in `slurm/site.sh`, cluster flags on
  the `sbatch` command line. Jobs of rejected experiments moved to `experiments/`.
- `label`: the endpoint is set with `LABEL_BASE_URL` / `LABEL_API_KEY`; fixed the repository-root path (it pointed at
  `src/` after the package move, so `.env` and `data/` were not found).
- `TINY_RESUME_PARSER_DATA` sets the data folder. `requirements.txt` removed: `pyproject.toml` lists the dependencies.
- The Colab notebooks moved to `docs/research/notebooks/`: they read a private Drive bundle.
