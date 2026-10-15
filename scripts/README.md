# Scripts

One-off data-preparation and analysis scripts, kept as they ran. Run from the repository root with the package
installed.

| Script | What it does |
|---|---|
| `reconstruct.py` | Group the Roboflow page images into CVs (`data/processed/documents.jsonl`) |
| `build_gold_set.py`, `apply_gold_decisions.py` | Build the 254-page review set and apply the reviewed fixes |
| `build_label_sets.py`, `check_labels.py`, `label_progress.py` | Split teacher labels into training / gold, check them against OCR, follow a labelling run |
| `prepare_hadikp.py` | Extra pages from `hadikp/resume-data-pdf` (licence unverified; not used for released weights) |
| `analyze_errors.py`, `failure_report.py` | Classify scoring errors; build a local HTML failure report (not committed: shows CV pages) |
| `make_figures.py` | Draw the README and docs figures into `docs/assets/` |
