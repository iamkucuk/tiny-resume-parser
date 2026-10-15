# Contributing

Issues and pull requests are welcome for noncommercial use (see [LICENSE.md](LICENSE.md)).

```bash
pip install -e ".[train,dev]"
pre-commit install          # ruff lint + format on every commit
pytest                      # fast tests; data-dependent tests skip when data/ is absent
```

- One change per pull request, with a test when it changes behaviour.
- A change to training or scoring needs evidence: the baseline run, the new run, and both development scores
  (see [docs/evaluation.md](docs/evaluation.md) for the acceptance rule).
- Keep docs in the same pull request as the code they describe. Scores belong in `docs/evaluation.md`, the effect
  of a change in `docs/findings.md`; other pages link to them. `docs/research/` is the lab log: append, do not edit.
