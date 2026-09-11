# Contributing

This repository is research-first: correctness of the mathematics and reproducibility of experiments take priority over feature count.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pre-commit install
pytest
```

## Expectations

- Add a test for every certificate or solver change.
- Never label a numerical estimator as **certified** unless its assumptions are explicit and its bound direction is justified.
- Separate *validation-only* exact quantities from quantities available to the online algorithm.
- Record experiment configurations and random seeds.
- Keep production-scale Devito/SubsurfaceGen adapters optional so the small mathematical test suite remains lightweight.
