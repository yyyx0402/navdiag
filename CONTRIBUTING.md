# Contributing

Install `python -m pip install -c requirements-lock.txt -e ".[dev]"` in a virtual environment. Run `python -m pytest -q`, `python -m navdiag demo`, and `python scripts/check_release.py` before submitting a change.

For algorithm changes, compare existing baselines under the same protocol. Keep complete acquisitions in one split, fit preprocessing on training data, and freeze thresholds/settings using validation before evaluating test data. Report label coverage, availability and failures alongside scores.

Document sources and licenses. Do not commit raw data, weights, environments, credentials or personal materials. Changes to allowlisted knowledge documents require rebuilding the corpus and running retrieval checks; source hits do not establish answer correctness.
