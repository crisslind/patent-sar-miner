# Contributing

Changes should preserve reproducibility and the audit trail from raw input to derived structure. Add tests for chemical transformations, file-format handling and report-facing calculations.

1. Create the Conda environment with `conda env create -f environment.yml`.
2. Install development dependencies with `python -m pip install -e '.[dev]'`.
3. Run `ruff check .` and `pytest` before submitting changes.

Do not contribute confidential structures, proprietary patent exports or datasets whose terms prohibit redistribution. Synthetic regression cases are preferred for tests.
