# Contributing

To keep the codebase consistent, install the development dependencies before committing:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

After installing the hooks, run the full suite once to set the baseline:

```bash
pre-commit run --all-files
```

The configured hooks format Python files with Black, lint with Ruff, and enforce standard whitespace checks.
