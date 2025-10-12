# Testing status

## Current status
The web integration tests exercise the FastAPI application via `fastapi.testclient.TestClient`, which in turn depends on Starlette and `httpx`. Those packages are now part of the core project dependencies, so a standard installation (`pip install .`) provides everything required for the suite.

## Local setup tips
- If you installed the project prior to the dependency update, rerun `pip install --upgrade .` (or `pip install -e .` for editable mode) so `fastapi>=0.119` and `httpx>=0.28` are available.
- After reinstalling, `pytest -q` should complete without import errors. Any remaining failures are unrelated to the missing FastAPI stack.
