# Testing status

## Pytest error summary
Running the test suite in a clean checkout currently fails during collection:

```
$ pytest -q
ImportError while importing test module '/workspace/DnD_Session_Transcribe/tests/test_web.py'.
Traceback:
tests/test_web.py:6: in <module>
    from fastapi.testclient import TestClient
ModuleNotFoundError: No module named 'fastapi'
```

The failure is triggered by `tests/test_web.py` importing `fastapi.testclient.TestClient`. The production code ships a FastAPI
application (`dnd_session_transcribe.web:app`), but the dependency is not part of the default development install, so a fresh
environment cannot import it when the tests run.

## Suggested fix
Install FastAPI (and its Starlette dependency) in the development environment before running the suite. Two easy options:

1. `pip install fastapi[standard]` – installs FastAPI plus Starlette, Uvicorn, and common extras.
2. Add `fastapi>=0.109,<1` to `requirements-dev.txt` (or your preferred dependency manager) so the package is always present when contributors set up the project.

Either approach restores the missing module so `pytest` can import the web tests successfully.
