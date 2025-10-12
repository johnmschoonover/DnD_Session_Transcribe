"""FastAPI application factory for the web UI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from .api.routes import router
from .services.jobs import JobRunner, JobService, build_cli_args, safe_filename

__all__ = ["create_app", "app", "build_cli_args", "safe_filename", "main"]

_WEB_ROOT_ENV = "DND_TRANSCRIBE_WEB_ROOT"


def _resolve_base_dir(base_dir: Optional[Path]) -> Path:
    if base_dir is not None:
        return Path(base_dir).expanduser().resolve()
    root = os.environ.get(_WEB_ROOT_ENV)
    return Path(root).expanduser().resolve() if root else Path.cwd() / "webui_runs"


def create_app(base_dir: Optional[Path] = None) -> FastAPI:
    resolved_dir = _resolve_base_dir(base_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="DnD Session Transcribe Web UI")
    runner = JobRunner()
    service = JobService(resolved_dir, runner=runner)
    app.state.job_service = service
    app.state.job_runner = runner
    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    """Launch the Web UI with Uvicorn."""

    import uvicorn

    host = os.environ.get("DND_TRANSCRIBE_WEB_HOST", "0.0.0.0")
    port_text = os.environ.get("DND_TRANSCRIBE_WEB_PORT", "8000")
    try:
        port = int(port_text)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise SystemExit(f"Invalid DND_TRANSCRIBE_WEB_PORT: {port_text}") from exc

    uvicorn.run(app, host=host, port=port)
