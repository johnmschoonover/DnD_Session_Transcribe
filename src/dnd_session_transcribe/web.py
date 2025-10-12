"""FastAPI-powered Web UI for remote DnD session transcription."""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import logging
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from . import cli

__all__ = ["create_app", "app", "build_cli_args", "safe_filename", "main"]

LOGGER = logging.getLogger(__name__)

_WEB_ROOT_ENV = "DND_TRANSCRIBE_WEB_ROOT"


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_filename(filename: str) -> str:
    """Return a filesystem-safe representation of *filename*."""

    name = Path(filename).name
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return sanitized or "upload"


def build_cli_args(
    audio: Path,
    *,
    outdir: Path,
    resume: bool = False,
    num_speakers: Optional[int] = None,
    asr_model: Optional[str] = None,
    asr_device: Optional[str] = None,
    asr_compute_type: Optional[str] = None,
    precise_rerun: bool = False,
    precise_model: Optional[str] = None,
    precise_device: Optional[str] = None,
    precise_compute_type: Optional[str] = None,
    vocal_extract: Optional[str] = None,
    log_level: str = cli.LOG.level,
    preview_start: Optional[float | int | str] = None,
    preview_duration: Optional[float | int | str] = None,
    preview_output: Optional[Path | str] = None,
) -> argparse.Namespace:
    """Construct an ``argparse.Namespace`` compatible with the CLI entry point."""

    return argparse.Namespace(
        audio=str(audio),
        outdir=str(outdir),
        ram=False,
        resume=resume,
        num_speakers=num_speakers,
        hotwords_file=None,
        initial_prompt_file=None,
        spelling_map=None,
        precise_rerun=precise_rerun,
        asr_model=asr_model,
        asr_device=asr_device,
        asr_compute_type=asr_compute_type,
        precise_model=precise_model,
        precise_device=precise_device,
        precise_compute_type=precise_compute_type,
        vocal_extract=vocal_extract,
        log_level=log_level,
        preview_start=preview_start,
        preview_duration=preview_duration,
        preview_output=str(preview_output) if preview_output is not None else None,
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _job_status_template(job_id: str, created_at: str) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": "running",
        "created_at": created_at,
        "updated_at": created_at,
        "output_dir": None,
        "error": None,
    }


def _generate_job_id(prefix: str = "job") -> str:
    token = secrets.token_hex(2)
    return f"{prefix}-{datetime.utcnow():%Y%m%d-%H%M%S}-{token}"


def _checkbox_to_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() not in {"0", "false", "off"}


def _build_home_html(jobs: Iterable[dict[str, Any]], message: str | None = None) -> str:
    rows = []
    for job in jobs:
        job_id = html.escape(job.get("job_id", ""))
        status = html.escape(job.get("status", "unknown"))
        created = html.escape(job.get("created_at", ""))
        updated = html.escape(job.get("updated_at", ""))
        error = html.escape(job.get("error", "") or "")
        rows.append(
            f"<tr><td><a href=\"/runs/{job_id}\">{job_id}</a></td>"
            f"<td>{status}</td><td>{created}</td><td>{updated}</td><td>{error}</td></tr>"
        )

    message_html = f"<div class='message'>{html.escape(message)}</div>" if message else ""

    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>DnD Session Transcribe Web UI</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; background: #f6f8fa; color: #24292f; }}
    h1 {{ margin-bottom: 1rem; }}
    form {{ background: #fff; padding: 1.5rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 2rem; }}
    fieldset {{ border: none; padding: 0; margin: 0; }}
    label {{ display: block; margin-top: 0.75rem; font-weight: 600; }}
    input[type='text'], select {{ width: 100%; padding: 0.5rem; border: 1px solid #d0d7de; border-radius: 6px; }}
    input[type='file'] {{ margin-top: 0.5rem; }}
    button {{ margin-top: 1.5rem; background: #2da44e; border: none; color: #fff; padding: 0.75rem 1.5rem; border-radius: 6px; cursor: pointer; font-size: 1rem; }}
    button:hover {{ background: #2c974b; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    th, td {{ padding: 0.75rem; border-bottom: 1px solid #d0d7de; text-align: left; }}
    th {{ background: #f0f3f6; }}
    .message {{ margin-bottom: 1rem; padding: 0.75rem; background: #fff3cd; border: 1px solid #ffe69c; border-radius: 6px; }}
    .preview-box {{ margin-top: 1rem; padding: 1rem; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; }}
    .preview-box label {{ margin-top: 0.5rem; font-weight: 500; }}
    .help-text {{ font-size: 0.85rem; color: #57606a; margin-top: 0.25rem; }}
  </style>
</head>
<body>
  <h1>DnD Session Transcribe</h1>
  {message_html}
  <form action=\"/transcribe\" method=\"post\" enctype=\"multipart/form-data\">
    <fieldset>
      <label for=\"audio_file\">Audio file</label>
      <input id=\"audio_file\" name=\"audio_file\" type=\"file\" required />

      <label for=\"log_level\">Log level</label>
      <select id=\"log_level\" name=\"log_level\">
        {''.join(f"<option value='{level}'>{level}</option>" for level in cli.LOG_LEVELS)}
      </select>

      <label for=\"asr_model\">Faster-Whisper model override (optional)</label>
      <input id=\"asr_model\" name=\"asr_model\" type=\"text\" placeholder=\"e.g. tiny, base, large-v3\" />

      <label for=\"num_speakers\">Number of speakers (optional)</label>
      <input id=\"num_speakers\" name=\"num_speakers\" type=\"text\" inputmode=\"numeric\" placeholder=\"e.g. 4\" />

      <label for=\"asr_device\">ASR device override</label>
      <select id=\"asr_device\" name=\"asr_device\">
        <option value=\"\">(default)</option>
        <option value=\"cpu\">cpu</option>
        <option value=\"cuda\">cuda</option>
        <option value=\"mps\">mps</option>
      </select>

      <label for=\"asr_compute_type\">ASR compute type override</label>
      <input id=\"asr_compute_type\" name=\"asr_compute_type\" type=\"text\" placeholder=\"e.g. float16\" />

      <label for=\"vocal_extract\">Preprocessing</label>
      <select id=\"vocal_extract\" name=\"vocal_extract\">
        <option value=\"\">(default)</option>
        <option value=\"off\">off</option>
        <option value=\"bandpass\">bandpass</option>
      </select>

      <div class=\"preview-box\">
        <label><input type=\"checkbox\" name=\"preview_enabled\" value=\"true\" /> Render preview snippet</label>
        <label for=\"preview_start\">Preview start (seconds or MM:SS)</label>
        <input id=\"preview_start\" name=\"preview_start\" type=\"text\" placeholder=\"e.g. 1:30\" />
        <label for=\"preview_duration\">Preview duration (seconds)</label>
        <input id=\"preview_duration\" name=\"preview_duration\" type=\"text\" placeholder=\"default: 10\" />
        <div class=\"help-text\">When enabled, the snippet audio and transcripts will be generated alongside the full outputs.</div>
      </div>

      <div style=\"margin-top: 0.75rem;\">
        <label><input type=\"checkbox\" name=\"resume\" value=\"true\" /> Resume from cached checkpoints</label>
      </div>
      <div>
        <label><input type=\"checkbox\" name=\"precise_rerun\" value=\"true\" /> Enable precise re-run</label>
      </div>
    </fieldset>
    <button type=\"submit\">Start transcription</button>
  </form>

  <h2>Jobs</h2>
  <table>
    <thead><tr><th>Job</th><th>Status</th><th>Created</th><th>Updated</th><th>Error</th></tr></thead>
    <tbody>
      {''.join(rows) if rows else "<tr><td colspan='5'>No jobs yet.</td></tr>"}
    </tbody>
  </table>
</body>
</html>
"""


def _build_job_html(
    job: dict[str, Any],
    files: list[tuple[str, str]],
    log_available: bool,
    preview: dict[str, Any] | None = None,
    preview_url: str | None = None,
) -> str:
    job_id = html.escape(job.get("job_id", ""))
    status = html.escape(job.get("status", "unknown"))
    created = html.escape(job.get("created_at", ""))
    updated = html.escape(job.get("updated_at", ""))
    error = html.escape(job.get("error", "") or "")
    audio_name = html.escape(job.get("audio_filename", ""))

    file_rows = []
    for label, url in files:
        file_rows.append(f"<li><a href=\"{url}\">{html.escape(label)}</a></li>")

    file_list = "<ul>" + "".join(file_rows) + "</ul>" if file_rows else "<p>No output files yet.</p>"
    log_link = "<p><a href=\"log\">Download job log</a></p>" if log_available else ""

    error_block = f"<div class='error'>Error: {error}</div>" if error else ""

    preview_block = ""
    preview_requested = bool(preview.get("requested")) if preview else False
    preview_details: list[str] = []
    if preview:
        start_val = preview.get("start")
        duration_val = preview.get("duration")
        if isinstance(start_val, (int, float)):
            preview_details.append(f"Start: {start_val:.2f}s")
        if isinstance(duration_val, (int, float)):
            preview_details.append(f"Duration: {duration_val:.2f}s")

    details_html = ""
    if preview_details:
        details_html = "<p>" + ", ".join(html.escape(item) for item in preview_details) + "</p>"

    if preview_url:
        escaped_url = html.escape(preview_url)
        preview_block = (
            "<div class='panel'>"
            "  <h2>Preview snippet</h2>"
            f"  <audio controls src=\"{escaped_url}\" preload=\"none\"></audio>"
            f"  {details_html}"
            "</div>"
        )
    elif preview_requested:
        preview_block = (
            "<div class='panel'>"
            "  <h2>Preview snippet</h2>"
            "  <p>Preview rendering in progress. The audio will appear once the job completes.</p>"
            f"  {details_html}"
            "</div>"
        )

    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Job {job_id}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; background: #f6f8fa; color: #24292f; }}
    h1 {{ margin-bottom: 1rem; }}
    .panel {{ background: #fff; padding: 1.5rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1.5rem; }}
    .error {{ padding: 0.75rem; background: #ffe3e6; border: 1px solid #ffccd5; border-radius: 6px; margin-top: 1rem; }}
    ul {{ padding-left: 1.25rem; }}
    a {{ color: #0969da; }}
  </style>
</head>
<body>
  <h1>Job {job_id}</h1>
  <div class='panel'>
    <p><strong>Status:</strong> {status}</p>
    <p><strong>Created:</strong> {created}</p>
    <p><strong>Updated:</strong> {updated}</p>
    <p><strong>Source audio:</strong> {audio_name}</p>
    {error_block}
  </div>
  {preview_block}
  <div class='panel'>
    <h2>Outputs</h2>
    {file_list}
    {log_link}
  </div>
  <p><a href=\"/\">Back to dashboard</a></p>
</body>
</html>
"""


def create_app(base_dir: Optional[Path] = None) -> FastAPI:
    """Create a configured FastAPI application."""

    if base_dir is None:
        root = os.environ.get(_WEB_ROOT_ENV)
        base_dir = Path(root).expanduser().resolve() if root else Path.cwd() / "webui_runs"
    else:
        base_dir = Path(base_dir).expanduser().resolve()

    base_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.debug("Web UI base directory: %s", base_dir)

    app = FastAPI(title="DnD Session Transcribe Web UI")
    app.state.runs_dir = base_dir

    def _job_dir(job_id: str) -> Path:
        return app.state.runs_dir / job_id

    def _list_jobs() -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for path in sorted(app.state.runs_dir.iterdir(), reverse=True):
            if not path.is_dir():
                continue
            status = _read_json(path / "status.json")
            if not status:
                continue
            meta = _read_json(path / "metadata.json")
            status.setdefault("audio_filename", meta.get("audio_filename", ""))
            jobs.append(status)
        jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
        return jobs

    def _collect_outputs(
        job_dir: Path, status: dict[str, Any]
    ) -> tuple[list[tuple[str, str]], str | None]:
        outputs: list[tuple[str, str]] = []
        preview_link: str | None = None
        output_dir = status.get("output_dir")
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = job_dir / "outputs"
        if out_path.exists():
            for child in sorted(out_path.glob("*")):
                if child.is_file():
                    rel = child.relative_to(job_dir)
                    url = f"files/{rel.as_posix()}"
                    outputs.append((child.name, url))
                    if preview_link is None and child.name.endswith("_preview.wav"):
                        preview_link = url
        return outputs, preview_link

    def _run_job(args: argparse.Namespace, job_id: str, job_dir: Path, created_at: str) -> None:
        log_path = job_dir / "job.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

        status = _job_status_template(job_id, created_at)
        _write_json(job_dir / "status.json", status)

        original_settings = {
            "ASR": {
                "hotwords_file": cli.ASR.hotwords_file,
                "initial_prompt_file": cli.ASR.initial_prompt_file,
                "model": cli.ASR.model,
                "device": cli.ASR.device,
                "compute_type": cli.ASR.compute_type,
            },
            "PRE": {"vocal_extract": cli.PRE.vocal_extract},
            "DIA": {"num_speakers": cli.DIA.num_speakers},
            "PREC": {
                "enabled": cli.PREC.enabled,
                "model": cli.PREC.model,
                "device": cli.PREC.device,
                "compute_type": cli.PREC.compute_type,
            },
        }

        resolved_outdir: Path | None = None

        try:
            resolved_outdir = cli.run_transcription(
                args, configure_logging=False, log_handlers=[handler]
            )
            status["status"] = "completed"
            if resolved_outdir is not None:
                status["output_dir"] = str(resolved_outdir)
            status["updated_at"] = _utc_now()
            _write_json(job_dir / "status.json", status)
        except BaseException as exc:  # pylint: disable=broad-except
            status["status"] = "failed"
            status["error"] = str(exc)
            status["updated_at"] = _utc_now()
            _write_json(job_dir / "status.json", status)
            LOGGER.exception("Job %s failed", job_id)
            raise
        finally:
            cli.ASR.hotwords_file = original_settings["ASR"]["hotwords_file"]
            cli.ASR.initial_prompt_file = original_settings["ASR"]["initial_prompt_file"]
            cli.ASR.model = original_settings["ASR"]["model"]
            cli.ASR.device = original_settings["ASR"]["device"]
            cli.ASR.compute_type = original_settings["ASR"]["compute_type"]
            cli.PRE.vocal_extract = original_settings["PRE"]["vocal_extract"]
            cli.DIA.num_speakers = original_settings["DIA"]["num_speakers"]
            cli.PREC.enabled = original_settings["PREC"]["enabled"]
            cli.PREC.model = original_settings["PREC"]["model"]
            cli.PREC.device = original_settings["PREC"]["device"]
            cli.PREC.compute_type = original_settings["PREC"]["compute_type"]

    @app.get("/", response_class=HTMLResponse)
    async def home() -> str:
        jobs = _list_jobs()
        return _build_home_html(jobs)

    @app.post("/transcribe")
    async def transcribe(
        audio_file: UploadFile = File(...),
        log_level: str = Form(cli.LOG.level),
        asr_model: str = Form(""),
        num_speakers: str = Form(""),
        asr_device: str = Form(""),
        asr_compute_type: str = Form(""),
        vocal_extract: str = Form(""),
        resume: Optional[str] = Form(None),
        precise_rerun: Optional[str] = Form(None),
        preview_enabled: Optional[str] = Form(None),
        preview_start: str = Form(""),
        preview_duration: str = Form(""),
    ) -> RedirectResponse:
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="Audio file is required")

        job_id = _generate_job_id()
        job_dir = _job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "inputs").mkdir(exist_ok=True)
        outputs_dir = job_dir / "outputs"
        outputs_dir.mkdir(exist_ok=True)

        clean_name = safe_filename(audio_file.filename)
        audio_path = job_dir / "inputs" / clean_name
        with audio_path.open("wb") as buffer:
            while chunk := await audio_file.read(1024 * 1024):
                buffer.write(chunk)
        await audio_file.close()

        created_at = _utc_now()
        preview_fields_supplied = preview_start.strip() or preview_duration.strip()
        preview_requested = _checkbox_to_bool(preview_enabled) or bool(preview_fields_supplied)

        preview_start_arg: Optional[float]
        preview_duration_arg: Optional[float]
        preview_meta: dict[str, Any] = {"requested": preview_requested}

        if preview_requested:
            start_text = preview_start.strip()
            duration_text = preview_duration.strip()
            try:
                start_value = (
                    cli.parse_time_spec(start_text)
                    if start_text
                    else 0.0
                )
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid preview_start value") from None

            try:
                duration_value = (
                    cli.parse_time_spec(duration_text)
                    if duration_text
                    else 10.0
                )
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid preview_duration value") from None

            if duration_value <= 0:
                raise HTTPException(status_code=400, detail="Preview duration must be positive")

            preview_start_arg = start_value
            preview_duration_arg = duration_value
            preview_meta.update({"start": start_value, "duration": duration_value})
        else:
            preview_start_arg = None
            preview_duration_arg = None

        metadata = {
            "job_id": job_id,
            "audio_filename": audio_file.filename,
            "created_at": created_at,
            "preview": preview_meta,
        }
        _write_json(job_dir / "metadata.json", metadata)

        parsed_num_speakers: Optional[int]
        try:
            parsed_num_speakers = int(num_speakers) if num_speakers.strip() else None
        except ValueError:
            raise HTTPException(status_code=400, detail="num_speakers must be an integer") from None

        if log_level not in cli.LOG_LEVELS:
            raise HTTPException(status_code=400, detail="Invalid log level")

        vocal_choice = vocal_extract or None
        if vocal_choice not in {None, "off", "bandpass"}:
            raise HTTPException(status_code=400, detail="Invalid vocal_extract value")

        args = build_cli_args(
            audio_path,
            outdir=outputs_dir,
            resume=_checkbox_to_bool(resume),
            num_speakers=parsed_num_speakers,
            asr_model=asr_model or None,
            asr_device=asr_device or None,
            asr_compute_type=asr_compute_type or None,
            precise_rerun=_checkbox_to_bool(precise_rerun),
            vocal_extract=vocal_choice,
            log_level=log_level,
            preview_start=preview_start_arg,
            preview_duration=preview_duration_arg,
        )

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, _run_job, args, job_id, job_dir, created_at)
        except SystemExit:
            return RedirectResponse(url=f"/runs/{job_id}", status_code=303)
        except Exception:
            return RedirectResponse(url=f"/runs/{job_id}", status_code=303)

        return RedirectResponse(url=f"/runs/{job_id}", status_code=303)

    @app.get("/runs/{job_id}", response_class=HTMLResponse)
    async def show_job(job_id: str) -> str:
        job_dir = _job_dir(job_id)
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        status = _read_json(job_dir / "status.json")
        meta = _read_json(job_dir / "metadata.json")
        status.setdefault("job_id", job_id)
        status.setdefault("created_at", meta.get("created_at", ""))
        status.setdefault("audio_filename", meta.get("audio_filename", ""))
        files, preview_link = _collect_outputs(job_dir, status)
        log_available = (job_dir / "job.log").exists()
        return _build_job_html(
            status,
            files,
            log_available,
            preview=meta.get("preview"),
            preview_url=preview_link,
        )

    @app.get("/runs/{job_id}/log")
    async def download_log(job_id: str) -> FileResponse:
        job_dir = _job_dir(job_id)
        log_path = job_dir / "job.log"
        if not log_path.exists():
            raise HTTPException(status_code=404, detail="Log not found")
        return FileResponse(log_path)

    @app.get("/runs/{job_id}/files/{file_path:path}")
    async def download_file(job_id: str, file_path: str) -> FileResponse:
        job_dir = _job_dir(job_id)
        target = (job_dir / file_path).resolve()
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        try:
            target.relative_to(job_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid file path") from exc
        return FileResponse(target)

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
