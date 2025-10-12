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
import shutil
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import quote, quote_plus

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
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


@lru_cache(maxsize=1)
def _cuda_available() -> bool:
    try:
        import torch

        try:
            return bool(torch.cuda.is_available())
        except Exception:  # pragma: no cover - defensive logging only
            LOGGER.debug("torch.cuda.is_available() raised an exception", exc_info=True)
            return False
    except Exception:  # ImportError, RuntimeError, etc.
        return False


def _normalize_device_choice(value: str) -> Optional[str]:
    selection = value.strip()
    if not selection:
        return None

    normalized = selection.lower()
    if normalized == "cuda":
        if not _cuda_available():
            LOGGER.warning(
                "CUDA device requested by web UI but CUDA runtime is unavailable; defaulting to auto"
            )
            return None
        return "cuda"

    return selection


def _build_home_html(jobs: Iterable[dict[str, Any]], message: str | None = None) -> str:
    def _options_html(options: list[tuple[str, str]], selected: str) -> str:
        rendered: list[str] = []
        for value, label in options:
            safe_value = html.escape(value, quote=True)
            safe_label = html.escape(label)
            selected_attr = " selected" if value == selected else ""
            rendered.append(f"<option value=\"{safe_value}\"{selected_attr}>{safe_label}</option>")
        return "".join(rendered)

    rows: list[str] = []
    for job in jobs:
        raw_job_id = job.get("job_id", "")
        job_id_text = str(raw_job_id)
        job_id = html.escape(job_id_text)
        job_href = html.escape(f"/runs/{quote(job_id_text, safe='')}")
        delete_action = html.escape(f"/runs/{quote(job_id_text, safe='')}/delete")
        status = html.escape(job.get("status", "unknown"))
        created = html.escape(job.get("created_at", ""))
        updated = html.escape(job.get("updated_at", ""))
        error = html.escape(job.get("error", "") or "")
        rows.append(
            "<tr>"
            f"<td><a href=\"{job_href}\" target=\"_blank\" rel=\"noopener noreferrer\">{job_id}</a></td>"
            f"<td><span class='status-badge'>{status}</span></td>"
            f"<td>{created}</td><td>{updated}</td><td>{error}</td>"
            "<td>"
            f"  <form action=\"{delete_action}\" method=\"post\" class=\"inline-form\" "
            "onsubmit=\"return confirm('Delete this job and all associated files?');\">"
            "    <button type=\"submit\" class=\"button button-danger\">Delete</button>"
            "  </form>"
            "</td>"
            "</tr>"
        )

    message_html = (
        f"<div class='flash success'><span>{html.escape(message)}</span></div>"
        if message
        else ""
    )

    log_default = "DEBUG" if "DEBUG" in cli.LOG_LEVELS else cli.LOG.level
    log_options = _options_html([(level, level) for level in cli.LOG_LEVELS], log_default)

    asr_model_options = [
        ("", "Auto (use configured default)"),
        ("large-v3", "large-v3 (maximum accuracy)"),
        ("distil-large-v3", "distil-large-v3"),
        ("large-v2", "large-v2"),
        ("medium", "medium"),
        ("small", "small"),
        ("base", "base"),
        ("tiny", "tiny"),
        ("turbo", "turbo"),
    ]
    asr_model_default = "large-v3"
    asr_model_select = _options_html(asr_model_options, asr_model_default)

    num_speakers_options = [("", "Auto-detect"), *[(str(count), f"{count} speakers") for count in range(2, 9)]]
    num_speakers_default = "8"
    num_speakers_select = _options_html(num_speakers_options, num_speakers_default)

    device_options = [
        ("", "Auto (prefer GPU if available)"),
        ("cuda", "CUDA"),
        ("mps", "Apple Metal"),
        ("cpu", "CPU"),
    ]
    device_default = "cuda"
    asr_device_select = _options_html(device_options, device_default)

    compute_type_options = [
        ("float16", "float16 (fast & precise)"),
        ("float32", "float32"),
        ("int8_float16", "int8_float16"),
        ("int8_float32", "int8_float32"),
        ("int8", "int8"),
        ("", "Auto"),
    ]
    compute_default = "float16"
    asr_compute_select = _options_html(compute_type_options, compute_default)

    precise_model_options = [
        ("", "Auto (reuse ASR model)"),
        ("large-v3", "large-v3"),
        ("distil-large-v3", "distil-large-v3"),
        ("large-v2", "large-v2"),
        ("medium", "medium"),
    ]
    precise_model_select = _options_html(precise_model_options, "large-v3")
    precise_device_select = _options_html(device_options, device_default)
    precise_compute_select = _options_html(compute_type_options, compute_default)

    vocal_extract_options = [
        ("bandpass", "Bandpass (emphasize dialogue)"),
        ("", "Disabled"),
        ("off", "Bypass isolation"),
    ]
    vocal_extract_select = _options_html(vocal_extract_options, "bandpass")

    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>DnD Session Transcribe Web UI</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: radial-gradient(circle at 15% 20%, rgba(61, 255, 108, 0.12), transparent 55%),
            radial-gradient(circle at 85% 15%, rgba(82, 220, 255, 0.08), transparent 50%),
            #050b07;
      --panel: rgba(6, 20, 12, 0.82);
      --panel-border: rgba(61, 255, 108, 0.35);
      --panel-hover: rgba(10, 34, 20, 0.92);
      --accent: #3dff6c;
      --accent-soft: rgba(61, 255, 108, 0.18);
      --accent-strong: rgba(61, 255, 108, 0.35);
      --muted: #79ffb1;
      --text: #e0ffe9;
      --danger: #ff4d6d;
      --danger-soft: rgba(255, 77, 109, 0.15);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: 'Fira Code', 'Source Code Pro', 'Courier New', monospace;
      margin: 0;
      padding: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      display: flex;
      justify-content: center;
      padding: 3rem 1.5rem;
    }}
    main {{
      width: min(1120px, 100%);
      display: grid;
      gap: 2rem;
    }}
    header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 1.5rem;
    }}
    header h1 {{
      font-size: clamp(1.8rem, 2.6vw, 2.4rem);
      text-shadow: 0 0 22px rgba(61, 255, 108, 0.45);
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin: 0;
    }}
    header .tagline {{
      color: var(--muted);
      font-size: 0.95rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 16px;
      padding: 2rem;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);
      backdrop-filter: blur(12px);
      transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .panel:hover {{
      border-color: var(--accent-strong);
      transform: translateY(-2px);
      box-shadow: 0 24px 48px rgba(0, 0, 0, 0.4);
      background: var(--panel-hover);
    }}
    form fieldset {{
      display: grid;
      gap: 1.35rem;
      border: none;
      padding: 0;
      margin: 0;
    }}
    .field-group {{
      display: grid;
      gap: 0.55rem;
    }}
    label {{
      font-size: 0.85rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    select, input[type='text'], input[type='file'] {{
      width: 100%;
      padding: 0.75rem 0.85rem;
      border-radius: 10px;
      border: 1px solid rgba(61, 255, 108, 0.25);
      background: rgba(5, 14, 9, 0.75);
      color: var(--text);
      font-family: inherit;
      font-size: 0.95rem;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    select:focus, input[type='text']:focus {{
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-soft);
    }}
    input[type='file'] {{
      padding: 0.5rem 0;
    }}
    .section-title {{
      font-size: 1rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      margin: 0;
    }}
    .grid-two {{
      display: grid;
      gap: 1.25rem;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .preview-box {{
      border: 1px dashed var(--accent-strong);
      border-radius: 14px;
      padding: 1.1rem 1.35rem;
      background: rgba(6, 22, 12, 0.6);
      display: grid;
      gap: 0.75rem;
    }}
    .preview-box label {{
      text-transform: none;
      letter-spacing: 0.02em;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.95rem;
    }}
    .preview-box input[type='checkbox'] {{
      width: 1.15rem;
      height: 1.15rem;
      accent-color: var(--accent);
    }}
    .help-text {{
      font-size: 0.85rem;
      color: rgba(224, 255, 233, 0.72);
      line-height: 1.4;
    }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 0.9rem 1.8rem;
      border-radius: 999px;
      border: 1px solid var(--accent);
      background: linear-gradient(135deg, rgba(61, 255, 108, 0.9), rgba(61, 255, 108, 0.55));
      color: #041b0b;
      cursor: pointer;
      transition: transform 0.18s ease, box-shadow 0.18s ease;
      box-shadow: 0 18px 36px rgba(61, 255, 108, 0.25);
    }}
    .button:hover {{
      transform: translateY(-2px);
      box-shadow: 0 22px 44px rgba(61, 255, 108, 0.32);
    }}
    .button:active {{
      transform: translateY(0);
      box-shadow: 0 16px 32px rgba(61, 255, 108, 0.24);
    }}
    .button-danger {{
      border-color: var(--danger);
      background: linear-gradient(135deg, rgba(255, 77, 109, 0.92), rgba(255, 77, 109, 0.6));
      color: #fff;
      box-shadow: 0 18px 36px var(--danger-soft);
    }}
    .inline-form {{ display: inline; }}
    .flash {{
      padding: 0.9rem 1.25rem;
      border-radius: 12px;
      border: 1px solid var(--accent-strong);
      background: rgba(6, 28, 16, 0.7);
      box-shadow: inset 0 0 0 1px rgba(61, 255, 108, 0.1);
      display: inline-flex;
      align-items: center;
      gap: 0.6rem;
      font-size: 0.9rem;
      letter-spacing: 0.04em;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 1.5rem;
    }}
    thead th {{
      font-size: 0.75rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: rgba(224, 255, 233, 0.64);
      text-align: left;
      padding: 0 0 0.75rem 0;
    }}
    tbody td {{
      padding: 0.75rem 0;
      border-top: 1px solid rgba(224, 255, 233, 0.08);
      font-size: 0.95rem;
    }}
    tbody tr:first-child td {{ border-top: none; }}
    tbody tr:hover td {{
      background: rgba(6, 24, 13, 0.55);
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
      transition: text-shadow 0.2s ease;
    }}
    a:hover {{
      text-shadow: 0 0 12px rgba(61, 255, 108, 0.6);
    }}
    .status-badge {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0.3rem 0.75rem;
      border-radius: 999px;
      font-size: 0.8rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      border: 1px solid var(--accent-strong);
      background: rgba(61, 255, 108, 0.12);
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>DnD Session Transcribe</h1>
      <span class="tagline">Hyper-focused inference control</span>
    </header>
    {message_html}
    <form class="panel" action="/transcribe" method="post" enctype="multipart/form-data">
      <fieldset>
        <div class="field-group">
          <span class="section-title">Upload</span>
          <input id="audio_file" name="audio_file" type="file" required />
        </div>
        <div class="field-group">
          <span class="section-title">Logging</span>
          <select id="log_level" name="log_level">{log_options}</select>
        </div>
        <div class="field-group">
          <span class="section-title">Recognition strategy</span>
          <div class="grid-two">
            <div>
              <label for="asr_model">Faster-Whisper model</label>
              <select id="asr_model" name="asr_model">{asr_model_select}</select>
            </div>
            <div>
              <label for="num_speakers">Speaker count hint</label>
              <select id="num_speakers" name="num_speakers">{num_speakers_select}</select>
            </div>
            <div>
              <label for="asr_device">Inference device</label>
              <select id="asr_device" name="asr_device">{asr_device_select}</select>
            </div>
            <div>
              <label for="asr_compute_type">Compute precision</label>
              <select id="asr_compute_type" name="asr_compute_type">{asr_compute_select}</select>
            </div>
          </div>
        </div>
        <div class="field-group">
          <span class="section-title">Precision rerun</span>
          <div class="grid-two">
            <div>
              <label for="precise_model">Model</label>
              <select id="precise_model" name="precise_model">{precise_model_select}</select>
            </div>
            <div>
              <label for="precise_device">Device</label>
              <select id="precise_device" name="precise_device">{precise_device_select}</select>
            </div>
            <div>
              <label for="precise_compute_type">Compute precision</label>
              <select id="precise_compute_type" name="precise_compute_type">{precise_compute_select}</select>
            </div>
            <div>
              <label for="vocal_extract">Pre-processing</label>
              <select id="vocal_extract" name="vocal_extract">{vocal_extract_select}</select>
            </div>
          </div>
          <label><input type="checkbox" name="precise_rerun" value="true" checked /> Enable precise rerun pass</label>
        </div>
        <div class="field-group">
          <span class="section-title">Preview</span>
          <div class="preview-box">
            <label><input type="checkbox" name="preview_enabled" value="true" checked /> Render a high-intensity preview snippet</label>
            <div class="grid-two">
              <div>
                <label for="preview_start">Preview start (seconds or MM:SS)</label>
                <input id="preview_start" name="preview_start" type="text" placeholder="e.g. 1:30" />
              </div>
              <div>
                <label for="preview_duration">Preview duration (seconds)</label>
                <input id="preview_duration" name="preview_duration" type="text" placeholder="default: 10" />
              </div>
            </div>
            <div class="help-text">Snippets help you quickly verify diarization and text quality before the full run finishes.</div>
          </div>
        </div>
        <label><input type="checkbox" name="resume" value="true" /> Resume from cached checkpoints</label>
      </fieldset>
      <button class="button" type="submit">Launch aggressive transcription</button>
    </form>

    <section class="panel">
      <header style="display:flex;justify-content:space-between;align-items:center;gap:1rem;">
        <h2 class="section-title">Active jobs</h2>
      </header>
      <table>
        <thead><tr><th>Job</th><th>Status</th><th>Created</th><th>Updated</th><th>Error</th><th>Actions</th></tr></thead>
        <tbody>
          {''.join(rows) if rows else "<tr><td colspan='6'>No jobs yet.</td></tr>"}
        </tbody>
      </table>
    </section>
  </main>
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
    raw_job_id = job.get("job_id", "")
    job_id_text = str(raw_job_id)
    job_id = html.escape(job_id_text)
    status = html.escape(job.get("status", "unknown"))
    created = html.escape(job.get("created_at", ""))
    updated = html.escape(job.get("updated_at", ""))
    error = html.escape(job.get("error", "") or "")
    audio_name = html.escape(job.get("audio_filename", ""))
    delete_action = html.escape(f"/runs/{quote(job_id_text, safe='')}/delete")

    file_rows = []
    for label, url in files:
        file_rows.append(
            f"<li><a href=\"{html.escape(url)}\">{html.escape(label)}</a></li>"
        )

    file_list = "<ul>" + "".join(file_rows) + "</ul>" if file_rows else "<p>No output files yet.</p>"
    log_href = f"/runs/{quote(job_id_text, safe='')}/log"
    log_link = (
        f"<p><a href=\"{html.escape(log_href)}\">Download job log</a></p>"
        if log_available
        else ""
    )

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
            "<section class='panel preview-panel'>"
            "  <h2>Preview snippet</h2>"
            f"  <audio controls src=\"{escaped_url}\" preload=\"none\"></audio>"
            f"  {details_html}"
            "</section>"
        )
    elif preview_requested:
        preview_block = (
            "<section class='panel preview-panel'>"
            "  <h2>Preview snippet</h2>"
            "  <p class='muted'>Preview rendering in progress. The audio will appear once the job completes.</p>"
            f"  {details_html}"
            "</section>"
        )

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Job {job_id}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: radial-gradient(circle at 20% 20%, rgba(61, 255, 108, 0.12), transparent 50%),
            radial-gradient(circle at 80% 10%, rgba(82, 220, 255, 0.1), transparent 55%),
            #050b07;
      --panel: rgba(6, 20, 12, 0.82);
      --panel-border: rgba(61, 255, 108, 0.35);
      --panel-hover: rgba(10, 34, 20, 0.92);
      --accent: #3dff6c;
      --accent-soft: rgba(61, 255, 108, 0.18);
      --accent-strong: rgba(61, 255, 108, 0.35);
      --muted: #79ffb1;
      --text: #e0ffe9;
      --danger: #ff4d6d;
      --danger-soft: rgba(255, 77, 109, 0.18);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: 'Fira Code', 'Source Code Pro', 'Courier New', monospace;
      margin: 0;
      padding: 3rem 1.5rem;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      display: flex;
      justify-content: center;
    }}
    main {{
      width: min(960px, 100%);
      display: grid;
      gap: 1.75rem;
    }}
    header {{
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      gap: 0.75rem 1.5rem;
      justify-content: space-between;
    }}
    header h1 {{
      font-size: clamp(1.8rem, 2.4vw, 2.3rem);
      text-shadow: 0 0 22px rgba(61, 255, 108, 0.45);
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin: 0;
    }}
    header .meta {{
      font-size: 0.9rem;
      color: rgba(224, 255, 233, 0.78);
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 16px;
      padding: 1.8rem 2rem;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);
      backdrop-filter: blur(12px);
      transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .panel:hover {{
      border-color: var(--accent-strong);
      transform: translateY(-2px);
      box-shadow: 0 24px 48px rgba(0, 0, 0, 0.4);
      background: var(--panel-hover);
    }}
    h2 {{
      margin-top: 0;
      margin-bottom: 1.1rem;
      font-size: 1rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .status-grid {{
      display: grid;
      gap: 0.75rem 1.25rem;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .status-item {{
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }}
    .status-label {{
      font-size: 0.75rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: rgba(224, 255, 233, 0.68);
    }}
    .status-value {{
      font-size: 1rem;
      letter-spacing: 0.04em;
    }}
    .status-chip {{
      display: inline-flex;
      padding: 0.4rem 0.85rem;
      border-radius: 999px;
      border: 1px solid var(--accent-strong);
      background: rgba(61, 255, 108, 0.12);
      font-size: 0.85rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }}
    .muted {{
      color: rgba(224, 255, 233, 0.72);
    }}
    .error {{
      padding: 0.95rem 1.1rem;
      border-radius: 12px;
      border: 1px solid var(--danger);
      background: var(--danger-soft);
      color: #ffe9ee;
      margin-top: 1.1rem;
      box-shadow: inset 0 0 0 1px rgba(255, 77, 109, 0.35);
    }}
    ul {{
      padding-left: 1.2rem;
      margin: 0;
      display: grid;
      gap: 0.4rem;
    }}
    li {{
      list-style: none;
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }}
    li::before {{
      content: '➜';
      color: var(--accent);
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
      transition: text-shadow 0.2s ease;
    }}
    a:hover {{
      text-shadow: 0 0 12px rgba(61, 255, 108, 0.6);
    }}
    .preview-panel audio {{
      margin-top: 1rem;
      border-radius: 12px;
    }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 0.9rem 1.8rem;
      border-radius: 999px;
      border: 1px solid var(--accent);
      background: linear-gradient(135deg, rgba(61, 255, 108, 0.9), rgba(61, 255, 108, 0.55));
      color: #041b0b;
      cursor: pointer;
      transition: transform 0.18s ease, box-shadow 0.18s ease;
      box-shadow: 0 18px 36px rgba(61, 255, 108, 0.25);
    }}
    .button:hover {{
      transform: translateY(-2px);
      box-shadow: 0 22px 44px rgba(61, 255, 108, 0.32);
    }}
    .button:active {{
      transform: translateY(0);
      box-shadow: 0 16px 32px rgba(61, 255, 108, 0.24);
    }}
    .button-danger {{
      border-color: var(--danger);
      background: linear-gradient(135deg, rgba(255, 77, 109, 0.92), rgba(255, 77, 109, 0.6));
      color: #fff;
      box-shadow: 0 18px 36px var(--danger-soft);
    }}
    .button-secondary {{
      border-color: var(--accent);
      background: rgba(6, 28, 16, 0.7);
      color: var(--text);
      box-shadow: 0 18px 36px rgba(61, 255, 108, 0.12);
    }}
    .button-secondary:hover {{
      box-shadow: 0 22px 44px rgba(61, 255, 108, 0.18);
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      justify-content: space-between;
      align-items: center;
    }}
    .toolbar form {{ margin: 0; }}
    .toolbar a.button {{ text-decoration: none; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Job {job_id}</h1>
      <span class="meta">Started {created}</span>
    </header>
    <section class="panel">
      <div class="status-grid">
        <div class="status-item">
          <span class="status-label">Current status</span>
          <span class="status-chip">{status}</span>
        </div>
        <div class="status-item">
          <span class="status-label">Updated</span>
          <span class="status-value">{updated}</span>
        </div>
        <div class="status-item">
          <span class="status-label">Source audio</span>
          <span class="status-value">{audio_name}</span>
        </div>
      </div>
      {error_block}
    </section>
    {preview_block}
    <section class="panel">
      <h2>Outputs</h2>
      {file_list}
      {log_link}
    </section>
    <div class="toolbar">
      <a class="button button-secondary" href="/">Back to dashboard</a>
      <form action="{delete_action}" method="post" onsubmit="return confirm('Delete this job and all associated files?');">
        <button class="button button-danger" type="submit">Delete job</button>
      </form>
    </div>
  </main>
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
        raw_job_id = str(status.get("job_id") or job_dir.name)
        quoted_job_id = quote(raw_job_id, safe="")

        output_dir = status.get("output_dir")
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = job_dir / "outputs"
        if out_path.exists():
            for child in sorted(out_path.glob("*")):
                if child.is_file():
                    rel = child.relative_to(job_dir)
                    rel_url = quote(rel.as_posix(), safe="/")
                    url = f"/runs/{quoted_job_id}/files/{rel_url}"
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
    async def home(request: Request) -> str:
        jobs = _list_jobs()
        message = request.query_params.get("message")
        return _build_home_html(jobs, message)

    @app.post("/transcribe")
    async def transcribe(
        audio_file: UploadFile = File(...),
        log_level: str = Form(cli.LOG.level),
        asr_model: str = Form(""),
        num_speakers: str = Form(""),
        asr_device: str = Form(""),
        asr_compute_type: str = Form(""),
        precise_model: str = Form(""),
        precise_device: str = Form(""),
        precise_compute_type: str = Form(""),
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

        resolved_asr_device = _normalize_device_choice(asr_device)
        resolved_precise_device = _normalize_device_choice(precise_device)

        args = build_cli_args(
            audio_path,
            outdir=outputs_dir,
            resume=_checkbox_to_bool(resume),
            num_speakers=parsed_num_speakers,
            asr_model=asr_model or None,
            asr_device=resolved_asr_device,
            asr_compute_type=asr_compute_type or None,
            precise_model=precise_model or None,
            precise_device=resolved_precise_device,
            precise_compute_type=precise_compute_type or None,
            precise_rerun=_checkbox_to_bool(precise_rerun),
            vocal_extract=vocal_choice,
            log_level=log_level,
            preview_start=preview_start_arg,
            preview_duration=preview_duration_arg,
        )

        status_snapshot = _job_status_template(job_id, created_at)
        status_snapshot["audio_filename"] = audio_file.filename
        _write_json(job_dir / "status.json", status_snapshot)

        loop = asyncio.get_running_loop()

        def _consume_future(fut: asyncio.Future[Any]) -> None:
            try:
                fut.result()
            except SystemExit:
                LOGGER.info("Job %s exited early", job_id)
            except Exception:  # pylint: disable=broad-except
                LOGGER.exception("Job %s raised an exception during execution", job_id)

        future = loop.run_in_executor(None, _run_job, args, job_id, job_dir, created_at)
        future.add_done_callback(_consume_future)

        message = quote_plus(f"Started job {job_id}")
        return RedirectResponse(url=f"/?message={message}", status_code=303)

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

    @app.post("/runs/{job_id}/delete")
    async def delete_job(job_id: str) -> RedirectResponse:
        job_dir = _job_dir(job_id)
        if not job_dir.exists() or not job_dir.is_dir():
            raise HTTPException(status_code=404, detail="Job not found")

        try:
            shutil.rmtree(job_dir)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Failed to delete job %s", job_id)
            raise HTTPException(status_code=500, detail="Failed to delete job") from exc

        message = quote_plus(f"Deleted job {job_id}")
        return RedirectResponse(url=f"/?message={message}", status_code=303)

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
