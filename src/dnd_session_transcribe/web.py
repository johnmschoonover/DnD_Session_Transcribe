"""FastAPI-powered Web UI for remote DnD session transcription."""

from __future__ import annotations

import argparse
import asyncio
import functools
import html
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import quote, quote_plus

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
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
    serialized = json.dumps(data, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as tmp_file:
        tmp_file.write(serialized)
        temp_name = tmp_file.name
    os.replace(temp_name, path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse JSON from %s: %s", path, exc)
        return {}


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


def _job_config_block(index: str, *, removable: bool) -> str:
    """Return the HTML markup for a single job configuration block."""

    prefix = f"job-{index}-"
    log_level_options = "".join(
        f"<option value='{level}'{' selected' if level == cli.LOG.level else ''}>{level}</option>"
        for level in cli.LOG_LEVELS
    )

    remove_button = (
        "<button type=\"button\" class=\"remove-job secondary-button\">Remove</button>"
        if removable
        else ""
    )

    return (
        "<div class=\"job-config\" data-job-index=\""
        + html.escape(index)
        + "\">"
        + "<div class=\"job-config-header\">"
        + "<h3>Configuration <span class=\"job-number\"></span></h3>"
        + remove_button
        + "</div>"
        + f"<label for=\"{prefix}log_level\">Log level</label>"
        + f"<select id=\"{prefix}log_level\" name=\"{prefix}log_level\">{log_level_options}</select>"
        + f"<label for=\"{prefix}asr_model\">Faster-Whisper model override (optional)</label>"
        + f"<input id=\"{prefix}asr_model\" name=\"{prefix}asr_model\" type=\"text\" placeholder=\"e.g. tiny, base, large-v3\" />"
        + f"<label for=\"{prefix}num_speakers\">Number of speakers (optional)</label>"
        + f"<input id=\"{prefix}num_speakers\" name=\"{prefix}num_speakers\" type=\"text\" inputmode=\"numeric\" placeholder=\"e.g. 4\" />"
        + f"<label for=\"{prefix}asr_device\">ASR device override</label>"
        + f"<select id=\"{prefix}asr_device\" name=\"{prefix}asr_device\">"
        + "<option value=\"\">(default)</option>"
        + "<option value=\"cpu\">cpu</option>"
        + "<option value=\"cuda\">cuda</option>"
        + "<option value=\"mps\">mps</option>"
        + "</select>"
        + f"<label for=\"{prefix}asr_compute_type\">ASR compute type override</label>"
        + f"<input id=\"{prefix}asr_compute_type\" name=\"{prefix}asr_compute_type\" type=\"text\" placeholder=\"e.g. float16\" />"
        + f"<label for=\"{prefix}vocal_extract\">Preprocessing</label>"
        + f"<select id=\"{prefix}vocal_extract\" name=\"{prefix}vocal_extract\">"
        + "<option value=\"\">(default)</option>"
        + "<option value=\"off\">off</option>"
        + "<option value=\"bandpass\">bandpass</option>"
        + "</select>"
        + "<div class=\"preview-box\">"
        + f"<label><input type=\"checkbox\" id=\"{prefix}preview_enabled\" name=\"{prefix}preview_enabled\" value=\"true\" /> Render preview snippet</label>"
        + f"<label for=\"{prefix}preview_start\">Preview start (seconds or MM:SS)</label>"
        + f"<input id=\"{prefix}preview_start\" name=\"{prefix}preview_start\" type=\"text\" placeholder=\"e.g. 1:30\" />"
        + f"<label for=\"{prefix}preview_duration\">Preview duration (seconds)</label>"
        + f"<input id=\"{prefix}preview_duration\" name=\"{prefix}preview_duration\" type=\"text\" placeholder=\"default: 10\" />"
        + "<div class=\"help-text\">When enabled, the snippet audio and transcripts will be generated alongside the full outputs.</div>"
        + "</div>"
        + "<div class=\"job-checkboxes\">"
        + f"<label><input type=\"checkbox\" name=\"{prefix}resume\" value=\"true\" /> Resume from cached checkpoints</label>"
        + f"<label><input type=\"checkbox\" name=\"{prefix}precise_rerun\" value=\"true\" /> Enable precise re-run</label>"
        + "</div>"
        + "</div>"
    )

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
            f"<td><span class=\"status-badge\">{status}</span></td>"
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

    initial_job_block = _job_config_block("0", removable=False)
    template_job_block = _job_config_block("__INDEX__", removable=True)

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>DnD Session Transcribe Web UI</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0c1117;
      --panel: #161b22;
      --panel-border: #30363d;
      --panel-hover: #1f242d;
      --accent: #58a6ff;
      --accent-soft: rgba(88, 166, 255, 0.2);
      --danger: #f85149;
      --danger-soft: rgba(248, 81, 73, 0.18);
      --text: #f0f6fc;
      --muted: #8b949e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: 'Inter', 'Segoe UI', sans-serif;
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--text);
    }}
    main.layout {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 2.5rem 1.5rem 3rem;
      display: grid;
      gap: 1.75rem;
    }}
    .header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: baseline;
      gap: 0.75rem 1.5rem;
    }}
    .header h1 {{
      margin: 0;
      font-size: clamp(1.8rem, 4vw, 2.4rem);
    }}
    .tagline {{
      color: var(--muted);
      font-size: 0.95rem;
      letter-spacing: 0.05em;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 12px;
      padding: 1.75rem;
      box-shadow: 0 12px 24px rgba(0, 0, 0, 0.25);
      transition: border-color 0.2s ease, background 0.2s ease;
    }}
    .panel:hover {{
      border-color: var(--accent);
      background: var(--panel-hover);
    }}
    form fieldset {{
      border: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 1.25rem;
    }}
    label {{
      font-weight: 600;
      font-size: 0.95rem;
      display: block;
      margin-bottom: 0.4rem;
    }}
    input[type='file'],
    input[type='text'],
    select {{
      width: 100%;
      border-radius: 8px;
      border: 1px solid var(--panel-border);
      background: #0d1117;
      color: var(--text);
      padding: 0.65rem 0.75rem;
      font-size: 0.95rem;
    }}
    input:focus,
    select:focus {{
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-soft);
    }}
    .job-configs {{
      margin-top: 1rem;
      display: grid;
      gap: 1rem;
    }}
    .job-config {{
      border: 1px solid var(--panel-border);
      border-radius: 10px;
      padding: 1rem 1.25rem;
      background: #0d1117;
      display: grid;
      gap: 0.75rem;
    }}
    .job-config-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .job-config h3 {{
      margin: 0;
      font-size: 1rem;
    }}
    .preview-box {{
      border: 1px dashed var(--panel-border);
      border-radius: 8px;
      padding: 0.75rem 1rem;
      background: #0f141c;
      display: grid;
      gap: 0.6rem;
    }}
    .job-checkboxes label {{
      font-weight: 500;
    }}
    .help-text {{
      color: var(--muted);
      font-size: 0.9rem;
      margin: 0;
    }}
    button {{
      cursor: pointer;
      border-radius: 999px;
      border: none;
      padding: 0.75rem 1.5rem;
      font-weight: 600;
      letter-spacing: 0.03em;
    }}
    form > button {{
      background: var(--accent);
      color: #0d1117;
    }}
    .secondary-button {{
      background: transparent;
      color: var(--accent);
      border: 1px solid var(--accent);
    }}
    .button-danger {{
      background: var(--danger);
      color: #fff;
      border: none;
    }}
    .jobs-table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 12px;
      overflow: hidden;
    }}
    .jobs-table th,
    .jobs-table td {{
      padding: 0.9rem 1rem;
      border-bottom: 1px solid var(--panel-border);
      text-align: left;
      font-size: 0.95rem;
    }}
    .jobs-table tr:last-child td {{ border-bottom: none; }}
    .status-badge {{
      display: inline-flex;
      align-items: center;
      padding: 0.25rem 0.7rem;
      border-radius: 999px;
      border: 1px solid var(--accent);
      font-size: 0.8rem;
    }}
    .flash {{
      background: var(--panel);
      border: 1px solid var(--accent);
      padding: 0.8rem 1rem;
      border-radius: 10px;
    }}
    .inline-form {{ display: inline; }}
  </style>
</head>
<body>
  <main class="layout">
    <header class="header">
      <h1>DnD Session Transcribe</h1>
      <p class="tagline">Batch job scheduling for iterative experiments</p>
    </header>
    {message_html}
    <form class="panel" action="/transcribe" method="post" enctype="multipart/form-data">
      <fieldset>
        <label for="audio_file">Audio file</label>
        <input id="audio_file" name="audio_file" type="file" required />
        <div class="job-configs">
          <p class="help-text">Configure one or more jobs to compare different settings for the same upload.</p>
          <div id="jobs-container">
            {initial_job_block}
          </div>
          <div class="job-controls">
            <button type="button" id="add-job" class="secondary-button">Add another configuration</button>
          </div>
        </div>
      </fieldset>
      <button type="submit">Start transcription</button>
    </form>
    <table class="jobs-table">
      <thead>
        <tr><th>Job</th><th>Status</th><th>Created</th><th>Updated</th><th>Error</th><th>Actions</th></tr>
      </thead>
      <tbody>
        {''.join(rows) if rows else "<tr><td colspan='6'>No jobs yet.</td></tr>"}
      </tbody>
    </table>
    <template id="job-template">
      {template_job_block}
    </template>
  </main>
  <script>
    (function() {{
      const container = document.getElementById('jobs-container');
      const template = document.getElementById('job-template');
      const addButton = document.getElementById('add-job');
      if (!container || !template || !addButton) {{
        return;
      }}

      let nextIndex = container.querySelectorAll('.job-config').length;

      function renumber() {{
        const blocks = container.querySelectorAll('.job-config');
        blocks.forEach((block, idx) => {{
          const numberEl = block.querySelector('.job-number');
          if (numberEl) {{
            numberEl.textContent = String(idx + 1);
          }}
        }});
      }}

      addButton.addEventListener('click', () => {{
        const markup = template.innerHTML.replace(/__INDEX__/g, String(nextIndex));
        const wrapper = document.createElement('div');
        wrapper.innerHTML = markup.trim();
        const newBlock = wrapper.firstElementChild;
        if (!newBlock) {{
          return;
        }}
        container.appendChild(newBlock);
        nextIndex += 1;
        renumber();
      }});

      container.addEventListener('click', (event) => {{
        const target = event.target;
        if (!(target instanceof Element)) {{
          return;
        }}
        if (!target.classList.contains('remove-job')) {{
          return;
        }}
        event.preventDefault();
        const block = target.closest('.job-config');
        if (!block) {{
          return;
        }}
        if (container.querySelectorAll('.job-config').length <= 1) {{
          return;
        }}
        block.remove();
        renumber();
      }});

      renumber();
    }})();
  </script>
</body>
</html>
"""

def _build_job_html(
    job: dict[str, Any],
    files: list[tuple[str, str]],
    log_available: bool,
    preview: dict[str, Any] | None = None,
    preview_url: str | None = None,
    settings: dict[str, Any] | None = None,
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

    file_rows: list[str] = []
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
    settings_block = ""
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

    if settings:
        rows: list[str] = []
        for key, value in settings.items():
            if isinstance(value, (dict, list)):
                formatted = json.dumps(value, indent=2, sort_keys=True)
                value_html = f"<pre>{html.escape(formatted)}</pre>"
            else:
                value_html = html.escape(str(value))
            rows.append(
                "<tr>"
                f"<th scope=\"row\">{html.escape(str(key))}</th>"
                f"<td>{value_html}</td>"
                "</tr>"
            )
        settings_table = "<table class='settings-table'><tbody>" + "".join(rows) + "</tbody></table>"
        settings_block = (
            "<section class='panel'>"
            "  <h2>Settings</h2>"
            f"  {settings_table}"
            "</section>"
        )

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
    .settings-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 0.75rem;
      background: rgba(6, 22, 12, 0.6);
    }}
    .settings-table th,
    .settings-table td {{
      padding: 0.5rem;
      border-bottom: 1px solid #d0d7de;
      vertical-align: top;
      text-align: left;
    }}
    .settings-table th {{
      width: 35%;
      background: rgba(224, 255, 233, 0.12);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.75rem;
    }}
    .settings-table pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: 'Fira Code', 'Source Code Pro', monospace;
      font-size: 0.85rem;
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
    {settings_block}
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
                resolved_outdir_str = str(resolved_outdir)
                status["output_dir"] = resolved_outdir_str

                metadata_path = job_dir / "metadata.json"
                metadata = _read_json(metadata_path)
                if metadata.get("output_dir") != resolved_outdir_str:
                    metadata["output_dir"] = resolved_outdir_str
                    _write_json(metadata_path, metadata)
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
    async def transcribe(request: Request, audio_file: UploadFile = File(...)) -> RedirectResponse:
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="Audio file is required")

        form_data = await request.form()
        form_values: dict[str, Any] = {}
        for key, value in form_data.multi_items():
            if key == "audio_file":
                continue
            form_values[key] = value

        job_indices: list[str | None] = []
        seen_indices: set[str] = set()
        for key in form_values:
            if not key.startswith("job-"):
                continue
            parts = key.split("-", 2)
            if len(parts) < 3:
                continue
            idx = parts[1]
            if idx and idx.isdigit() and idx not in seen_indices:
                seen_indices.add(idx)
                job_indices.append(idx)

        job_indices.sort(key=lambda item: int(item))
        if not job_indices:
            job_indices = [None]

        temp_audio_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("wb", delete=False) as tmp_file:
                while chunk := await audio_file.read(1024 * 1024):
                    tmp_file.write(chunk)
                temp_audio_path = Path(tmp_file.name)
        finally:
            await audio_file.close()

        if temp_audio_path is None:
            raise HTTPException(status_code=400, detail="Failed to read audio file")

        clean_name = safe_filename(audio_file.filename)

        def _field_key(job_idx: str | None, name: str) -> str:
            return f"job-{job_idx}-{name}" if job_idx is not None else name

        def _get_value(job_idx: str | None, name: str, default: str = "") -> str:
            raw = form_values.get(_field_key(job_idx, name), default)
            if isinstance(raw, str):
                return raw
            if raw is None:
                return default
            return str(raw)

        def _get_checkbox(job_idx: str | None, name: str) -> bool:
            key = _field_key(job_idx, name)
            if key not in form_values:
                return False
            return _checkbox_to_bool(str(form_values[key]))

        job_plans: list[dict[str, Any]] = []

        try:
            for order, index in enumerate(job_indices):
                job_id = _generate_job_id()
                job_dir = _job_dir(job_id)
                inputs_dir = job_dir / "inputs"
                outputs_dir = job_dir / "outputs"
                audio_target = inputs_dir / clean_name

                def _detail(message: str) -> str:
                    return f"{message} for configuration {order + 1}"

                log_level_value = _get_value(index, "log_level", cli.LOG.level).strip() or cli.LOG.level
                if log_level_value not in cli.LOG_LEVELS:
                    raise HTTPException(status_code=400, detail=_detail("Invalid log level"))

                asr_model_value = _get_value(index, "asr_model").strip()
                num_speakers_text = _get_value(index, "num_speakers").strip()
                asr_device_value = _get_value(index, "asr_device").strip()
                asr_compute_type_value = _get_value(index, "asr_compute_type").strip()
                precise_model_value = _get_value(index, "precise_model").strip()
                precise_device_value = _get_value(index, "precise_device").strip()
                precise_compute_type_value = _get_value(index, "precise_compute_type").strip()
                vocal_extract_value = _get_value(index, "vocal_extract").strip()

                if num_speakers_text:
                    try:
                        parsed_num_speakers = int(num_speakers_text)
                    except ValueError:
                        raise HTTPException(status_code=400, detail=_detail("num_speakers must be an integer")) from None
                else:
                    parsed_num_speakers = None

                vocal_choice = vocal_extract_value or None
                if vocal_choice not in {None, "off", "bandpass"}:
                    raise HTTPException(status_code=400, detail=_detail("Invalid vocal_extract value"))

                resume_enabled = _get_checkbox(index, "resume")
                precise_rerun_enabled = _get_checkbox(index, "precise_rerun")

                preview_start_text = _get_value(index, "preview_start").strip()
                preview_duration_text = _get_value(index, "preview_duration").strip()
                preview_enabled = _get_checkbox(index, "preview_enabled")
                preview_fields_supplied = bool(preview_start_text or preview_duration_text)
                preview_requested = preview_enabled or preview_fields_supplied
                preview_meta: dict[str, Any] = {"requested": preview_requested}

                if preview_requested:
                    try:
                        start_value = (
                            cli.parse_time_spec(preview_start_text)
                            if preview_start_text
                            else 0.0
                        )
                    except ValueError:
                        raise HTTPException(status_code=400, detail=_detail("Invalid preview_start value")) from None

                    try:
                        duration_value = (
                            cli.parse_time_spec(preview_duration_text)
                            if preview_duration_text
                            else 10.0
                        )
                    except ValueError:
                        raise HTTPException(status_code=400, detail=_detail("Invalid preview_duration value")) from None

                    if duration_value <= 0:
                        raise HTTPException(status_code=400, detail=_detail("Preview duration must be positive"))

                    preview_start_arg: Optional[float] = start_value
                    preview_duration_arg: Optional[float] = duration_value
                    preview_meta.update({"start": start_value, "duration": duration_value})
                else:
                    preview_start_arg = None
                    preview_duration_arg = None

                resolved_asr_device = _normalize_device_choice(asr_device_value)
                resolved_precise_device = _normalize_device_choice(precise_device_value)

                args = build_cli_args(
                    audio_target,
                    outdir=outputs_dir,
                    resume=resume_enabled,
                    num_speakers=parsed_num_speakers,
                    asr_model=asr_model_value or None,
                    asr_device=resolved_asr_device,
                    asr_compute_type=asr_compute_type_value or None,
                    precise_model=precise_model_value or None,
                    precise_device=resolved_precise_device,
                    precise_compute_type=precise_compute_type_value or None,
                    precise_rerun=precise_rerun_enabled,
                    vocal_extract=vocal_choice,
                    log_level=log_level_value,
                    preview_start=preview_start_arg,
                    preview_duration=preview_duration_arg,
                )

                created_at = _utc_now()
                settings_snapshot = {
                    key: (str(val) if isinstance(val, Path) else val)
                    for key, val in vars(args).items()
                }
                settings_snapshot["preview_requested"] = preview_requested
                settings_snapshot["batch_index"] = order
                settings_snapshot["job_index"] = index

                metadata = {
                    "job_id": job_id,
                    "audio_filename": audio_file.filename,
                    "created_at": created_at,
                    "preview": preview_meta,
                    "settings": settings_snapshot,
                    "batch_index": order,
                    "job_index": index,
                }

                status_snapshot = _job_status_template(job_id, created_at)
                status_snapshot["audio_filename"] = audio_file.filename

                job_plans.append(
                    {
                        "job_id": job_id,
                        "job_dir": job_dir,
                        "inputs_dir": inputs_dir,
                        "outputs_dir": outputs_dir,
                        "audio_path": audio_target,
                        "args": args,
                        "metadata": metadata,
                        "status": status_snapshot,
                        "created_at": created_at,
                    }
                )

            if not job_plans:
                raise HTTPException(status_code=400, detail="No job configurations provided")

            loop = asyncio.get_running_loop()

            def _consume_future(job_label: str, fut: asyncio.Future[Any]) -> None:
                try:
                    fut.result()
                except SystemExit:
                    LOGGER.info("Job %s exited early", job_label)
                except Exception:  # pylint: disable=broad-except
                    LOGGER.exception("Job %s raised an exception during execution", job_label)

            job_ids: list[str] = []
            for plan in job_plans:
                job_dir = plan["job_dir"]
                inputs_dir = plan["inputs_dir"]
                outputs_dir = plan["outputs_dir"]
                audio_path = plan["audio_path"]

                job_dir.mkdir(parents=True, exist_ok=True)
                inputs_dir.mkdir(exist_ok=True)
                outputs_dir.mkdir(exist_ok=True)
                shutil.copyfile(temp_audio_path, audio_path)

                _write_json(job_dir / "metadata.json", plan["metadata"])
                _write_json(job_dir / "status.json", plan["status"])

                future = loop.run_in_executor(
                    None,
                    _run_job,
                    plan["args"],
                    plan["job_id"],
                    job_dir,
                    plan["created_at"],
                )
                future.add_done_callback(functools.partial(_consume_future, plan["job_id"]))
                job_ids.append(plan["job_id"])

            if not job_ids:
                raise HTTPException(status_code=400, detail="No jobs were scheduled")

            if len(job_ids) == 1:
                message = quote_plus(f"Started job {job_ids[0]}")
            else:
                message = quote_plus("Started jobs " + ", ".join(job_ids))

            return RedirectResponse(url=f"/?message={message}", status_code=303)
        finally:
            if temp_audio_path is not None:
                temp_audio_path.unlink(missing_ok=True)

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
            settings=meta.get("settings"),
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
