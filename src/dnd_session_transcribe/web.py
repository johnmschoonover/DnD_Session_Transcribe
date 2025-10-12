"""FastAPI-powered Web UI for remote DnD session transcription."""

from __future__ import annotations

import argparse
import asyncio
import functools
import html
import itertools
import json
import logging
import os
import random
import re
import secrets
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
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
    ram: bool = False,
    resume: bool = False,
    num_speakers: Optional[int] = None,
    hotwords_file: Optional[Path | str] = None,
    initial_prompt_file: Optional[Path | str] = None,
    spelling_map: Optional[Path | str] = None,
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
        ram=ram,
        resume=resume,
        num_speakers=num_speakers,
        hotwords_file=str(hotwords_file) if hotwords_file is not None else None,
        initial_prompt_file=
        str(initial_prompt_file) if initial_prompt_file is not None else None,
        spelling_map=str(spelling_map) if spelling_map is not None else None,
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


_ASR_MODEL_SUGGESTIONS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v2",
    "large-v3",
    "turbo",
    "distil-large-v2",
    "distil-large-v3",
    "distil-medium.en",
    "distil-small.en",
]

_COMPUTE_TYPE_SUGGESTIONS = [
    "auto",
    "float16",
    "float32",
    "int8",
    "int8_float16",
    "int8_float32",
    "int8_float16_float32",
]


_CHOICE_RANDOM = "random"
_CHOICE_ALL = "all"

_DEVICE_OPTIONS = [
    ("", "Config default"),
    ("cpu", "cpu"),
    ("cuda", "cuda"),
    ("mps", "mps"),
]
_DEVICE_VALUES = [value for value, _ in _DEVICE_OPTIONS]

_VOCAL_OPTIONS = [
    ("", "Config default"),
    ("off", "Off"),
    ("bandpass", "Band-pass filter"),
]
_VOCAL_VALUES = [value for value, _ in _VOCAL_OPTIONS]


def _resolve_selection(value: str, allowed: Sequence[str]) -> tuple[str, list[str]]:
    """Resolve a dropdown selection to a selection mode and available choices."""

    if value == _CHOICE_ALL:
        return "all", list(allowed)
    if value == _CHOICE_RANDOM:
        return "random", list(allowed)
    if value in allowed:
        return "single", [value]
    raise ValueError(f"Invalid selection: {value}")


def _checkbox_to_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() not in {"0", "false", "off"}


def _job_config_block(index: str, *, removable: bool) -> str:
    """Return the HTML for an individual job configuration block."""

    prefix = f"job-{index}-"
    log_level_options = "".join(
        (
            "<option value='{level}'{selected}>{label}</option>".format(
                level=html.escape(level),
                label=html.escape(level.title()),
                selected=" selected" if level == cli.LOG.level else "",
            )
        )
        for level in cli.LOG_LEVELS
    )
    log_level_options += (
        f"<option value=\"{_CHOICE_RANDOM}\">Random</option>"
        f"<option value=\"{_CHOICE_ALL}\">All</option>"
    )

    device_options = "".join(
        (
            "<option value='{value}'{selected}>{label}</option>".format(
                value=html.escape(value),
                label=html.escape(label),
                selected=" selected" if value == "" else "",
            )
        )
        for value, label in _DEVICE_OPTIONS
    )
    device_options += (
        f"<option value=\"{_CHOICE_RANDOM}\">Random</option>"
        f"<option value=\"{_CHOICE_ALL}\">All</option>"
    )

    vocal_options = "".join(
        (
            "<option value='{value}'{selected}>{label}</option>".format(
                value=html.escape(value),
                label=html.escape(label),
                selected=" selected" if value == "" else "",
            )
        )
        for value, label in _VOCAL_OPTIONS
    )
    vocal_options += (
        f"<option value=\"{_CHOICE_RANDOM}\">Random</option>"
        f"<option value=\"{_CHOICE_ALL}\">All</option>"
    )

    remove_button = (
        "<button type=\"button\" class=\"remove-job neon-button secondary\">Remove</button>"
        if removable
        else ""
    )

    return (
        "<div class=\"job-config\" data-job-index=\""
        + html.escape(index)
        + "\">"
        + "<div class=\"job-config-header\">"
        + "<h3><span class=\"job-number\"></span> Loadout</h3>"
        + remove_button
        + "</div>"
        + "<div class=\"config-grid\">"
        + "<section class=\"config-card\">"
        + "<h4>Control Center</h4>"
        + f"<label for=\"{prefix}log_level\">Log level</label>"
        + f"<select id=\"{prefix}log_level\" name=\"{prefix}log_level\">{log_level_options}</select>"
        + f"<label for=\"{prefix}num_speakers\">Number of speakers</label>"
        + f"<input id=\"{prefix}num_speakers\" name=\"{prefix}num_speakers\" type=\"number\" min=\"1\" placeholder=\"auto\" />"
        + "<div class=\"checkbox-grid\">"
        + f"<label><input type=\"checkbox\" name=\"{prefix}ram\" value=\"true\" checked /> Stage audio in RAM</label>"
        + f"<label><input type=\"checkbox\" name=\"{prefix}resume\" value=\"true\" checked /> Resume from checkpoints</label>"
        + f"<label><input type=\"checkbox\" name=\"{prefix}precise_rerun\" value=\"true\" checked /> Precise re-run</label>"
        + "</div>"
        + "</section>"
        + "<section class=\"config-card\">"
        + "<h4>ASR Engine</h4>"
        + f"<label for=\"{prefix}asr_model\">Faster-Whisper model</label>"
        + f"<input id=\"{prefix}asr_model\" name=\"{prefix}asr_model\" type=\"text\" list=\"asr-models\" placeholder=\"config default\" />"
        + f"<label for=\"{prefix}asr_device\">Device override</label>"
        + f"<select id=\"{prefix}asr_device\" name=\"{prefix}asr_device\">{device_options}</select>"
        + f"<label for=\"{prefix}asr_compute_type\">Compute type</label>"
        + f"<input id=\"{prefix}asr_compute_type\" name=\"{prefix}asr_compute_type\" type=\"text\" list=\"compute-types\" placeholder=\"auto\" />"
        + "</section>"
        + "<section class=\"config-card\">"
        + "<h4>Prompt Engineering</h4>"
        + f"<label for=\"{prefix}hotwords\">Hotwords</label>"
        + f"<textarea id=\"{prefix}hotwords\" name=\"{prefix}hotwords\" rows=\"3\" placeholder=\"One keyword per line\"></textarea>"
        + f"<label for=\"{prefix}initial_prompt\">Initial prompt</label>"
        + f"<textarea id=\"{prefix}initial_prompt\" name=\"{prefix}initial_prompt\" rows=\"3\" placeholder=\"Drop in scene context\"></textarea>"
        + f"<label for=\"{prefix}spelling_map\">Spelling corrections (CSV)</label>"
        + f"<textarea id=\"{prefix}spelling_map\" name=\"{prefix}spelling_map\" rows=\"3\" placeholder=\"wrong,right\nStrahd,Strahd von Zarovich\"></textarea>"
        + "</section>"
        + "<section class=\"config-card\">"
        + "<h4>Enhancements</h4>"
        + f"<label for=\"{prefix}vocal_extract\">Preprocessing</label>"
        + f"<select id=\"{prefix}vocal_extract\" name=\"{prefix}vocal_extract\">{vocal_options}</select>"
        + f"<label for=\"{prefix}precise_model\">Precise model</label>"
        + f"<input id=\"{prefix}precise_model\" name=\"{prefix}precise_model\" type=\"text\" list=\"asr-models\" placeholder=\"auto\" />"
        + f"<label for=\"{prefix}precise_device\">Precise device</label>"
        + f"<select id=\"{prefix}precise_device\" name=\"{prefix}precise_device\">{device_options}</select>"
        + f"<label for=\"{prefix}precise_compute_type\">Precise compute</label>"
        + f"<input id=\"{prefix}precise_compute_type\" name=\"{prefix}precise_compute_type\" type=\"text\" list=\"compute-types\" placeholder=\"auto\" />"
        + "</section>"
        + "<section class=\"config-card wide\">"
        + "<h4>Preview Builder</h4>"
        + f"<label class=\"checkbox-inline\"><input type=\"checkbox\" id=\"{prefix}preview_enabled\" name=\"{prefix}preview_enabled\" value=\"true\" checked /> Generate 10s teaser</label>"
        + "<div class=\"preview-grid\">"
        + "<div class=\"preview-field\">"
        + f"<label for=\"{prefix}preview_start\">Start</label>"
        + f"<input id=\"{prefix}preview_start\" name=\"{prefix}preview_start\" type=\"text\" placeholder=\"0 or MM:SS\" />"
        + "</div>"
        + "<div class=\"preview-field\">"
        + f"<label for=\"{prefix}preview_duration\">Duration</label>"
        + f"<input id=\"{prefix}preview_duration\" name=\"{prefix}preview_duration\" type=\"text\" placeholder=\"10\" />"
        + "</div>"
        + "<div class=\"preview-field full\">"
        + f"<label for=\"{prefix}preview_output\">Custom WAV name</label>"
        + f"<input id=\"{prefix}preview_output\" name=\"{prefix}preview_output\" type=\"text\" placeholder=\"leave blank for default\" />"
        + "</div>"
        + "</div>"
        + "</section>"
        + "</div>"
        + "</div>"
    )

def _build_home_html(jobs: Iterable[dict[str, Any]], message: str | None = None) -> str:
    rows = []
    for job in jobs:
        raw_job_id = job.get("job_id", "")
        job_id_text = str(raw_job_id)
        job_id = html.escape(job_id_text)
        job_href = html.escape(f"/runs/{quote(job_id_text, safe='')}")
        delete_action = html.escape(
            f"/runs/{quote(job_id_text, safe='')}/delete"
        )
        status = html.escape(job.get("status", "unknown"))
        created = html.escape(job.get("created_at", ""))
        updated = html.escape(job.get("updated_at", ""))
        error = html.escape(job.get("error", "") or "")
        rows.append(
            "<tr>"
            f"<td><a href=\"{job_href}\" target=\"_blank\" rel=\"noopener noreferrer\">{job_id}</a></td>"
            f"<td>{status}</td><td>{created}</td><td>{updated}</td><td>{error}</td>"
            f"<td><form action=\"{delete_action}\" method=\"post\" class=\"inline-form\" "
            "onsubmit=\"return confirm('Delete this job and all associated files?');\">"
            "<button type=\"submit\" class=\"delete-button\">Delete</button></form></td>"
            "</tr>"
        )

    message_html = f"<div class='message'>{html.escape(message)}</div>" if message else ""

    initial_job_block = _job_config_block("0", removable=False)
    template_job_block = _job_config_block("__INDEX__", removable=True)

    asr_model_datalist = "".join(
        f"<option value=\"{html.escape(model)}\"></option>"
        for model in _ASR_MODEL_SUGGESTIONS
    )
    compute_type_datalist = "".join(
        f"<option value=\"{html.escape(value)}\"></option>"
        for value in _COMPUTE_TYPE_SUGGESTIONS
    )

    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>DnD Session Transcribe Web UI</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #020305;
      --panel: rgba(8, 18, 26, 0.82);
      --border: rgba(57, 255, 20, 0.35);
      --text: #d7ffda;
      --muted: rgba(141, 214, 145, 0.8);
      --accent: #39ff14;
      --accent-soft: rgba(57, 255, 20, 0.15);
      --danger: #ff3c7f;
      font-family: 'Fira Code', 'JetBrains Mono', Menlo, Consolas, monospace;
    }}

    body {{
      margin: 0;
      padding: clamp(1.5rem, 3vw + 1rem, 3.5rem);
      background: radial-gradient(circle at top left, rgba(57,255,20,0.08), transparent 45%),
                  radial-gradient(circle at bottom right, rgba(0,180,255,0.08), transparent 40%),
                  var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 2rem;
      box-sizing: border-box;
    }}

    h1 {{
      font-size: 2.75rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 2rem;
      text-shadow: 0 0 18px rgba(57,255,20,0.7), 0 0 32px rgba(0, 255, 255, 0.35);
      text-align: center;
    }}

    h2 {{
      margin-top: 3rem;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 1.1rem;
      color: var(--muted);
      text-align: center;
    }}

    form {{
      background: var(--panel);
      border: 1px solid var(--border);
      box-shadow: 0 0 25px rgba(57,255,20,0.08);
      border-radius: 18px;
      padding: 2rem;
      backdrop-filter: blur(12px);
      width: min(100%, 1100px);
      margin: 0 auto;
    }}

    fieldset {{
      border: none;
      padding: 0;
      margin: 0;
    }}

    label {{
      display: block;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: var(--muted);
      margin-bottom: 0.35rem;
    }}

    input[type='text'],
    input[type='number'],
    input[type='file'],
    select,
    textarea {{
      width: 100%;
      background: rgba(2, 4, 6, 0.65);
      border: 1px solid rgba(57,255,20,0.35);
      border-radius: 10px;
      padding: 0.65rem 0.75rem;
      color: var(--text);
      font-family: inherit;
      box-shadow: inset 0 0 8px rgba(57,255,20,0.12);
    }}

    input[type='file'] {{
      padding: 0.45rem 0.75rem;
      cursor: pointer;
      border-style: dashed;
    }}

    textarea {{
      min-height: 5rem;
      resize: vertical;
    }}

    .help-text {{
      margin: 0 0 1.75rem;
      color: var(--muted);
      font-size: 0.85rem;
    }}

    .job-configs {{
      margin-top: 1.5rem;
    }}

    .job-config {{
      margin-top: 1.5rem;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.5rem;
      background: rgba(3, 10, 14, 0.85);
      box-shadow: inset 0 0 0 1px rgba(57,255,20,0.08), 0 0 22px rgba(0,0,0,0.35);
    }}

    .job-config-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1.25rem;
    }}

    .job-config-header h3 {{
      margin: 0;
      text-transform: uppercase;
      letter-spacing: 0.22em;
      font-size: 0.95rem;
      color: var(--accent);
    }}

    .config-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 1.25rem;
    }}

    .config-card {{
      border: 1px solid rgba(57,255,20,0.25);
      border-radius: 12px;
      padding: 1rem;
      background: rgba(1, 8, 10, 0.6);
      position: relative;
      overflow: hidden;
    }}

    .config-card::after {{
      content: '';
      position: absolute;
      inset: 0;
      border-radius: inherit;
      pointer-events: none;
      box-shadow: 0 0 18px rgba(57,255,20,0.1);
      opacity: 0;
      transition: opacity 0.3s ease;
    }}

    .config-card:hover::after {{
      opacity: 1;
    }}

    .config-card h4 {{
      margin: 0 0 0.75rem;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.24em;
      color: var(--accent);
    }}

    .config-card.wide {{
      grid-column: 1 / -1;
    }}

    .checkbox-grid {{
      display: grid;
      gap: 0.5rem;
      margin-top: 0.75rem;
    }}

    .job-controls {{
      margin-top: 1.25rem;
      text-align: right;
    }}

    .form-actions {{
      margin-top: 2rem;
      text-align: right;
    }}

    .checkbox-grid label,
    .checkbox-inline {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
    }}

    input[type='checkbox'] {{
      width: auto;
      accent-color: var(--accent);
      transform: scale(1.15);
    }}

    .preview-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-top: 0.75rem;
    }}

    .preview-field {{
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }}

    .preview-field.full {{
      grid-column: 1 / -1;
    }}

    .neon-button {{
      border: 1px solid var(--accent);
      background: linear-gradient(135deg, rgba(57,255,20,0.25), rgba(57,255,20,0.05));
      color: var(--text);
      padding: 0.65rem 1.75rem;
      border-radius: 999px;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.28em;
      cursor: pointer;
      transition: transform 0.18s ease, box-shadow 0.18s ease;
      box-shadow: 0 0 12px rgba(57,255,20,0.35);
    }}

    .neon-button:hover {{
      transform: translateY(-2px);
      box-shadow: 0 0 22px rgba(57,255,20,0.55);
    }}

    .neon-button.secondary {{
      border-color: rgba(0,255,255,0.55);
      color: rgba(152, 244, 255, 0.95);
      box-shadow: 0 0 12px rgba(0,255,255,0.32);
      background: linear-gradient(135deg, rgba(0,255,255,0.25), rgba(0,255,255,0.05));
    }}

    .neon-button.secondary:hover {{
      box-shadow: 0 0 22px rgba(0,255,255,0.52);
    }}

    table {{
      width: min(100%, 1100px);
      margin: 1rem auto 0;
      border-collapse: collapse;
      border: 1px solid var(--border);
      background: rgba(3, 12, 14, 0.78);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 0 18px rgba(0,0,0,0.45);
    }}

    th, td {{
      padding: 0.85rem 1rem;
      border-bottom: 1px solid rgba(57,255,20,0.15);
      font-size: 0.85rem;
    }}

    th {{
      text-transform: uppercase;
      letter-spacing: 0.2em;
      color: rgba(152, 244, 255, 0.85);
      background: rgba(0,255,255,0.05);
    }}

    tr:last-child td {{
      border-bottom: none;
    }}

    a {{
      color: var(--accent);
    }}

    .inline-form {{
      display: inline;
    }}

    .delete-button {{
      background: linear-gradient(135deg, rgba(255, 60, 127, 0.5), rgba(255, 0, 128, 0.15));
      border-color: rgba(255, 60, 127, 0.6);
      box-shadow: 0 0 16px rgba(255, 60, 127, 0.32);
    }}

    .message {{
      margin: 0 auto 1.5rem;
      width: min(100%, 1100px);
      padding: 0.85rem 1.2rem;
      border-radius: 999px;
      border: 1px solid rgba(0,255,255,0.45);
      background: rgba(0,255,255,0.12);
      color: rgba(191, 250, 255, 0.95);
      letter-spacing: 0.14em;
      text-transform: uppercase;
      font-size: 0.8rem;
    }}

    @media (max-width: 960px) {{
      form {{
        padding: 1.5rem;
      }}
    }}
  </style>
</head>
<body>
  <h1>DnD Session Transcribe</h1>
  {message_html}
  <form action=\"/transcribe\" method=\"post\" enctype=\"multipart/form-data\">
    <fieldset>
      <label for=\"audio_file\">Audio signal</label>
      <input id=\"audio_file\" name=\"audio_file\" type=\"file\" required />
      <div class=\"job-configs\">
        <p class=\"help-text\">Queue parallel loadouts to A/B models, prompts, and compute knobs. Every configuration reuses the uploaded audio for rapid experiments.</p>
        <div id=\"jobs-container\">
          {initial_job_block}
        </div>
        <div class=\"job-controls\">
          <button type=\"button\" id=\"add-job\" class=\"neon-button secondary\">Clone config</button>
        </div>
      </div>
    </fieldset>
    <div class=\"form-actions\">
      <button type=\"submit\" class=\"neon-button\">Execute Stack</button>
    </div>
  </form>

  <h2>Run console</h2>
  <table>
    <thead><tr><th>Job</th><th>Status</th><th>Created</th><th>Updated</th><th>Error</th><th>Actions</th></tr></thead>
    <tbody>
      {''.join(rows) if rows else "<tr><td colspan='6'>No jobs yet.</td></tr>"}
    </tbody>
  </table>
  <template id=\"job-template\">
    {template_job_block}
  </template>
  <datalist id=\"asr-models\">
    {asr_model_datalist}
  </datalist>
  <datalist id=\"compute-types\">
    {compute_type_datalist}
  </datalist>
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

      function createBlock(index) {{
        const markup = template.innerHTML.replace(/__INDEX__/g, String(index));
        const wrapper = document.createElement('div');
        wrapper.innerHTML = markup.trim();
        return wrapper.firstElementChild;
      }}

      function copyValues(source, target) {{
        if (!source || !target) {{
          return;
        }}
        const fieldMap = new Map();
        target.querySelectorAll('input, select, textarea').forEach((element) => {{
          const name = element.getAttribute('name');
          if (!name) {{
            return;
          }}
          const segments = name.includes('-') ? name.split('-').slice(2) : [name];
          fieldMap.set(segments.join('-'), element);
        }});

        source.querySelectorAll('input, select, textarea').forEach((element) => {{
          const name = element.getAttribute('name');
          if (!name) {{
            return;
          }}
          const segments = name.includes('-') ? name.split('-').slice(2) : [name];
          const key = segments.join('-');
          const destination = fieldMap.get(key);
          if (!destination) {{
            return;
          }}
          if (element instanceof HTMLInputElement && element.type === 'file') {{
            return;
          }}
          if (element instanceof HTMLInputElement && element.type === 'checkbox') {{
            if (destination instanceof HTMLInputElement) {{
              destination.checked = element.checked;
            }}
            return;
          }}
          if (destination instanceof HTMLInputElement && destination.type === 'checkbox') {{
            destination.checked = false;
            return;
          }}
          if ('value' in destination) {{
            destination.value = element.value;
          }}
        }});
      }}

      addButton.addEventListener('click', () => {{
        const newBlock = createBlock(nextIndex);
        if (!newBlock) {{
          return;
        }}
        const blocks = container.querySelectorAll('.job-config');
        const source = blocks.length ? blocks[blocks.length - 1] : null;
        container.appendChild(newBlock);
        if (source) {{
          copyValues(source, newBlock);
        }}
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

    settings_block = ""
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
            "<div class='panel'>"
            "  <h2>Settings</h2>"
            f"  {settings_table}"
            "</div>"
        )

    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Job {job_id}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #010204;
      --panel: rgba(6, 15, 21, 0.85);
      --accent: #39ff14;
      --muted: rgba(152, 244, 255, 0.85);
      --text: #e9ffee;
      --danger: #ff3c7f;
      --border: rgba(57,255,20,0.35);
      font-family: 'Fira Code', 'JetBrains Mono', Menlo, Consolas, monospace;
    }}

    body {{
      margin: 0;
      padding: 3rem;
      background: radial-gradient(circle at 15% 20%, rgba(57,255,20,0.12), transparent 45%),
                  radial-gradient(circle at 80% 85%, rgba(0,255,255,0.12), transparent 40%),
                  var(--bg);
      color: var(--text);
      min-height: 100vh;
    }}

    a {{
      color: var(--accent);
    }}

    h1 {{
      text-transform: uppercase;
      letter-spacing: 0.24em;
      font-size: 2.4rem;
      margin-bottom: 2rem;
      text-shadow: 0 0 18px rgba(57,255,20,0.68);
    }}

    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.75rem;
      margin-bottom: 1.75rem;
      box-shadow: 0 0 22px rgba(0,0,0,0.45);
      backdrop-filter: blur(10px);
    }}

    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin: 0;
      padding: 0;
      list-style: none;
    }}

    .meta-grid li {{
      border: 1px solid rgba(57,255,20,0.25);
      border-radius: 12px;
      padding: 0.75rem 1rem;
      font-size: 0.85rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      background: rgba(0,0,0,0.35);
    }}

    .meta-grid strong {{
      display: block;
      color: var(--muted);
      margin-bottom: 0.35rem;
    }}

    .error {{
      margin-top: 1.25rem;
      border: 1px solid rgba(255,60,127,0.65);
      border-radius: 12px;
      padding: 0.85rem 1rem;
      background: rgba(255, 60, 127, 0.2);
      color: #ffe9f2;
      letter-spacing: 0.16em;
      text-transform: uppercase;
    }}

    .panel h2 {{
      text-transform: uppercase;
      letter-spacing: 0.22em;
      color: var(--muted);
      margin-top: 0;
      margin-bottom: 1rem;
      font-size: 1rem;
    }}

    .settings-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 0.75rem;
      border: 1px solid rgba(57,255,20,0.25);
      border-radius: 12px;
      overflow: hidden;
    }}

    .settings-table th,
    .settings-table td {{
      padding: 0.65rem 0.85rem;
      border-bottom: 1px solid rgba(57,255,20,0.12);
      vertical-align: top;
      font-size: 0.8rem;
      letter-spacing: 0.08em;
    }}

    .settings-table th {{
      width: 28%;
      text-transform: uppercase;
      color: var(--muted);
      background: rgba(0,255,255,0.08);
    }}

    .settings-table pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
    }}

    ul {{
      padding-left: 1.25rem;
    }}

    .neon-button {{
      border: 1px solid var(--accent);
      background: linear-gradient(135deg, rgba(57,255,20,0.25), rgba(57,255,20,0.05));
      color: var(--text);
      padding: 0.6rem 1.65rem;
      border-radius: 999px;
      text-transform: uppercase;
      letter-spacing: 0.28em;
      cursor: pointer;
      transition: transform 0.18s ease, box-shadow 0.18s ease;
      box-shadow: 0 0 12px rgba(57,255,20,0.32);
      font-size: 0.8rem;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }}

    .neon-button:hover {{
      transform: translateY(-2px);
      box-shadow: 0 0 20px rgba(57,255,20,0.52);
    }}

    .delete-button {{
      border-color: rgba(255,60,127,0.75);
      background: linear-gradient(135deg, rgba(255,60,127,0.5), rgba(255,60,127,0.1));
      box-shadow: 0 0 16px rgba(255,60,127,0.32);
    }}

    .delete-form {{
      margin-top: 1.5rem;
    }}

    footer {{
      margin-top: 2.5rem;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.22em;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <h1>Job {job_id}</h1>
  <div class='panel'>
    <ul class='meta-grid'>
      <li><strong>Status</strong>{status}</li>
      <li><strong>Created</strong>{created}</li>
      <li><strong>Updated</strong>{updated}</li>
      <li><strong>Source audio</strong>{audio_name}</li>
    </ul>
    {error_block}
  </div>
  {preview_block}
  {settings_block}
  <div class='panel'>
    <h2>Outputs</h2>
    {file_list}
    {log_link}
  </div>
  <form action=\"{delete_action}\" method=\"post\" class=\"delete-form\" onsubmit=\"return confirm('Delete this job and all associated files?');\">
    <button type=\"submit\" class=\"neon-button delete-button\">Delete job</button>
  </form>
  <footer><a href=\"/\">Return to command console</a></footer>
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
            if idx and idx not in seen_indices and idx.isdigit():
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
            key = _field_key(job_idx, name)
            raw = form_values.get(key, default)
            if isinstance(raw, str):
                return raw
            return default if raw is None else str(raw)

        def _get_checkbox(job_idx: str | None, name: str) -> bool:
            key = _field_key(job_idx, name)
            if key not in form_values:
                return False
            raw = form_values[key]
            return _checkbox_to_bool(str(raw))

        try:
            rng = random.Random(secrets.randbits(64))
            job_plans: list[dict[str, Any]] = []
            skipped_jobs: list[dict[str, Any]] = []
            seen_signatures: set[tuple[Any, ...]] = set()
            batch_counter = 0

            def _signature(base: dict[str, Any], selections: dict[str, str]) -> tuple[tuple[str, Any], ...]:
                return (
                    ("log_level", selections["log_level"]),
                    ("num_speakers", base["num_speakers"]),
                    ("ram", base["ram"]),
                    ("resume", base["resume"]),
                    ("precise_rerun", base["precise_rerun"]),
                    ("asr_model", base["asr_model"]),
                    ("asr_device", selections["asr_device"] or None),
                    ("asr_compute_type", base["asr_compute_type"]),
                    ("precise_model", base["precise_model"]),
                    ("precise_device", selections["precise_device"] or None),
                    ("precise_compute_type", base["precise_compute_type"]),
                    ("vocal_extract", selections["vocal_extract"] or None),
                    ("hotwords_text", base["hotwords_text"]),
                    ("initial_prompt_text", base["initial_prompt_text"]),
                    ("spelling_map_text", base["spelling_map_text"]),
                    ("preview_requested", base["preview_requested"]),
                    (
                        "preview_start",
                        base["preview_start"] if base["preview_requested"] else None,
                    ),
                    (
                        "preview_duration",
                        base["preview_duration"] if base["preview_requested"] else None,
                    ),
                    (
                        "preview_output_name",
                        base["preview_output_name"] if base["preview_requested"] else None,
                    ),
                )

            for order, index in enumerate(job_indices):
                def _detail(message: str) -> str:
                    return f"{message} for configuration {order + 1}"

                log_level_text = _get_value(index, "log_level", cli.LOG.level).strip()
                if log_level_text and log_level_text not in {_CHOICE_RANDOM, _CHOICE_ALL}:
                    log_level_text = log_level_text.upper()
                log_level_requested = log_level_text or cli.LOG.level

                try:
                    log_level_mode, log_level_choices = _resolve_selection(
                        log_level_requested, tuple(cli.LOG_LEVELS.keys())
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=_detail("Invalid log level")) from exc

                asr_model_value = _get_value(index, "asr_model").strip()
                num_speakers_text = _get_value(index, "num_speakers").strip()
                asr_device_value = _get_value(index, "asr_device").strip()
                asr_compute_type_value = _get_value(index, "asr_compute_type").strip()
                precise_model_value = _get_value(index, "precise_model").strip()
                precise_device_value = _get_value(index, "precise_device").strip()
                precise_compute_type_value = _get_value(index, "precise_compute_type").strip()
                hotwords_text = _get_value(index, "hotwords").strip()
                initial_prompt_text = _get_value(index, "initial_prompt").strip()
                spelling_map_text = _get_value(index, "spelling_map").strip()
                preview_output_text = _get_value(index, "preview_output").strip()
                vocal_extract_value = _get_value(index, "vocal_extract").strip()

                try:
                    asr_device_mode, asr_device_choices = _resolve_selection(
                        asr_device_value, _DEVICE_VALUES
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=_detail("Invalid asr_device value")) from exc

                try:
                    precise_device_mode, precise_device_choices = _resolve_selection(
                        precise_device_value, _DEVICE_VALUES
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=_detail("Invalid precise_device value")) from exc

                try:
                    vocal_mode, vocal_choices = _resolve_selection(
                        vocal_extract_value, _VOCAL_VALUES
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=_detail("Invalid vocal_extract value")) from exc

                parsed_num_speakers: Optional[int]
                if num_speakers_text:
                    try:
                        parsed_num_speakers = int(num_speakers_text)
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail=_detail("num_speakers must be an integer")) from exc
                else:
                    parsed_num_speakers = None

                ram_enabled = _get_checkbox(index, "ram")
                resume_enabled = _get_checkbox(index, "resume")
                precise_rerun_enabled = _get_checkbox(index, "precise_rerun")
                preview_enabled_flag = _get_checkbox(index, "preview_enabled")
                preview_start_text = _get_value(index, "preview_start").strip()
                preview_duration_text = _get_value(index, "preview_duration").strip()

                preview_fields_supplied = bool(preview_start_text or preview_duration_text)
                preview_requested = preview_enabled_flag or preview_fields_supplied
                preview_meta: dict[str, Any] = {"requested": preview_requested}

                preview_start_arg: Optional[float]
                preview_duration_arg: Optional[float]
                if preview_requested:
                    try:
                        start_value = (
                            cli.parse_time_spec(preview_start_text)
                            if preview_start_text
                            else 0.0
                        )
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail=_detail("Invalid preview_start value")) from exc

                    try:
                        duration_value = (
                            cli.parse_time_spec(preview_duration_text)
                            if preview_duration_text
                            else 10.0
                        )
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail=_detail("Invalid preview_duration value")) from exc

                    if duration_value <= 0:
                        raise HTTPException(status_code=400, detail=_detail("Preview duration must be positive"))

                    preview_start_arg = start_value
                    preview_duration_arg = duration_value
                    preview_meta.update({"start": start_value, "duration": duration_value})
                else:
                    preview_start_arg = None
                    preview_duration_arg = None

                preview_output_name: Optional[str] = None
                if preview_output_text:
                    preview_output_name = safe_filename(preview_output_text)
                    if not preview_output_name.lower().endswith(".wav"):
                        preview_output_name += ".wav"

                selection_modes = {
                    "log_level": (log_level_mode, log_level_choices),
                    "asr_device": (asr_device_mode, asr_device_choices),
                    "precise_device": (precise_device_mode, precise_device_choices),
                    "vocal_extract": (vocal_mode, vocal_choices),
                }

                base_config = {
                    "order": order,
                    "index": index,
                    "num_speakers": parsed_num_speakers,
                    "ram": ram_enabled,
                    "resume": resume_enabled,
                    "precise_rerun": precise_rerun_enabled,
                    "asr_model": asr_model_value or None,
                    "asr_compute_type": asr_compute_type_value or None,
                    "precise_model": precise_model_value or None,
                    "precise_compute_type": precise_compute_type_value or None,
                    "hotwords_text": hotwords_text,
                    "initial_prompt_text": initial_prompt_text,
                    "spelling_map_text": spelling_map_text,
                    "preview_requested": preview_requested,
                    "preview_start": preview_start_arg,
                    "preview_duration": preview_duration_arg,
                    "preview_output_name": preview_output_name,
                    "preview_meta": preview_meta,
                    "log_level_requested": log_level_requested,
                    "asr_device_requested": asr_device_value,
                    "precise_device_requested": precise_device_value,
                    "vocal_extract_requested": vocal_extract_value,
                }

                base_values = {
                    name: choices[0]
                    for name, (mode, choices) in selection_modes.items()
                    if mode == "single"
                }
                all_field_names = [
                    name for name, (mode, _) in selection_modes.items() if mode == "all"
                ]
                all_combinations = (
                    list(
                        itertools.product(
                            *(selection_modes[name][1] for name in all_field_names)
                        )
                    )
                    if all_field_names
                    else [()]
                )
                random_fields = {
                    name: choices
                    for name, (mode, choices) in selection_modes.items()
                    if mode == "random"
                }
                random_field_names = list(random_fields)

                for combo in all_combinations:
                    resolved_values = base_values.copy()
                    for field_name, chosen in zip(all_field_names, combo):
                        resolved_values[field_name] = chosen

                    random_combos = [()]
                    if random_field_names:
                        random_combos = list(
                            itertools.product(
                                *(random_fields[name] for name in random_field_names)
                            )
                        )
                        rng.shuffle(random_combos)

                    assigned = False
                    attempted_candidate: dict[str, str] | None = None
                    for random_choice in random_combos:
                        candidate_values = resolved_values.copy()
                        for field_name, chosen in zip(random_field_names, random_choice):
                            candidate_values[field_name] = chosen
                        signature = _signature(base_config, candidate_values)
                        if signature not in seen_signatures:
                            seen_signatures.add(signature)
                            assigned = True
                            final_values = candidate_values
                            break
                        attempted_candidate = candidate_values

                    if not assigned:
                        reason = "Skipped duplicate configuration"
                        if random_field_names:
                            reason = "Skipped duplicate configuration (random exhausted)"
                        skip_settings = {
                            "log_level": base_config["log_level_requested"],
                            "num_speakers": base_config["num_speakers"],
                            "ram": base_config["ram"],
                            "resume": base_config["resume"],
                            "precise_rerun": base_config["precise_rerun"],
                            "asr_model": base_config["asr_model"],
                            "asr_device": base_config["asr_device_requested"],
                            "asr_compute_type": base_config["asr_compute_type"],
                            "precise_model": base_config["precise_model"],
                            "precise_device": base_config["precise_device_requested"],
                            "precise_compute_type": base_config["precise_compute_type"],
                            "vocal_extract": base_config["vocal_extract_requested"],
                            "preview_requested": base_config["preview_requested"],
                            "preview_start": base_config["preview_start"],
                            "preview_duration": base_config["preview_duration"],
                            "preview_output": base_config["preview_output_name"],
                            "hotwords": base_config["hotwords_text"],
                            "initial_prompt": base_config["initial_prompt_text"],
                            "spelling_map": base_config["spelling_map_text"],
                            "batch_index": None,
                            "job_index": index,
                        }
                        if attempted_candidate:
                            skip_settings["resolved_log_level"] = attempted_candidate.get("log_level")
                            skip_settings["resolved_asr_device"] = attempted_candidate.get("asr_device")
                            skip_settings["resolved_precise_device"] = attempted_candidate.get(
                                "precise_device"
                            )
                            skip_settings["resolved_vocal_extract"] = attempted_candidate.get(
                                "vocal_extract"
                            )

                        skip_job_id = _generate_job_id()
                        skip_dir = _job_dir(skip_job_id)
                        created_at = _utc_now()
                        status_snapshot = _job_status_template(skip_job_id, created_at)
                        status_snapshot["status"] = "skipped"
                        status_snapshot["error"] = reason
                        metadata = {
                            "job_id": skip_job_id,
                            "audio_filename": audio_file.filename,
                            "created_at": created_at,
                            "preview": dict(base_config["preview_meta"]),
                            "settings": skip_settings,
                            "batch_index": None,
                            "job_index": index,
                            "skip_reason": reason,
                        }
                        skipped_jobs.append(
                            {
                                "job_id": skip_job_id,
                                "job_dir": skip_dir,
                                "metadata": metadata,
                                "status": status_snapshot,
                                "reason": reason,
                            }
                        )
                        continue

                    job_id = _generate_job_id()
                    job_dir = _job_dir(job_id)
                    inputs_dir = job_dir / "inputs"
                    outputs_dir = job_dir / "outputs"
                    audio_target = inputs_dir / clean_name

                    hotwords_path = inputs_dir / "hotwords.txt" if hotwords_text else None
                    initial_prompt_path = inputs_dir / "initial_prompt.txt" if initial_prompt_text else None
                    spelling_map_path = inputs_dir / "spelling_map.csv" if spelling_map_text else None
                    preview_output_path = (
                        outputs_dir / preview_output_name if preview_output_name else None
                    )

                    file_payloads: list[tuple[Path, str]] = []
                    if hotwords_path:
                        file_payloads.append((hotwords_path, hotwords_text))
                    if initial_prompt_path:
                        file_payloads.append((initial_prompt_path, initial_prompt_text))
                    if spelling_map_path:
                        file_payloads.append((spelling_map_path, spelling_map_text))

                    args = build_cli_args(
                        audio_target,
                        outdir=outputs_dir,
                        ram=ram_enabled,
                        resume=resume_enabled,
                        num_speakers=parsed_num_speakers,
                        hotwords_file=hotwords_path,
                        initial_prompt_file=initial_prompt_path,
                        spelling_map=spelling_map_path,
                        asr_model=asr_model_value or None,
                        asr_device=final_values["asr_device"] or None,
                        asr_compute_type=asr_compute_type_value or None,
                        precise_rerun=precise_rerun_enabled,
                        precise_model=precise_model_value or None,
                        precise_device=final_values["precise_device"] or None,
                        precise_compute_type=precise_compute_type_value or None,
                        vocal_extract=final_values["vocal_extract"] or None,
                        log_level=final_values["log_level"],
                        preview_start=preview_start_arg,
                        preview_duration=preview_duration_arg,
                        preview_output=preview_output_path,
                    )

                    settings_snapshot = {
                        key: (str(val) if isinstance(val, Path) else val)
                        for key, val in vars(args).items()
                    }
                    settings_snapshot["preview_requested"] = preview_requested
                    settings_snapshot["batch_index"] = batch_counter
                    settings_snapshot["job_index"] = index
                    if random_field_names:
                        settings_snapshot["randomized_fields"] = sorted(random_field_names)

                    created_at = _utc_now()
                    status_snapshot = _job_status_template(job_id, created_at)
                    status_snapshot["audio_filename"] = audio_file.filename

                    metadata = {
                        "job_id": job_id,
                        "audio_filename": audio_file.filename,
                        "created_at": created_at,
                        "preview": dict(base_config["preview_meta"]),
                        "settings": settings_snapshot,
                        "batch_index": batch_counter,
                        "job_index": index,
                    }

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
                            "file_payloads": file_payloads,
                        }
                    )
                    batch_counter += 1

            if not job_plans and not skipped_jobs:
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

                for payload_path, payload_content in plan.get("file_payloads", []):
                    payload_path.parent.mkdir(parents=True, exist_ok=True)
                    payload_path.write_text(payload_content, encoding="utf-8")

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

            skipped_ids: list[str] = []
            for skipped in skipped_jobs:
                job_dir = skipped["job_dir"]
                job_dir.mkdir(parents=True, exist_ok=True)
                _write_json(job_dir / "metadata.json", skipped["metadata"])
                _write_json(job_dir / "status.json", skipped["status"])
                skipped_ids.append(skipped["job_id"])

            parts: list[str] = []
            if job_ids:
                if len(job_ids) == 1:
                    parts.append(f"Started job {job_ids[0]}")
                else:
                    parts.append("Started jobs " + ", ".join(job_ids))
            if skipped_ids:
                parts.append("Skipped duplicate configs " + ", ".join(skipped_ids))
            if parts:
                message = quote_plus("; ".join(parts))
            else:
                message = quote_plus("No jobs were scheduled")

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
