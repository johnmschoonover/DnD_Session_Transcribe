"""HTML rendering helpers for the web UI."""

from __future__ import annotations

import html
import json
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

from .. import cli
from .services.jobs import (
    _ASR_MODEL_SUGGESTIONS,
    _COMPUTE_TYPE_SUGGESTIONS,
    _DEVICE_OPTIONS,
    _VOCAL_OPTIONS,
)

__all__ = ["render_home", "render_job_detail"]


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
        for level in cli.LOG_LEVELS.keys()
    )
    log_level_options += (
        f"<option value=\"random\">Random</option>"
        f"<option value=\"all\">All</option>"
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
        f"<option value=\"random\">Random</option>"
        f"<option value=\"all\">All</option>"
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
        f"<option value=\"random\">Random</option>"
        f"<option value=\"all\">All</option>"
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


def render_home(jobs: Iterable[dict[str, Any]], message: str | None = None) -> str:
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
      background: radial-gradient(circle at top left, rgba(57,255,20,0.12), transparent 60%);
      opacity: 0;
      transition: opacity 0.3s ease;
    }}

    .config-card:hover::after {{
      opacity: 1;
    }}

    .checkbox-grid {{
      display: grid;
      gap: 0.5rem;
      margin-top: 1rem;
    }}

    .checkbox-inline {{
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      margin-top: 0.75rem;
    }}

    .job-controls {{
      display: flex;
      justify-content: flex-end;
      margin-top: 1rem;
    }}

    .neon-button {{
      background: linear-gradient(135deg, rgba(57,255,20,0.35), rgba(57,255,20,0.05));
      border: 1px solid rgba(57,255,20,0.5);
      border-radius: 12px;
      padding: 0.75rem 1.5rem;
      color: var(--text);
      text-transform: uppercase;
      letter-spacing: 0.22em;
      cursor: pointer;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
      box-shadow: 0 0 18px rgba(57,255,20,0.28);
    }}

    .neon-button.secondary {{
      background: linear-gradient(135deg, rgba(0,200,255,0.25), rgba(0,200,255,0.05));
      border-color: rgba(0,200,255,0.4);
    }}

    .neon-button:hover {{
      transform: translateY(-2px);
      box-shadow: 0 0 20px rgba(57,255,20,0.52);
    }}

    .form-actions {{
      margin-top: 2rem;
      display: flex;
      justify-content: center;
    }}

    table {{
      width: min(100%, 1100px);
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 0 25px rgba(57,255,20,0.08);
    }}

    th,
    td {{
      padding: 0.85rem 1rem;
      border-bottom: 1px solid rgba(57,255,20,0.18);
      text-align: left;
    }}

    thead th {{
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: var(--muted);
      font-size: 0.75rem;
    }}

    tr:last-child td {{
      border-bottom: none;
    }}

    .message {{
      background: rgba(57,255,20,0.15);
      border: 1px solid rgba(57,255,20,0.35);
      padding: 1rem 1.5rem;
      border-radius: 12px;
      color: var(--text);
      text-transform: uppercase;
      letter-spacing: 0.18em;
    }}

    .inline-form {{
      display: inline;
    }}

    .delete-button {{
      border-color: rgba(255,60,127,0.75);
      background: linear-gradient(135deg, rgba(255,60,127,0.5), rgba(255,60,127,0.1));
      box-shadow: 0 0 16px rgba(255,60,127,0.32);
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


def render_job_detail(
    job: Mapping[str, Any],
    files: Sequence[tuple[str, str]],
    log_available: bool,
    *,
    preview: Mapping[str, Any] | None = None,
    preview_url: str | None = None,
    settings: Mapping[str, Any] | None = None,
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

    file_rows = [
        f"<li><a href=\"{html.escape(url)}\">{html.escape(label)}</a></li>"
        for label, url in files
    ]
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
      font-size: 2.5rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 1.5rem;
      text-shadow: 0 0 18px rgba(57,255,20,0.7), 0 0 32px rgba(0, 255, 255, 0.35);
      text-align: center;
    }}

    .panel {{
      width: min(100%, 960px);
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 1.75rem;
      box-shadow: 0 0 25px rgba(57,255,20,0.08);
    }}

    .meta-grid {{
      list-style: none;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      padding: 0;
      margin: 0;
    }}

    .meta-grid li {{
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 0.75rem;
    }}

    .meta-grid strong {{
      display: block;
      color: var(--muted);
      font-size: 0.7rem;
      margin-bottom: 0.35rem;
    }}

    .settings-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 1rem;
    }}

    .settings-table th,
    .settings-table td {{
      border: 1px solid rgba(57,255,20,0.18);
      padding: 0.75rem;
      text-align: left;
      vertical-align: top;
    }}

    .settings-table th {{
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 0.7rem;
      color: var(--muted);
      width: 25%;
    }}

    .error {{
      margin-top: 1rem;
      padding: 0.75rem 1rem;
      border-radius: 12px;
      background: rgba(255,60,127,0.15);
      border: 1px solid rgba(255,60,127,0.35);
    }}

    .delete-form {{
      margin-top: 1.5rem;
    }}

    .neon-button {{
      background: linear-gradient(135deg, rgba(57,255,20,0.35), rgba(57,255,20,0.05));
      border: 1px solid rgba(57,255,20,0.5);
      border-radius: 12px;
      padding: 0.75rem 1.5rem;
      color: var(--text);
      text-transform: uppercase;
      letter-spacing: 0.22em;
      cursor: pointer;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
      box-shadow: 0 0 18px rgba(57,255,20,0.28);
    }}

    .neon-button.delete-button {{
      border-color: rgba(255,60,127,0.75);
      background: linear-gradient(135deg, rgba(255,60,127,0.5), rgba(255,60,127,0.1));
      box-shadow: 0 0 16px rgba(255,60,127,0.32);
    }}

    .neon-button:hover {{
      transform: translateY(-2px);
      box-shadow: 0 0 20px rgba(57,255,20,0.52);
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
  <form action=\"{delete_action}\" method=\"post\" class=\"delete-form\" onsubmit=\"return confirm('Delete this job and all associated files?');\">\n    <button type=\"submit\" class=\"neon-button delete-button\">Delete job</button>\n  </form>\n  <footer><a href=\"/\">Return to command console</a></footer>\n</body>\n</html>\n"""
