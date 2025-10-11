# DnD Session Transcribe – Agent Guidelines (2024 refresh)

Welcome! The project was recently restructured into an installable package. Please read the notes below before changing code in this repository.

## Installation & CLI
- Install the project in editable mode during development: `pip install -e .[uvr5]` (add `--extra-index-url` if you need CUDA wheels).
- The CLI entry point is `dnd-transcribe`, exposed via `src/dnd_session_transcribe/cli.py`. Use this command (not `python run_whisperx.py`) for smoke tests, e.g. `dnd-transcribe --help`.
- Modules should import configuration by going through `dnd_session_transcribe.util.config`; avoid assuming the working directory.

## Code Organization
- `src/dnd_session_transcribe/cli.py` keeps argument parsing and the high-level pipeline wiring. Push heavy logic into feature modules.
- `features/` contains the core stages: `asr.py`, `alignment.py`, `diarization.py`, and `precise_rerun.py`.
- `adapters/` holds integration glue for external libraries (WhisperX, pyannote, UVR). Keep external API calls isolated here.
- `util/` provides shared helpers: configuration defaults (`util/config`), path helpers (`next_outdir.py`), preprocessing (`processing.py`), and writers (`write_files.py`).
- Tests live under `tests/` mirroring the util/helper they cover. Add tests alongside any behavior change.

## Coding Standards
- Target Python 3.10+. Prefer `pathlib.Path` and keep type hints accurate.
- Import grouping: stdlib → third-party → local. No try/except around imports.
- Use module-level `logger = logging.getLogger(__name__)` and structured log messages with feature prefixes (e.g., `[ASR]`, `[ALIGN]`).
- When writing outputs, favor helpers in `util.write_files` or `util.helpers.atomic_json` for atomic I/O.
- CLI help text in `cli.py` must stay in sync with README examples.

## Dependencies
- Runtime dependencies belong in `pyproject.toml`; optional ones in the relevant extras (`[project.optional-dependencies]`).
- If you gate optional imports (e.g., UVR), document the requirement in the module docstring and README.

## Testing & Verification
- Run unit tests with `pytest`. Use targeted paths for quicker feedback (e.g., `pytest tests/util/test_processing.py`).
- For CLI-affecting changes, run `dnd-transcribe --help` to ensure arguments and defaults are correct.
- If you change long-running stages, ensure resume/checkpointing via JSON artifacts still works.
- ffmpeg will need to be installed via apt.

## Performance Constraints & Model Selection
- Environment: CPU-only; no CUDA/GPU; limited RAM and wall-clock.
- Audio length: <= 60 seconds for test runs.
- Whisper backend (local, no network calls):
  - Default: `tiny`
  - Upgrade to `base` only if the clip is < 60s and prior runs complete < 90s CPU time.
- Absolutely do not select large models in this sandbox.
- If internet egress is available and permitted (rare), you may use a hosted STT model; otherwise **must** use local tiny/base.

## Configuration Knobs
- `WHISPER_MODEL` env var controls the local model size. Allowed: `tiny`, `base`.
- If unset, assume `tiny`.
- The agent must set `WHISPER_MODEL=tiny` for first run; can retry with `base` only if CPU budget allows.

## Documentation
- Update `README.md` when user workflows or CLI flags change.
- Maintain docstrings for shared helpers and adapters, especially where external APIs are invoked.

## Pull Requests
- Summarize pipeline impacts (latency, accuracy, outputs) in the PR body.
- Reference relevant tests and include reproduction steps for manual flows when applicable.

Thanks for keeping the transcription pipeline polished!
