# DnD Session Transcribe – Agent Guidelines

Welcome to the project! This repository wires together preprocessing, Faster-Whisper ASR, WhisperX alignment, and pyannote diarization to build polished transcripts for long-form tabletop sessions. Follow the conventions below whenever you touch code under this repo.

## Code Organization
- **Entry point:** `run_whisperx.py` contains the CLI and orchestrates the pipeline. Keep it focused on wiring and lightweight flow-control; push non-trivial logic into the `functions/`, `helpers/`, or `utilities/` packages.
- **Core stages:**
  - `functions/` – heavy operations (ASR, alignment, diarization, precise reruns).
  - `utilities/` – infrastructure helpers (I/O, Hugging Face token checks, preprocessing, output writers).
  - `helpers/` – lightweight data helpers (VAD parameters, hotwords, initial prompts, spelling maps, atomic JSON writes).
  - `constants/` – dataclass-style configuration defaults.
- **Tests:** Live under `tests/` mirroring the helper or utility they target. Add/update tests alongside any behavioral change.

## Coding Standards
- Target Python **3.10+**. Use `pathlib.Path` for filesystem work and keep type hints up-to-date.
- Import grouping mirrors the existing modules: stdlib → third-party → local packages, with section comments when helpful.
- Use module-level `logger = logging.getLogger(__name__)` and structured logging (no bare `print`). Prefer informative messages with prefixes (`[ASR]`, `[DIA]`, etc.) that match existing style.
- When writing files, prefer the dedicated helpers (`utilities.write_srt_vtt_txt_json`, `helpers.atomic_json`, etc.) rather than ad-hoc `open()` calls. This keeps writes atomic and consistent.
- Honor the existing progress bar approach via `tqdm.auto` and `PROGRESS_STREAM`. If you add new progress reporting, match the configuration already in `run_whisperx.py` or `functions/asr.py`.
- Fail fast with explicit exceptions or `SystemExit` for user-facing CLI issues; avoid silent fall-throughs.
- Keep CLI surfaces self-documenting: update `parse_args()` help text and README snippets if flags change.
- Stick to f-strings for formatting and avoid try/except guards around imports. Import late only for truly heavy dependencies (see `run_whisperx.py`).

## Adding Dependencies
- Declare runtime dependencies in `pyproject.toml`. If CUDA-specific wheels are required, mirror the guidance already present in `requirements*.txt`.
- If a helper only needs a dependency optionally (e.g., UVR), gate the import locally and document the requirement in docstrings/README.

## Testing & Verification
- Run unit tests with `pytest`. Use targeted invocations when working on a subset, e.g. `pytest tests/helpers/test_hotwords.py`.
- For CLI-affecting changes, smoke-test `python run_whisperx.py --help` to confirm argument wiring and default text remain sane.
- If you modify long-running stages (ASR/diarization), ensure they can still be resumed (`--resume`) by exercising the relevant JSON checkpoints.

## Documentation Expectations
- Update `README.md` when user-facing workflow or options change.
- Keep docstrings succinct but descriptive, especially for new helpers/utilities consumed by multiple modules.
- Mention environment prerequisites (FFmpeg, Hugging Face tokens, UVR) if you make them more/less strict.

## Pull Request Tips
- Describe pipeline impacts clearly: note effects on latency, accuracy, and file outputs.
- Reference relevant tests in the PR body and include reproduction steps for manual flows when applicable.

Adhering to these guidelines keeps the transcription pipeline predictable and approachable for future contributors. Thanks! 
