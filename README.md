# DnD Session Transcribe

## Project Overview
This repository wraps an end-to-end speech transcription workflow optimized for tabletop roleplaying game sessions and similarly long recordings. The installable CLI entry point, `dnd-transcribe`, orchestrates preprocessing, automatic speech recognition (ASR), alignment, and diarization into a single command so you can go from raw audio to timecoded, speaker-attributed transcripts with minimal manual effort.

### Pipeline Summary
`dnd-transcribe` wires together the following stages:
1. **Optional preprocessing** – Copies audio into RAM for faster IO (`--ram`) and can run vocal extraction (`--vocal-extract` with `off` or `bandpass`).
2. **Faster-Whisper ASR** – Generates initial segments, optionally primed with hotwords, an initial prompt, and post-run spelling correction maps.
3. **Optional precise re-run** – Reruns difficult spans with more exact ASR settings when `--precise-rerun` is enabled.
4. **WhisperX alignment** – Refines word-level timestamps for the recognized text.
5. **pyannote diarization** – Separates speakers and aligns segments to diarization turns.
6. **Speaker assignment & output writing** – Merges transcripts with speaker turns and emits `.srt`, `.vtt`, `.txt`, and `.json` artifacts per session.

## Installation
After cloning the repository, install the project in an isolated environment once the `pyproject.toml` is available:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
```

This pulls in the dependencies declared in the project metadata so the CLI can run without additional manual steps.

> **GPU wheels:** If you instead install from `requirements.txt`, note that the project pins CUDA-enabled builds of
> `torch`, `torchaudio`, and `torchvision` (`*+cu121`). Those wheels live on the PyTorch extra index, so standard pip
> will need a finder override:
>
> ```bash
> pip install -r requirements.txt \
>   -f https://download.pytorch.org/whl/cu121
> ```
>
> Without the `-f`/`--extra-index-url` hint, pip reports “No matching distribution found” because the CUDA wheels are
> not published to the default PyPI index.

### External Requirements
- **FFmpeg** – Required for audio decoding, clipping, and vocal extraction helpers. Ensure `ffmpeg` is available on your `PATH` (e.g., via `apt`, `brew`, or manual downloads).
- **Bandpass filtering** – A lightweight high/low-pass filter that boosts dialog intelligibility without external dependencies.

## Environment Variables
- `HUGGINGFACE_TOKEN` – A mandatory token with access to the pyannote diarization models. The pipeline calls `utilities.ensure_hf_token()` before diarization and aborts if the variable is not set. Export the token in your shell or configure it via your environment manager so the diarization stage can authenticate against Hugging Face.

```bash
export HUGGINGFACE_TOKEN=hf_your_token_here
```

## CLI Usage
Run the end-to-end pipeline by pointing to an audio file:

```bash
dnd-transcribe /path/to/audio.wav
```

Commonly used options (consult `dnd-transcribe --help` for the full list):
- `--ram` – Copy the input audio to `/dev/shm` (tmpfs) for faster processing on systems with ample memory.
- `--precise-rerun` – Enable the high-accuracy ASR pass over spans flagged as difficult after the first transcription.
- `--vocal-extract {off,bandpass}` – Override the preprocessing strategy. `bandpass` applies a 50–7800 Hz filter with loudness normalization.
- `--num-speakers` – Override the diarization speaker count when the automatic estimate needs correction.
- `--resume` – Reuse cached JSON checkpoints to avoid recomputing completed stages.
- `--hotwords-file`, `--initial-prompt-file`, `--spelling-map` – Provide customization files for ASR biasing and post-processing.
- `--asr-model`, `--asr-device`, `--asr-compute-type` – Override the Faster-Whisper model id, device, or compute precision.
- `--precise-model`, `--precise-device`, `--precise-compute-type` – Tune the optional precise rerun pass when GPU resources differ from the defaults.
- `--preview-start`, `--preview-duration`, `--preview-output` – Render a standalone WAV snippet for preview playback and run the full transcription stack on that excerpt. The start time accepts seconds or `MM:SS`/`HH:MM:SS` values, the duration defaults to 10s, and the snippet is always copied into the preview output directory (auto-named `preview_<prefix>N` unless you supply `--outdir`). A custom `--preview-output` path receives a duplicate copy in addition to the managed folder.

### Rendering preview snippets

Supply any of the preview flags to export a temporary snippet that GUI integrations or manual spot-checks can play back. The command continues on to produce transcript artifacts (`.srt`, `.vtt`, `.txt`, `.json`) for just that excerpt once the preview WAV is written:

```bash
dnd-transcribe sample_audio/test.wav \
  --preview-start 1:23 --preview-duration 5 \
  --preview-output /tmp/test_preview.wav
```

The CLI reports every preview copy it writes (managed output directory plus any custom path) and the actual length of the rendered snippet when it completes. Preview runs keep their artifacts—snippet audio and transcripts—inside the `preview_` output directory so the association is unambiguous.

### CPU-friendly overrides

The default configuration targets GPU-equipped systems (`large-v3`, `cuda`, `float16`). On CPU-only hosts, select a lighter model and precision explicitly:

```bash
dnd-transcribe sample_audio/test.wav \
  --asr-model tiny --asr-device cpu --asr-compute-type int8_float32
```

When enabling `--precise-rerun` on CPU, pair it with `--precise-model tiny --precise-device cpu --precise-compute-type float32` to avoid CUDA requirements.

The script automatically chooses an output directory (`textN` alongside the audio) unless `--outdir` is supplied. Preview runs instead create `preview_textN` next to the source audio (or `preview_<custom>` if you provide `--outdir`).

## Web UI

Prefer a browser-based workflow? Install the package (which now includes FastAPI and Uvicorn) and launch the hosted interface:

```bash
dnd-transcribe-web  # binds to 0.0.0.0:8000 by default
```

The server exposes a simple dashboard where you can upload audio, toggle resume/precise rerun, monitor job status, and download the generated transcripts/logs. Because it binds to `0.0.0.0`, you can access it from other machines on your network via `http://<server-ip>:8000`.

Key behaviors:

- Jobs are stored beneath `webui_runs/` in the current working directory. Override the storage root with `DND_TRANSCRIBE_WEB_ROOT=/path/to/runs`.
- Adjust the bind host/port via `DND_TRANSCRIBE_WEB_HOST` and `DND_TRANSCRIBE_WEB_PORT` environment variables if you need different network settings.
- Each job records its log to `job.log`; the Web UI links to it alongside the output files for quick download.
- The FastAPI app can also be served manually: `uvicorn dnd_session_transcribe.web:app --host 0.0.0.0 --port 8000`.

## Outputs
Each run writes artifacts prefixed by the audio stem inside the resolved output directory:
- `<name>.srt` – Subtitle file with speaker tags and timestamps for video players.
- `<name>.vtt` – WebVTT subtitles suitable for browsers and web editors.
- `<name>.txt` – Plain text transcript for quick review.
- `<name>.json` – Structured output containing word-level timings and speaker annotations for programmatic use.

Intermediate JSON checkpoints (e.g., `*_fw_segments_scrubbed.json`) are also saved when applicable, enabling resume and debugging workflows.

## Testing

Refer to [docs/testing.md](docs/testing.md) for the current pytest status. In short, a clean environment hits an import error for
`fastapi.testclient` because FastAPI is not installed by default; installing `fastapi[standard]` or adding `fastapi>=0.109,<1`
to your development requirements resolves the missing module so the web integration tests can run.

## Troubleshooting
- **Missing Hugging Face access** – If the script stops with `[HF] Missing HUGGINGFACE_TOKEN`, generate or renew a token on Hugging Face, grant it access to the pyannote models, and export it before retrying.
- **Diarization failures** – Poor audio quality or aggressive preprocessing can yield empty diarization tracks. Try disabling vocal extraction, supplying `--num-speakers`, or verifying the token has pyannote access.
- **Preprocessing errors** – Fall back to `--vocal-extract off` if bandpass filtering fails unexpectedly.

With the prerequisites satisfied, a single CLI invocation handles the entire transcription-to-diarization pipeline end-to-end.
