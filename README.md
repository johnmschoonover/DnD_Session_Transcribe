# DnD Session Transcribe

## Project Overview
This repository wraps an end-to-end speech transcription workflow optimized for tabletop roleplaying game sessions and similarly long recordings. The main entry point, [`run_whisperx.py`](run_whisperx.py), orchestrates preprocessing, automatic speech recognition (ASR), alignment, and diarization into a single command so you can go from raw audio to timecoded, speaker-attributed transcripts with minimal manual effort.

### Pipeline Summary
`run_whisperx.py` wires together the following stages:
1. **Optional preprocessing** – Copies audio into RAM for faster IO (`--ram`) and can run vocal extraction (`--vocal-extract` with `off`, `bandpass`, or `mdx_kim2`).
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

### External Requirements
- **FFmpeg** – Required for audio decoding, clipping, and vocal extraction helpers. Ensure `ffmpeg` is available on your `PATH` (e.g., via `apt`, `brew`, or manual downloads).
- **UVR5 CLI (optional)** – Needed when invoking `--vocal-extract mdx_kim2` to isolate vocals before transcription. Install the Ultimate Vocal Remover v5 command-line tools and verify they are discoverable from the shell running the script.

## Environment Variables
- `HUGGINGFACE_TOKEN` – A mandatory token with access to the pyannote diarization models. The pipeline calls `utilities.ensure_hf_token()` before diarization and aborts if the variable is not set. Export the token in your shell or configure it via your environment manager so the diarization stage can authenticate against Hugging Face.

```bash
export HUGGINGFACE_TOKEN=hf_your_token_here
```

## CLI Usage
Run the end-to-end pipeline by pointing to an audio file:

```bash
python run_whisperx.py /path/to/audio.wav
```

Commonly used options (consult `python run_whisperx.py --help` for the full list):
- `--ram` – Copy the input audio to `/dev/shm` (tmpfs) for faster processing on systems with ample memory.
- `--precise-rerun` – Enable the high-accuracy ASR pass over spans flagged as difficult after the first transcription.
- `--vocal-extract {off,bandpass,mdx_kim2}` – Override the preprocessing strategy, including optional UVR5 separation via `mdx_kim2`.
- `--num-speakers` – Override the diarization speaker count when the automatic estimate needs correction.
- `--resume` – Reuse cached JSON checkpoints to avoid recomputing completed stages.
- `--hotwords-file`, `--initial-prompt-file`, `--spelling-map` – Provide customization files for ASR biasing and post-processing.

The script automatically chooses an output directory (`textN` alongside the audio) unless `--outdir` is supplied.

## Outputs
Each run writes artifacts prefixed by the audio stem inside the resolved output directory:
- `<name>.srt` – Subtitle file with speaker tags and timestamps for video players.
- `<name>.vtt` – WebVTT subtitles suitable for browsers and web editors.
- `<name>.txt` – Plain text transcript for quick review.
- `<name>.json` – Structured output containing word-level timings and speaker annotations for programmatic use.

Intermediate JSON checkpoints (e.g., `*_fw_segments_scrubbed.json`) are also saved when applicable, enabling resume and debugging workflows.

## Troubleshooting
- **Missing Hugging Face access** – If the script stops with `[HF] Missing HUGGINGFACE_TOKEN`, generate or renew a token on Hugging Face, grant it access to the pyannote models, and export it before retrying.
- **Diarization failures** – Poor audio quality or aggressive preprocessing can yield empty diarization tracks. Try disabling vocal extraction, supplying `--num-speakers`, or verifying the token has pyannote access.
- **UVR5 extraction errors** – Confirm the UVR5 CLI is installed and callable when using `--vocal-extract mdx_kim2`; fall back to `bandpass` or `off` otherwise.

With the prerequisites satisfied, a single CLI invocation handles the entire transcription-to-diarization pipeline end-to-end.
