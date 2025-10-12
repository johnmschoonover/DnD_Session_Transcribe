import asyncio
import json
from pathlib import Path

from dnd_session_transcribe import web


def test_safe_filename_sanitizes_paths() -> None:
    assert web.safe_filename("../../bad name.wav") == "bad_name.wav"
    assert web.safe_filename("My Audio File.mp3") == "My_Audio_File.mp3"
    assert web.safe_filename("") == "upload"


def test_build_cli_args_defaults(tmp_path: Path) -> None:
    audio = tmp_path / "input.wav"
    audio.touch()
    outdir = tmp_path / "outputs"
    args = web.build_cli_args(audio, outdir=outdir)

    assert args.audio == str(audio)
    assert args.outdir == str(outdir)
    assert args.resume is False
    assert args.precise_rerun is False
    assert args.hotwords_file is None
    assert args.log_level == web.cli.LOG.level
    assert args.preview_start is None
    assert args.preview_duration is None
    assert args.preview_output is None


def test_build_cli_args_with_preview(tmp_path: Path) -> None:
    audio = tmp_path / "input.wav"
    audio.touch()
    outdir = tmp_path / "outputs"
    preview_copy = tmp_path / "custom_preview.wav"

    args = web.build_cli_args(
        audio,
        outdir=outdir,
        preview_start=1.5,
        preview_duration=5,
        preview_output=preview_copy,
    )

    assert args.preview_start == 1.5
    assert args.preview_duration == 5
    assert args.preview_output == str(preview_copy)


def test_create_app_uses_supplied_directory(tmp_path: Path) -> None:
    base_dir = tmp_path / "runs"
    app = web.create_app(base_dir)
    assert app.state.runs_dir == base_dir.resolve()


def test_show_job_lists_outputs_from_reported_directory(tmp_path: Path) -> None:
    app = web.create_app(tmp_path)

    job_id = "job-123"
    job_dir = tmp_path / job_id
    output_dir = job_dir / "preview_outputs"
    output_dir.mkdir(parents=True)

    preview_file = output_dir / "example_preview.wav"
    preview_file.write_bytes(b"not real audio")

    status = {
        "job_id": job_id,
        "status": "completed",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:10:00Z",
        "output_dir": str(output_dir),
        "error": None,
    }
    metadata = {
        "job_id": job_id,
        "created_at": "2024-01-01T00:00:00Z",
        "audio_filename": "sample.wav",
        "preview": {"requested": True, "start": 0.0, "duration": 10.0},
    }

    (job_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (job_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    show_job_route = next(
        route for route in app.routes if getattr(route, "path", None) == "/runs/{job_id}"
    )
    html = asyncio.run(show_job_route.endpoint(job_id=job_id))

    assert "example_preview.wav" in html
    assert "Preview snippet" in html
