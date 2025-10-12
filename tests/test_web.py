import asyncio
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

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
    text_file = output_dir / "notes final.txt"
    text_file.write_text("hello", encoding="utf-8")
    (job_dir / "job.log").write_text("log entry", encoding="utf-8")

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
        "settings": {"log_level": "INFO", "resume": False},
    }

    (job_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (job_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    show_job_route = next(
        route for route in app.routes if getattr(route, "path", None) == "/runs/{job_id}"
    )
    html = asyncio.run(show_job_route.endpoint(job_id=job_id))

    assert "example_preview.wav" in html
    assert "Preview snippet" in html
    assert 'href="/runs/job-123/files/preview_outputs/example_preview.wav"' in html
    assert 'href="/runs/job-123/files/preview_outputs/notes%20final.txt"' in html
    assert 'href="/runs/job-123/log"' in html
    assert 'audio controls src="/runs/job-123/files/preview_outputs/example_preview.wav"' in html
    assert "Settings" in html
    assert "log_level" in html


def test_transcribe_redirects_home_and_lists_job(
    tmp_path: Path, monkeypatch
) -> None:
    app = web.create_app(tmp_path)

    def fake_run_transcription(args, configure_logging=False, log_handlers=None):  # type: ignore[override]
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "result.txt").write_text("done", encoding="utf-8")
        return outdir

    monkeypatch.setattr(web.cli, "run_transcription", fake_run_transcription)

    with TestClient(app, follow_redirects=False) as client:
        response = client.post(
            "/transcribe",
            data={
                "job-0-log_level": web.cli.LOG.level,
                "job-0-preview_start": "",
                "job-0-preview_duration": "",
            },
            files={"audio_file": ("session.wav", b"fake-bytes", "audio/wav")},
        )

        assert response.status_code == 303
        assert response.headers["location"].startswith("/?message=Started+job+")

        job_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert job_dirs, "expected job directory to be created"
        job_dir = job_dirs[0]
        status_path = job_dir / "status.json"
        assert status_path.exists()

        metadata_path = job_dir / "metadata.json"
        assert metadata_path.exists()

        status = None
        for _ in range(10):
            content = status_path.read_text(encoding="utf-8")
            if content.strip():
                status = json.loads(content)
                break
            time.sleep(0.05)

        assert status is not None, "status.json remained empty"
        assert status["job_id"] == job_dir.name
        assert status["status"] in {"running", "completed"}

        metadata_content = metadata_path.read_text(encoding="utf-8")
        metadata = json.loads(metadata_content)
        assert "settings" in metadata
        settings = metadata["settings"]
        assert settings["log_level"] == web.cli.LOG.level
        assert settings["resume"] is False
        assert settings["preview_requested"] is False
        assert settings["batch_index"] == 0
        assert settings["job_index"] == "0"
        assert metadata["batch_index"] == 0
        assert metadata["job_index"] == "0"

        dashboard_html = client.get("/").text
        link_snippet = (
            f'<a href="/runs/{job_dir.name}" target="_blank" rel="noopener noreferrer">'
        )
        assert link_snippet in dashboard_html
        assert job_dir.name in dashboard_html


def test_transcribe_preview_updates_output_directory(
    tmp_path: Path, monkeypatch
) -> None:
    app = web.create_app(tmp_path)

    def fake_run_transcription(args, configure_logging=False, log_handlers=None):  # type: ignore[override]
        requested_outdir = Path(args.outdir)
        preview_dir = requested_outdir.parent / f"preview_{requested_outdir.name}"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_file = preview_dir / "session_preview.wav"
        preview_file.write_text("preview", encoding="utf-8")
        return preview_dir

    monkeypatch.setattr(web.cli, "run_transcription", fake_run_transcription)

    with TestClient(app, follow_redirects=False) as client:
        response = client.post(
            "/transcribe",
            data={
                "job-0-log_level": web.cli.LOG.level,
                "job-0-preview_enabled": "1",
                "job-0-preview_start": "2",
                "job-0-preview_duration": "5",
            },
            files={"audio_file": ("session.wav", b"fake-bytes", "audio/wav")},
        )

        assert response.status_code == 303

        job_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert job_dirs, "expected job directory to be created"
        job_dir = job_dirs[0]
        status_path = job_dir / "status.json"

        status: dict[str, object] | None = None
        for _ in range(50):
            raw = status_path.read_text(encoding="utf-8")
            if raw.strip():
                parsed = json.loads(raw)
                if parsed.get("status") == "completed":
                    status = parsed
                    break
            time.sleep(0.05)

        assert status is not None, "job did not complete"
        expected_preview_dir = job_dir / "preview_outputs"
        assert status["output_dir"] == str(expected_preview_dir)

        job_page = client.get(f"/runs/{job_dir.name}").text
        assert "session_preview.wav" in job_page
        assert f"/runs/{job_dir.name}/files/preview_outputs/session_preview.wav" in job_page


def test_transcribe_schedules_multiple_jobs(tmp_path: Path, monkeypatch) -> None:
    app = web.create_app(tmp_path)

    recorded_args: list[object] = []

    def fake_run_transcription(args, configure_logging=False, log_handlers=None):  # type: ignore[override]
        recorded_args.append(args)
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "result.txt").write_text("done", encoding="utf-8")
        return outdir

    monkeypatch.setattr(web.cli, "run_transcription", fake_run_transcription)

    alternate_log_level = next(
        (level for level in web.cli.LOG_LEVELS if level != web.cli.LOG.level),
        web.cli.LOG.level,
    )

    with TestClient(app, follow_redirects=False) as client:
        response = client.post(
            "/transcribe",
            data={
                "job-0-log_level": web.cli.LOG.level,
                "job-0-asr_model": "tiny",
                "job-1-log_level": alternate_log_level,
                "job-1-num_speakers": "3",
                "job-1-vocal_extract": "bandpass",
                "job-1-precise_rerun": "true",
                "job-1-preview_enabled": "true",
                "job-1-preview_start": "1:30",
                "job-1-preview_duration": "8",
            },
            files={"audio_file": ("session.wav", b"fake-bytes", "audio/wav")},
        )

        assert response.status_code == 303
        assert "Started+jobs+" in response.headers["location"]

        job_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(job_dirs) == 2

        metadata_by_batch: dict[int, dict[str, object]] = {}
        for job_dir in job_dirs:
            inputs = list((job_dir / "inputs").iterdir())
            assert inputs and inputs[0].name == "session.wav"

            status_path = job_dir / "status.json"
            for _ in range(50):
                raw = status_path.read_text(encoding="utf-8")
                if raw.strip():
                    parsed = json.loads(raw)
                    if parsed.get("status") == "completed":
                        break
                time.sleep(0.05)

            metadata_path = job_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_by_batch[int(metadata["batch_index"])] = metadata

        assert 0 in metadata_by_batch and 1 in metadata_by_batch

        first_job = metadata_by_batch[0]
        assert first_job["settings"]["asr_model"] == "tiny"
        assert first_job["settings"]["preview_requested"] is False
        assert first_job["settings"]["batch_index"] == 0
        assert first_job["settings"]["job_index"] == "0"

        second_job = metadata_by_batch[1]
        assert second_job["settings"]["log_level"] == alternate_log_level
        assert second_job["settings"]["num_speakers"] == 3
        assert second_job["settings"]["vocal_extract"] == "bandpass"
        assert second_job["settings"]["precise_rerun"] is True
        assert second_job["preview"]["requested"] is True
        assert second_job["preview"]["duration"] == 8.0
        assert second_job["settings"]["batch_index"] == 1
        assert second_job["settings"]["job_index"] == "1"

    assert len(recorded_args) == 2
    assert any(getattr(args, "log_level", None) == web.cli.LOG.level for args in recorded_args)
    assert any(
        getattr(args, "log_level", None) == alternate_log_level
        and getattr(args, "precise_rerun", None) is True
        and getattr(args, "num_speakers", None) == 3
        for args in recorded_args
    )
def test_delete_job_removes_directory(tmp_path: Path) -> None:
    app = web.create_app(tmp_path)
    job_id = "job-delete"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / "status.json").write_text(json.dumps({"job_id": job_id}), encoding="utf-8")
    (job_dir / "metadata.json").write_text(json.dumps({"job_id": job_id}), encoding="utf-8")
    (job_dir / "outputs").mkdir()

    delete_route = next(
        route for route in app.routes if getattr(route, "path", None) == "/runs/{job_id}/delete"
    )
    response = asyncio.run(delete_route.endpoint(job_id=job_id))

    assert response.status_code == 303
    assert response.headers["location"] == f"/?message=Deleted+job+{job_id}"
    assert not job_dir.exists()


def test_read_json_handles_partial_file(tmp_path: Path, caplog) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{\"status\":", encoding="utf-8")

    with caplog.at_level("WARNING"):
        parsed = web._read_json(path)

    assert parsed == {}
    assert "Failed to parse JSON" in caplog.text
