from __future__ import annotations

import argparse
import asyncio
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.datastructures import UploadFile

from dnd_session_transcribe.web.services import jobs as job_services


@pytest.fixture(autouse=True)
def _patch_cli_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a predictable CLI configuration for job tests."""

    monkeypatch.setattr(job_services.cli, "ASR", SimpleNamespace(
        hotwords_file=None,
        initial_prompt_file=None,
        model="tiny",
        device="cuda",
        compute_type="float16",
    ))
    monkeypatch.setattr(job_services.cli, "PRE", SimpleNamespace(vocal_extract="off"))
    monkeypatch.setattr(job_services.cli, "DIA", SimpleNamespace(num_speakers=None))
    monkeypatch.setattr(job_services.cli, "PREC", SimpleNamespace(
        enabled=True,
        model="tiny",
        device="cuda",
        compute_type="float16",
    ))
    monkeypatch.setattr(job_services.cli, "LOG", SimpleNamespace(level="WARNING"))


def test_resolve_selection_normalizes_sentinel_values() -> None:
    mode, choices = job_services._resolve_selection("ALL", ("one", "two"))
    assert mode == "all"
    assert choices == ["one", "two"]

    mode, choices = job_services._resolve_selection("Random", ("alpha", "beta"))
    assert mode == "random"
    assert choices == ["alpha", "beta"]


def test_resolve_selection_matches_allowed_case_insensitively() -> None:
    mode, choices = job_services._resolve_selection("CPU", ("", "cpu", "cuda"))
    assert mode == "single"
    assert choices == ["cpu"]


@pytest.mark.asyncio()
async def test_job_runner_success(tmp_path: Path) -> None:
    loop = asyncio.get_running_loop()

    def fake_runner(args: argparse.Namespace, *, configure_logging: bool, log_handlers: list[Any]):
        return Path(args.outdir)

    runner = job_services.JobRunner(loop_factory=lambda: loop, transcription_runner=fake_runner)
    job_dir = tmp_path / "job-success"
    job_dir.mkdir()
    args = job_services.build_cli_args(job_dir / "audio.wav", outdir=job_dir / "outputs")
    future = runner.submit(args, "job-success", job_dir, "2024-01-01T00:00:00Z")
    await future

    status = json.loads((job_dir / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["output_dir"].endswith("outputs")


@pytest.mark.asyncio()
async def test_job_runner_failure(tmp_path: Path) -> None:
    loop = asyncio.get_running_loop()

    def failing_runner(*args: Any, **kwargs: Any) -> None:  # type: ignore[return-value]
        raise RuntimeError("boom")

    runner = job_services.JobRunner(loop_factory=lambda: loop, transcription_runner=failing_runner)
    job_dir = tmp_path / "job-failure"
    job_dir.mkdir()
    args = job_services.build_cli_args(job_dir / "audio.wav", outdir=job_dir / "outputs")
    future = runner.submit(args, "job-failure", job_dir, "2024-01-01T00:00:00Z")
    await future

    status = json.loads((job_dir / "status.json").read_text())
    assert status["status"] == "failed"
    assert "boom" in status["error"]


class _StubRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[argparse.Namespace, str, Path, str]] = []

    def submit(self, args: argparse.Namespace, job_id: str, job_dir: Path, created_at: str):
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        self.calls.append((args, job_id, job_dir, created_at))
        return fut


@pytest.mark.asyncio()
async def test_job_service_schedule_single_job(tmp_path: Path) -> None:
    runner = _StubRunner()
    service = job_services.JobService(tmp_path, runner=runner)

    upload = UploadFile(filename="session.wav", file=io.BytesIO(b"audio"))
    form_items = [
        ("job-0-log_level", "WARNING"),
        ("job-0-ram", "true"),
        ("job-0-resume", "true"),
        ("job-0-precise_rerun", "false"),
    ]

    result = await service.schedule_jobs(form_items, upload)

    assert len(result.job_ids) == 1
    assert result.skipped_ids == []
    assert runner.calls[0][1] == result.job_ids[0]

    job_dir = tmp_path / result.job_ids[0]
    assert (job_dir / "metadata.json").exists()
    assert (job_dir / "status.json").exists()
