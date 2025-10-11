from __future__ import annotations

import subprocess

from dnd_session_transcribe.adapters import ffmpeg as ffmpeg_module


def test_ffmpeg_invokes_subprocess(monkeypatch):
    recorded: dict[str, object] = {}

    def fake_run(cmd, *, shell, check, stdout, stderr):  # type: ignore[no-untyped-def]
        recorded["cmd"] = cmd
        recorded["shell"] = shell
        recorded["check"] = check
        recorded["stdout"] = stdout
        recorded["stderr"] = stderr

    monkeypatch.setattr(subprocess, "run", fake_run)

    ffmpeg_module.ffmpeg("ffmpeg -version")

    assert recorded == {
        "cmd": "ffmpeg -version",
        "shell": True,
        "check": True,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.STDOUT,
    }


def test_ffmpeg_cut_formats_command(monkeypatch):
    calls: list[str] = []

    def fake_ffmpeg(cmd: str) -> None:
        calls.append(cmd)

    monkeypatch.setattr(ffmpeg_module, "ffmpeg", fake_ffmpeg)

    ffmpeg_module.ffmpeg_cut("input path.wav", start=1.2349, end=5.4321, out_wav="out path.wav")

    assert calls == [
        "ffmpeg -y -ss 1.235 -to 5.432 -i 'input path.wav' -ac 1 -ar 16000 -sample_fmt s16 'out path.wav'",
    ]
