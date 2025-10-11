from __future__ import annotations

import subprocess

from dnd_session_transcribe.adapters.ffmpeg import ffmpeg, ffmpeg_cut


def test_ffmpeg_invokes_subprocess(monkeypatch):
    called = {}

    def fake_run(cmd, *, shell, check, stdout, stderr):
        called["args"] = (cmd, shell, check, stdout, stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)

    ffmpeg("echo hi")

    assert called["args"] == (
        "echo hi",
        True,
        True,
        subprocess.DEVNULL,
        subprocess.STDOUT,
    )


def test_ffmpeg_cut_formats_command(monkeypatch):
    recorded = {}

    monkeypatch.setattr("dnd_session_transcribe.adapters.ffmpeg.ffmpeg", lambda cmd: recorded.setdefault("cmd", cmd))

    ffmpeg_cut("/path/with space.wav", 1.234, 5.678, "/tmp/out.wav")

    expected = (
        "ffmpeg -y -ss 1.234 -to 5.678 -i '/path/with space.wav' "
        "-ac 1 -ar 16000 -sample_fmt s16 /tmp/out.wav"
    )
    assert recorded["cmd"] == expected
