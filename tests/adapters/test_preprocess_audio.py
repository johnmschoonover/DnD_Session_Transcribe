import types

import pytest

from dnd_session_transcribe.adapters import preprocess_audio


@pytest.fixture
def fake_tmp(monkeypatch, tmp_path):
    output_path = tmp_path / "processed.wav"

    def fake_named_tempfile(*args, **kwargs):
        return types.SimpleNamespace(name=str(output_path))

    monkeypatch.setattr(preprocess_audio.tempfile, "NamedTemporaryFile", fake_named_tempfile)
    return output_path


def test_preprocess_off_returns_original(monkeypatch, tmp_path):
    original = tmp_path / "clip.wav"
    original.write_bytes(b"audio")

    def boom(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("ffmpeg should not be invoked when mode is off")

    monkeypatch.setattr(preprocess_audio, "ffmpeg", boom)

    result = preprocess_audio.preprocess_audio(str(original), mode="off")

    assert result == str(original)


def test_preprocess_bandpass_invokes_ffmpeg(fake_tmp, monkeypatch, tmp_path):
    original = tmp_path / "clip.wav"
    original.write_text("wave")
    called = {}

    def fake_ffmpeg(cmd):
        called["cmd"] = cmd

    monkeypatch.setattr(preprocess_audio, "ffmpeg", fake_ffmpeg)

    result = preprocess_audio.preprocess_audio(str(original), mode="bandpass")

    assert result == str(fake_tmp)
    assert called["cmd"].startswith("ffmpeg -y -i ")
    assert "highpass=f=50" in called["cmd"]
    assert "lowpass=f=7800" in called["cmd"]


def test_preprocess_mdx_falls_back_to_bandpass(caplog, fake_tmp, monkeypatch, tmp_path):
    original = tmp_path / "clip.wav"
    original.write_text("wave")
    called = {}

    def fake_ffmpeg(cmd):
        called["cmd"] = cmd

    monkeypatch.setattr(preprocess_audio, "ffmpeg", fake_ffmpeg)

    with caplog.at_level("WARNING"):
        result = preprocess_audio.preprocess_audio(str(original), mode="mdx_kim2")

    assert result == str(fake_tmp)
    assert "highpass=f=50" in called["cmd"]
    assert "mdx_kim2 mode has been removed" in caplog.text


def test_preprocess_unknown_mode_returns_original(caplog, monkeypatch, tmp_path):
    original = tmp_path / "clip.wav"
    original.write_bytes(b"audio")

    def boom(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("ffmpeg should not run for unknown modes")

    monkeypatch.setattr(preprocess_audio, "ffmpeg", boom)

    with caplog.at_level("WARNING"):
        result = preprocess_audio.preprocess_audio(str(original), mode="mystery")

    assert result == str(original)
    assert "Unknown preprocess mode" in caplog.text
