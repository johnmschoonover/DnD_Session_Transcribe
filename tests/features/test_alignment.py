import json
from pathlib import Path

import pytest

from dnd_session_transcribe.features.alignment import run_alignment


@pytest.fixture
def dummy_segments():
    return [{"text": "hello"}, {"text": "world"}]


def test_run_alignment_invokes_whisperx_and_persists_json(monkeypatch, tmp_path, dummy_segments):
    audio_path = "audio.wav"
    device = "cpu"
    out_base = tmp_path / "aligned/output"
    out_base.parent.mkdir(parents=True, exist_ok=True)
    sentinel = {"segments": [1, 2, 3]}

    load_calls = {}

    def fake_load_align_model(*, language_code, device):
        load_calls["language_code"] = language_code
        load_calls["device"] = device
        return "align_model", {"foo": "bar"}

    captured = {}

    def fake_align(segments, align_model, metadata, audio_path_arg, *, device):
        captured["segments"] = segments
        captured["align_model"] = align_model
        captured["metadata"] = metadata
        captured["audio_path"] = audio_path_arg
        captured["device"] = device
        return sentinel

    monkeypatch.setattr(
        "dnd_session_transcribe.features.alignment.whisperx.load_align_model",
        fake_load_align_model,
    )
    monkeypatch.setattr(
        "dnd_session_transcribe.features.alignment.whisperx.align",
        fake_align,
    )

    result = run_alignment(dummy_segments, audio_path, device, out_base, resume=False)

    assert load_calls == {"language_code": "en", "device": device}
    assert captured == {
        "segments": dummy_segments,
        "align_model": "align_model",
        "metadata": {"foo": "bar"},
        "audio_path": audio_path,
        "device": device,
    }
    assert result is sentinel

    aligned_path = Path(f"{out_base}_aligned.json")
    with aligned_path.open("r", encoding="utf-8") as fh:
        assert json.load(fh) == sentinel


def test_run_alignment_uses_cached_alignment_when_resume(monkeypatch, tmp_path, dummy_segments):
    out_base = tmp_path / "existing/base"
    aligned_path = Path(f"{out_base}_aligned.json")
    aligned_path.parent.mkdir(parents=True, exist_ok=True)
    cached = {"cached": True}
    aligned_path.write_text(json.dumps(cached), encoding="utf-8")
    before_stat = aligned_path.stat()

    def _fail(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("whisperx helper should not be called when resume=True")

    monkeypatch.setattr(
        "dnd_session_transcribe.features.alignment.whisperx.load_align_model",
        _fail,
    )
    monkeypatch.setattr(
        "dnd_session_transcribe.features.alignment.whisperx.align",
        _fail,
    )

    result = run_alignment(dummy_segments, "audio.wav", "cpu", out_base, resume=True)

    assert result == cached
    assert aligned_path.read_text(encoding="utf-8") == json.dumps(cached)
    assert aligned_path.stat().st_mtime == before_stat.st_mtime
    assert aligned_path.stat().st_size == before_stat.st_size
