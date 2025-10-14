"""Tests for the ASR feature module."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from dnd_session_transcribe.util.config import ASRConfig


@pytest.fixture()
def asr_module(monkeypatch):
    """Provide a reloaded ``asr`` module with a stubbed faster_whisper import."""

    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = object  # replaced within each test as needed
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

    import dnd_session_transcribe.features.asr as asr

    asr = importlib.reload(asr)
    return asr


def test_run_asr_uses_cached_segments_when_resume(asr_module, tmp_path, monkeypatch):
    """``run_asr`` should reuse cached segments when resume is enabled."""

    out_base = tmp_path / "session"
    cached_segments = [
        {"id": 0, "start": 0.0, "end": 1.5, "text": "hello", "avg_logprob": None},
        {"id": 1, "start": 1.5, "end": 3.0, "text": "world", "avg_logprob": -0.5},
    ]
    raw_path = Path(f"{out_base}_fw_segments_raw.json")
    raw_path.write_text(json.dumps({"segments": cached_segments}), encoding="utf-8")

    def _unexpected_whisper(*_args, **_kwargs):  # pragma: no cover - defensive
        raise AssertionError("WhisperModel should not be instantiated when resuming")

    monkeypatch.setattr(asr_module, "WhisperModel", _unexpected_whisper)

    cfg = ASRConfig()
    result = asr_module.run_asr(
        audio_path="dummy.wav",
        out_base=out_base,
        asr_cfg=cfg,
        hotwords=None,
        init_prompt=None,
        resume=True,
        total_sec=10.0,
    )

    assert result == cached_segments


def test_run_asr_retries_with_float32_on_init_failure(asr_module, tmp_path, monkeypatch):
    """``run_asr`` retries model init with float32 and persists emitted segments."""

    sentinel_vad = {"sentinel": "vad"}
    monkeypatch.setattr(asr_module, "build_vad_params", lambda *_args, **_kwargs: sentinel_vad)

    class DummyTqdm:
        def __init__(self, *args, **kwargs):
            self.n = 0

        def update(self, n=1):
            self.n += n

        def refresh(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(asr_module, "tqdm", DummyTqdm)

    instantiate_requests = []
    instance_holder: dict[str, object] = {}

    class DummyWhisper:
        def __init__(self):
            self.calls = []

        def transcribe(self, audio_path, **kwargs):
            self.calls.append((audio_path, kwargs))
            segments = [
                SimpleNamespace(start=0.0, end=1.25, text="Hello there"),
                SimpleNamespace(start=1.25, end=2.5, text="General Kenobi"),
            ]
            return segments, {"language": "en"}

    def fake_whisper(model, device, compute_type):
        instantiate_requests.append(
            {
                "model": model,
                "device": device,
                "compute_type": compute_type,
            }
        )
        if len(instantiate_requests) == 1:
            raise RuntimeError("init failure")
        instance = DummyWhisper()
        instance_holder["instance"] = instance
        return instance

    monkeypatch.setattr(asr_module, "WhisperModel", fake_whisper)

    cfg = ASRConfig()
    out_base = tmp_path / "session"
    hotwords = "orc goblin"
    init_prompt = "Once upon a time"

    result = asr_module.run_asr(
        audio_path="dummy.wav",
        out_base=out_base,
        asr_cfg=cfg,
        hotwords=hotwords,
        init_prompt=init_prompt,
        resume=False,
        total_sec=3.0,
    )

    assert [call["compute_type"] for call in instantiate_requests] == [cfg.compute_type, "float32"]

    instance = instance_holder["instance"]
    assert isinstance(instance, DummyWhisper)
    assert len(instance.calls) == 1
    transcribe_path, transcribe_kwargs = instance.calls[0]
    assert transcribe_path == "dummy.wav"
    assert transcribe_kwargs["vad_parameters"] is sentinel_vad
    assert transcribe_kwargs["hotwords"] == hotwords
    assert transcribe_kwargs["initial_prompt"] == init_prompt
    assert transcribe_kwargs["beam_size"] == cfg.beam_size
    assert transcribe_kwargs["vad_filter"] == cfg.use_vad

    raw_path = Path(f"{out_base}_fw_segments_raw.json")
    with raw_path.open(encoding="utf-8") as fh:
        persisted = json.load(fh)

    assert persisted["segments"] == result
    assert [seg["text"] for seg in result] == ["Hello there", "General Kenobi"]


def test_run_asr_exits_when_same_line_repeats(asr_module, tmp_path, monkeypatch):
    """``run_asr`` should exit early when the decoder loops on identical text."""

    sentinel_vad = {"sentinel": "vad"}
    monkeypatch.setattr(asr_module, "build_vad_params", lambda *_args, **_kwargs: sentinel_vad)

    created_bars = []

    class DummyTqdm:
        def __init__(self, *args, **kwargs):
            self.n = 0
            self.closed = False
            created_bars.append(self)

        def update(self, n=1):
            self.n += n

        def refresh(self):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(asr_module, "tqdm", DummyTqdm)

    repeated_text = "Looping line"

    class DummyWhisper:
        def transcribe(self, *_args, **_kwargs):
            def _iter():
                for i in range(8):
                    yield SimpleNamespace(start=float(i), end=float(i + 1), text=repeated_text)

            return _iter(), {"language": "en"}

    monkeypatch.setattr(asr_module, "WhisperModel", lambda *_args, **_kwargs: DummyWhisper())

    cfg = ASRConfig()
    out_base = tmp_path / "session"

    with pytest.raises(SystemExit) as excinfo:
        asr_module.run_asr(
            audio_path="dummy.wav",
            out_base=out_base,
            asr_cfg=cfg,
            hotwords="dragon, lich",
            init_prompt="You are the narrator",
            resume=False,
            total_sec=10.0,
        )

    message = str(excinfo.value)
    assert "Detected the same line 5 times" in message
    assert repeated_text in message
    assert "vad_min_speech_ms=250" in message
    assert "hotwords_length" in message
    assert "initial_prompt_length" in message
    assert all(bar.closed for bar in created_bars)
