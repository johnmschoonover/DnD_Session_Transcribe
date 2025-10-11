from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def asr_module(monkeypatch):
    class _Sentinel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("WhisperModel not patched")

    stub_fw = types.SimpleNamespace(WhisperModel=_Sentinel)
    monkeypatch.setitem(sys.modules, "faster_whisper", stub_fw)

    module = importlib.import_module("dnd_session_transcribe.features.asr")
    return importlib.reload(module)


def test_run_asr_uses_cached_segments(asr_module, monkeypatch, tmp_path):
    cache_path = tmp_path / "session_fw_segments_raw.json"
    cache_path.write_text(json.dumps({"segments": [{"id": 1, "text": "hi"}]}), encoding="utf-8")

    def fail(*args, **kwargs):
        raise AssertionError("resume should bypass inference")

    monkeypatch.setattr(asr_module, "WhisperModel", fail)

    cfg = asr_module.ASRConfig(model="tiny", device="cpu", compute_type="int8")

    segments = asr_module.run_asr(
        audio_path="audio.wav",
        out_base=tmp_path / "session",
        asr_cfg=cfg,
        hotwords=None,
        init_prompt=None,
        resume=True,
        total_sec=1.0,
    )

    assert segments == [{"id": 1, "text": "hi"}]


def test_run_asr_initializes_model_with_retry(asr_module, monkeypatch, tmp_path):
    calls: list[tuple[str, str, str]] = []

    class DummySegment:
        def __init__(self, start: float, end: float, text: str):
            self.start = start
            self.end = end
            self.text = text
            self.avg_logprob = -0.5
            self.compression_ratio = 0.9
            self.no_speech_prob = 0.01

    class DummyModel:
        def __init__(self, model: str, device: str, compute_type: str):
            calls.append((model, device, compute_type))
            if compute_type != "float32":
                raise RuntimeError("bad precision")

        def transcribe(self, *args, **kwargs):
            return [
                DummySegment(0.0, 1.25, "hello there"),
                DummySegment(1.25, 1.25, "echo"),
            ], {"language": "en"}

    monkeypatch.setattr(asr_module, "WhisperModel", DummyModel)

    cfg = asr_module.ASRConfig(model="tiny", device="cpu", compute_type="float16", beam_size=5)

    result = asr_module.run_asr(
        audio_path="clip.wav",
        out_base=tmp_path / "clip",
        asr_cfg=cfg,
        hotwords="sword",
        init_prompt="Hear ye",
        resume=False,
        total_sec=5.0,
    )

    assert calls == [("tiny", "cpu", "float16"), ("tiny", "cpu", "float32")]
    assert result == [
        {
            "id": 0,
            "start": 0.0,
            "end": 1.25,
            "text": "hello there",
            "avg_logprob": -0.5,
            "compression_ratio": 0.9,
            "no_speech_prob": 0.01,
        },
        {
            "id": 1,
            "start": 1.25,
            "end": 1.25,
            "text": "echo",
            "avg_logprob": -0.5,
            "compression_ratio": 0.9,
            "no_speech_prob": 0.01,
        },
    ]

    saved = Path(f"{tmp_path / 'clip'}_fw_segments_raw.json")
    assert saved.exists()
    with saved.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload == {"segments": result}
