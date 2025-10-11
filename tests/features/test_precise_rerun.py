from __future__ import annotations

import importlib
import sys
import types

import pytest


@pytest.fixture
def precise_module(monkeypatch):
    class _Sentinel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("WhisperModel not patched")

    stub_fw = types.SimpleNamespace(WhisperModel=_Sentinel)
    monkeypatch.setitem(sys.modules, "faster_whisper", stub_fw)

    module = importlib.import_module("dnd_session_transcribe.features.precise_rerun")
    return importlib.reload(module)


def test_rerun_precise_splits_windows_and_cleans_tempfiles(precise_module, monkeypatch):
    cut_calls: list[tuple[str, float, float, str]] = []
    removed: list[str] = []

    def fake_cut(src: str, start: float, end: float, out_path: str) -> None:
        cut_calls.append((src, start, end, out_path))

    class DummySegment:
        def __init__(self, start: float, end: float, text: str) -> None:
            self.start = start
            self.end = end
            self.text = text
            self.avg_logprob = None
            self.compression_ratio = None
            self.no_speech_prob = None

    class DummyModel:
        def __init__(self, model: str, device: str, compute_type: str) -> None:
            self.calls: list[tuple[str, str, str]] = [(model, device, compute_type)]

        def transcribe(self, path: str, **kwargs):
            return [DummySegment(0.0, 0.5, f"segment@{path}")], {}

    def fake_remove(path: str) -> None:
        removed.append(path)
        raise OSError("busy")

    monkeypatch.setattr(precise_module, "ffmpeg_cut", fake_cut)
    monkeypatch.setattr(precise_module.os, "remove", fake_remove)
    monkeypatch.setattr(precise_module, "WhisperModel", DummyModel)

    spans = [(0.0, 3.0)]

    result = precise_module.rerun_precise_on_spans(
        src_audio="story.wav",
        spans=spans,
        lang="en",
        model="tiny",
        compute="float32",
        device="cpu",
        beam=5,
        patience=1.0,
        max_window_s=1.0,
    )

    assert cut_calls, "ffmpeg_cut should be invoked for each chunk"
    assert removed, "cleanup should attempt to remove temporary files"

    assert result
    start, end, segments = result[0]
    assert start == 0.0 and end == pytest.approx(1.0)
    assert segments[0]["text"].startswith("segment@")
