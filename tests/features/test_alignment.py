from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def alignment_module(monkeypatch):
    stub_whisperx = types.SimpleNamespace(
        load_align_model=lambda **_: (object(), {}),
        align=lambda *args, **kwargs: {},
    )
    monkeypatch.setitem(sys.modules, "whisperx", stub_whisperx)

    module = importlib.import_module("dnd_session_transcribe.features.alignment")
    return importlib.reload(module), stub_whisperx


def test_run_alignment_uses_cached_result(alignment_module, monkeypatch, tmp_path):
    module, stub_whisperx = alignment_module

    aligned_path = tmp_path / "clip_aligned.json"
    aligned_path.write_text(json.dumps({"segments": [1, 2, 3]}), encoding="utf-8")

    def fail(*args, **kwargs):
        raise AssertionError("cache path should avoid new inference")

    stub_whisperx.load_align_model = fail  # type: ignore[attr-defined]

    result = module.run_alignment([], "audio.wav", "cpu", tmp_path / "clip", resume=True)

    assert result == {"segments": [1, 2, 3]}


def test_run_alignment_invokes_whisperx_and_persists(alignment_module, monkeypatch, tmp_path):
    module, stub_whisperx = alignment_module

    called = {}

    def fake_load_align_model(language_code: str, device: str):
        called["load_align_model"] = (language_code, device)
        return object(), {"lang": language_code}

    def fake_align(segments, model, metadata, audio_path, *, device):
        called["align"] = {
            "segments": segments,
            "metadata": metadata,
            "audio_path": audio_path,
            "device": device,
        }
        return {"segments": segments, "meta": metadata}

    stub_whisperx.load_align_model = fake_load_align_model  # type: ignore[attr-defined]
    stub_whisperx.align = fake_align  # type: ignore[attr-defined]

    base = tmp_path / "demo"
    segments = [{"start": 0.0, "end": 1.0, "text": "hi"}]

    result = module.run_alignment(segments, "demo.wav", "cpu", base, resume=False)

    assert called["load_align_model"] == ("en", "cpu")
    assert called["align"]["segments"] == segments
    assert called["align"]["metadata"] == {"lang": "en"}

    out_file = Path(f"{base}_aligned.json")
    assert out_file.exists()
    assert json.loads(out_file.read_text(encoding="utf-8")) == result
