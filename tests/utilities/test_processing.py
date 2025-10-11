import importlib
import sys
import types

import pytest

from dnd_session_transcribe.util.config import ScrubConfig


def _load_processing(monkeypatch, diarize_cls=None, top_level_cls=None):
    module_name = "dnd_session_transcribe.util.processing"

    # Ensure a clean import each time the helper is invoked.
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.delitem(sys.modules, "whisperx", raising=False)
    monkeypatch.delitem(sys.modules, "whisperx.diarize", raising=False)

    torch_stub = types.ModuleType("torch")

    class _Device:
        def __init__(self, name):
            self.name = name

        def __repr__(self) -> str:  # pragma: no cover - repr aids debugging
            return f"device({self.name})"

    torch_stub.device = _Device
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    whisperx_stub = types.ModuleType("whisperx")
    if diarize_cls is not None:
        diarize_stub = types.ModuleType("whisperx.diarize")
        diarize_stub.DiarizationPipeline = diarize_cls
        monkeypatch.setitem(sys.modules, "whisperx.diarize", diarize_stub)
        whisperx_stub.diarize = diarize_stub
    if top_level_cls is not None:
        whisperx_stub.DiarizationPipeline = top_level_cls
    monkeypatch.setitem(sys.modules, "whisperx", whisperx_stub)

    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    return module, torch_stub, whisperx_stub


@pytest.fixture
def processing_module(monkeypatch):
    module, _, _ = _load_processing(monkeypatch, diarize_cls=lambda *args, **kwargs: None)
    return module


def test_make_diarization_pipeline_prefers_nested_class(monkeypatch):
    class NestedPipeline:
        def __init__(self, use_auth_token, device):
            self.use_auth_token = use_auth_token
            self.device = device

    module, torch_stub, whisperx_stub = _load_processing(
        monkeypatch, diarize_cls=NestedPipeline
    )
    # Guard against accidentally falling back to the top-level attribute.
    whisperx_stub.DiarizationPipeline = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("should not use top-level fallback")
    )

    pipeline = module.make_diarization_pipeline("hf_token", "cpu")

    assert isinstance(pipeline, NestedPipeline)
    assert pipeline.use_auth_token == "hf_token"
    assert isinstance(pipeline.device, torch_stub.device)
    assert pipeline.device.name == "cpu"


def test_make_diarization_pipeline_falls_back_to_top_level(monkeypatch):
    class TopLevelPipeline:
        def __init__(self, use_auth_token, device):
            self.args = (use_auth_token, device)

    module, torch_stub, _ = _load_processing(
        monkeypatch, top_level_cls=TopLevelPipeline
    )

    device_obj = torch_stub.device("cuda")
    pipeline = module.make_diarization_pipeline("token", device_obj)

    assert isinstance(pipeline, TopLevelPipeline)
    assert pipeline.args == ("token", device_obj)


def test_make_diarization_pipeline_raises_when_no_options(monkeypatch):
    class BrokenTopLevel:
        def __init__(self, *_, **__):
            raise RuntimeError("boom")

    module, _, _ = _load_processing(monkeypatch, top_level_cls=BrokenTopLevel)

    with pytest.raises(RuntimeError) as excinfo:
        module.make_diarization_pipeline("token", "cpu")

    message = str(excinfo.value)
    assert "Could not construct WhisperX diarization pipeline" in message
    assert "whisperx.DiarizationPipeline" in message


def test_scrub_segments_filters_problematic_entries(processing_module):
    cfg = ScrubConfig(
        min_segment_len_s=0.5,
        drop_if_avg_logprob_lt=-0.2,
        drop_if_compratio_gt=1.2,
        drop_if_nospeech_gt=0.5,
        unique_char_ratio_min=0.3,
    )
    segments = [
        {"start": 0.0, "end": 0.3, "text": "tiny"},
        {"start": 0.0, "end": 2.0, "text": "avg", "avg_logprob": -0.3},
        {"start": 0.0, "end": 2.0, "text": "compr", "compression_ratio": 1.5},
        {"start": 0.0, "end": 2.0, "text": "ns", "no_speech_prob": 0.7},
        {"start": 0.0, "end": 3.0, "text": "aaaaaaaaaabbbb"},
        {"start": 1.0, "end": 2.0, "text": "Keep one", "avg_logprob": -0.1},
        {"start": 3.0, "end": 4.0, "text": "Also keep", "avg_logprob": 0.0},
    ]

    result = processing_module.scrub_segments(segments, cfg)

    assert result == [segments[5], segments[6]]


def test_find_hard_spans_merges_and_pads(processing_module):
    segments = [
        {"start": 0.0, "end": 1.0, "avg_logprob": -2.0},
        {"start": 1.8, "end": 2.2, "compression_ratio": 3.0},
        {"start": 3.0, "end": 3.5, "no_speech_prob": 0.9},
        {"start": 7.9, "end": 8.3, "avg_logprob": -2.5},
        {"start": 8.6, "end": 9.0, "avg_logprob": -0.5},
    ]

    spans = processing_module.find_hard_spans(
        segments,
        dur=9.0,
        logprob_thr=-1.0,
        cr_thr=2.5,
        nospeech_thr=0.8,
        pad=0.25,
        merge_gap=1.0,
    )

    assert spans == [(0.0, 3.75), (7.65, 8.55)]

    none_spans = processing_module.find_hard_spans(
        segments,
        dur=9.0,
        logprob_thr=-5.0,
        cr_thr=5.0,
        nospeech_thr=1.0,
    )
    assert none_spans == []


def test_splice_segments_replaces_overlaps(processing_module):
    original = [
        {"id": 0, "start": 0.0, "end": 0.4, "text": "intro"},
        {"id": 1, "start": 0.6, "end": 1.2, "text": "mid"},
        {"id": 2, "start": 2.5, "end": 3.0, "text": "end"},
    ]
    replacements = [
        (0.5, 1.5, [{"id": 99, "start": 0.7, "end": 1.0, "text": "new_mid"}]),
        (2.4, 3.5, [{"id": 100, "start": 2.6, "end": 3.2, "text": "new_end"}]),
    ]

    merged = processing_module.splice_segments(original, replacements)

    assert [seg["text"] for seg in merged] == ["intro", "new_mid", "new_end"]
    assert [seg["id"] for seg in merged] == [0, 1, 2]


def test_clamp_to_duration_adjusts_and_drops(processing_module):
    segments = [
        {"id": 10, "start": -0.2, "end": 0.2, "text": "before"},
        {"id": 11, "start": 4.8, "end": 5.5, "text": "tail"},
        {"id": 12, "start": 5.0, "end": 5.5, "text": "beyond"},
        {"id": 13, "start": 0.5, "end": 0.6, "text": "mid"},
        {"id": 14, "start": 0.0, "end": 0.0, "text": "zero"},
    ]

    clamped = processing_module.clamp_to_duration(segments, dur=5.0)

    assert len(clamped) == 3
    assert clamped[0]["start"] == pytest.approx(0.0)
    assert clamped[0]["end"] == pytest.approx(0.2)
    assert clamped[1]["start"] == pytest.approx(4.8)
    assert clamped[1]["end"] < 5.0
    assert clamped[2]["start"] == pytest.approx(0.5)
    assert clamped[2]["end"] == pytest.approx(0.6)
