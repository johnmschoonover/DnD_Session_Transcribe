from __future__ import annotations

import json
from pathlib import Path
import wave

import pandas as pd
from pandas import testing as pdt
import pytest

from dnd_session_transcribe.features.diarization import (
    normalize_diarization_to_df,
    run_diarization,
)
from dnd_session_transcribe.util.config import DiarizationConfig


class _DummyInnerPipeline:
    def __init__(self) -> None:
        self.applied_overrides: dict[str, dict[str, float | int]] | None = None

    def parameters(self) -> dict[str, dict[str, float | int]]:
        return {
            "segmentation": {},
            "speech_turn": {},
            "clustering": {},
        }

    def instantiate(self, overrides: dict[str, dict[str, float | int]]):
        # Mirror pyannote's instantiate API by returning a configured copy.
        self.applied_overrides = overrides
        return self


class _DummyDiarizationPipeline:
    def __init__(self, expected_audio: Path) -> None:
        self.pipeline = _DummyInnerPipeline()
        self._expected_audio = expected_audio
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        audio_path: str,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ):
        call = {
            "audio_path": Path(audio_path),
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }
        self.calls.append(call)

        # Always succeed for the expected clip; return alternating turns
        # resembling the bundled README dialogue.
        if call["audio_path"].resolve() != self._expected_audio:
            raise AssertionError("Unexpected audio path passed to pipeline")

        if min_speakers is not None or max_speakers is not None:
            return []  # we should not need the fallback path in this test

        assert num_speakers == 2, "run_diarization should request two speakers"
        return [
            {"start": 0.00, "end": 1.50, "label": "SPEAKER_00"},
            {"start": 1.50, "end": 3.00, "label": "SPEAKER_01"},
            {"start": 3.00, "end": 4.50, "label": "SPEAKER_00"},
            {"start": 4.50, "end": 6.00, "label": "SPEAKER_01"},
        ]


def _load_audio_duration(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


class _SegmentStub:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _AnnotationLike:
    def __init__(self, rows: list[tuple[float, float, str]]) -> None:
        self._rows = rows
        self.calls = 0

    def itertracks(self, *, yield_label: bool = False):
        self.calls += 1
        assert yield_label, "normalize_diarization_to_df should request labels"
        for start, end, label in self._rows:
            yield _SegmentStub(start, end), None, label


def test_normalize_diarization_handles_annotation_like_object():
    ann = _AnnotationLike([
        (0.0, 1.0, "SPEAKER_00"),
        (1.0, 2.5, "SPEAKER_01"),
    ])

    df = normalize_diarization_to_df(ann, audio_dur=5.0, speaker_prefix="PLAYER")

    assert ann.calls == 1
    expected = pd.DataFrame(
        [
            {"start": 0.0, "end": 1.0, "speaker": "PLAYER_00"},
            {"start": 1.0, "end": 2.5, "speaker": "PLAYER_01"},
        ]
    )
    pdt.assert_frame_equal(df.reset_index(drop=True), expected)


def test_normalize_diarization_handles_errors():
    class _BrokenAnnotation:
        def itertracks(self, *, yield_label: bool = False):
            raise RuntimeError("boom")

    df = normalize_diarization_to_df(_BrokenAnnotation(), audio_dur=2.0, speaker_prefix="NPC")

    assert df.empty


def test_sample_clip_diarization_uses_two_speakers(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    audio_path = repo_root / "sample_audio" / "test.wav"

    dummy_pipeline = _DummyDiarizationPipeline(audio_path.resolve())

    def _fake_make_diarization_pipeline(token: str, device: str):
        # The real WhisperX/pyannote pipeline pulls gated weights and is
        # impractical for unit tests, so we substitute a deterministic stub.
        # Ensure we exercise the code path that instantiates overrides.
        assert device == "cpu"
        return dummy_pipeline

    monkeypatch.setattr(
        "dnd_session_transcribe.features.diarization.make_diarization_pipeline",
        _fake_make_diarization_pipeline,
    )

    cfg = DiarizationConfig(
        num_speakers=2,
        seg_min_off=0.05,
        turn_min_off=0.05,
    )
    duration = _load_audio_duration(audio_path)

    out_base = tmp_path / "sample"
    df = run_diarization(
        str(audio_path),
        device="cpu",
        cfg=cfg,
        token="",
        audio_dur=duration,
        out_base=out_base,
        resume=False,
    )

    assert dummy_pipeline.calls == [
        {
            "audio_path": audio_path,
            "num_speakers": 2,
            "min_speakers": None,
            "max_speakers": None,
        }
    ]

    # The relaxed overrides should flow into the instantiated pipeline.
    assert dummy_pipeline.pipeline.applied_overrides == {
        "segmentation": {"min_duration_on": 0.05, "min_duration_off": 0.05},
        "speech_turn": {"min_duration_on": 0.10, "min_duration_off": 0.05},
        "clustering": {"max_speakers_per_frame": 2},
    }

    # Speakers should be rewritten using the PLAYER prefix and only two unique IDs.
    assert set(df["speaker"].unique()) == {"PLAYER_00", "PLAYER_01"}
    assert len(df) == 4

    diarization_json = Path(f"{out_base}_diarization_df.json")
    assert diarization_json.exists(), "run_diarization should persist the dataframe"
    with diarization_json.open("r", encoding="utf-8") as fh:
        saved_df = pd.DataFrame(json.load(fh))
    pdt.assert_frame_equal(saved_df[df.columns], df)


def test_run_diarization_resume_uses_cached_file(monkeypatch, tmp_path):
    out_base = tmp_path / "cached"
    cached = [
        {"start": 0.0, "end": 1.0, "speaker": "PLAYER_00"},
        {"start": 1.0, "end": 2.0, "speaker": "PLAYER_01"},
    ]
    (tmp_path / "cached_diarization_df.json").write_text(json.dumps(cached), encoding="utf-8")

    def fail(*args, **kwargs):
        raise AssertionError("resume should bypass pipeline construction")

    monkeypatch.setattr(
        "dnd_session_transcribe.features.diarization.make_diarization_pipeline",
        fail,
    )

    df = run_diarization(
        audio_path="clip.wav",
        device="cpu",
        cfg=DiarizationConfig(num_speakers=2),
        token="",
        audio_dur=3.0,
        out_base=out_base,
        resume=True,
    )

    expected = pd.DataFrame(cached)
    pdt.assert_frame_equal(df[expected.columns], expected)


def test_run_diarization_single_speaker_covers_duration(tmp_path):
    out_base = tmp_path / "solo"

    df = run_diarization(
        audio_path="clip.wav",
        device="cpu",
        cfg=DiarizationConfig(num_speakers=1),
        token="",
        audio_dur=4.0,
        out_base=out_base,
        resume=False,
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert row["start"] == 0.0
    assert row["end"] == pytest.approx(3.999)
    assert row["speaker"] == "PLAYER00"


def test_run_diarization_range_fallback_when_empty(monkeypatch, tmp_path):
    class _FallbackPipeline:
        def __init__(self):
            self.pipeline = _DummyInnerPipeline()
            self.calls: list[dict[str, object]] = []

        def __call__(self, audio_path, num_speakers=None, min_speakers=None, max_speakers=None):
            call = {
                "num_speakers": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            }
            self.calls.append(call)
            if min_speakers is not None or max_speakers is not None:
                return [
                    {"start": 0.0, "end": 1.0, "label": "SPEAKER_02"},
                    {"start": 1.0, "end": 2.0, "label": "SPEAKER_03"},
                ]
            return []

    dummy = _FallbackPipeline()

    monkeypatch.setattr(
        "dnd_session_transcribe.features.diarization.make_diarization_pipeline",
        lambda token, device: dummy,
    )

    cfg = DiarizationConfig(num_speakers=3, allow_range_fallback=True)

    df = run_diarization(
        audio_path="clip.wav",
        device="cpu",
        cfg=cfg,
        token="token",
        audio_dur=2.5,
        out_base=tmp_path / "range",
        resume=False,
    )

    assert dummy.calls == [
        {"num_speakers": 3, "min_speakers": None, "max_speakers": None},
        {"num_speakers": None, "min_speakers": 2, "max_speakers": 4},
    ]
    assert not df.empty
    assert set(df["speaker"]) == {"PLAYER_02", "PLAYER_03"}
