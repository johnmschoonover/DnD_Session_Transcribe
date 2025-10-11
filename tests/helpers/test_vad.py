import pytest

from dnd_session_transcribe.util.helpers import build_vad_params


def test_build_vad_params_casts_required_values_to_ints():
    params = build_vad_params(100.0, 200.0, 300.0)

    assert params.keys() == {
        "min_speech_duration_ms",
        "min_silence_duration_ms",
        "speech_pad_ms",
    }
    assert params["min_speech_duration_ms"] == 100
    assert isinstance(params["min_speech_duration_ms"], int)
    assert params["min_silence_duration_ms"] == 200
    assert isinstance(params["min_silence_duration_ms"], int)
    assert params["speech_pad_ms"] == 300
    assert isinstance(params["speech_pad_ms"], int)


def test_build_vad_params_includes_optional_duration_as_float():
    params = build_vad_params(10, 20, 30, max_speech_s=4)

    assert params["max_speech_duration_s"] == pytest.approx(4.0)
    assert isinstance(params["max_speech_duration_s"], float)


def test_build_vad_params_omits_optional_duration_when_none():
    params = build_vad_params(10, 20, 30, max_speech_s=None)

    assert "max_speech_duration_s" not in params
