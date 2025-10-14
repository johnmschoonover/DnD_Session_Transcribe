from __future__ import annotations

from dnd_session_transcribe.helpers.audio_stats import AudioDiagnostics
from dnd_session_transcribe.helpers.auto_tune import AutoTuneSuggestion, suggest_config


def make_diag(**overrides) -> AudioDiagnostics:
    base = dict(
        duration_s=10.0,
        sr=16_000,
        rms_dbfs=-25.0,
        peak_dbfs=-3.0,
        clipping_ratio=0.0,
        snr_db=20.0,
        spectral_flatness_mean=0.1,
        p25_speech_s=1.2,
        median_speech_s=1.6,
        p90_speech_s=4.0,
        p95_speech_s=5.0,
        median_gap_s=0.6,
        p90_gap_s=1.5,
        micro_segment_ratio=0.1,
        num_segments=5,
        pre_norm_recommended=False,
    )
    base.update(overrides)
    return AudioDiagnostics(**base)


def test_suggest_config_for_clean_audio_prefers_beam():
    suggestion = suggest_config(make_diag())
    assert isinstance(suggestion, AutoTuneSuggestion)
    assert suggestion.cfg["beam_size"] == 3
    assert suggestion.cfg["temperature"] == 0.0
    assert suggestion.cfg["pre_norm"] == "off"


def test_suggest_config_for_noisy_audio_switches_to_sampling():
    diag = make_diag(snr_db=5.0, spectral_flatness_mean=0.6, micro_segment_ratio=0.4, pre_norm_recommended=True)
    suggestion = suggest_config(diag)
    assert suggestion.cfg["beam_size"] is None
    assert suggestion.cfg["best_of"] == 5
    assert suggestion.cfg["temperature"] == [0.2, 0.4, 0.6]
    assert suggestion.cfg["pre_norm"] == "apply"


def test_suggest_config_for_mixed_audio_uses_hybrid_sampling():
    diag = make_diag(snr_db=15.0, micro_segment_ratio=0.22)
    suggestion = suggest_config(diag)
    assert suggestion.cfg["beam_size"] is None
    assert suggestion.cfg["temperature"] == [0.2, 0.4]
    assert suggestion.cfg["no_speech_threshold"] > 0.6
