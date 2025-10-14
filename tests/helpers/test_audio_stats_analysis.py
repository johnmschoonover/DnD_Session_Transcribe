from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from dnd_session_transcribe.helpers.audio_stats import analyze_audio


def _make_synthetic_audio(path):
    sr = 16_000
    tone = 0.2 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.6, int(sr * 0.6), endpoint=False))
    pause = np.zeros(int(sr * 0.3))
    waveform = np.concatenate([tone, pause, tone])
    sf.write(path, waveform, sr)
    return sr, waveform


def test_analyze_audio_reports_core_metrics(tmp_path):
    audio_path = tmp_path / "synthetic.wav"
    sr, waveform = _make_synthetic_audio(audio_path)

    diag = analyze_audio(audio_path)

    assert diag.sr == 16_000
    assert diag.duration_s == pytest.approx(len(waveform) / sr, rel=0.01)
    assert diag.num_segments >= 1
    assert 0.0 <= diag.micro_segment_ratio <= 1.0
    assert -120.0 <= diag.rms_dbfs <= 0.0
    assert 0.0 <= diag.spectral_flatness_mean <= 1.0

    metrics = diag.to_dict()
    assert metrics["duration_s"] == pytest.approx(diag.duration_s)
    assert metrics["micro_segment_ratio"] == diag.micro_segment_ratio
    assert isinstance(metrics["pre_norm_recommended"], bool)
