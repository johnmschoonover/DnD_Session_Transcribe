from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dnd_session_transcribe.helpers import preflight_analyze_and_suggest
from dnd_session_transcribe.helpers import preflight as preflight_mod


def _write_test_tone(path: Path) -> None:
    sr = 16_000
    tone = 0.1 * np.sin(2 * np.pi * 220 * np.linspace(0, 1.2, int(sr * 1.2), endpoint=False))
    pause = np.zeros(int(sr * 0.4))
    waveform = np.concatenate([tone, pause, tone])
    sf.write(path, waveform, sr)


@pytest.fixture()
def temp_audio(tmp_path: Path) -> Path:
    audio_path = tmp_path / "tone.wav"
    _write_test_tone(audio_path)
    return audio_path


def test_preflight_returns_diagnostics_and_respects_cache(tmp_path: Path, temp_audio: Path, monkeypatch: pytest.MonkeyPatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(preflight_mod, "_CACHE_DIR", cache_dir)

    cfg, diag = preflight_analyze_and_suggest(temp_audio, mode="suggest", pre_norm_mode="off")
    assert cfg == {}
    assert diag["cache"]["hit"] is False
    assert "metrics" in diag
    assert diag["pre_norm"]["final"] in {"off", "suggest", "apply"}

    cfg_repeat, diag_repeat = preflight_analyze_and_suggest(temp_audio, mode="suggest", pre_norm_mode="off")
    assert cfg_repeat == {}
    assert diag_repeat["cache"]["hit"] is True


def test_preflight_apply_mode_generates_final_config(tmp_path: Path, temp_audio: Path, monkeypatch: pytest.MonkeyPatch):
    cache_dir = tmp_path / "cache_apply"
    monkeypatch.setattr(preflight_mod, "_CACHE_DIR", cache_dir)

    cfg, diag = preflight_analyze_and_suggest(
        temp_audio,
        mode="apply",
        pre_norm_mode="off",
        no_cache=True,
    )

    assert "use_vad" in cfg or cfg == {}
    assert "final_config" in diag
    assert "suggestion" in diag
