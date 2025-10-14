"""Translate audio diagnostics into decoder/VAD suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..constants.autotune_defaults import GUARDRAILS, THRESHOLDS
from .audio_stats import AudioDiagnostics


@dataclass
class AutoTuneSuggestion:
    """Structured representation of auto-tune recommendations."""

    cfg: Dict[str, object]
    rationale: Dict[str, object]

    def to_dict(self) -> dict:
        return {
            "cfg": dict(self.cfg),
            "rationale": dict(self.rationale),
        }


def suggest_config(diag: AudioDiagnostics) -> AutoTuneSuggestion:
    """Apply heuristics to derive decoder and VAD configuration suggestions."""

    rationale: Dict[str, object] = {
        "snr_db": diag.snr_db,
        "spectral_flatness_mean": diag.spectral_flatness_mean,
        "micro_segment_ratio": diag.micro_segment_ratio,
        "p95_speech_s": diag.p95_speech_s,
        "median_gap_s": diag.median_gap_s,
    }

    cfg: Dict[str, object] = {
        "use_vad": True,
        "compression_ratio_threshold": GUARDRAILS.compression_ratio_threshold,
        "log_prob_threshold": GUARDRAILS.log_prob_threshold,
        "condition_on_previous_text": False,
    }

    noisy_audio = (
        diag.snr_db < THRESHOLDS.low_snr_db
        or diag.spectral_flatness_mean > THRESHOLDS.high_flatness
        or diag.micro_segment_ratio >= THRESHOLDS.micro_segment_ratio_high
    )
    clean_audio = (
        diag.snr_db >= THRESHOLDS.high_snr_db
        and diag.micro_segment_ratio < THRESHOLDS.micro_segment_ratio_low
    )

    if noisy_audio:
        cfg.update(
            {
                "beam_size": None,
                "best_of": 5,
                "temperature": [0.2, 0.4, 0.6],
                "patience": 1.0,
            }
        )
        rationale["decoding_mode"] = "sampling"
    elif clean_audio:
        cfg.update(
            {
                "beam_size": 3,
                "patience": 1.2,
                "temperature": 0.0,
            }
        )
        rationale["decoding_mode"] = "beam"
    else:
        cfg.update(
            {
                "beam_size": None,
                "best_of": 5,
                "temperature": [0.2, 0.4],
                "patience": 1.0,
            }
        )
        rationale["decoding_mode"] = "sampling_mixed"

    no_speech_threshold = (
        GUARDRAILS.no_speech_threshold_noisy
        if noisy_audio
        else GUARDRAILS.no_speech_threshold_clean
    )
    cfg["no_speech_threshold"] = no_speech_threshold

    vad_max = _clamp(
        diag.p95_speech_s + 10.0,
        GUARDRAILS.vad_max_speech_min,
        GUARDRAILS.vad_max_speech_max,
    )
    cfg["vad_max_speech_s"] = vad_max

    if diag.micro_segment_ratio >= THRESHOLDS.micro_segment_ratio_high:
        cfg.update(
            {
                "vad_min_speech_ms": 2000,
                "vad_min_silence_ms": 1900,
            }
        )
        rationale["vad_profile"] = "micro_segments"
    else:
        min_speech = _clamp(
            max(diag.p25_speech_s * 0.9, 0.9) * 1000,
            GUARDRAILS.vad_min_speech_ms_min,
            GUARDRAILS.vad_min_speech_ms_max,
        )
        min_silence = _clamp(
            max(diag.median_gap_s * 0.9, 1.2) * 1000,
            GUARDRAILS.vad_min_silence_ms_min,
            GUARDRAILS.vad_min_silence_ms_max,
        )
        cfg["vad_min_speech_ms"] = int(min_speech)
        cfg["vad_min_silence_ms"] = int(min_silence)
        rationale["vad_profile"] = "balanced"

    speech_pad = (
        GUARDRAILS.vad_speech_pad_short_gap
        if diag.median_gap_s < 0.5
        else GUARDRAILS.vad_speech_pad_default
    )
    cfg["vad_speech_pad_ms"] = speech_pad

    pre_norm_mode = "apply" if diag.pre_norm_recommended else "off"
    cfg["pre_norm"] = pre_norm_mode

    return AutoTuneSuggestion(cfg=cfg, rationale=rationale)


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return float(max(min_value, min(max_value, value)))


__all__ = ["AutoTuneSuggestion", "suggest_config"]
