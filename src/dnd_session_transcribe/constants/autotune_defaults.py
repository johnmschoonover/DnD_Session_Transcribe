"""Default thresholds and guardrail values for the auto-tune preflight pipeline."""

from __future__ import annotations

from dataclasses import dataclass

AUTO_TUNE_VERSION = "1.0.0"


@dataclass(frozen=True)
class Thresholds:
    """Collection of heuristic thresholds used during preflight analysis."""

    low_snr_db: float = 10.0
    high_snr_db: float = 20.0
    high_flatness: float = 0.5
    micro_segment_ratio_high: float = 0.30
    micro_segment_ratio_low: float = 0.20
    micro_segment_limit_s: float = 1.2
    pre_norm_peak_dbfs: float = -2.0
    pre_norm_rms_dbfs: float = -28.0
    pre_norm_snr_db: float = 12.0
    clipping_ratio: float = 0.005


THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class Guardrails:
    """Decoder guardrail defaults to enforce safe decoding."""

    compression_ratio_threshold: float = 2.3
    log_prob_threshold: float = -0.8
    no_speech_threshold_noisy: float = 0.78
    no_speech_threshold_clean: float = 0.68
    vad_min_speech_ms_min: int = 900
    vad_min_speech_ms_max: int = 2200
    vad_min_silence_ms_min: int = 1000
    vad_min_silence_ms_max: int = 2200
    vad_max_speech_min: float = 25.0
    vad_max_speech_max: float = 60.0
    vad_speech_pad_short_gap: int = 300
    vad_speech_pad_default: int = 450


GUARDRAILS = Guardrails()

__all__ = [
    "AUTO_TUNE_VERSION",
    "THRESHOLDS",
    "Guardrails",
    "Thresholds",
    "GUARDRAILS",
]
