"""Audio analysis helpers for the auto-tune preflight pipeline."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import soundfile as sf

try:  # pragma: no cover - optional dependency for CPU-friendly environments
    import webrtcvad  # type: ignore
except ImportError:  # pragma: no cover
    webrtcvad = None  # type: ignore

try:  # librosa is optional but available in project deps
    import librosa
except ImportError:  # pragma: no cover
    librosa = None  # type: ignore

from ..constants.autotune_defaults import THRESHOLDS

logger = logging.getLogger(__name__)

_ANALYSIS_SR = 16_000
_FRAME_MS = 30


def _dbfs(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        return float("-inf")
    return 20.0 * np.log10(value)


@dataclass
class AudioDiagnostics:
    """Aggregate metrics describing an audio file."""

    duration_s: float
    sr: int
    rms_dbfs: float
    peak_dbfs: float
    clipping_ratio: float
    snr_db: float
    spectral_flatness_mean: float
    p25_speech_s: float
    median_speech_s: float
    p90_speech_s: float
    p95_speech_s: float
    median_gap_s: float
    p90_gap_s: float
    micro_segment_ratio: float
    num_segments: int
    pre_norm_recommended: bool

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of the diagnostics."""

        data = asdict(self)
        return data


def analyze_audio(path: Path) -> AudioDiagnostics:
    """Load audio (mono @ 16 kHz) and compute diagnostics for auto-tuning."""

    path = Path(path)
    audio, sr = sf.read(path, always_2d=False)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    original_duration = len(audio) / float(sr)

    if sr != _ANALYSIS_SR:
        if librosa is None:
            raise RuntimeError(
                "librosa is required for resampling to 16 kHz but is not available"
            )
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=_ANALYSIS_SR)
        sr = _ANALYSIS_SR
    else:
        audio = audio.astype(np.float32)

    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 0.0

    rms_dbfs = _dbfs(rms)
    peak_dbfs = _dbfs(peak)

    speech_segments, speech_mask = _segment_speech(audio, sr)

    speech_durations = [end - start for start, end in speech_segments]
    gap_durations = _compute_gaps(speech_segments, original_duration)

    snr_db = _estimate_snr(audio, speech_mask)

    spectral_flatness = _spectral_flatness(audio, sr)

    micro_ratio = _micro_segment_ratio(speech_durations)

    pre_norm_recommended = _should_pre_norm(
        snr_db,
        spectral_flatness,
        clipping_ratio,
        peak_dbfs,
        rms_dbfs,
    )

    return AudioDiagnostics(
        duration_s=original_duration,
        sr=sr,
        rms_dbfs=float(rms_dbfs),
        peak_dbfs=float(peak_dbfs),
        clipping_ratio=float(clipping_ratio),
        snr_db=float(snr_db),
        spectral_flatness_mean=float(spectral_flatness),
        p25_speech_s=_percentile(speech_durations, 25),
        median_speech_s=_percentile(speech_durations, 50),
        p90_speech_s=_percentile(speech_durations, 90),
        p95_speech_s=_percentile(speech_durations, 95),
        median_gap_s=_percentile(gap_durations, 50),
        p90_gap_s=_percentile(gap_durations, 90),
        micro_segment_ratio=float(micro_ratio),
        num_segments=len(speech_segments),
        pre_norm_recommended=bool(pre_norm_recommended),
    )


def _segment_speech(audio: np.ndarray, sr: int) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """Return speech segments using WebRTC VAD where available."""

    frame_size = int(sr * _FRAME_MS / 1000)
    if frame_size <= 0:
        raise ValueError("Frame size must be positive")

    total_frames = int(np.ceil(len(audio) / frame_size))
    padded = np.zeros(total_frames * frame_size, dtype=np.float32)
    padded[: len(audio)] = audio

    pcm16 = np.clip(padded * 32768.0, -32768, 32767).astype(np.int16).tobytes()

    if webrtcvad is not None:
        vad = webrtcvad.Vad(2)
        decisions = []
        for idx in range(total_frames):
            start = idx * frame_size * 2  # int16 bytes
            end = start + frame_size * 2
            frame = pcm16[start:end]
            if len(frame) < frame_size * 2:
                frame += b"\x00" * (frame_size * 2 - len(frame))
            try:
                decisions.append(vad.is_speech(frame, sr))
            except Exception as exc:  # pragma: no cover - guard against rare VAD issues
                logger.debug("WebRTC VAD failed on frame %d: %s", idx, exc)
                decisions.append(False)
    else:  # pragma: no cover - exercised only when dependency missing
        logger.warning(
            "webrtcvad not installed; falling back to energy-based segmentation"
        )
        energies = []
        for idx in range(total_frames):
            frame = padded[idx * frame_size : (idx + 1) * frame_size]
            energies.append(np.sqrt(np.mean(np.square(frame))) + 1e-8)
        threshold = max(1e-7, float(np.percentile(energies, 75) * 0.5))
        decisions = [energy > threshold for energy in energies]

    segments: List[Tuple[float, float]] = []
    mask = np.zeros(len(padded), dtype=bool)
    current_start: float | None = None

    frame_duration = _FRAME_MS / 1000.0
    for idx, speech in enumerate(decisions):
        start_time = idx * frame_duration
        end_time = start_time + frame_duration
        if speech:
            mask[idx * frame_size : (idx + 1) * frame_size] = True
            if current_start is None:
                current_start = start_time
        else:
            if current_start is not None:
                segments.append((current_start, start_time))
                current_start = None
    if current_start is not None:
        segments.append((current_start, len(decisions) * frame_duration))

    trimmed_mask = mask[: len(audio)]

    return segments, trimmed_mask


def _compute_gaps(segments: Sequence[Tuple[float, float]], duration: float) -> List[float]:
    if not segments:
        return []
    gaps: List[float] = []
    prev_end = 0.0
    for start, end in segments:
        if start > prev_end:
            gaps.append(start - prev_end)
        prev_end = end
    if prev_end < duration:
        gaps.append(max(0.0, duration - prev_end))
    return gaps


def _estimate_snr(audio: np.ndarray, speech_mask: np.ndarray) -> float:
    speech = audio[speech_mask]
    nonspeech = audio[~speech_mask]
    if speech.size == 0:
        return 0.0
    speech_rms = float(np.sqrt(np.mean(np.square(speech))) + 1e-12)
    if nonspeech.size < _ANALYSIS_SR * 0.3:  # <300ms of silence → fallback
        return 15.0
    noise_rms = float(np.sqrt(np.mean(np.square(nonspeech))) + 1e-12)
    if noise_rms <= 0:
        return 60.0
    snr = 20.0 * np.log10(speech_rms / noise_rms)
    return float(np.clip(snr, -10.0, 60.0))


def _spectral_flatness(audio: np.ndarray, sr: int) -> float:
    if audio.size == 0:
        return 0.0
    if librosa is None:  # pragma: no cover - librosa shipped with deps but guard anyway
        return 0.0
    flatness = librosa.feature.spectral_flatness(y=audio, hop_length=512, n_fft=1024)
    return float(np.clip(np.mean(flatness), 0.0, 1.0))


def _micro_segment_ratio(speech_durations: Sequence[float]) -> float:
    if not speech_durations:
        return 0.0
    tiny = sum(1 for dur in speech_durations if dur < THRESHOLDS.micro_segment_limit_s)
    return tiny / float(len(speech_durations))


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def _should_pre_norm(
    snr_db: float,
    spectral_flatness: float,
    clipping_ratio: float,
    peak_dbfs: float,
    rms_dbfs: float,
) -> bool:
    return (
        snr_db < THRESHOLDS.pre_norm_snr_db
        or spectral_flatness > THRESHOLDS.high_flatness
        or clipping_ratio > THRESHOLDS.clipping_ratio
        or peak_dbfs > THRESHOLDS.pre_norm_peak_dbfs
        or rms_dbfs < THRESHOLDS.pre_norm_rms_dbfs
    )


__all__ = ["AudioDiagnostics", "analyze_audio"]
