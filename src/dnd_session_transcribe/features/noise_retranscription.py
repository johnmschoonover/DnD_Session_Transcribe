"""Noise-aware re-transcription queue management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, MutableMapping, Optional


@dataclass(slots=True)
class QualitySegment:
    """Quality metrics for a portion of the transcript."""

    start: float
    end: float
    text: str
    asr_confidence: float
    noise_level: float

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class RetranscriptionItem:
    segment: QualitySegment
    reason: str
    priority: str
    attempts: int = 0
    corrections: list[str] = field(default_factory=list)


class SegmentQualityAnalyzer:
    """Identify segments that should be re-transcribed."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.85,
        noise_threshold: float = 0.35,
        min_duration: float = 2.0,
    ) -> None:
        if confidence_threshold <= 0 or confidence_threshold > 1:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if noise_threshold < 0 or noise_threshold > 1:
            raise ValueError("noise_threshold must be within [0, 1]")
        if min_duration <= 0:
            raise ValueError("min_duration must be positive")

        self._confidence_threshold = confidence_threshold
        self._noise_threshold = noise_threshold
        self._min_duration = min_duration

    def score_segment(self, segment: QualitySegment) -> float:
        """Return a score indicating how poor the audio quality is."""

        duration_penalty = 1.0 if segment.duration() >= self._min_duration else 0.5
        confidence_gap = max(0.0, self._confidence_threshold - segment.asr_confidence)
        noise_excess = max(0.0, segment.noise_level - self._noise_threshold)
        return (confidence_gap * 0.6 + noise_excess * 0.4) * duration_penalty

    def should_flag(self, segment: QualitySegment) -> bool:
        return segment.asr_confidence < self._confidence_threshold or segment.noise_level > self._noise_threshold

    def flag_segments(self, segments: Iterable[QualitySegment]) -> list[RetranscriptionItem]:
        flagged: list[RetranscriptionItem] = []
        for segment in segments:
            if self.should_flag(segment):
                confidence_gap = max(0.0, self._confidence_threshold - segment.asr_confidence)
                noise_excess = max(0.0, segment.noise_level - self._noise_threshold)
                score = self.score_segment(segment)
                if confidence_gap >= 0.1 or noise_excess >= 0.1 or score >= 0.18:
                    priority = "high"
                else:
                    priority = "normal"
                reason = self._describe_reason(segment)
                flagged.append(
                    RetranscriptionItem(
                        segment=segment,
                        reason=reason,
                        priority=priority,
                    )
                )
        return flagged

    def _describe_reason(self, segment: QualitySegment) -> str:
        reasons: list[str] = []
        if segment.asr_confidence < self._confidence_threshold:
            reasons.append(
                f"low confidence ({segment.asr_confidence:.2f} < {self._confidence_threshold:.2f})"
            )
        if segment.noise_level > self._noise_threshold:
            reasons.append(
                f"noisy ({segment.noise_level:.2f} > {self._noise_threshold:.2f})"
            )
        return ", ".join(reasons) or "manual review"


class RetranscriptionQueue:
    """Queue handling for iterative re-transcription passes."""

    def __init__(self) -> None:
        self._items: MutableMapping[tuple[float, float], RetranscriptionItem] = {}

    def enqueue(self, item: RetranscriptionItem) -> None:
        key = (round(item.segment.start, 3), round(item.segment.end, 3))
        self._items[key] = item

    def bulk_enqueue(self, items: Iterable[RetranscriptionItem]) -> None:
        for item in items:
            self.enqueue(item)

    def mark_attempt(self, start: float, end: float, *, corrected_text: Optional[str] = None) -> None:
        key = (round(start, 3), round(end, 3))
        entry = self._items.get(key)
        if entry is None:
            raise KeyError(f"No retranscription item for span {start}-{end}")
        entry.attempts += 1
        if corrected_text:
            entry.corrections.append(corrected_text)

    def resolve(self, start: float, end: float) -> None:
        key = (round(start, 3), round(end, 3))
        self._items.pop(key, None)

    def pending(self) -> list[RetranscriptionItem]:
        return sorted(
            self._items.values(),
            key=lambda item: (0 if item.priority == "high" else 1, item.segment.start),
        )

    def __iter__(self) -> Iterator[RetranscriptionItem]:
        return iter(self.pending())


__all__ = [
    "QualitySegment",
    "RetranscriptionItem",
    "RetranscriptionQueue",
    "SegmentQualityAnalyzer",
]
