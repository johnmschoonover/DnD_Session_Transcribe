from __future__ import annotations

import pytest

from dnd_session_transcribe.features.noise_retranscription import (
    QualitySegment,
    RetranscriptionQueue,
    SegmentQualityAnalyzer,
)


def make_segment(start: float, end: float, confidence: float, noise: float, text: str = "sample") -> QualitySegment:
    return QualitySegment(start=start, end=end, text=text, asr_confidence=confidence, noise_level=noise)


def test_flag_segments_returns_priority_items():
    analyzer = SegmentQualityAnalyzer(confidence_threshold=0.9, noise_threshold=0.4)
    segments = [
        make_segment(0.0, 3.0, 0.92, 0.2),
        make_segment(3.0, 8.0, 0.75, 0.5),
        make_segment(8.0, 12.0, 0.6, 0.2),
    ]

    flagged = analyzer.flag_segments(segments)

    assert len(flagged) == 2
    assert flagged[0].priority == "high"
    assert "low confidence" in flagged[0].reason


def test_retranscription_queue_tracks_attempts_and_corrections():
    queue = RetranscriptionQueue()
    analyzer = SegmentQualityAnalyzer(confidence_threshold=0.9, noise_threshold=0.4)
    flagged = analyzer.flag_segments([make_segment(0.0, 5.0, 0.8, 0.3)])

    queue.bulk_enqueue(flagged)
    entry = queue.pending()[0]
    queue.mark_attempt(entry.segment.start, entry.segment.end, corrected_text="Corrected line")
    queue.mark_attempt(entry.segment.start, entry.segment.end)
    assert entry.attempts == 2
    assert entry.corrections == ["Corrected line"]

    queue.resolve(entry.segment.start, entry.segment.end)
    assert queue.pending() == []


def test_mark_attempt_raises_for_unknown_segment():
    queue = RetranscriptionQueue()
    with pytest.raises(KeyError):
        queue.mark_attempt(1.0, 2.0)
