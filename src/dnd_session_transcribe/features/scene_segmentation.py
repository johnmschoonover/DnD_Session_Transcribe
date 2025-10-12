"""Utilities for adaptive scene segmentation visualizations.

The real application would surface these results inside a dashboard; here we
provide the data modelling and heuristics that power such a view.  The module
groups transcript segments into narrative scenes by combining lightweight
signals such as speaker changes, keyword detection, and any campaign-specific
tags that may be present on segments.  The resulting ``SceneSnapshot`` objects
summarise the content so a caller can render a timeline or jump list without
performing heavyweight analysis.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import logging
import re


LOGGER = logging.getLogger(__name__)


_SCENE_KEYWORDS: dict[str, set[str]] = {
    "combat": {
        "initiative",
        "attack",
        "damage",
        "hit",
        "miss",
        "critical",
        "battle",
        "fight",
        "strikes",
        "blood",
        "roll initiative",
        "armor class",
        "rage",
        "swing",
        "slash",
        "goblin",
    },
    "exploration": {
        "search",
        "investigate",
        "door",
        "trap",
        "room",
        "corridor",
        "map",
        "travel",
        "wilderness",
        "track",
        "navigate",
        "terrain",
    },
    "roleplay": {
        "persuade",
        "deception",
        "insight",
        "conversation",
        "negotiation",
        "npc",
        "dialogue",
        "appeal",
        "charisma",
        "speech",
        "promise",
        "deal",
    },
}


_TAG_ALIASES: dict[str, str] = {
    "social": "roleplay",
    "downtime": "roleplay",
    "shopping": "roleplay",
    "travel": "exploration",
    "investigation": "exploration",
    "combat": "combat",
    "skirmish": "combat",
}


def _normalise_scene_tag(tag: str) -> str | None:
    key = tag.strip().lower()
    if not key:
        return None
    if key in _SCENE_KEYWORDS:
        return key
    return _TAG_ALIASES.get(key)


@dataclass(slots=True)
class TranscriptSegment:
    """Minimal representation of an aligned transcript segment."""

    start: float
    end: float
    speaker: str
    text: str
    tags: tuple[str, ...] = ()

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class SceneSnapshot:
    """Summary of a single scene suitable for dashboard visualisation."""

    label: str
    start: float
    end: float
    segments: list[TranscriptSegment] = field(default_factory=list)
    keywords: Counter[str] = field(default_factory=Counter)
    speakers: Counter[str] = field(default_factory=Counter)
    confidence: float = 1.0

    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_timeline_entry(self) -> dict[str, object]:
        """Return a serialisable dictionary for a UI timeline."""

        dominant_speakers = [speaker for speaker, _ in self.speakers.most_common(3)]
        top_keywords = [word for word, _ in self.keywords.most_common(5)]
        return {
            "label": self.label,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration(), 2),
            "segments": len(self.segments),
            "dominant_speakers": dominant_speakers,
            "highlight_terms": top_keywords,
            "confidence": round(self.confidence, 3),
        }


class SceneSegmentationDashboard:
    """Heuristic scene segmentation for long-form D&D sessions."""

    def __init__(
        self,
        *,
        gap_threshold: float = 90.0,
        min_scene_duration: float = 12.0,
    ) -> None:
        if gap_threshold <= 0:
            raise ValueError("gap_threshold must be positive")
        if min_scene_duration <= 0:
            raise ValueError("min_scene_duration must be positive")
        self._gap_threshold = gap_threshold
        self._min_scene_duration = min_scene_duration

    @staticmethod
    def _classify_segment(segment: TranscriptSegment) -> tuple[str, Counter[str]]:
        """Return (label, keyword_counter) for *segment*."""

        keyword_counter: Counter[str] = Counter()
        scores: Counter[str] = Counter()

        # Campaign-provided tags have the highest weight.
        for raw_tag in segment.tags:
            tag = _normalise_scene_tag(raw_tag)
            if tag is None:
                continue
            scores[tag] += 2

        # Keyword lookup inside the transcript text is a softer signal.
        lowered_text = segment.text.lower()
        tokens = set(re.findall(r"[a-zA-Z][a-zA-Z']+", lowered_text))
        for label, keywords in _SCENE_KEYWORDS.items():
            overlap = tokens.intersection(keywords)
            if overlap:
                keyword_counter.update(overlap)
                scores[label] += len(overlap)

        if not scores:
            scores["roleplay"] = 1  # Default to roleplay style narration.

        label, label_score = scores.most_common(1)[0]

        total_score = sum(scores.values())
        confidence = label_score / total_score if total_score else 1.0
        keyword_counter["__confidence__"] = confidence
        return label, keyword_counter

    def _should_split(
        self,
        previous: SceneSnapshot | None,
        current_label: str,
        current_start: float,
    ) -> bool:
        if previous is None:
            return True

        label_changed = current_label != previous.label
        gap = current_start - previous.end
        if gap > self._gap_threshold:
            LOGGER.debug(
                "Starting new scene due to gap %.2fs > %.2fs", gap, self._gap_threshold
            )
            return True

        if label_changed:
            LOGGER.debug("Scene label changed from %s to %s", previous.label, current_label)
            return True

        return False

    def segment(self, segments: Sequence[TranscriptSegment]) -> list[SceneSnapshot]:
        """Group ``segments`` into scene snapshots.

        Parameters
        ----------
        segments:
            Iterable of transcript segments.  Segments are sorted by ``start`` to
            ensure sequential processing.
        """

        ordered_segments = sorted(segments, key=lambda seg: seg.start)
        scenes: list[SceneSnapshot] = []

        active_scene: SceneSnapshot | None = None
        for segment in ordered_segments:
            label, keyword_counter = self._classify_segment(segment)
            confidence = keyword_counter.pop("__confidence__", 1.0)

            if self._should_split(active_scene, label, segment.start):
                if active_scene is not None:
                    self._finalise_scene(active_scene)
                    scenes.append(active_scene)
                active_scene = SceneSnapshot(
                    label=label,
                    start=segment.start,
                    end=segment.end,
                    confidence=confidence,
                )
            else:
                assert active_scene is not None  # For type-checkers.
                active_scene.end = max(active_scene.end, segment.end)
                active_scene.confidence = (active_scene.confidence + confidence) / 2

            active_scene.segments.append(segment)
            active_scene.keywords.update(keyword_counter)
            active_scene.speakers[segment.speaker] += 1

        if active_scene is not None:
            self._finalise_scene(active_scene)
            scenes.append(active_scene)

        return scenes

    def build_timeline(self, scenes: Iterable[SceneSnapshot]) -> list[dict[str, object]]:
        """Create a timeline representation of ``scenes`` for dashboards."""

        return [scene.to_timeline_entry() for scene in scenes]

    def _finalise_scene(self, scene: SceneSnapshot) -> None:
        duration = scene.duration()
        if duration < self._min_scene_duration and len(scene.segments) > 1:
            LOGGER.debug(
                "Extending short scene %.2fs to min duration %.2fs",
                duration,
                self._min_scene_duration,
            )
            scene.end = scene.start + self._min_scene_duration
        # Normalise the internal counters to remove helper values.
        if "__confidence__" in scene.keywords:
            del scene.keywords["__confidence__"]


__all__ = [
    "SceneSegmentationDashboard",
    "SceneSnapshot",
    "TranscriptSegment",
]
