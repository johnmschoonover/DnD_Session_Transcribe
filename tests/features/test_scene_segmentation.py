from __future__ import annotations

from dnd_session_transcribe.features.scene_segmentation import (
    SceneSegmentationDashboard,
    TranscriptSegment,
)


def test_segments_are_grouped_by_scene_classification():
    dashboard = SceneSegmentationDashboard(gap_threshold=60.0, min_scene_duration=5.0)
    segments = [
        TranscriptSegment(0.0, 10.0, "DM", "You travel down the corridor quietly."),
        TranscriptSegment(10.0, 18.0, "Rogue", "I check the door for traps."),
        TranscriptSegment(18.0, 35.0, "DM", "Roll initiative! The goblins attack immediately."),
        TranscriptSegment(35.0, 50.0, "Barbarian", "I rage and swing at the closest goblin."),
        TranscriptSegment(120.0, 140.0, "DM", "The mayor greets you and asks about your travels."),
    ]

    scenes = dashboard.segment(segments)

    assert [scene.label for scene in scenes] == ["exploration", "combat", "roleplay"]
    assert scenes[0].start == 0.0
    assert scenes[0].end >= 18.0  # Should extend slightly to cover min duration
    assert len(scenes[1].segments) == 2
    assert scenes[2].start == 120.0


def test_timeline_highlights_dominant_speakers_and_keywords():
    dashboard = SceneSegmentationDashboard()
    segments = [
        TranscriptSegment(0.0, 5.0, "Cleric", "I persuade the guard to open the gate.", tags=("social",)),
        TranscriptSegment(5.0, 10.0, "DM", "The guard is convinced by your charisma."),
    ]

    timeline = dashboard.build_timeline(dashboard.segment(segments))

    assert timeline[0]["label"] == "roleplay"
    assert "Cleric" in timeline[0]["dominant_speakers"]
    assert "persuade" in timeline[0]["highlight_terms"]
