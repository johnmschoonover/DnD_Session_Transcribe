from __future__ import annotations

from pathlib import Path

from dnd_session_transcribe.util.helpers.spelling_map import (
    apply_spelling_rules,
    load_spelling_map,
)


def test_load_spelling_map_returns_empty_when_missing(tmp_path):
    missing = tmp_path / "absent.csv"

    assert load_spelling_map(str(missing)) == []


def test_load_spelling_map_handles_none():
    assert load_spelling_map(None) == []


def test_apply_spelling_rules_replaces_whole_words(tmp_path):
    csv_path = tmp_path / "spelling.csv"
    csv_path.write_text("wrong,right\ncolour,color\nWi-Fi,WiFi\n", encoding="utf-8")

    rules = load_spelling_map(str(csv_path))

    corrected = apply_spelling_rules("The COLOUR of the wi-fi router", rules)

    assert corrected == "The color of the WiFi router"
