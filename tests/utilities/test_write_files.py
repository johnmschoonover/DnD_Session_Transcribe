import json
from pathlib import Path

from dnd_session_transcribe.util.write_files import write_srt_vtt_txt_json


def test_write_srt_vtt_txt_json(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    final = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.234,
                "text": " Hello world ",
                "speaker": "Speaker 1",
            },
            {
                "start": 65.5,
                "end": 70.0,
                "text": "Second line",
                "speaker": "",
            },
        ]
    }

    base: Path = tmp_path / "transcript"
    write_srt_vtt_txt_json(final, base)

    srt_path = base.with_suffix(".srt")
    vtt_path = base.with_suffix(".vtt")
    txt_path = base.with_suffix(".txt")
    json_path = base.with_suffix(".json")

    assert srt_path.exists()
    assert vtt_path.exists()
    assert txt_path.exists()
    assert json_path.exists()

    expected_srt = (
        "1\n"
        "00:00:00,000 --> 00:00:01,234\n"
        "[Speaker 1] Hello world\n\n"
        "2\n"
        "00:01:05,500 --> 00:01:10,000\n"
        "Second line\n\n"
    )
    assert srt_path.read_text(encoding="utf-8") == expected_srt

    expected_vtt = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.234\n"
        "[Speaker 1] Hello world\n\n"
        "00:01:05.500 --> 00:01:10.000\n"
        "Second line\n\n"
    )
    assert vtt_path.read_text(encoding="utf-8") == expected_vtt

    expected_txt = "[Speaker 1] Hello world\nSecond line\n"
    assert txt_path.read_text(encoding="utf-8") == expected_txt

    assert json.loads(json_path.read_text(encoding="utf-8")) == final
    # Ensure the JSON file is valid UTF-8 encoded
    json_path.read_bytes().decode("utf-8")
