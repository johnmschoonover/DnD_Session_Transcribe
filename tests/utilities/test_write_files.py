import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "utilities" / "write_files.py"
_spec = importlib.util.spec_from_file_location("write_files_module", MODULE_PATH)
write_files = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader is not None
_spec.loader.exec_module(write_files)


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
    write_files.write_srt_vtt_txt_json(final, base)

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
