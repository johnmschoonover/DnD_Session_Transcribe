import json

from dnd_session_transcribe.util.helpers import atomic_json


def test_atomic_json_writes_data_and_removes_tmp(tmp_path):
    target = tmp_path / "data.json"
    payload = {"name": "session", "count": 42, "tags": ["dnd", "transcribe"]}

    atomic_json(target, payload)

    with target.open("r", encoding="utf-8") as fh:
        assert json.load(fh) == payload

    assert not target.with_suffix(target.suffix + ".tmp").exists()
