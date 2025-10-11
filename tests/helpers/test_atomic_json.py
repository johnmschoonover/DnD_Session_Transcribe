import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from helpers.atomic_json import atomic_json


def test_atomic_json_writes_data_and_removes_tmp(tmp_path):
    target = tmp_path / "data.json"
    payload = {"name": "session", "count": 42, "tags": ["dnd", "transcribe"]}

    atomic_json(target, payload)

    with target.open("r", encoding="utf-8") as fh:
        assert json.load(fh) == payload

    assert not target.with_suffix(target.suffix + ".tmp").exists()
