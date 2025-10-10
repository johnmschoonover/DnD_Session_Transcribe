from __future__ import annotations
import json
import pathlib

def atomic_json(path: str | pathlib.Path, data) -> None:
    """Safe JSON write: write to .tmp then replace."""
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)
