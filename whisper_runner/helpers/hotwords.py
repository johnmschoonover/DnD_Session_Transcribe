from __future__ import annotations
import os

def load_hotwords(path: str | None) -> str | None:
    """Load comma/newline separated hotwords → single comma string."""
    if not path:
        return None
    p = os.path.expanduser(path)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        txt = f.read()
    words = [w.strip() for line in txt.splitlines() for w in line.split(",")]
    words = [w for w in words if w]
    return ", ".join(words) if words else None
