from __future__ import annotations
import os

def load_initial_prompt(path: str | None) -> str | None:
    """Load a short initial context prompt (first decode window only)."""
    if not path:
        return None
    p = os.path.expanduser(path)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return text or None
