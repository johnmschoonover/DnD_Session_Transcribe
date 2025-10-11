from __future__ import annotations
from typing import List, Tuple
import os, csv, re

def load_spelling_map(path: str | None) -> List[Tuple[str, str]]:
    """CSV with columns: wrong,right"""
    if not path:
        return []
    p = os.path.expanduser(path)
    if not os.path.exists(p):
        return []
    rules: List[Tuple[str, str]] = []
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = (row.get("wrong") or "").strip()
            r = (row.get("right") or "").strip()
            if w and r:
                rules.append((w, r))
    return rules

def apply_spelling_rules(text: str, rules: List[Tuple[str, str]]) -> str:
    """Apply case-insensitive whole-word replacements to text."""
    out = text
    for wrong, replacement in rules:
        pattern = rf"\b{re.escape(wrong)}\b"
        out = re.sub(
            pattern,
            lambda match, repl=replacement: repl,
            out,
            flags=re.IGNORECASE,
        )
    return out
