"""DnD Session Transcribe package."""

from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version("dnd-session-transcribe")
except metadata.PackageNotFoundError:  # pragma: no cover - fallback for local development
    __version__ = "0.0.0"

__all__ = ["__version__"]
