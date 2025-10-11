"""Adapter layer for external dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "copy_to_ram_if_requested",
    "ffmpeg",
    "ffmpeg_cut",
    "ensure_hf_token",
    "preprocess_audio",
    "read_duration_seconds",
]

_ATTR_TO_MODULE = {
    "copy_to_ram_if_requested": (
        "dnd_session_transcribe.adapters.copy_to_ram",
        "copy_to_ram_if_requested",
    ),
    "ffmpeg": ("dnd_session_transcribe.adapters.ffmpeg", "ffmpeg"),
    "ffmpeg_cut": ("dnd_session_transcribe.adapters.ffmpeg", "ffmpeg_cut"),
    "ensure_hf_token": (
        "dnd_session_transcribe.adapters.huggingface",
        "ensure_hf_token",
    ),
    "preprocess_audio": (
        "dnd_session_transcribe.adapters.preprocess_audio",
        "preprocess_audio",
    ),
    "read_duration_seconds": (
        "dnd_session_transcribe.adapters.read_duration_seconds",
        "read_duration_seconds",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _ATTR_TO_MODULE[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__} has no attribute {name}") from exc
    module = import_module(module_name)
    return getattr(module, attr_name)
