"""Preview helpers for GUI integrations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterator, Literal, Optional

from .ffmpeg import ffmpeg_cut
from .read_duration_seconds import read_duration_seconds


PreviewHookEvent = Literal["start", "finish"]


@dataclass
class PreviewSnippet:
    """Information about a rendered preview snippet."""

    path: Path
    duration: float


@contextmanager
def render_preview(
    audio_path: Path,
    *,
    start: float = 0.0,
    duration: float = 10.0,
    hook: Optional[Callable[[PreviewHookEvent], None]] = None,
) -> Iterator[PreviewSnippet]:
    """Render a temporary WAV snippet for preview playback.

    Parameters
    ----------
    audio_path:
        Source audio file to preview.
    start:
        Start time in seconds for the preview window.
    duration:
        Requested duration in seconds for the preview window.
    hook:
        Optional callback invoked with "start" before rendering and "finish"
        after cleanup. Useful for progress bars or logging.

    Yields
    ------
    PreviewSnippet
        Metadata describing the rendered snippet. The temporary file is
        cleaned up automatically when the context exits.
    """

    audio_path = Path(audio_path)

    if duration <= 0:
        raise ValueError("duration must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")

    if hook is not None:
        hook("start")

    try:
        with TemporaryDirectory() as tmpdir:
            snippet_path = Path(tmpdir) / "preview.wav"
            end = start + duration
            ffmpeg_cut(str(audio_path), start, end, str(snippet_path))
            snippet_duration = read_duration_seconds(str(snippet_path))
            yield PreviewSnippet(path=snippet_path, duration=snippet_duration)
    finally:
        if hook is not None:
            hook("finish")
