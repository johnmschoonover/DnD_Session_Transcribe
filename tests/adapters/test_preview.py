from pathlib import Path
import wave

import pytest

from dnd_session_transcribe.adapters.preview import PreviewSnippet, render_preview
from dnd_session_transcribe.adapters.read_duration_seconds import read_duration_seconds


AUDIO_PATH = Path("sample_audio/test.wav")


@pytest.fixture(autouse=True)
def stub_ffmpeg_cut(monkeypatch):
    def _fake_ffmpeg_cut(_src: str, start: float, end: float, out_wav: str) -> None:
        frames = max(int(round((end - start) * 16000)), 1)
        with wave.open(out_wav, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * frames)

    monkeypatch.setattr("dnd_session_transcribe.adapters.preview.ffmpeg_cut", _fake_ffmpeg_cut)


def test_render_preview_creates_snippet_and_cleans_up():
    events: list[str] = []
    full_duration = read_duration_seconds(str(AUDIO_PATH))
    start = 0.1
    window = min(1.0, full_duration - start - 0.1)
    if window <= 0:
        pytest.skip("sample audio too short for preview test")

    with render_preview(AUDIO_PATH, start=start, duration=window, hook=events.append) as snippet:
        assert isinstance(snippet, PreviewSnippet)
        assert snippet.path.exists()
        assert snippet.duration > 0

        expected_end = min(start + window, full_duration)
        expected_duration = max(0.0, expected_end - start)
        assert snippet.duration == pytest.approx(expected_duration, abs=0.05)
        assert events == ["start"]

    assert events == ["start", "finish"]
    assert not snippet.path.exists()


def test_render_preview_validates_arguments():
    with pytest.raises(ValueError):
        with render_preview(AUDIO_PATH, duration=0):
            pass

    with pytest.raises(ValueError):
        with render_preview(AUDIO_PATH, start=-0.1):
            pass
