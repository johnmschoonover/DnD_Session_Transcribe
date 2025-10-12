import importlib.util
import pathlib

from dnd_session_transcribe.util.next_outdir import next_outdir_for


def test_next_outdir_returns_first_gap(tmp_path: pathlib.Path) -> None:
    audio_path = tmp_path / "audio" / "recording.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.touch()

    base = audio_path.parent
    (base / "text0").mkdir()
    (base / "text1").mkdir()

    result = next_outdir_for(str(audio_path), "text")

    assert result == base / "text2"


def test_next_outdir_defaults_to_zero(tmp_path: pathlib.Path) -> None:
    audio_path = tmp_path / "session" / "clip.mp3"
    audio_path.parent.mkdir(parents=True)
    audio_path.touch()

    base = audio_path.parent

    result = next_outdir_for(str(audio_path), "text")

    assert result == base / "text0"
