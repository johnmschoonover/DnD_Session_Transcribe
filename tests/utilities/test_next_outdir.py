import importlib.util
import pathlib
from types import ModuleType


def load_next_outdir_module() -> ModuleType:
    root = pathlib.Path(__file__).resolve().parents[2]
    module_path = root / "utilities" / "next_outdir.py"
    spec = importlib.util.spec_from_file_location("next_outdir", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_next_outdir_returns_first_gap(tmp_path: pathlib.Path) -> None:
    next_outdir = load_next_outdir_module()
    audio_path = tmp_path / "audio" / "recording.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.touch()

    base = audio_path.parent
    (base / "text0").mkdir()
    (base / "text1").mkdir()

    result = next_outdir.next_outdir_for(str(audio_path), "text")

    assert result == base / "text2"


def test_next_outdir_defaults_to_zero(tmp_path: pathlib.Path) -> None:
    next_outdir = load_next_outdir_module()
    audio_path = tmp_path / "session" / "clip.mp3"
    audio_path.parent.mkdir(parents=True)
    audio_path.touch()

    base = audio_path.parent

    result = next_outdir.next_outdir_for(str(audio_path), "text")

    assert result == base / "text0"
