import types
import pathlib

import pytest

from dnd_session_transcribe.adapters import copy_to_ram


@pytest.fixture
def fake_ramdir(monkeypatch, tmp_path):
    original_path = pathlib.Path
    ram_root = tmp_path / "ramdisk"

    def fake_path(value):
        if isinstance(value, original_path):
            return value
        if value == "/dev/shm/whx":
            return ram_root
        return original_path(value)

    monkeypatch.setattr(copy_to_ram, "pathlib", types.SimpleNamespace(Path=fake_path))
    return ram_root


def test_copy_disabled_returns_original(tmp_path):
    src = tmp_path / "clip.wav"
    src.write_bytes(b"abc")

    result = copy_to_ram.copy_to_ram_if_requested(str(src), enable=False)

    assert result == str(src)
    assert src.read_bytes() == b"abc"


def test_copy_to_ram_creates_copy(tmp_path, fake_ramdir):
    src = tmp_path / "clip.wav"
    src.write_bytes(b"test-bytes")

    result = copy_to_ram.copy_to_ram_if_requested(str(src), enable=True)

    dst = fake_ramdir / src.name
    assert result == str(dst)
    assert dst.exists()
    assert dst.read_bytes() == b"test-bytes"


def test_copy_to_ram_reuses_existing_copy(tmp_path, fake_ramdir):
    src = tmp_path / "clip.wav"
    src.write_bytes(b"initial")

    first_result = copy_to_ram.copy_to_ram_if_requested(str(src), enable=True)
    dst = fake_ramdir / src.name
    first_mtime = dst.stat().st_mtime_ns
    assert first_result == str(dst)

    second_result = copy_to_ram.copy_to_ram_if_requested(str(src), enable=True)

    assert second_result == first_result
    assert dst.read_bytes() == b"initial"
    assert dst.stat().st_mtime_ns == first_mtime
