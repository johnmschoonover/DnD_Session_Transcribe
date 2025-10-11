import types

from dnd_session_transcribe.adapters import read_duration_seconds


def test_read_duration_seconds(monkeypatch):
    fake_info = types.SimpleNamespace(frames=16000, samplerate=8000)

    monkeypatch.setattr(read_duration_seconds.sf, "info", lambda path: fake_info)

    duration = read_duration_seconds.read_duration_seconds("audio.wav")

    assert duration == 2.0
