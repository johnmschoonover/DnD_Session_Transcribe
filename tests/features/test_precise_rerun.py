from types import SimpleNamespace

import pytest

from dnd_session_transcribe.features import precise_rerun


def test_rerun_precise_on_spans_splits_windows(monkeypatch):
    model_inits = []
    transcribe_calls = []

    class FakeModel:
        def __init__(self, model_name, *, device, compute_type):
            model_inits.append((model_name, device, compute_type))

        def transcribe(self, audio_path, **kwargs):
            transcribe_calls.append((audio_path, kwargs))
            segments = [
                SimpleNamespace(
                    start=0.0,
                    end=0.5,
                    text="alpha",
                    avg_logprob=-0.1,
                    compression_ratio=0.5,
                    no_speech_prob=0.01,
                ),
                SimpleNamespace(
                    start=0.5,
                    end=1.0,
                    text="beta",
                    avg_logprob=-0.2,
                    compression_ratio=0.6,
                    no_speech_prob=0.02,
                ),
            ]
            return iter(segments), {"language": "en"}

    monkeypatch.setattr(precise_rerun, "WhisperModel", FakeModel)

    ffmpeg_calls = []

    def fake_ffmpeg(src, start, end, out):
        ffmpeg_calls.append((src, start, end, out))

    monkeypatch.setattr(precise_rerun, "ffmpeg_cut", fake_ffmpeg)

    tmp_paths = []

    class FakeTmp:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_named_temporary_file(*args, **kwargs):
        name = f"/tmp/fake-{len(tmp_paths)}.wav"
        tmp_paths.append(name)
        return FakeTmp(name)

    monkeypatch.setattr(precise_rerun.tempfile, "NamedTemporaryFile", fake_named_temporary_file)

    removed_paths = []

    def fake_remove(path):
        removed_paths.append(path)

    monkeypatch.setattr(precise_rerun.os, "remove", fake_remove)

    class FakeProgress:
        def __init__(self, *_, **__):
            self.updates = []

        def update(self, amount):
            self.updates.append(amount)

        def close(self):
            pass

    monkeypatch.setattr(precise_rerun, "tqdm", FakeProgress)

    spans = [(1.0, 6.2), (8.0, 13.6)]
    max_window = 2.5

    result = precise_rerun.rerun_precise_on_spans(
        src_audio="audio.wav",
        spans=spans,
        lang="en",
        model="test-model",
        compute="float32",
        device="cpu",
        beam=5,
        patience=1.0,
        max_window_s=max_window,
    )

    expected_windows = [
        (1.0, 3.5),
        (3.5, 6.0),
        (6.0, 6.2),
        (8.0, 10.5),
        (10.5, 13.0),
        (13.0, 13.6),
    ]

    assert len(ffmpeg_calls) == len(expected_windows)
    for (src, start, end, out_path), (exp_start, exp_end), tmp_path in zip(
        ffmpeg_calls, expected_windows, tmp_paths
    ):
        assert src == "audio.wav"
        assert start == pytest.approx(exp_start)
        assert end == pytest.approx(exp_end)
        assert out_path == tmp_path

    assert [call[0] for call in transcribe_calls] == tmp_paths

    assert [start for start, end, _ in result] == pytest.approx([w[0] for w in expected_windows])
    assert [end for start, end, _ in result] == pytest.approx([w[1] for w in expected_windows])

    for (cursor, _window_end, repl) in result:
        rel_starts = [entry["start"] - cursor for entry in repl]
        rel_ends = [entry["end"] - cursor for entry in repl]
        assert rel_starts == pytest.approx([0.0, 0.5])
        assert rel_ends == pytest.approx([0.5, 1.0])

    assert removed_paths == tmp_paths

    assert model_inits == [("test-model", "cpu", "float32")]
