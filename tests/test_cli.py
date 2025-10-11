import importlib
import sys
import types

import pytest


@pytest.fixture
def cli_module(monkeypatch):
    fake_whisperx = types.SimpleNamespace(assign_word_speakers=lambda dia, aligned: [])
    fake_torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
            cudnn=types.SimpleNamespace(allow_tf32=False),
        )
    )

    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    module = importlib.import_module("dnd_session_transcribe.cli")
    return importlib.reload(module)


def test_parse_args_round_trip(cli_module, monkeypatch):
    argv = [
        "dnd-transcribe",
        "sample.wav",
        "--log-level",
        "DEBUG",
        "--ram",
        "--resume",
        "--num-speakers",
        "2",
        "--vocal-extract",
        "bandpass",
    ]
    monkeypatch.setattr(cli_module.sys, "argv", argv)

    args = cli_module.parse_args()

    assert args.audio == "sample.wav"
    assert args.log_level == "DEBUG"
    assert args.ram is True
    assert args.resume is True
    assert args.num_speakers == 2
    assert args.vocal_extract == "bandpass"


def test_main_exits_when_audio_missing(cli_module, monkeypatch, tmp_path):
    missing = tmp_path / "missing.wav"
    args = types.SimpleNamespace(
        audio=str(missing),
        outdir=None,
        ram=False,
        resume=False,
        num_speakers=None,
        hotwords_file=None,
        initial_prompt_file=None,
        spelling_map=None,
        precise_rerun=False,
        asr_model=None,
        asr_device=None,
        asr_compute_type=None,
        precise_model=None,
        precise_device=None,
        precise_compute_type=None,
        vocal_extract=None,
        log_level="INFO",
    )

    monkeypatch.setattr(cli_module, "parse_args", lambda: args)

    with pytest.raises(SystemExit) as excinfo:
        cli_module.main()

    assert "Audio not found" in str(excinfo.value)
