import importlib
from pathlib import Path
import sys
import types

import pandas as pd

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


def test_main_runs_pipeline_with_overrides(cli_module, monkeypatch, tmp_path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"RIFF")

    hotwords_path = tmp_path / "hotwords.txt"
    hotwords_path.write_text("dragon, lich", encoding="utf-8")
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("You are the narrator", encoding="utf-8")
    spelling_path = tmp_path / "spelling.csv"
    spelling_path.write_text("wrong,right\ncolour,color\n", encoding="utf-8")

    args = types.SimpleNamespace(
        audio=str(audio),
        outdir=None,
        ram=True,
        resume=True,
        num_speakers=3,
        hotwords_file=str(hotwords_path),
        initial_prompt_file=str(prompt_path),
        spelling_map=str(spelling_path),
        precise_rerun=True,
        asr_model="base.en",
        asr_device="cpu",
        asr_compute_type="int8",
        precise_model="large-v3",
        precise_device="cpu",
        precise_compute_type="float32",
        vocal_extract="bandpass",
        log_level="DEBUG",
    )

    monkeypatch.setattr(cli_module, "parse_args", lambda: args)

    # Replace module-level configs with fresh instances so the test can assert mutations.
    monkeypatch.setattr(cli_module, "ASR", cli_module.ASRConfig())
    monkeypatch.setattr(cli_module, "DIA", cli_module.DiarizationConfig())
    monkeypatch.setattr(cli_module, "PREC", cli_module.PreciseRerunConfig(enabled=False))
    monkeypatch.setattr(cli_module, "PRE", cli_module.PreprocessConfig())
    monkeypatch.setattr(cli_module, "SCR", cli_module.ScrubConfig())
    monkeypatch.setattr(cli_module, "WR", cli_module.WritingConfig())
    monkeypatch.setattr(cli_module, "PROF", cli_module.ProfilesConfig())

    outdir = tmp_path / "auto-out"
    monkeypatch.setattr(cli_module, "next_outdir_for", lambda audio_path, prefix: outdir)

    copy_calls: list[tuple[str, bool]] = []

    def fake_copy(path: str, use_ram: bool) -> str:
        copy_calls.append((path, use_ram))
        return str(tmp_path / "rammed.wav")

    monkeypatch.setattr(cli_module, "copy_to_ram_if_requested", fake_copy)

    processed_audio = str(tmp_path / "processed.wav")

    def fake_preprocess(audio_path: str, mode: str | None) -> str:
        assert mode == "bandpass"
        return processed_audio

    monkeypatch.setattr(cli_module, "preprocess_audio", fake_preprocess)

    atomic_calls: list[tuple[str, dict]] = []

    def fake_atomic(path: str, payload: dict) -> None:
        atomic_calls.append((path, payload))

    monkeypatch.setattr(cli_module, "atomic_json", fake_atomic)

    run_asr_calls: list[dict[str, object]] = []

    def fake_run_asr(
        audio_path: str,
        base,
        cfg,
        hotwords,
        init_prompt,
        *,
        resume: bool,
        total_sec: float,
    ):
        run_asr_calls.append(
            {
                "audio": audio_path,
                "base": base,
                "cfg": cfg,
                "hotwords": hotwords,
                "prompt": init_prompt,
                "resume": resume,
                "total_sec": total_sec,
            }
        )
        return [
            {"start": 0.0, "end": 1.0, "text": "colour"},
            {"start": 1.0, "end": 2.0, "text": "dragon"},
        ]

    monkeypatch.setattr(cli_module, "run_asr", fake_run_asr)
    monkeypatch.setattr(cli_module, "scrub_segments", lambda segs, cfg: segs)
    monkeypatch.setattr(cli_module, "read_duration_seconds", lambda _: 6.0)
    monkeypatch.setattr(cli_module, "find_hard_spans", lambda segs, **_: [(0.5, 1.5)])

    splice_calls: list[tuple[list[dict], list[dict]]] = []

    def fake_splice(original, replacement):
        splice_calls.append((original, replacement))
        return replacement

    monkeypatch.setattr(cli_module, "splice_segments", fake_splice)

    monkeypatch.setattr(
        cli_module,
        "rerun_precise_on_spans",
        lambda *_, **__: [
            {"start": 0.2, "end": 0.8, "text": "colour"},
            {"start": 1.2, "end": 1.8, "text": "dragon"},
        ],
    )

    spell_calls: list[tuple[str, list[tuple[str, str]]]] = []

    def fake_apply_spelling(text: str, rules):
        spell_calls.append((text, list(rules)))
        return text.upper()

    monkeypatch.setattr(cli_module, "apply_spelling_rules", fake_apply_spelling)

    monkeypatch.setattr(cli_module, "clamp_to_duration", lambda segs, _: segs)
    monkeypatch.setattr(cli_module, "run_alignment", lambda *_, **__: [{"text": "aligned"}])
    monkeypatch.setattr(cli_module, "ensure_hf_token", lambda: "hf_token")

    diarization_calls: list[dict[str, object]] = []

    def fake_run_diarization(audio_path, device, cfg, token, audio_dur, out_base, resume):
        diarization_calls.append(
            {
                "audio": audio_path,
                "device": device,
                "cfg": cfg,
                "token": token,
                "dur": audio_dur,
                "base": out_base,
                "resume": resume,
            }
        )
        return pd.DataFrame([
            {"start": 0.0, "end": 1.0, "speaker": "PLAYER_00"},
        ])

    monkeypatch.setattr(cli_module, "run_diarization", fake_run_diarization)
    monkeypatch.setattr(cli_module.whisperx, "assign_word_speakers", lambda dia, aligned: ["final"])

    written: list[tuple[list[str], object]] = []

    def fake_write(final, base):
        written.append((final, base))

    monkeypatch.setattr(cli_module, "write_srt_vtt_txt_json", fake_write)

    cli_module.main()

    assert copy_calls == [(str(audio.resolve()), True)]
    assert run_asr_calls[0]["audio"] == processed_audio
    assert run_asr_calls[0]["hotwords"] == "dragon, lich"
    assert run_asr_calls[0]["prompt"] == "You are the narrator"
    assert run_asr_calls[0]["resume"] is True
    assert run_asr_calls[0]["total_sec"] == 6.0

    assert cli_module.ASR.model == "base.en"
    assert cli_module.ASR.device == "cpu"
    assert cli_module.ASR.compute_type == "int8"
    assert cli_module.DIA.num_speakers == 3
    assert cli_module.PREC.enabled is True
    assert cli_module.PREC.model == "large-v3"
    assert cli_module.PREC.compute_type == "float32"

    assert splice_calls
    assert spell_calls == [
        ("colour", [("colour", "color")]),
        ("dragon", [("colour", "color")]),
    ]

    assert any(path.endswith("_fw_segments_postspell.json") for path, _ in atomic_calls)
    assert diarization_calls[0]["token"] == "hf_token"

    assert cli_module.torch.backends.cuda.matmul.allow_tf32 is True
    assert cli_module.torch.backends.cudnn.allow_tf32 is True

    final_call = written.pop()
    assert final_call[0] == ["final"]
    assert Path(final_call[1]).name == "input"
