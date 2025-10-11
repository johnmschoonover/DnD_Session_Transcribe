import importlib
import os
import runpy
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


def _stub_torch():
    return types.SimpleNamespace(
        backends=types.SimpleNamespace(
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
            cudnn=types.SimpleNamespace(allow_tf32=False),
        )
    )


@pytest.fixture
def cli_module(monkeypatch):
    fake_whisperx = types.SimpleNamespace(assign_word_speakers=lambda dia, aligned: [])
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "torch", _stub_torch())

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


def _fresh_cli_state(cli_module, monkeypatch):
    monkeypatch.setattr(cli_module, "LOG", cli_module.LoggingConfig())
    monkeypatch.setattr(cli_module, "ASR", cli_module.ASRConfig())
    monkeypatch.setattr(cli_module, "DIA", cli_module.DiarizationConfig())
    monkeypatch.setattr(cli_module, "PREC", cli_module.PreciseRerunConfig(enabled=False))
    monkeypatch.setattr(cli_module, "PRE", cli_module.PreprocessConfig())
    monkeypatch.setattr(cli_module, "SCR", cli_module.ScrubConfig())
    monkeypatch.setattr(cli_module, "WR", cli_module.WritingConfig())
    monkeypatch.setattr(cli_module, "PROF", cli_module.ProfilesConfig())


def test_main_runs_full_pipeline_with_precise(monkeypatch, cli_module, tmp_path):
    _fresh_cli_state(cli_module, monkeypatch)

    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"RIFF")
    auto_outdir = tmp_path / "text1"

    args = types.SimpleNamespace(
        audio=str(audio),
        outdir=None,
        ram=True,
        resume=True,
        num_speakers=3,
        hotwords_file="hot.txt",
        initial_prompt_file="prompt.txt",
        spelling_map="map.csv",
        precise_rerun=True,
        asr_model="tiny",
        asr_device="cpu",
        asr_compute_type="int8",
        precise_model="precise",
        precise_device="cpu",
        precise_compute_type="float32",
        vocal_extract="bandpass",
        log_level="DEBUG",
    )
    monkeypatch.setattr(cli_module, "parse_args", lambda: args)

    call_log: dict[str, object] = {}

    def fake_next_outdir(audio_path: str, prefix: str):
        call_log["next_outdir"] = (audio_path, prefix)
        return auto_outdir

    copy_calls: list[tuple[str, bool]] = []

    def fake_copy(path: str, copy_flag: bool) -> str:
        copy_calls.append((path, copy_flag))
        return str(Path(path).with_suffix(".ram.wav"))

    preprocess_calls: list[tuple[str, str]] = []

    def fake_preprocess(path: str, mode: str) -> str:
        preprocess_calls.append((path, mode))
        processed = tmp_path / "processed.wav"
        processed.write_text("ok", encoding="utf-8")
        return str(processed)

    monkeypatch.setattr(cli_module, "next_outdir_for", fake_next_outdir)
    monkeypatch.setattr(cli_module, "copy_to_ram_if_requested", fake_copy)
    monkeypatch.setattr(cli_module, "preprocess_audio", fake_preprocess)
    monkeypatch.setattr(cli_module, "load_hotwords", lambda path: "dragons")
    monkeypatch.setattr(cli_module, "load_initial_prompt", lambda path: "Begin!")
    monkeypatch.setattr(cli_module, "load_spelling_map", lambda path: [("orc", "Orc")])

    applied_rules: list[str] = []
    monkeypatch.setattr(
        cli_module,
        "apply_spelling_rules",
        lambda text, rules: applied_rules.append(text) or text.upper(),
    )

    monkeypatch.setattr(cli_module, "read_duration_seconds", lambda path: 4.0)

    def fake_run_asr(audio_path, base, cfg, hotwords, prompt, resume, total_sec):
        call_log["run_asr"] = {
            "audio": audio_path,
            "base": str(base),
            "model": cfg.model,
            "compute": cfg.compute_type,
            "hotwords": hotwords,
            "prompt": prompt,
            "resume": resume,
            "total": total_sec,
        }
        return [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "orc"},
            {"id": 1, "start": 1.0, "end": 2.5, "text": "encounter"},
        ]

    scrub_calls: list[list[dict[str, object]]] = []

    def fake_scrub(segments, cfg):
        scrub_calls.append([seg.copy() for seg in segments])
        return segments

    monkeypatch.setattr(cli_module, "run_asr", fake_run_asr)
    monkeypatch.setattr(cli_module, "scrub_segments", fake_scrub)
    monkeypatch.setattr(cli_module, "find_hard_spans", lambda segs, **_: [])
    monkeypatch.setattr(cli_module, "rerun_precise_on_spans", lambda *args, **kwargs: pytest.fail("rerun should be skipped"))

    monkeypatch.setattr(cli_module, "clamp_to_duration", lambda segs, _: segs)

    aligned_payload = {"segments": ["aligned"]}
    monkeypatch.setattr(cli_module, "run_alignment", lambda *args, **kwargs: aligned_payload)
    monkeypatch.setattr(cli_module, "ensure_hf_token", lambda: "token")

    dia_df = pd.DataFrame([
        {"start": 0.0, "end": 1.0, "speaker": "PLAYER_00"}
    ])
    monkeypatch.setattr(cli_module, "run_diarization", lambda *args, **kwargs: dia_df)

    assigned: list[object] = []

    def fake_assign(dia, aligned):
        assigned.append((dia, aligned))
        return [{"text": "FINAL"}]

    cli_module.whisperx.assign_word_speakers = fake_assign

    written: list[object] = []
    monkeypatch.setattr(
        cli_module,
        "write_srt_vtt_txt_json",
        lambda data, base: written.append((data, str(base))),
    )

    cli_module.main()

    assert call_log["next_outdir"][0].endswith("voice.wav")
    assert copy_calls == [(str(audio.resolve()), True)]
    assert preprocess_calls and preprocess_calls[0][1] == "bandpass"
    processed_path = Path(preprocess_calls[0][0])
    assert processed_path.name == "voice.ram.wav"
    assert applied_rules == ["orc", "encounter"]
    assert scrub_calls, "scrub_segments should run at least once"
    assert assigned and assigned[0][1] is aligned_payload
    assert written and "text1" in written[0][1]


def test_main_precise_rerun_replaces_segments(monkeypatch, cli_module, tmp_path):
    _fresh_cli_state(cli_module, monkeypatch)

    audio = tmp_path / "story.wav"
    audio.write_bytes(b"RIFF")
    outdir = tmp_path / "outdir"

    args = types.SimpleNamespace(
        audio=str(audio),
        outdir=str(outdir),
        ram=False,
        resume=False,
        num_speakers=2,
        hotwords_file=None,
        initial_prompt_file=None,
        spelling_map=None,
        precise_rerun=True,
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

    monkeypatch.setattr(cli_module, "next_outdir_for", lambda *args, **kwargs: pytest.fail("next_outdir should not run"))
    monkeypatch.setattr(cli_module, "copy_to_ram_if_requested", lambda path, _: path)
    monkeypatch.setattr(cli_module, "preprocess_audio", lambda path, mode: path)
    monkeypatch.setattr(cli_module, "load_hotwords", lambda _: None)
    monkeypatch.setattr(cli_module, "load_initial_prompt", lambda _: None)
    monkeypatch.setattr(cli_module, "load_spelling_map", lambda _: [])
    monkeypatch.setattr(cli_module, "read_duration_seconds", lambda _: 6.0)

    base_segments = [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "one"},
        {"id": 1, "start": 1.0, "end": 2.0, "text": "two"},
    ]

    monkeypatch.setattr(cli_module, "run_asr", lambda *args, **kwargs: base_segments[:])

    scrubbed: list[list[dict[str, object]]] = []

    def fake_scrub(segs, _cfg):
        scrubbed.append([seg.copy() for seg in segs])
        return segs

    monkeypatch.setattr(cli_module, "scrub_segments", fake_scrub)

    spans = [(0.0, 2.0)]
    monkeypatch.setattr(cli_module, "find_hard_spans", lambda *args, **kwargs: spans)

    replacements = [(0.0, 2.0, [{"id": 0, "start": 0.0, "end": 2.0, "text": "redo"}])]
    monkeypatch.setattr(cli_module, "rerun_precise_on_spans", lambda *args, **kwargs: replacements)
    monkeypatch.setattr(cli_module, "clamp_to_duration", lambda segs, _: segs)
    monkeypatch.setattr(cli_module, "run_alignment", lambda *args, **kwargs: [])
    monkeypatch.setattr(cli_module, "ensure_hf_token", lambda: "token")

    dia_df = pd.DataFrame([{ "start": 0.0, "end": 2.0, "speaker": "PLAYER_00" }])
    monkeypatch.setattr(cli_module, "run_diarization", lambda *args, **kwargs: dia_df)

    cli_module.whisperx.assign_word_speakers = lambda dia, aligned: dia.to_dict("records")
    outputs: list[object] = []
    monkeypatch.setattr(
        cli_module,
        "write_srt_vtt_txt_json",
        lambda data, base: outputs.append((data, str(base))),
    )

    cli_module.main()

    assert len(scrubbed) >= 2, "scrub should run after precise splice"
    assert outputs and "outdir" in outputs[0][1]
    assert scrubbed[-1][0]["text"] == "redo"


def test_main_exits_when_diarization_empty(monkeypatch, cli_module, tmp_path):
    _fresh_cli_state(cli_module, monkeypatch)

    audio = tmp_path / "empty.wav"
    audio.write_bytes(b"RIFF")
    outdir = tmp_path / "out"

    args = types.SimpleNamespace(
        audio=str(audio),
        outdir=str(outdir),
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

    monkeypatch.setattr(cli_module, "copy_to_ram_if_requested", lambda path, _: path)
    monkeypatch.setattr(cli_module, "preprocess_audio", lambda path, mode: path)
    monkeypatch.setattr(cli_module, "load_hotwords", lambda _: None)
    monkeypatch.setattr(cli_module, "load_initial_prompt", lambda _: None)
    monkeypatch.setattr(cli_module, "load_spelling_map", lambda _: [])
    monkeypatch.setattr(cli_module, "read_duration_seconds", lambda _: 2.0)

    base_segments = [
        {"id": 0, "start": 0.0, "end": 0.5, "text": "a"},
        {"id": 1, "start": 0.5, "end": 1.0, "text": "b"},
    ]

    monkeypatch.setattr(cli_module, "run_asr", lambda *args, **kwargs: base_segments[:])
    monkeypatch.setattr(cli_module, "scrub_segments", lambda segs, cfg: segs)
    monkeypatch.setattr(cli_module, "clamp_to_duration", lambda segs, _: segs)
    monkeypatch.setattr(cli_module, "run_alignment", lambda *args, **kwargs: [])
    monkeypatch.setattr(cli_module, "ensure_hf_token", lambda: "token")
    monkeypatch.setattr(cli_module, "run_diarization", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(cli_module, "write_srt_vtt_txt_json", lambda *args, **kwargs: pytest.fail("should not write output"))

    with pytest.raises(SystemExit) as excinfo:
        cli_module.main()

    assert "no speaker regions" in str(excinfo.value)


def test_cli_entry_point_sets_pythonunbuffered(monkeypatch):
    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)
    monkeypatch.setattr(sys, "argv", ["dnd-transcribe", "--help"])

    sys.modules.pop("dnd_session_transcribe.cli", None)
    monkeypatch.setitem(sys.modules, "whisperx", types.SimpleNamespace(assign_word_speakers=lambda *args, **kwargs: []))
    monkeypatch.setitem(sys.modules, "torch", _stub_torch())

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("dnd_session_transcribe.cli", run_name="__main__", alter_sys=True)

    assert excinfo.value.code == 0
    assert os.environ["PYTHONUNBUFFERED"] == "1"
