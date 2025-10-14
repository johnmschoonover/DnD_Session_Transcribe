#!/usr/bin/env python3
# run_whisperx.py
# Pipeline: (optional preprocess) → Faster-Whisper ASR → (optional precise re-run)
# → WhisperX alignment → pyannote diarization → assign speakers → write outputs

from __future__ import annotations
import os, sys, pathlib
import json
import logging

# --- progress bars ---
from tqdm.auto import tqdm
PROGRESS_STREAM = sys.stdout
IS_TTY = PROGRESS_STREAM.isatty()

# --- configs ---
from .util.config import (
    ASRConfig, DiarizationConfig, LoggingConfig, PreciseRerunConfig,
    PreprocessConfig, ProfilesConfig, ScrubConfig, WritingConfig
)

# --- helpers (data helpers) ---
from .util.helpers import (
    load_hotwords, load_initial_prompt,
    load_spelling_map, apply_spelling_rules, atomic_json
)
from .helpers import preflight_analyze_and_suggest

# --- utilities (infra helpers) ---
from .adapters.copy_to_ram import copy_to_ram_if_requested
from .adapters.huggingface import ensure_hf_token
from .adapters.preprocess_audio import preprocess_audio
from .adapters.preview import render_preview
from .adapters.read_duration_seconds import read_duration_seconds
from .util.next_outdir import next_outdir_for
from .util.processing import clamp_to_duration, find_hard_spans, scrub_segments, splice_segments
from .util.write_files import write_srt_vtt_txt_json

# --- functions (core steps) ---
from .features import (
    run_asr,
    rerun_precise_on_spans,
    run_alignment,
    run_diarization,
)

# --- heavy libs (import late) ---
import whisperx
import torch


# ========================= CLI =========================
import argparse
import contextlib
import shutil
import time
from dataclasses import replace, fields


logger = logging.getLogger(__name__)


LOG = LoggingConfig()

LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def parse_time_spec(value: str | float | int) -> float:
    """Parse a flexible time specification into seconds."""

    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip()
        try:
            seconds = float(text)
        except ValueError as exc:
            parts = text.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid time specification: {value!r}") from exc

            try:
                seconds = float(parts[-1])
                minutes = int(parts[-2])
                hours = int(parts[-3]) if len(parts) == 3 else 0
            except ValueError as inner_exc:
                raise ValueError(f"Invalid time specification: {value!r}") from inner_exc

            seconds += minutes * 60
            seconds += hours * 3600

    if seconds < 0:
        raise ValueError("Time specification must be non-negative")

    return seconds


def parse_args():
    ap = argparse.ArgumentParser(
        description="Faster-Whisper → WhisperX alignment → pyannote diarization"
    )
    ap.add_argument("audio", help="Path to input audio (wav/mp3/flac)")
    ap.add_argument("--outdir", default=None, help="Output dir (default: auto textN next to audio)")
    ap.add_argument("--ram", action="store_true", help="Copy audio to /dev/shm (tmpfs) for faster IO")
    ap.add_argument("--resume", action="store_true", help="Reuse cached JSON checkpoints if present")
    ap.add_argument("--num-speakers", type=int, default=None, help="Override speaker count")
    ap.add_argument("--hotwords-file", default=None, help="Comma/line-separated hotwords file")
    ap.add_argument("--initial-prompt-file", default=None, help="Short context prompt for first window")
    ap.add_argument("--spelling-map", default=None, help="CSV wrong,right for post-correction")
    ap.add_argument("--precise-rerun", action="store_true", help="Re-ASR hard spans with ultra-precise settings")
    ap.add_argument("--asr-model", default=None, help="Override Faster-Whisper model id (e.g., tiny, turbo, large-v3)")
    ap.add_argument("--asr-device", choices=["cuda", "cpu", "mps"], default=None,
                    help="Select device for Faster-Whisper (default: config)")
    ap.add_argument("--asr-compute-type", default=None,
                    help="Override compute type (float16, float32, int8_float32, etc.)")
    ap.add_argument("--precise-model", default=None,
                    help="Override precise rerun model when --precise-rerun is used")
    ap.add_argument("--precise-device", choices=["cuda", "cpu", "mps"], default=None,
                    help="Select device for the precise rerun (default: config)")
    ap.add_argument("--precise-compute-type", default=None,
                    help="Override compute type for the precise rerun")
    ap.add_argument("--vocal-extract", choices=["off", "bandpass"], default=None,
                    help="Override preprocessing (off/bandpass)")
    ap.add_argument(
        "--log-level",
        choices=tuple(LOG_LEVELS.keys()),
        default=LOG.level,
        help="Logging verbosity (default: %(default)s)",
    )
    ap.add_argument(
        "--preview-start",
        default=None,
        help="Start time for preview rendering (seconds or MM:SS / HH:MM:SS)",
    )
    ap.add_argument(
        "--preview-duration",
        default=None,
        help="Duration for preview rendering in seconds (default: 10s)",
    )
    ap.add_argument(
        "--preview-output",
        default=None,
        help="Optional path for the rendered preview WAV (default: alongside audio)",
    )
    ap.add_argument(
        "--auto-tune",
        action="store_true",
        help="Run preflight analysis to suggest ASR/VAD parameters (disabled by default)",
    )
    ap.add_argument(
        "--auto-tune-mode",
        choices=["suggest", "apply"],
        default="apply",
        help="Choose whether to apply suggestions or only report them",
    )
    ap.add_argument(
        "--pre-norm",
        choices=["off", "suggest", "apply"],
        default="suggest",
        help="Control automatic pre-normalisation handling",
    )
    ap.add_argument(
        "--autotune-cache-ttl",
        type=int,
        default=86400,
        help="Seconds to retain preflight cache entries (default: 1 day)",
    )
    ap.add_argument(
        "--autotune-dump",
        default=None,
        help="Directory for preflight diagnostics (default: output directory)",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        dest="autotune_no_cache",
        help="Disable the preflight analysis cache",
    )
    ap.add_argument(
        "--log-preflight",
        action="store_true",
        help="Emit detailed preflight diagnostics to the log",
    )
    ap.add_argument(
        "--redact-paths",
        action="store_true",
        help="Redact file paths in preflight artifacts",
    )
    return ap.parse_args()


# ====================== CONFIGS ========================
ASR  = ASRConfig()
DIA  = DiarizationConfig()
PREC = PreciseRerunConfig(enabled=False)
PRE  = PreprocessConfig()
SCR  = ScrubConfig()
WR   = WritingConfig()
PROF = ProfilesConfig()   # reserved for profile matching


def _clone_pipeline_configs() -> tuple[ASRConfig, DiarizationConfig, PreciseRerunConfig, PreprocessConfig]:
    """Return independent config instances for a single run."""

    return (
        replace(ASR),
        replace(DIA),
        replace(PREC),
        replace(PRE),
    )


# ======================= MAIN =======================
def _apply_custom_logging(log_level: str | None, handlers: list[logging.Handler]) -> contextlib.AbstractContextManager[None]:
    """Attach custom log handlers for the duration of a run."""

    @contextlib.contextmanager
    def _manager():
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        level = LOG_LEVELS.get(log_level, logging.WARNING)
        root_logger.setLevel(level)
        for handler in handlers:
            root_logger.addHandler(handler)
        try:
            yield
        finally:
            for handler in handlers:
                root_logger.removeHandler(handler)
                handler.flush()
                handler.close()
            root_logger.setLevel(previous_level)

    return _manager()


def run_transcription(
    args: argparse.Namespace,
    *,
    configure_logging: bool = True,
    log_handlers: list[logging.Handler] | None = None,
) -> pathlib.Path:
    """Execute the transcription pipeline using CLI-style arguments."""

    handlers = list(log_handlers or [])

    log_context: contextlib.AbstractContextManager[None]
    if configure_logging:
        logging.basicConfig(
            level=LOG_LEVELS.get(getattr(args, "log_level", LOG.level), logging.WARNING),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            force=True,
        )
        log_context = contextlib.nullcontext()
    else:
        log_context = _apply_custom_logging(getattr(args, "log_level", LOG.level), handlers)

    with log_context:
        logger.debug("Logging initialized at %s", getattr(args, "log_level", LOG.level))

        audio = pathlib.Path(args.audio).resolve()
        logger.debug("Resolved audio path: %s", audio)
        if not audio.exists():
            raise SystemExit(f"Audio not found: {audio}")

        original_audio = audio

        preview_requested = (
            args.preview_start is not None
            or args.preview_duration is not None
            or args.preview_output is not None
        )

        outdir: pathlib.Path | None = None

        # Preview rendering short-circuit
        if preview_requested:
            if args.outdir:
                requested_outdir = pathlib.Path(args.outdir).expanduser().resolve()
                if requested_outdir.name.startswith("preview_"):
                    outdir = requested_outdir
                else:
                    outdir = requested_outdir.parent / f"preview_{requested_outdir.name}"
            else:
                outdir = next_outdir_for(str(original_audio), f"preview_{WR.out_prefix}")

            outdir.mkdir(parents=True, exist_ok=True)

            start_sec = parse_time_spec(args.preview_start) if args.preview_start is not None else 0.0
            duration_sec = (
                parse_time_spec(args.preview_duration)
                if args.preview_duration is not None
                else 10.0
            )

            preview_primary = outdir / f"{original_audio.stem}_preview.wav"
            destinations: list[pathlib.Path] = []

            if args.preview_output is not None:
                user_path = pathlib.Path(args.preview_output).expanduser().resolve()
                user_path.parent.mkdir(parents=True, exist_ok=True)
                destinations.append(user_path)

            destinations.append(preview_primary)

            logger.info(
                "Rendering preview from %.2fs for %.2fs", start_sec, duration_sec
            )

            with render_preview(audio, start=start_sec, duration=duration_sec) as snippet:
                seen: set[pathlib.Path] = set()
                for dest in destinations:
                    if dest in seen:
                        continue
                    shutil.copyfile(snippet.path, dest)
                    logger.info(
                        "Preview snippet ready: %s (%.2fs)",
                        dest,
                        snippet.duration,
                    )
                    seen.add(dest)

            audio = preview_primary
            logger.info(
                "Continuing with transcription for preview snippet from %s", original_audio
            )

        asr_cfg, dia_cfg, prec_cfg, pre_cfg = _clone_pipeline_configs()
        baseline_asr_cfg = replace(asr_cfg)

        # CLI → config overrides
        if args.vocal_extract is not None:
            pre_cfg.vocal_extract = args.vocal_extract
        if args.hotwords_file:
            asr_cfg.hotwords_file = args.hotwords_file
        if args.initial_prompt_file:
            asr_cfg.initial_prompt_file = args.initial_prompt_file
        if args.asr_model:
            asr_cfg.model = args.asr_model
        if args.asr_device:
            asr_cfg.device = args.asr_device
        if args.asr_compute_type:
            asr_cfg.compute_type = args.asr_compute_type
        if args.num_speakers:
            dia_cfg.num_speakers = args.num_speakers
        if args.precise_rerun:
            prec_cfg.enabled = True
        if args.precise_model:
            prec_cfg.model = args.precise_model
        if args.precise_device:
            prec_cfg.device = args.precise_device
        if args.precise_compute_type:
            prec_cfg.compute_type = args.precise_compute_type

        user_override_fields: dict[str, object] = {}
        for field in fields(asr_cfg):
            base_value = getattr(baseline_asr_cfg, field.name)
            current_value = getattr(asr_cfg, field.name)
            if current_value != base_value:
                user_override_fields[field.name] = current_value

        # outdir (auto textN if not provided)
        if outdir is None:
            outdir = (
                pathlib.Path(args.outdir).expanduser().resolve()
                if args.outdir
                else next_outdir_for(str(audio), WR.out_prefix)
            )
            outdir.mkdir(parents=True, exist_ok=True)
        base = outdir / audio.stem
        logger.debug("Output directory resolved: %s", outdir)
        logger.debug("Output base path: %s", base)

        logger.info("Audio path: %s", audio)
        logger.info("Output directory: %s", outdir)

        # RAM copy (optional)
        src_for_io = copy_to_ram_if_requested(str(audio), args.ram)
        logger.debug("Source for I/O: %s", src_for_io)

        # preprocess (utilities)
        mode = args.vocal_extract if args.vocal_extract is not None else pre_cfg.vocal_extract
        VOCAL_AUDIO = preprocess_audio(src_for_io, mode)
        if VOCAL_AUDIO != str(audio):
            logger.info("Preprocessed audio via %s → %s", mode, VOCAL_AUDIO)
        else:
            logger.debug("Preprocessing skipped; using original audio")

        # hotwords / prompt / spelling
        hotwords = load_hotwords(asr_cfg.hotwords_file)
        if hotwords:
            logger.debug("Loaded hotwords (%d chars)", len(hotwords))
        init_prompt = load_initial_prompt(asr_cfg.initial_prompt_file)
        if init_prompt:
            logger.debug("Loaded initial prompt (%d chars)", len(init_prompt))
        sp_rules = load_spelling_map(args.spelling_map)
        if sp_rules:
            logger.debug("Loaded %d spelling correction rules", len(sp_rules))

        preflight_diag: dict | None = None
        if args.auto_tune:
            preflight_start = time.perf_counter()
            dump_dir = (
                pathlib.Path(args.autotune_dump).expanduser().resolve()
                if args.autotune_dump
                else outdir
            )
            dump_dir.mkdir(parents=True, exist_ok=True)

            preflight_cfg, preflight_diag = preflight_analyze_and_suggest(
                pathlib.Path(VOCAL_AUDIO),
                user_overrides=user_override_fields,
                mode=args.auto_tune_mode,
                pre_norm_mode=args.pre_norm,
                cache_ttl=args.autotune_cache_ttl,
                no_cache=getattr(args, "autotune_no_cache", False),
                artifact_dir=dump_dir,
            )
            elapsed_ms = int((time.perf_counter() - preflight_start) * 1000)
            preflight_diag["elapsed_ms"] = elapsed_ms

            if args.auto_tune_mode == "apply":
                for key, value in preflight_cfg.items():
                    if key == "pre_norm":
                        continue
                    if hasattr(asr_cfg, key):
                        setattr(asr_cfg, key, value)

            pre_norm_meta = preflight_diag.get("pre_norm", {})
            if (
                pre_norm_meta.get("final") == "apply"
                and pre_norm_meta.get("applied_path")
            ):
                VOCAL_AUDIO = pre_norm_meta["applied_path"]

            summary_payload = {
                "audio": audio.name if args.redact_paths else str(audio),
                "autotune": {
                    "snr_db": preflight_diag["metrics"].get("snr_db"),
                    "micro_ratio": preflight_diag["metrics"].get("micro_segment_ratio"),
                    "decision": preflight_diag["suggestion"]["rationale"].get("decoding_mode"),
                    "pre_norm": pre_norm_meta.get("final"),
                },
                "final_config": preflight_diag.get("final_config", {}),
                "elapsed_ms": elapsed_ms,
            }
            if args.log_preflight:
                logger.info("Preflight summary: %s", json.dumps(summary_payload, ensure_ascii=False))

            diag_artifact = json.loads(json.dumps(preflight_diag, ensure_ascii=False))
            if args.redact_paths and diag_artifact.get("pre_norm", {}).get("applied_path"):
                diag_artifact["pre_norm"]["applied_path"] = None

            diagnostics_path = dump_dir / f"{base.name}.preflight.json"
            suggested_path = dump_dir / f"{base.name}.autotune.suggested.json"
            final_path = dump_dir / f"{base.name}.autotune.final.json"

            atomic_json(diagnostics_path, diag_artifact)
            atomic_json(suggested_path, preflight_diag["suggestion"])
            atomic_json(final_path, preflight_diag.get("final_config", {}))

        # duration (MUST be the file passed to ASR for proper %)
        total_sec = read_duration_seconds(VOCAL_AUDIO)
        logger.debug("Total audio duration (s): %.2f", total_sec)

        # ---------------- ASR ----------------
        fw_segments = run_asr(VOCAL_AUDIO, base, asr_cfg, hotwords, init_prompt, resume=args.resume, total_sec=total_sec)
        fw_segments = scrub_segments(fw_segments, SCR)
        fw_segments = [s for s in fw_segments if s["end"] > s["start"]]
        atomic_json(f"{base}_fw_segments_scrubbed.json", {"segments": fw_segments})
        logger.debug("Scrubbed segments retained: %d", len(fw_segments))

        # ------------- precise re-run (optional) -------------
        if prec_cfg.enabled:
            spans = find_hard_spans(
                fw_segments, dur=total_sec,
                logprob_thr=prec_cfg.thr_logprob, cr_thr=prec_cfg.thr_compratio,
                nospeech_thr=prec_cfg.thr_nospeech, pad=prec_cfg.pad_s, merge_gap=prec_cfg.merge_gap_s
            )
            logger.debug("Hard span candidates: %s", spans)
            if spans:
                total = sum(e - s for s, e in spans)
                logger.info(
                    "Precise rerun on %d span(s) (~%.1fs) using %s beam=%s patience=%s",
                    len(spans), total, prec_cfg.model, prec_cfg.beam_size, prec_cfg.patience,
                )
                repl = rerun_precise_on_spans(
                    VOCAL_AUDIO, spans, "en", prec_cfg.model, prec_cfg.compute_type,
                    prec_cfg.device, prec_cfg.beam_size, prec_cfg.patience, prec_cfg.window_max_s
                )
                fw_segments = splice_segments(fw_segments, repl)
                fw_segments = scrub_segments(fw_segments, SCR)
                fw_segments = [s for s in fw_segments if s["end"] > s["start"]]
                atomic_json(f"{base}_fw_segments_after_precise.json", {"segments": fw_segments})
                logger.debug("Segments after precise rerun: %d", len(fw_segments))
            else:
                logger.info("Precise rerun skipped; no hard spans detected")

        # post-correct (optional)
        if sp_rules:
            for s in fw_segments:
                s["text"] = apply_spelling_rules(s.get("text","") or "", sp_rules)
            atomic_json(f"{base}_fw_segments_postspell.json", {"segments": fw_segments})
            logger.info("Applied spelling map (%d rules)", len(sp_rules))

        # ---------------- alignment ----------------
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        fw_segments = clamp_to_duration(fw_segments, total_sec)
        aligned = run_alignment(fw_segments, VOCAL_AUDIO, asr_cfg.device, base, resume=args.resume)

        # ---------------- diarization ----------------
        token = ensure_hf_token()
        dia_df = run_diarization(VOCAL_AUDIO, asr_cfg.device, dia_cfg, token, total_sec, base, resume=args.resume)
        if dia_df.empty:
            raise SystemExit("[diarize] no speaker regions produced")

        # ---------------- assign speakers ----------------
        logger.info("Assigning speakers to words…")
        final = whisperx.assign_word_speakers(dia_df, aligned)

        # ---------------- write outputs ----------------
        write_srt_vtt_txt_json(final, base)
        logger.info("Processing complete → %s", outdir)

        return outdir


def main():
    args = parse_args()
    run_transcription(args)

if __name__ == "__main__":
    # Encourage unbuffered output so bars animate in more shells
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
