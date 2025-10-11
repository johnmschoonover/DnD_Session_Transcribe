#!/usr/bin/env python3
# run_whisperx.py
# Pipeline: (optional preprocess) → Faster-Whisper ASR → (optional precise re-run)
# → WhisperX alignment → pyannote diarization → assign speakers → write outputs

from __future__ import annotations
import os, sys, pathlib
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

# --- utilities (infra helpers) ---
from .adapters.copy_to_ram import copy_to_ram_if_requested
from .adapters.huggingface import ensure_hf_token
from .adapters.preprocess_audio import preprocess_audio
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


logger = logging.getLogger(__name__)


LOG = LoggingConfig()

LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


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
    ap.add_argument("--vocal-extract", choices=["off", "bandpass", "mdx_kim2"], default=None,
                    help="Override preprocessing (off/bandpass/mdx_kim2)")
    ap.add_argument(
        "--log-level",
        choices=tuple(LOG_LEVELS.keys()),
        default=LOG.level,
        help="Logging verbosity (default: %(default)s)",
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


# ======================= MAIN =======================
def main():
    args = parse_args()

    logging.basicConfig(
        level=LOG_LEVELS.get(args.log_level, logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logger.debug("Logging initialized at %s", args.log_level)

    # CLI → config overrides
    if args.vocal_extract is not None:
        PRE.vocal_extract = args.vocal_extract
    if args.hotwords_file:
        ASR.hotwords_file = args.hotwords_file
    if args.initial_prompt_file:
        ASR.initial_prompt_file = args.initial_prompt_file
    if args.asr_model:
        ASR.model = args.asr_model
    if args.asr_device:
        ASR.device = args.asr_device
    if args.asr_compute_type:
        ASR.compute_type = args.asr_compute_type
    if args.num_speakers:
        DIA.num_speakers = args.num_speakers
    if args.precise_rerun:
        PREC.enabled = True
    if args.precise_model:
        PREC.model = args.precise_model
    if args.precise_device:
        PREC.device = args.precise_device
    if args.precise_compute_type:
        PREC.compute_type = args.precise_compute_type

    audio = pathlib.Path(args.audio).resolve()
    logger.debug("Resolved audio path: %s", audio)
    if not audio.exists():
        raise SystemExit(f"Audio not found: {audio}")

    # outdir (auto textN if not provided)
    outdir = pathlib.Path(args.outdir).resolve() if args.outdir else next_outdir_for(str(audio), WR.out_prefix)
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
    mode = args.vocal_extract if args.vocal_extract is not None else PRE.vocal_extract
    VOCAL_AUDIO = preprocess_audio(src_for_io, mode)
    if VOCAL_AUDIO != str(audio):
        logger.info("Preprocessed audio via %s → %s", mode, VOCAL_AUDIO)
    else:
        logger.debug("Preprocessing skipped; using original audio")

    # hotwords / prompt / spelling
    hotwords = load_hotwords(ASR.hotwords_file)
    if hotwords:
        logger.debug("Loaded hotwords (%d chars)", len(hotwords))
    init_prompt = load_initial_prompt(ASR.initial_prompt_file)
    if init_prompt:
        logger.debug("Loaded initial prompt (%d chars)", len(init_prompt))
    sp_rules = load_spelling_map(args.spelling_map)
    if sp_rules:
        logger.debug("Loaded %d spelling correction rules", len(sp_rules))

    # duration (MUST be the file passed to ASR for proper %)
    total_sec = read_duration_seconds(VOCAL_AUDIO)
    logger.debug("Total audio duration (s): %.2f", total_sec)

    # ---------------- ASR ----------------
    fw_segments = run_asr(VOCAL_AUDIO, base, ASR, hotwords, init_prompt, resume=args.resume, total_sec=total_sec)
    fw_segments = scrub_segments(fw_segments, SCR)
    fw_segments = [s for s in fw_segments if s["end"] > s["start"]]
    atomic_json(f"{base}_fw_segments_scrubbed.json", {"segments": fw_segments})
    logger.debug("Scrubbed segments retained: %d", len(fw_segments))

    # ------------- precise re-run (optional) -------------
    if PREC.enabled:
        spans = find_hard_spans(
            fw_segments, dur=total_sec,
            logprob_thr=PREC.thr_logprob, cr_thr=PREC.thr_compratio,
            nospeech_thr=PREC.thr_nospeech, pad=PREC.pad_s, merge_gap=PREC.merge_gap_s
        )
        logger.debug("Hard span candidates: %s", spans)
        if spans:
            total = sum(e - s for s, e in spans)
            logger.info(
                "Precise rerun on %d span(s) (~%.1fs) using %s beam=%s patience=%s",
                len(spans), total, PREC.model, PREC.beam_size, PREC.patience,
            )
            repl = rerun_precise_on_spans(
                VOCAL_AUDIO, spans, "en", PREC.model, PREC.compute_type,
                PREC.device, PREC.beam_size, PREC.patience, PREC.window_max_s
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
    aligned = run_alignment(fw_segments, VOCAL_AUDIO, ASR.device, base, resume=args.resume)

    # ---------------- diarization ----------------
    token = ensure_hf_token()
    dia_df = run_diarization(VOCAL_AUDIO, ASR.device, DIA, token, total_sec, base, resume=args.resume)
    if dia_df.empty:
        raise SystemExit("[diarize] no speaker regions produced")

    # ---------------- assign speakers ----------------
    logger.info("Assigning speakers to words…")
    final = whisperx.assign_word_speakers(dia_df, aligned)

    # ---------------- write outputs ----------------
    write_srt_vtt_txt_json(final, base)
    logger.info("Processing complete → %s", outdir)


if __name__ == "__main__":
    # Encourage unbuffered output so bars animate in more shells
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
