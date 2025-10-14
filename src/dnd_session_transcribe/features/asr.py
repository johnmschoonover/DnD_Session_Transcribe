import pathlib
import json
import os, sys
import logging
from collections import deque

from typing import List, Optional
from faster_whisper import WhisperModel

from ..util.config import ASRConfig
from ..util.helpers import build_vad_params, atomic_json

# --- progress bars ---
from tqdm.auto import tqdm
PROGRESS_STREAM = sys.stdout
IS_TTY = PROGRESS_STREAM.isatty()


logger = logging.getLogger(__name__)

def run_asr(audio_path: str, out_base: pathlib.Path,
            asr_cfg: ASRConfig, hotwords: Optional[str], init_prompt: Optional[str],
            resume: bool, total_sec: float) -> List[dict]:

    raw_json = f"{out_base}_fw_segments_raw.json"
    if resume and os.path.exists(raw_json):
        logger.info("[ASR] resume: using cached raw segments")
        logger.debug("Loading segments from %s", raw_json)
        return json.load(open(raw_json, "r", encoding="utf-8"))["segments"]

    vad_params = build_vad_params(
        asr_cfg.vad_min_speech_ms, asr_cfg.vad_min_silence_ms,
        asr_cfg.vad_speech_pad_ms, asr_cfg.vad_max_speech_s
    )

    logger.info(
        "[ASR] model=%s device=%s compute=%s beam=%s patience=%s vad=%s",
        asr_cfg.model,
        asr_cfg.device,
        asr_cfg.compute_type,
        asr_cfg.beam_size,
        asr_cfg.patience,
        asr_cfg.use_vad,
    )
    logger.debug("VAD parameters: %s", vad_params)

    try:
        fw = WhisperModel(asr_cfg.model, device=asr_cfg.device, compute_type=asr_cfg.compute_type)
    except Exception as e:
        logger.warning("[ASR] init failed on %s: %s → retry float32", asr_cfg.compute_type, e)
        fw = WhisperModel(asr_cfg.model, device=asr_cfg.device, compute_type="float32")

    # Stream decode
    segs_iter, info = fw.transcribe(
        audio_path,
        language="en",
        beam_size=asr_cfg.beam_size,
        patience=asr_cfg.patience,
        temperature=asr_cfg.temperature,
        vad_filter=asr_cfg.use_vad,
        vad_parameters=vad_params,
        no_speech_threshold=asr_cfg.no_speech_threshold,
        compression_ratio_threshold=asr_cfg.compression_ratio_threshold,
        log_prob_threshold=asr_cfg.log_prob_threshold,
        hotwords=hotwords,
        initial_prompt=init_prompt,
    )

    # True 0–100%: drive bar by seconds processed vs total_sec
    p_time = tqdm(
        total=total_sec, desc="[ASR] audio", unit="s",
        dynamic_ncols=True, mininterval=0.2, smoothing=0.1,
        leave=True, disable=not IS_TTY, file=PROGRESS_STREAM,
        bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}s ({percentage:3.0f}%) [{elapsed}<{remaining}, {rate_fmt}]",
    )
    p_cnt = tqdm(
        desc="[ASR] segments", unit="seg",
        dynamic_ncols=True, mininterval=0.2, leave=False,
        disable=not IS_TTY, file=PROGRESS_STREAM,
    )

    recent_texts: deque[str] = deque(maxlen=5)

    segments, max_end = [], 0.0

    try:
        for i, s in enumerate(segs_iter):
            seg_text = s.text or ""
            seg = {
                "id": i,
                "start": float(s.start),
                "end": float(s.end),
                "text": seg_text,
                "avg_logprob": getattr(s, "avg_logprob", None),
                "compression_ratio": getattr(s, "compression_ratio", None),
                "no_speech_prob": getattr(s, "no_speech_prob", None),
            }
            segments.append(seg)
            p_cnt.update(1)
            if seg["end"] > max_end:
                max_end = seg["end"]
                p_time.n = min(max_end, total_sec)
                p_time.refresh()

            _track_repeated_lines(
                seg_text,
                recent_texts,
                asr_cfg,
                hotwords,
                init_prompt,
            )
    finally:
        p_cnt.close()
        p_time.close()
    atomic_json(raw_json, {"segments": segments})
    logger.debug("Wrote raw segments to %s", raw_json)
    logger.info("[ASR] produced %d segments", len(segments))
    return segments


def _track_repeated_lines(
    text: str,
    recent_texts: deque[str],
    asr_cfg: ASRConfig,
    hotwords: Optional[str],
    init_prompt: Optional[str],
) -> None:
    """Exit early if the ASR decoder is looping on the same line."""

    if not recent_texts.maxlen:
        return

    normalized = (text or "").strip()
    if not normalized:
        recent_texts.clear()
        return

    recent_texts.append(normalized)
    if len(recent_texts) < recent_texts.maxlen:
        return

    if len(set(recent_texts)) > 1:
        return

    repeated = normalized
    logger.error("[ASR] repeated output detected: %s", repeated)
    suggestion = _format_repetition_message(repeated, asr_cfg, hotwords, init_prompt)
    raise SystemExit(suggestion)


def _format_repetition_message(
    repeated: str,
    asr_cfg: ASRConfig,
    hotwords: Optional[str],
    init_prompt: Optional[str],
) -> str:
    """Create a helpful error message with tuning suggestions."""

    suggestions: list[str] = []
    if asr_cfg.use_vad:
        suggestions.append(
            "Increase `vad_min_speech_ms` (currently {speech} ms) or `vad_min_silence_ms` (currently {silence} ms) to avoid looping on tiny fragments.".format(
                speech=asr_cfg.vad_min_speech_ms,
                silence=asr_cfg.vad_min_silence_ms,
            )
        )
    else:
        suggestions.append("Enable VAD (`use_vad=True`) so repeated frames are trimmed automatically.")

    suggestions.append(
        "Relax decoding guardrails if the search is too rigid (temperature={temp}, beam_size={beam}, patience={pat}).".format(
            temp=asr_cfg.temperature,
            beam=asr_cfg.beam_size,
            pat=asr_cfg.patience,
        )
    )

    if hotwords:
        suggestions.append("Hotwords are enabled; try narrowing or removing them if they over-bias the transcription.")
    if init_prompt:
        suggestions.append("An initial prompt is loaded; consider simplifying it or running without one.")

    cfg_lines = [
        "model={model}",
        "device={device}",
        "compute_type={compute}",
        "beam_size={beam}",
        "patience={pat}",
        "temperature={temp}",
        "use_vad={vad}",
        "vad_min_speech_ms={speech}",
        "vad_min_silence_ms={silence}",
        "vad_speech_pad_ms={pad}",
        "vad_max_speech_s={max_speech}",
        "no_speech_threshold={no_speech}",
        "compression_ratio_threshold={compression}",
        "log_prob_threshold={logprob}",
    ]

    cfg_dump = "\n".join(
        line.format(
            model=asr_cfg.model,
            device=asr_cfg.device,
            compute=asr_cfg.compute_type,
            beam=asr_cfg.beam_size,
            pat=asr_cfg.patience,
            temp=asr_cfg.temperature,
            vad=asr_cfg.use_vad,
            speech=asr_cfg.vad_min_speech_ms,
            silence=asr_cfg.vad_min_silence_ms,
            pad=asr_cfg.vad_speech_pad_ms,
            max_speech=asr_cfg.vad_max_speech_s,
            no_speech=asr_cfg.no_speech_threshold,
            compression=asr_cfg.compression_ratio_threshold,
            logprob=asr_cfg.log_prob_threshold,
        )
        for line in cfg_lines
    )

    if hotwords:
        cfg_dump += f"\nhotwords_length={len(hotwords)}"
    if init_prompt:
        cfg_dump += f"\ninitial_prompt_length={len(init_prompt)}"

    message_lines = [
        f"[ASR] Detected the same line 5 times in a row: \"{repeated}\".",
        "The decoder is looping on the audio; tweak one of these settings:",
    ]
    message_lines.extend(f"  • {tip}" for tip in suggestions)
    message_lines.append("\nIf you need to ask for help, share this context:")
    message_lines.append("```\n" + cfg_dump + "\n```")

    return "\n".join(message_lines)
