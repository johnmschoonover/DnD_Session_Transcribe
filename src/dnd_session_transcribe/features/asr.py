import pathlib
import json
import os, sys
import logging

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

    segments, max_end = [], 0.0
    for i, s in enumerate(segs_iter):
        seg = {
            "id": i,
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text or "",
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

    p_cnt.close(); p_time.close()
    atomic_json(raw_json, {"segments": segments})
    logger.debug("Wrote raw segments to %s", raw_json)
    logger.info("[ASR] produced %d segments", len(segments))
    return segments
