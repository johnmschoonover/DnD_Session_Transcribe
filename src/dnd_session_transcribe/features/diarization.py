import json
import logging
import os
import pathlib

import pandas as pd

from ..util.config import DiarizationConfig
from ..util.config import WritingConfig as WR
from ..util.helpers import atomic_json
from ..util.processing import make_diarization_pipeline


logger = logging.getLogger(__name__)

def normalize_diarization_to_df(d, audio_dur: float, speaker_prefix: str) -> pd.DataFrame:
    try:
        from pyannote.core import Annotation
        if isinstance(d, Annotation):
            rows = []
            for seg, _, lab in d.itertracks(yield_label=True):
                rows.append({"start": float(seg.start), "end": float(seg.end), "speaker": str(lab)})
            df = pd.DataFrame(rows)
        else:
            if isinstance(d, list) and d and isinstance(d[0], dict):
                df = pd.DataFrame([
                    {"start": float(x["start"]), "end": float(x["end"]), "speaker": str(x.get("label","SPEAKER_00"))}
                    for x in d
                ])
            else:
                df = pd.DataFrame(columns=["start","end","speaker"])
    except Exception:
        df = pd.DataFrame(columns=["start","end","speaker"])

    if df.empty:
        return df

    df["speaker"] = df["speaker"].astype(str).str.replace("^SPEAKER", speaker_prefix, n=1, regex=True)
    df["start"] = df["start"].clip(lower=0, upper=audio_dur - 1e-3)
    df["end"]   = df["end"]  .clip(lower=0, upper=audio_dur - 1e-3)
    df = df[df["end"] > df["start"]].copy()
    df = df.sort_values(["start","end"]).reset_index(drop=True)
    return df


def run_diarization(audio_path: str, device: str, cfg: DiarizationConfig,
                    token: str, audio_dur: float, out_base: pathlib.Path, resume: bool) -> pd.DataFrame:

    dia_df_json = f"{out_base}_diarization_df.json"
    if resume and os.path.exists(dia_df_json):
        logger.info("[diarize] resume: using cached diarization df")
        logger.debug("Loading diarization DataFrame from %s", dia_df_json)
        return pd.DataFrame(json.load(open(dia_df_json, "r", encoding="utf-8")))

    if cfg.num_speakers <= 1:
        logger.info("[diarize] skipping model inference; assuming single speaker")
        df = pd.DataFrame(
            [{
                "start": 0.0,
                "end": max(1e-3, audio_dur - 1e-3),
                "speaker": f"{WR.speaker_tag_prefix}00",
            }]
        ) if audio_dur > 0 else pd.DataFrame(columns=["start", "end", "speaker"])
        atomic_json(dia_df_json, df.to_dict(orient="records"))
        logger.debug("Wrote diarization DataFrame to %s", dia_df_json)
        logger.info("[diarize-prepare] rows=%d", len(df))
        return df

    pipe = make_diarization_pipeline(token=token, device=device)
    logger.debug("Constructed diarization pipeline with token=%s device=%s", bool(token), device)

    # Loosen tiny splits if the pipeline exposes parameters
    try:
        defaults = pipe.pipeline.parameters()
        overrides = {}
        if "segmentation" in defaults:
            overrides.setdefault("segmentation", {})
            overrides["segmentation"]["min_duration_on"]  = cfg.seg_min_on
            overrides["segmentation"]["min_duration_off"] = cfg.seg_min_off
        if "speech_turn" in defaults:
            overrides.setdefault("speech_turn", {})
            overrides["speech_turn"]["min_duration_on"]  = cfg.turn_min_on
            overrides["speech_turn"]["min_duration_off"] = cfg.turn_min_off
        if "clustering" in defaults:
            overrides.setdefault("clustering", {})
            overrides["clustering"]["max_speakers_per_frame"] = cfg.max_speakers_per_frame
        if overrides:
            pipe.pipeline = pipe.pipeline.instantiate(overrides)
            logger.debug("Applied diarization parameter overrides: %s", overrides)
    except Exception:
        pass

    spk = cfg.num_speakers
    try:
        logger.debug("Running diarization on %s with num_speakers=%s", audio_path, spk)
        ann = pipe(audio_path, num_speakers=spk)
        df = normalize_diarization_to_df(ann, audio_dur, WR.speaker_tag_prefix)
        if df.empty and cfg.allow_range_fallback and spk and spk > 1:
            logger.warning("[diarize] 0 regions at %s; retry %s–%s…", spk, max(1, spk-1), spk+1)
            ann = pipe(audio_path, min_speakers=max(1, spk-1), max_speakers=spk+1)
            df = normalize_diarization_to_df(ann, audio_dur, WR.speaker_tag_prefix)
    except Exception as e:
        raise SystemExit(f"[diarize] failed: {e}")

    if df.empty:
        logger.warning("[diarize] pipeline returned no regions; defaulting to single speaker")
        if audio_dur <= 0:
            df = pd.DataFrame(columns=["start", "end", "speaker"])
        else:
            df = pd.DataFrame(
                [{
                    "start": 0.0,
                    "end": max(1e-3, audio_dur - 1e-3),
                    "speaker": f"{WR.speaker_tag_prefix}00",
                }]
            )

    atomic_json(dia_df_json, df.to_dict(orient="records"))
    logger.debug("Wrote diarization DataFrame to %s", dia_df_json)
    logger.info("[diarize-prepare] rows=%d", len(df))
    return df
