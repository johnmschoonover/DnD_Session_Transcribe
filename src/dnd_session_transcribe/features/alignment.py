import os, pathlib
import json
import logging
import whisperx

from typing import List

from ..util.helpers import atomic_json


logger = logging.getLogger(__name__)

def run_alignment(segments: List[dict], audio_path: str, device: str, out_base: pathlib.Path, resume: bool):
    aligned_json = f"{out_base}_aligned.json"
    if resume and os.path.exists(aligned_json):
        logger.info("[align] resume: using cached alignment")
        logger.debug("Loading alignment from %s", aligned_json)
        return json.load(open(aligned_json, "r", encoding="utf-8"))
    align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
    logger.debug("Loaded alignment model metadata: %s", metadata)
    aligned = whisperx.align(segments, align_model, metadata, audio_path, device=device)
    atomic_json(aligned_json, aligned)
    logger.debug("Wrote alignment to %s", aligned_json)
    logger.info("[align] aligned %d segments", len(segments))
    return aligned
