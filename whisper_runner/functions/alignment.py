import os, pathlib
import json
import whisperx

from typing import List

from helpers import atomic_json

def run_alignment(segments: List[dict], audio_path: str, device: str, out_base: pathlib.Path, resume: bool):
    aligned_json = f"{out_base}_aligned.json"
    if resume and os.path.exists(aligned_json):
        print("[align] resume: using cached alignment")
        return json.load(open(aligned_json, "r", encoding="utf-8"))
    align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
    aligned = whisperx.align(segments, align_model, metadata, audio_path, device=device)
    atomic_json(aligned_json, aligned)
    return aligned