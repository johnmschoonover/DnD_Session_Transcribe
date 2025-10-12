from __future__ import annotations
from typing import Optional, Dict, Any

def build_vad_params(
    min_speech_ms: int,
    min_silence_ms: int,
    speech_pad_ms: int,
    max_speech_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Return VAD params; omit max_speech_duration_s when None."""
    p = {
        "min_speech_duration_ms": int(min_speech_ms),
        "min_silence_duration_ms": int(min_silence_ms),
        "speech_pad_ms": int(speech_pad_ms),
    }
    if max_speech_s is not None:
        p["max_speech_duration_s"] = float(max_speech_s)
    return p
