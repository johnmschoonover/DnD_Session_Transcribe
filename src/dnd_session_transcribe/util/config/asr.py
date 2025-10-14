from __future__ import annotations

# ASR (Faster-Whisper) settings
from dataclasses import dataclass
from typing import Optional, Sequence

@dataclass
class ASRConfig:
    model: str = "large-v3"         # "large-v2" | "large-v3" | "distil-large-v3"
    device: str = "cuda"            # "cuda" | "cpu" | "mps"
    compute_type: str = "float16"   # "float16" | "float32"
    beam_size: Optional[int] = 10             # 8–12 sweet spot (None → sampling)
    patience: float = 1.5           # 1.2–2.0 typical
    best_of: Optional[int] = None
    temperature: float | Sequence[float] = 0.0        # Scalar or schedule
    use_vad: bool = True

    # VAD params (use build_vad_params in helpers)
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 200
    vad_max_speech_s: Optional[float] = None   # None → omit (don’t cap long monologues)

    # intrinsic guardrails
    no_speech_threshold: float = 0.58
    compression_ratio_threshold: float = 2.6
    log_prob_threshold: float = -1.1
    condition_on_previous_text: bool = True

    # optional biasing
    hotwords_file: Optional[str] = None
    initial_prompt_file: Optional[str] = None
