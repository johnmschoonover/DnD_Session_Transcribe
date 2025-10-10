# Pyannote diarization settings
from dataclasses import dataclass

@dataclass
class DiarizationConfig:
    model_id: str = "pyannote/speaker-diarization-3.1"
    num_speakers: int = 5
    allow_range_fallback: bool = True  # try [num-1, num+1] if empty

    # light relax overrides (applied if keys exist in pipeline params)
    seg_min_on: float = 0.05
    seg_min_off: float = 0.10
    turn_min_on: float = 0.10
    turn_min_off: float = 0.10
    max_speakers_per_frame: int = 2
