# Second-pass “ultra-precise” re-ASR for hard spans
from dataclasses import dataclass

@dataclass
class PreciseRerunConfig:
    enabled: bool = False            # set True to always re-run hard spans
    model: str = "large-v3"
    compute_type: str = "float16"
    device: str = "cuda"
    beam_size: int = 16
    patience: float = 2.0
    window_max_s: float = 60.0       # split long spans into ≤ this
    pad_s: float = 0.5               # context per span
    # thresholds (mark span “hard” if any trip)
    thr_logprob: float = -1.0
    thr_compratio: float = 2.6
    thr_nospeech: float = 0.60
    merge_gap_s: float = 3.0
