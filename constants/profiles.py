# Speaker profile matching (optional)
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProfilesConfig:
    dir: Optional[str] = None                 # default to <wav_dir>/audio_profiles if None
    sim_threshold: float = 0.65               # cosine similarity
    max_speech_per_spk_s: float = 30.0
    min_seg_len_s: float = 0.8
    embedding_model_id: str = "pyannote/embedding"
