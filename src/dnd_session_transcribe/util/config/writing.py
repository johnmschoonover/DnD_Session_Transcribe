# Output naming/formatting
from dataclasses import dataclass

@dataclass
class WritingConfig:
    speaker_tag_prefix: str = "PLAYER"  # SPEAKER_01 → PLAYER_01
    out_prefix: str = "text"            # auto out dir: <wav_dir>/textN
    preview_limit_lines: int = 200
