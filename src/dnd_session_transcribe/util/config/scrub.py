# Post-ASR scrub controls (removing loops/junk)
from dataclasses import dataclass

@dataclass
class ScrubConfig:
    drop_if_avg_logprob_lt: float = -1.1
    drop_if_compratio_gt: float = 2.6
    drop_if_nospeech_gt: float = 0.60
    unique_char_ratio_min: float = 0.15   # drop if below & long text
    short_repeat_word_len_max: int = 8
    min_segment_len_s: float = 0.2
