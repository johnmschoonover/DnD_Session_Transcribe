# Input pre-processing (vocal extraction / bandpass)
from dataclasses import dataclass

@dataclass
class PreprocessConfig:
    # "off" | "bandpass"
    vocal_extract: str = "off"
    # bandpass pipeline is: highpass 50 Hz + lowpass 7800 Hz + loudnorm → mono 16k
