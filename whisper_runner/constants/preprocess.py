# Input pre-processing (vocal extraction / bandpass)
from dataclasses import dataclass

@dataclass
class PreprocessConfig:
    # "off" | "bandpass" | "mdx_kim2" (UVR5 CLI)
    vocal_extract: str = "off"
    # bandpass pipeline is: highpass 50 Hz + lowpass 7800 Hz + loudnorm → mono 16k
