import soundfile as sf

def read_duration_seconds(path: str) -> float:
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)
