import pathlib

def next_outdir_for(audio_path: str, prefix: str) -> pathlib.Path:
    p = pathlib.Path(audio_path).resolve()
    base = p.parent
    i = 0
    while True:
        cand = base / f"{prefix}{i}"
        if not cand.exists():
            return cand
        i += 1
