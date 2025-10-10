import tempfile
import shlex
import pathlib

from .ffmpeg import ffmpeg

def preprocess_audio(src_audio: str, mode: str) -> str:
    """Return a path to the audio that should be fed to ASR/diarization."""
    if mode == "off":
        return src_audio
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    if mode == "bandpass":
        # 50–7800 Hz band + loudnorm; mono 16k; s16
        cmd = (
            f'ffmpeg -y -i {shlex.quote(src_audio)} -af '
            f'"highpass=f=50, lowpass=f=7800, loudnorm" '
            f'-ac 1 -ar 16000 -sample_fmt s16 {shlex.quote(out)}'
        )
        ffmpeg(cmd)
        return out
    if mode == "mdx_kim2":
        # Requires UVR5 CLI in PATH; fallback: bandpass if fail
        try:
            stem_dir = tempfile.mkdtemp(prefix="uvr_")
            cmd = (
                f'uvr5 -i {shlex.quote(src_audio)} -o {shlex.quote(stem_dir)} '
                f'--model mdx_kim2 --format wav --sr 16000 --mono'
            )
            ffmpeg(cmd)  # reuse ffmpeg runner for simplicity
            # choose vocals stem
            for cand in pathlib.Path(stem_dir).glob("**/*Vocals*.wav"):
                return str(cand)
            raise RuntimeError("UVR output missing vocals stem.")
        except Exception as e:
            print(f"[preproc] UVR mdx_kim2 failed: {e} → falling back to bandpass")
            return preprocess_audio(src_audio, "bandpass")
    return src_audio
