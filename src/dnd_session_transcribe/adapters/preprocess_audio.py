import tempfile
import shlex
import pathlib
import logging

from .ffmpeg import ffmpeg


logger = logging.getLogger(__name__)


def preprocess_audio(src_audio: str, mode: str) -> str:
    """Return a path to the audio that should be fed to ASR/diarization."""
    if mode == "off":
        return src_audio
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    if mode == "mdx_kim2":
        logger.warning(
            "[preproc] mdx_kim2 mode has been removed; defaulting to bandpass filtering."
        )
        mode = "bandpass"

    if mode == "bandpass":
        # 50–7800 Hz band + loudnorm; mono 16k; s16
        cmd = (
            f'ffmpeg -y -i {shlex.quote(src_audio)} -af '
            f'"highpass=f=50, lowpass=f=7800, loudnorm" '
            f'-ac 1 -ar 16000 -sample_fmt s16 {shlex.quote(out)}'
        )
        ffmpeg(cmd)
        logger.debug("Bandpass preprocessed audio written to %s", out)
        return out
    if mode not in {"off"}:
        logger.warning("[preproc] Unknown preprocess mode '%s'; returning original audio", mode)
    return src_audio
