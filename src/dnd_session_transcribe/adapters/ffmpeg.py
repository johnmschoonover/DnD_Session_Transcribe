import shlex
import subprocess


def ffmpeg(cmd: str) -> None:
    subprocess.run(
        cmd,
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def ffmpeg_cut(src: str, start: float, end: float, out_wav: str) -> None:
    cmd = (
        f"ffmpeg -y -ss {start:.3f} -to {end:.3f} -i {shlex.quote(src)} "
        f"-ac 1 -ar 16000 -sample_fmt s16 {shlex.quote(out_wav)}"
    )
    ffmpeg(cmd)
