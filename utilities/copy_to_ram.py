import pathlib
import os
import logging


logger = logging.getLogger(__name__)

def copy_to_ram_if_requested(src_path: str, enable: bool) -> str:
    if not enable:
        logger.debug("RAM copy disabled; using %s", src_path)
        return src_path
    ramdir = pathlib.Path("/dev/shm/whx")
    ramdir.mkdir(parents=True, exist_ok=True)
    dst = ramdir / pathlib.Path(src_path).name
    # If already exists with same size, reuse
    if not dst.exists() or os.path.getsize(dst) != os.path.getsize(src_path):
        logger.info("[ram] copying → %s", dst)
        with open(src_path, "rb") as fin, open(dst, "wb") as fout:
            for chunk in iter(lambda: fin.read(8 * 1024 * 1024), b""):
                fout.write(chunk)
        logger.debug("Copied %s bytes to RAM", os.path.getsize(dst))
    else:
        logger.debug("Reusing RAM copy at %s", dst)
    return str(dst)
