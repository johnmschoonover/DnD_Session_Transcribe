import pathlib
import os

def copy_to_ram_if_requested(src_path: str, enable: bool) -> str:
    if not enable:
        return src_path
    ramdir = pathlib.Path("/dev/shm/whx")
    ramdir.mkdir(parents=True, exist_ok=True)
    dst = ramdir / pathlib.Path(src_path).name
    # If already exists with same size, reuse
    if not dst.exists() or os.path.getsize(dst) != os.path.getsize(src_path):
        print(f"[ram] copying → {dst}")
        with open(src_path, "rb") as fin, open(dst, "wb") as fout:
            for chunk in iter(lambda: fin.read(8 * 1024 * 1024), b""):
                fout.write(chunk)
    return str(dst)
