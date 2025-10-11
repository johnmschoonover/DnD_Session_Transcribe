import pathlib
import os
import json
import logging
from helpers import atomic_json


logger = logging.getLogger(__name__)

def write_srt_vtt_txt_json(final: dict, base: pathlib.Path):
    def fmt_ts(t: float, vtt: bool=False) -> str:
        if t < 0: t = 0.0
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        sep = '.' if vtt else ','
        return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"

    # SRT
    srt_path = f"{base}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(final.get("segments", []), 1):
            st, en = float(seg["start"]), float(seg["end"])
            sp = seg.get("speaker", "")
            txt = (seg.get("text") or "").strip()
            label = f"[{sp}] " if sp else ""
            f.write(f"{i}\n{fmt_ts(st)} --> {fmt_ts(en)}\n{label}{txt}\n\n")
    logger.debug("Wrote SRT to %s", srt_path)

    # VTT
    vtt_path = f"{base}.vtt"
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in final.get("segments", []):
            st, en = float(seg["start"]), float(seg["end"])
            sp = seg.get("speaker", "")
            txt = (seg.get("text") or "").strip()
            label = f"[{sp}] " if sp else ""
            f.write(f"{fmt_ts(st, vtt=True)} --> {fmt_ts(en, vtt=True)}\n{label}{txt}\n\n")
    logger.debug("Wrote VTT to %s", vtt_path)

    # TXT
    txt_path = f"{base}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in final.get("segments", []):
            sp = seg.get("speaker", "")
            txt = (seg.get("text") or "").strip()
            f.write(f"[{sp}] {txt}\n" if sp else txt + "\n")
    logger.debug("Wrote TXT to %s", txt_path)

    # JSON (full)
    json_path = f"{base}.json"
    atomic_json(json_path, final)
    logger.debug("Wrote JSON to %s", json_path)

    sizes = {
        "srt": os.path.getsize(srt_path),
        "vtt": os.path.getsize(vtt_path),
        "txt": os.path.getsize(txt_path),
        "json": os.path.getsize(json_path),
    }
    logger.info(
        "[write] sizes → srt:%s vtt:%s txt:%s json:%s bytes",
        f"{sizes['srt']:,}",
        f"{sizes['vtt']:,}",
        f"{sizes['txt']:,}",
        f"{sizes['json']:,}",
    )
