import pathlib
import os
import json
from helpers import atomic_json

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

    # TXT
    txt_path = f"{base}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in final.get("segments", []):
            sp = seg.get("speaker", "")
            txt = (seg.get("text") or "").strip()
            f.write(f"[{sp}] {txt}\n" if sp else txt + "\n")

    # JSON (full)
    json_path = f"{base}.json"
    atomic_json(json_path, final)

    sizes = {
        "srt": os.path.getsize(srt_path),
        "vtt": os.path.getsize(vtt_path),
        "txt": os.path.getsize(txt_path),
        "json": os.path.getsize(json_path),
    }
    print(f"[write] sizes → srt:{sizes['srt']:,} vtt:{sizes['vtt']:,} txt:{sizes['txt']:,} json:{sizes['json']:,} bytes")
