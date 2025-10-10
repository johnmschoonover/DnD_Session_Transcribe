import os, sys, tempfile


from typing import List, Tuple
from faster_whisper import WhisperModel

from utilities import ffmpeg_cut

# --- progress bars ---
from tqdm.auto import tqdm
PROGRESS_STREAM = sys.stdout
IS_TTY = PROGRESS_STREAM.isatty()

def rerun_precise_on_spans(src_audio: str, spans: List[Tuple[float,float]],
                           lang: str, model: str, compute: str,
                           beam: int, patience: float, max_window_s: float) -> List[Tuple[float,float,List[dict]]]:
    precise = WhisperModel(model, device="cuda", compute_type=compute)
    replacements = []

    total_precise = sum(e - s for (s, e) in spans)
    p_time = tqdm(
        total=total_precise, desc="[precise] audio", unit="s",
        dynamic_ncols=True, mininterval=0.2, smoothing=0.1,
        leave=True, disable=not IS_TTY, file=PROGRESS_STREAM,
        bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}s ({percentage:3.0f}%) [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for (s,e) in spans:
        t = s
        while t < e:
            w_end = min(e, t + max_window_s)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                ffmpeg_cut(src_audio, t, w_end, tmp_path)
                segs_iter, _ = precise.transcribe(
                    tmp_path, language=lang, beam_size=beam, patience=patience,
                    temperature=0.0, vad_filter=False
                )
                repl = []
                for i, seg in enumerate(segs_iter):
                    repl.append({
                        "id": i,
                        "start": float(seg.start) + t,
                        "end":   float(seg.end)   + t,
                        "text": seg.text or "",
                        "avg_logprob": getattr(seg, "avg_logprob", None),
                        "compression_ratio": getattr(seg, "compression_ratio", None),
                        "no_speech_prob": getattr(seg, "no_speech_prob", None),
                    })
                replacements.append((t, w_end, repl))
            finally:
                try: os.remove(tmp_path)
                except Exception: pass
            p_time.update(w_end - t)
            t = w_end

    p_time.close()
    return replacements