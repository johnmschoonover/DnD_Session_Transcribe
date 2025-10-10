import torch
import whisperx
from typing import List, Tuple

from constants import ScrubConfig


def make_diarization_pipeline(token: str, device: str | torch.device):
    """
    Return a pyannote 3.x diarization pipeline compatible with various WhisperX versions.
    - Tries whisperx.diarize.DiarizationPipeline first (newer layout)
    - Falls back to whisperx.DiarizationPipeline (older layout)
    - Normalizes device to torch.device (some versions require it)
    """
    # normalize device
    if isinstance(device, str):
        dev_obj = torch.device(device)
    else:
        dev_obj = device

    # Try module path first (most reliable across versions)
    try:
        from whisperx.diarize import DiarizationPipeline as _DP
        return _DP(use_auth_token=token, device=dev_obj)
    except Exception as e1:
        # Fall back to top-level attribute (older builds)
        try:
            _DP = getattr(whisperx, "DiarizationPipeline")
            return _DP(use_auth_token=token, device=dev_obj)
        except Exception as e2:
            raise RuntimeError(
                "Could not construct WhisperX diarization pipeline.\n"
                f"- whisperx at: {getattr(whisperx, '__file__', 'unknown')}\n"
                f"- Tried whisperx.diarize.DiarizationPipeline -> {e1}\n"
                f"- Tried whisperx.DiarizationPipeline        -> {e2}\n"
                "Fixes:\n"
                "  • Ensure your HUGGINGFACE_TOKEN is set and accepted access for "
                "    pyannote/speaker-diarization-3.1 on Hugging Face.\n"
                "  • Or upgrade/downgrade whisperx to a stable version in your env.\n"
            )
        
def scrub_segments(segments: List[dict], cfg: ScrubConfig) -> List[dict]:
    out = []
    for s in segments:
        st, en = float(s["start"]), float(s["end"])
        if en - st < cfg.min_segment_len_s:
            continue
        txt = (s.get("text") or "").strip()
        lp  = s.get("avg_logprob")
        cr  = s.get("compression_ratio")
        ns  = s.get("no_speech_prob")
        if lp is not None and lp < cfg.drop_if_avg_logprob_lt:   continue
        if cr is not None and cr > cfg.drop_if_compratio_gt:     continue
        if ns is not None and ns > cfg.drop_if_nospeech_gt:      continue
        if len(txt) >= 12:
            uniq = len(set(txt.replace(" ", ""))) / max(1, len(txt.replace(" ", "")))
            if uniq < cfg.unique_char_ratio_min:
                continue
        out.append(s)
    return out

def overlaps(a0,a1,b0,b1) -> bool:
    return (a0 < b1) and (b0 < a1)


def find_hard_spans(segments: List[dict], dur: float, logprob_thr: float, cr_thr: float,
                    nospeech_thr: float, pad: float=0.5, merge_gap: float=3.0) -> List[Tuple[float,float]]:
    marks = []
    for s in segments:
        bad = False
        lp = s.get("avg_logprob"); cr = s.get("compression_ratio"); ns = s.get("no_speech_prob")
        if lp is not None and lp < logprob_thr: bad = True
        if cr is not None and cr > cr_thr:      bad = True
        if ns is not None and ns > nospeech_thr:bad = True
        if bad: marks.append((float(s["start"]), float(s["end"])))
    if not marks: return []
    marks.sort()
    merged = []
    cs, ce = marks[0]
    for s,e in marks[1:]:
        if s - ce <= merge_gap: ce = max(ce, e)
        else: merged.append((cs, ce)); cs, ce = s, e
    merged.append((cs, ce))
    out = []
    for s,e in merged:
        s = max(0.0, s - pad); e = min(dur, e + pad)
        if e - s >= 0.2: out.append((s,e))
    return out


def splice_segments(original: List[dict], replacements: List[Tuple[float,float,List[dict]]]) -> List[dict]:
    kept = []
    for s in original:
        s0, s1 = float(s["start"]), float(s["end"])
        if any(overlaps(s0, s1, r0, r1) for (r0, r1, _) in replacements):
            continue
        kept.append(s)
    new_list = kept[:]
    for _, _, repl in replacements:
        new_list.extend(repl)
    new_list.sort(key=lambda x: (float(x["start"]), float(x["end"])))
    for i, s in enumerate(new_list):
        s["id"] = i
    return new_list


def clamp_to_duration(segments: List[dict], dur: float) -> List[dict]:
    fixed, adjusted, dropped = [], 0, 0
    eps = 1e-3
    for s in segments:
        st, en = float(s["start"]), float(s["end"])
        if st >= dur - eps or en <= 0:
            dropped += 1; continue
        nst = max(0.0, min(st, dur - eps))
        nen = max(nst + 1e-3, min(en,  dur - eps))
        if abs(nst - st) > 1e-3 or abs(nen - en) > 1e-3: adjusted += 1
        fixed.append({**s, "start": nst, "end": nen})
    print(f"[align-prepare] kept={len(fixed)} adjusted={adjusted} dropped={dropped} (dur={dur:.2f}s)")
    return fixed