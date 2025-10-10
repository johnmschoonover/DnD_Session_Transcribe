#!/usr/bin/env python3
# run_whisperx_resume.py
# GPU ASR (Faster-Whisper) → WhisperX alignment → pyannote diarization → optional profile naming → writers
# Checkpoints + --resume/--force + optional --ram + vocal-extract + hallucination scrub
# Auto output dir: <wav_dir>/<prefix><N>, N increments; --resume reuses latest N

import os, sys, re, json, pathlib, hashlib, pickle, shutil, tempfile, subprocess, shlex
import argparse
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from faster_whisper import WhisperModel

from constants import ASRConfig, DiarizationConfig, PreciseRerunConfig
from helpers import build_vad_params, load_hotwords, atomic_json

# =============== CLI & OUTDIR LOGIC =================
ap = argparse.ArgumentParser(description="WhisperX pipeline with checkpoints, RAM cache, and auto out-dir")
ap.add_argument("audio", help="Path to source WAV/MP3/etc.")
ap.add_argument("--resume", action="store_true", help="resume from existing artifacts when possible (reuse latest textN if auto)")
ap.add_argument("--force",  action="store_true", help="ignore artifacts; recompute everything")
ap.add_argument("--ram",    action="store_true", help="copy WAV to /dev/shm/wx first (fast I/O)")
ap.add_argument("--num-speakers", type=int, default=5, help="exact number of speakers (fallback tries +/-1)")
ap.add_argument("--vocal-extract", choices=["off","mdx_kim2","bandpass"], default="off",
                help="pre-process audio to isolate vocals before ASR/align/diarize")
ap.add_argument("--pause-before-write", action="store_true",
                help="stop after assign/profile-naming; write PREVIEW and exit (no SRT/VTT/TXT/JSON)")
ap.add_argument("--out-prefix", default="text", help="output folder prefix inside WAV’s folder (default: text)")
ap.add_argument("--out-dir", default=None, help="explicit output directory (overrides auto)")
ap.add_argument("--profiles-dir", default=None, help="directory with profile WAVs (e.g., /path/audio_profiles)")
args = ap.parse_args()

AUDIO_SRC = os.path.abspath(args.audio)
if not os.path.exists(AUDIO_SRC):
    sys.exit(f"[io] Source audio missing: {AUDIO_SRC}")

wav_dir  = os.path.dirname(AUDIO_SRC)
wav_stem = pathlib.Path(AUDIO_SRC).stem

def pick_output_dir():
    if args.out_dir:  # explicit
        out = pathlib.Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out
    # auto under the wav dir, prefix like textN
    existing = []
    for name in os.listdir(wav_dir):
        if not name.startswith(args.out_prefix): continue
        m = re.fullmatch(rf"{re.escape(args.out_prefix)}(\d+)", name)
        if m:
            existing.append(int(m.group(1)))
    if args.resume and existing:
        N = max(existing)  # reuse latest for resume
    else:
        N = (max(existing) + 1) if existing else 0
    out = pathlib.Path(wav_dir) / f"{args.out_prefix}{N}"
    out.mkdir(parents=True, exist_ok=True)
    return out

OUTDIR = pick_output_dir()
print(f"[out] Output directory: {OUTDIR}")

# =============== CONFIG ============================
LANG        = "en"
ASR_MODEL   = "large-v3"
DEVICE      = "cuda"
COMPUTE     = "float16"      # fallback to float32 if init fails
BEAM_SIZE   = 12
SPEAKER_TAG = "PLAYER"
MODEL_DIA   = "pyannote/speaker-diarization-3.1"   # or "pyannote/speaker-diarization-3.0"
PROFILES_DIR = args.profiles_dir or os.path.join(wav_dir, "audio_profiles")

# Profile naming knobs (optional step)
SIM_THRESHOLD       = 0.65
MAX_SPEECH_PER_SPK  = 30.0
MIN_SEG_LEN         = 0.8

print("GUARD: FW->WX pipeline (checkpointable, RAM cache optional)")

# =============== Helpers ===========================
def atomic_write(path, data=None, mode="json", binary_obj: bytes | None = None):
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if mode == "json":
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif mode == "text":
        with tmp.open("w", encoding="utf-8") as f:
            f.write(data)
    elif mode == "pickle":
        with tmp.open("wb") as f:
            pickle.dump(data, f)
    elif mode == "bytes":
        with tmp.open("wb") as f:
            f.write(binary_obj)
    else:
        raise ValueError("unknown atomic_write mode")
    tmp.replace(path)

def file_signature(p):
    st = os.stat(p)
    return {"size": st.st_size, "mtime": int(st.st_mtime)}

def config_signature():
    cfg = dict(
        LANG=LANG, ASR_MODEL=ASR_MODEL, DEVICE=DEVICE, COMPUTE=COMPUTE, BEAM_SIZE=BEAM_SIZE,
        SPEAKER_TAG=SPEAKER_TAG, MODEL_DIA=MODEL_DIA,
        SIM_THRESHOLD=SIM_THRESHOLD, MAX_SPEECH_PER_SPK=MAX_SPEECH_PER_SPK, MIN_SEG_LEN=MIN_SEG_LEN,
        NUM_SPEAKERS=args.num_speakers, VOCAL_EXTRACT=args.vocal_extract
    )
    s = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest()

def load_state(base):
    p = pathlib.Path(f"{base}.state.json")
    if not p.exists(): return {}
    try: return json.load(p.open("r", encoding="utf-8"))
    except Exception: return {}

def save_state(base, state):
    atomic_write(f"{base}.state.json", state, mode="json")

def ok_to_reuse(step_key, state, audio_sig, cfg_sig):
    if not args.resume: return False
    if state.get("audio_sig") != audio_sig: return False
    if state.get("cfg_sig")   != cfg_sig:   return False
    return bool(state.get("steps", {}).get(step_key) is True)

def mark_done(state, step_key, extra=None):
    state.setdefault("steps", {})[step_key] = True
    if extra:
        state.setdefault("meta", {}).update(extra)

# =============== Optional RAM copy =================
def ensure_local_audio_in_ram(src):
    os.makedirs("/dev/shm/wx", exist_ok=True)
    dst = os.path.join("/dev/shm/wx", pathlib.Path(src).name)
    need_copy = True
    try:
        if os.path.exists(dst):
            s_src = file_signature(src)
            s_dst = file_signature(dst)
            if s_src == s_dst: need_copy = False
    except Exception:
        need_copy = True
    if need_copy:
        print("[io] copying to RAM:", dst)
        shutil.copy2(src, dst)
    else:
        print("[io] RAM copy already up-to-date:", dst)
    return dst

if args.ram:
    LOCAL_AUDIO = ensure_local_audio_in_ram(AUDIO_SRC)
else:
    LOCAL_AUDIO = AUDIO_SRC

# =============== Pre-processing (vocal extract) ================
def run_ffmpeg_bandpass(src_path: str) -> str:
    """Light SNR boost: 50Hz highpass + 7800Hz lowpass + loudnorm + mono 16k."""
    out_wav = os.path.join(tempfile.gettempdir(), f"wx_bp_{os.path.basename(src_path)}.wav")
    cmd = (
        f'ffmpeg -y -i {shlex.quote(src_path)} '
        f'-af "highpass=f=50, lowpass=f=7800, loudnorm=i=-18:lra=7:tp=-2" '
        f'-ac 1 -ar 16000 -sample_fmt s16 {shlex.quote(out_wav)}'
    )
    print("[pre] ffmpeg bandpass:", cmd)
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return out_wav

def try_uvr_mdx_kim2(src_path: str) -> str | None:
    """Try UVR5 CLI with Kim_Vocal_2; returns vocals WAV or None."""
    tmp_dir = tempfile.mkdtemp(prefix="uvr_")
    cmd = [
        "uvr5", "-m", "Kim_Vocal_2", "-i", src_path, "-o", tmp_dir, "--vocals-only"
    ]
    print("[pre] UVR5 mdx_kim2:", " ".join(shlex.quote(c) for c in cmd))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        print("[pre] UVR5 not found on PATH; falling back to bandpass.")
        return None
    except subprocess.CalledProcessError:
        print("[pre] UVR5 failed; falling back to bandpass.")
        return None
    for fn in os.listdir(tmp_dir):
        if fn.lower().endswith(".wav") and "vocal" in fn.lower():
            return os.path.join(tmp_dir, fn)
    print("[pre] UVR output not found; falling back to bandpass.")
    return None

def maybe_preprocess_audio(src_path: str, mode: str) -> str:
    if mode == "off":
        return src_path
    if mode == "mdx_kim2":
        v = try_uvr_mdx_kim2(src_path)
        if v: return v
        try:
            return run_ffmpeg_bandpass(src_path)
        except Exception as e:
            print("[pre] bandpass fallback failed:", e, "; using original audio.")
            return src_path
    if mode == "bandpass":
        try:
            return run_ffmpeg_bandpass(src_path)
        except Exception as e:
            print("[pre] bandpass failed:", e, "; using original audio.")
            return src_path
    return src_path

if args.vocal_extract != "off":
    print(f"[pre] vocal-extract mode: {args.vocal_extract}")
VOCAL_AUDIO = maybe_preprocess_audio(LOCAL_AUDIO, args.vocal_extract) if args.vocal_extract != "off" else LOCAL_AUDIO

# =============== HF token/access check ===============
def check_hf_token_and_access():
    from huggingface_hub import HfApi
    try:
        from huggingface_hub.errors import HfHubHTTPError
    except Exception:
        try:
            from huggingface_hub.utils._errors import HfHubHTTPError
        except Exception:
            HfHubHTTPError = Exception

    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        sys.exit(
            "\n[HF] Missing HUGGINGFACE_TOKEN.\n"
            "  conda env config vars set HUGGINGFACE_TOKEN=hf_xxx\n"
            "  conda deactivate && conda activate whisperx\n"
        )
    api = HfApi()
    try:
        user = api.whoami(token=token).get("name", "<unknown>")
        print(f"[HF] Auth OK as: {user}")
    except Exception as e:
        sys.exit(f"[HF] whoami failed: {e}")
    try:
        api.model_info(MODEL_DIA, token=token)
        print(f"[HF] Access OK → {MODEL_DIA}")
    except HfHubHTTPError as e:
        sys.exit(
            f"[HF] Access denied for {MODEL_DIA}. Accept terms:\n"
            f"  https://huggingface.co/{MODEL_DIA}\n{e}\n"
        )

# =============== Output base / state =================
base = OUTDIR / pathlib.Path(AUDIO_SRC).stem
state = ({} if args.force else None) or {}
if not args.force:
    # try load if exists
    st = load_state(base)
    if st: state = st

audio_sig = file_signature(AUDIO_SRC)  # tie checkpoints to original file
cfg_sig = config_signature()
state["audio_sig"] = audio_sig
state["cfg_sig"] = cfg_sig
state.setdefault("meta", {})["local_audio"] = LOCAL_AUDIO
save_state(base, state)

# =============== Hallucination scrub =================
def scrub_hallucinations(segments):
    cleaned = []
    drop_count = 0

    def suspicious_text(txt: str) -> bool:
        t = (txt or "").strip().lower()
        if not t: return True
        words = [w for w in re.split(r"[^\w']+", t) if w]
        if len(words) <= 2:
            if words and len(set(words)) == 1 and len(words[0]) <= 8:
                return True
        unique_ratio = len(set(t)) / max(1, len(t))
        if unique_ratio < 0.15 and len(t) >= 6:
            return True
        return False

    for s in segments:
        ok = True
        if s.get("avg_logprob") is not None and s["avg_logprob"] < -1.1: ok = False
        if s.get("compression_ratio") is not None and s["compression_ratio"] > 2.6: ok = False
        if s.get("no_speech_prob") is not None and s["no_speech_prob"] > 0.6: ok = False
        if suspicious_text(s.get("text", "")): ok = False
        if ok: cleaned.append(s)
        else: drop_count += 1

    merged = []
    for s in cleaned:
        if merged and (s["text"].strip().lower() == merged[-1]["text"].strip().lower()):
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append(s)

    if drop_count:
        print(f"[asr-scrub] dropped {drop_count} suspicious segments; kept {len(merged)}")

    from collections import Counter
    texts = [(seg["text"] or "").strip().lower() for seg in merged if seg.get("text")]
    top = Counter(texts).most_common(1)
    if top and len(merged) >= 100:
        phrase, count = top[0]
        if count / len(merged) >= 0.3 and len(phrase.split()) <= 3:
            print(f"[asr-scrub] WARNING: top phrase '{phrase}' = {count}/{len(merged)} "
                  f"({100*count/len(merged):.1f}%). Looks loopish.")
    return merged

# =============== ASR ================================
ASR_JSON = f"{base}_fw_segments_raw.json"
if ok_to_reuse("asr", state, audio_sig, cfg_sig) and os.path.exists(ASR_JSON):
    print("[ASR] resume: loading", ASR_JSON)
    fw_segments = json.load(open(ASR_JSON, "r", encoding="utf-8"))["segments"]
else:
    check_hf_token_and_access()
    print(f"[ASR] Faster-Whisper: model={ASR_MODEL}, device={DEVICE}, compute={COMPUTE}")
    try:
        fw = WhisperModel(ASR_MODEL, device=DEVICE, compute_type=COMPUTE)
    except Exception as e:
        print(f"[ASR] init on {COMPUTE} failed: {e} → retry float32")
        fw = WhisperModel(ASR_MODEL, device=DEVICE, compute_type="float32")

    segs_iter, info = fw.transcribe(
        VOCAL_AUDIO,
        language=LANG,
        beam_size=BEAM_SIZE,
        patience=2.0,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={
            "min_speech_duration_ms": 250,
            # "max_speech_duration_s": None,
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        },
        no_speech_threshold=0.58,
        compression_ratio_threshold=2.6,
        log_prob_threshold=-1.1,
    )

    fw_segments = []
    last_end = 0.0
    total = float(getattr(info, "duration", 0.0)) or None
    with tqdm(total=total, unit="s", desc="ASR", leave=True) as pbar:
        for i, s in enumerate(segs_iter):
            fw_segments.append({
                "id": i, "start": s.start, "end": s.end, "text": s.text or "",
                "avg_logprob": getattr(s, "avg_logprob", None),
                "compression_ratio": getattr(s, "compression_ratio", None),
                "no_speech_prob": getattr(s, "no_speech_prob", None),
            })
            inc = max(0.0, (s.end or 0.0) - last_end)
            if total: pbar.update(inc)
            last_end = max(last_end, s.end or last_end)

    atomic_write(ASR_JSON, {"segments": fw_segments}, mode="json")
    mark_done(state, "asr", extra={"audio_used": VOCAL_AUDIO}); save_state(base, state)

# Scrub hallucinations before align
fw_segments = scrub_hallucinations(fw_segments)

# =============== WhisperX ALIGN =====================
import whisperx, torch, soundfile as sf
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

info_aud = sf.info(VOCAL_AUDIO)
AUDIO_DUR = float(info_aud.frames) / float(info_aud.samplerate)

def clamp_and_filter(segments, dur, eps=1e-3):
    fixed, dropped, adjusted = [], 0, 0
    for s in segments:
        st, en = float(s["start"]), float(s["end"])
        if st >= dur - eps or en <= 0:
            dropped += 1; continue
        nst = max(0.0, min(st, dur - eps))
        nen = max(nst + 1e-3, min(en, dur - eps))
        if abs(nst-st)>1e-3 or abs(nen-en)>1e-3: adjusted += 1
        fixed.append({"id": s["id"], "start": nst, "end": nen, "text": s.get("text","")})
    print(f"[align-prepare] kept={len(fixed)} adjusted={adjusted} dropped={dropped} (dur={dur:.2f}s)")
    return fixed

fw_segments = clamp_and_filter(fw_segments, AUDIO_DUR)

ALIGNED_JSON = f"{base}_aligned_segments.json"
if ok_to_reuse("align", state, audio_sig, cfg_sig) and os.path.exists(ALIGNED_JSON):
    print("[ALIGN] resume: loading", ALIGNED_JSON)
    aligned_segments = json.load(open(ALIGNED_JSON, "r", encoding="utf-8"))["segments"]
else:
    print("[ALIGN] loading alignment model…")
    align_model, metadata = whisperx.load_align_model(language_code=LANG, device=DEVICE)

    def chunk_by_seconds(segments, max_secs=600.0):
        batch, acc = [], 0.0
        for s in segments:
            dur = float(s["end"]) - float(s["start"])
            if batch and acc + dur > max_secs:
                yield batch; batch, acc = [], 0.0
            batch.append(s); acc += max(0.0, dur)
        if batch: yield batch

    print("[ALIGN] aligning with progress…")
    aligned_segments = []
    total_secs = AUDIO_DUR
    covered_align = 0.0
    with tqdm(total=total_secs, unit="s", desc="Align", leave=True) as pbar:
        for seg_batch in chunk_by_seconds(fw_segments, max_secs=600.0):
            batch_result = whisperx.align(seg_batch, align_model, metadata, VOCAL_AUDIO, device=DEVICE)
            aligned_segments.extend(batch_result["segments"])
            batch_end = max(s["end"] for s in seg_batch) if seg_batch else covered_align
            covered_align = max(covered_align, batch_end)
            pbar.n = min(covered_align, total_secs); pbar.refresh()

    atomic_write(ALIGNED_JSON, {"segments": aligned_segments}, mode="json")
    mark_done(state, "align"); save_state(base, state)

aligned = {"segments": aligned_segments}

# =============== DIARIZATION ========================
from pyannote.audio import Pipeline
token = os.getenv("HUGGINGFACE_TOKEN")
device_obj = torch.device(DEVICE) if not isinstance(DEVICE, torch.device) else DEVICE

DIA_RAW = f"{base}_diarization_raw.json"
DIA_PKL = f"{base}_dia_df.pkl"

def build_relaxed_params(pipe):
    defaults = pipe.parameters()
    overrides = {}
    seg_defaults = defaults.get("segmentation", {})
    seg_over = {}
    if "onset" in seg_defaults:  seg_over["onset"]  = 0.28
    if "offset" in seg_defaults: seg_over["offset"] = 0.12
    if "min_duration_on"  in seg_defaults: seg_over["min_duration_on"]  = 0.05
    if "min_duration_off" in seg_defaults: seg_over["min_duration_off"] = 0.10
    if seg_over: overrides["segmentation"] = seg_over

    turn_defaults = defaults.get("speech_turn", {})
    turn_over = {}
    if "min_duration_on"  in turn_defaults: turn_over["min_duration_on"]  = 0.05
    if "min_duration_off" in turn_defaults: turn_over["min_duration_off"] = 0.10
    if turn_over: overrides["speech_turn"] = turn_over

    clust_defaults = defaults.get("clustering", {})
    clust_over = {}
    if "max_speakers_per_frame" in clust_defaults: clust_over["max_speakers_per_frame"] = 2
    if clust_over: overrides["clustering"] = clust_over
    return overrides

def ann_to_list(ann):
    out = []
    for seg, _, lab in ann.itertracks(yield_label=True):
        out.append({"start": float(seg.start), "end": float(seg.end), "label": str(lab)})
    return out

if ok_to_reuse("diarize", state, audio_sig, cfg_sig) and os.path.exists(DIA_PKL):
    print("[DIA] resume: loading", DIA_PKL)
    with open(DIA_PKL, "rb") as f:
        dia_df = pickle.load(f)
else:
    print(f"[DIA] running {MODEL_DIA}…")
    pipe = Pipeline.from_pretrained(MODEL_DIA, use_auth_token=token).to(device_obj)
    params = build_relaxed_params(pipe)
    if params: pipe = pipe.instantiate(params)

    try:
        ann = pipe(VOCAL_AUDIO, num_speakers=args.num_speakers)
        dia_list = ann_to_list(ann)
        if not dia_list:
            print(f"[DIA] no segments; retry with {args.num_speakers-1}–{args.num_speakers+1} speakers…")
            ann = pipe(VOCAL_AUDIO, min_speakers=max(1, args.num_speakers-1), max_speakers=args.num_speakers+1)
            dia_list = ann_to_list(ann)
    except Exception as e:
        sys.exit(f"[DIA] pipeline error: {e}")

    if not dia_list:
        sys.exit("[DIA] No speaker regions even after fallback. Check audio loudness/SNR.")

    atomic_write(DIA_RAW, {"diarization": dia_list}, mode="json")

    rows = []
    for d in dia_list:
        st = float(d["start"]); en = float(d["end"])
        lab = str(d.get("label", "SPEAKER_00"))
        if en > st:
            rows.append({"start": st, "end": en, "speaker": lab})
    dia_df = pd.DataFrame(rows, columns=["start", "end", "speaker"])

    import soundfile as sf
    dur = AUDIO_DUR
    dia_df["start"] = dia_df["start"].clip(lower=0, upper=dur - 1e-3)
    dia_df["end"]   = dia_df["end"]  .clip(lower=0, upper=dur - 1e-3)
    dia_df = dia_df[dia_df["end"] > dia_df["start"]].copy().sort_values(["start","end"]).reset_index(drop=True)
    dia_df["speaker"] = dia_df["speaker"].astype(str).str.replace("SPEAKER", SPEAKER_TAG, n=1)

    atomic_write(DIA_PKL, dia_df, mode="pickle")
    mark_done(state, "diarize"); save_state(base, state)

print(f"[DIA] rows={len(dia_df)}")

# =============== ASSIGN ============================
FINAL_JSON = f"{base}_final_segments.json"
if ok_to_reuse("assign", state, audio_sig, cfg_sig) and os.path.exists(FINAL_JSON):
    print("[ASSIGN] resume: loading", FINAL_JSON)
    final = json.load(open(FINAL_JSON, "r", encoding="utf-8"))
else:
    print("[ASSIGN] assigning speakers to words…")
    final = whisperx.assign_word_speakers(dia_df, {"segments": aligned["segments"] if "segments" in aligned else aligned})
    atomic_write(FINAL_JSON, final, mode="json")
    mark_done(state, "assign"); save_state(base, state)

# =============== Optional: PROFILE NAMING ==========
def do_profile_naming():
    import glob, math
    from pyannote.audio import Model, Inference
    import torch as _torch

    if not PROFILES_DIR or not os.path.isdir(PROFILES_DIR):
        print("[profiles] none found; skipping"); return

    wavs = sorted(glob.glob(os.path.join(PROFILES_DIR, "*.wav")))
    if not wavs:
        print("[profiles] none found; skipping"); return

    names = [os.path.splitext(os.path.basename(p))[0] for p in wavs]
    print(f"[profiles] {names}")

    try:
        spk_model = Model.from_pretrained("pyannote/embedding", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))
    except Exception as e:
        print(f"[profiles] embedding model unavailable: {e} (skip)"); return

    infer = Inference(spk_model, window="whole", device=_torch.device(DEVICE) if not isinstance(DEVICE, _torch.device) else DEVICE)

    def emb_file(p):
        try: return np.asarray(infer(p))
        except Exception as e: print("[profiles] fail:", p, e); return None

    prof_embs = [emb_file(p) for p in wavs]
    pairs = [(n,e) for n,e in zip(names, prof_embs) if e is not None]
    if not pairs:
        print("[profiles] no usable profile embeddings"); return
    names, prof_embs = zip(*pairs)
    prof_mat = np.vstack([e.reshape(1,-1) for e in prof_embs])

    from collections import defaultdict as _dd
    clusters = _dd(list)
    for _, row in dia_df.iterrows():
        clusters[str(row["speaker"])].append((float(row["start"]), float(row["end"])))
    for k in clusters:
        clusters[k].sort(key=lambda t:(t[1]-t[0]), reverse=True)

    def emb_cluster(audio_path, segs, budget=MAX_SPEECH_PER_SPK):
        used=0.; embs=[]
        for st,en in segs:
            dur=max(0.,en-st)
            if dur<MIN_SEG_LEN: continue
            step=2.0; win=3.0
            if dur<=win:
                crop={"uri":audio_path,"start":float(st),"end":float(en)}
                try: embs.append(np.asarray(infer(crop)))
                except: pass
                used+=dur
            else:
                n=int(np.ceil((dur-win)/step))+1
                for i in range(n):
                    cst=st+i*step; cen=min(st+i*step+win,en)
                    if cen-cst<MIN_SEG_LEN: continue
                    crop={"uri":audio_path,"start":float(cst),"end":float(cen)}
                    try: embs.append(np.asarray(infer(crop)))
                    except: pass
                    used+=(cen-cst)
                    if used>=budget: break
            if used>=budget: break
        if not embs: return None
        E=np.vstack([e.reshape(1,-1) for e in embs])
        E=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-8)
        c=E.mean(axis=0); c=c/(np.linalg.norm(c)+1e-8)
        return c.reshape(1,-1)

    def cos(A,B):
        if A is None: return None
        A=A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-8)
        B=B/(np.linalg.norm(B,axis=1,keepdims=True)+1e-8)
        return A@B.T

    label_map={}
    for lab in sorted(clusters.keys()):
        cem=emb_cluster(VOCAL_AUDIO, clusters[lab])
        sims=cos(cem, prof_mat)
        if sims is None: continue
        j=int(np.argmax(sims)); best=float(sims.flatten()[j]); nm=names[j]
        print(f"[profiles] {lab} -> {nm} (cos={best:.3f})")
        if best>=SIM_THRESHOLD: label_map[lab]=nm

    if not label_map:
        print("[profiles] no matches passed threshold; leaving PLAYER_*"); return

    for seg in final.get("segments", []):
        sp=seg.get("speaker")
        if isinstance(sp,str) and sp in label_map:
            seg["speaker"]=label_map[sp]
    dia_df["speaker"]=dia_df["speaker"].astype(str).map(lambda s: label_map.get(s,s))
    print("[profiles] applied:", label_map)

# Profiles step is quick; always attempt it.
do_profile_naming()

# =============== Optional pause before write =========
if args.pause_before_write:
    from collections import defaultdict
    talk = defaultdict(float)
    for s in final["segments"]:
        talk[s.get("speaker","UNK")] += float(s["end"]) - float(s["start"])
    print("\n[preview] talk time by speaker:")
    for sp, secs in sorted(talk.items(), key=lambda x: -x[1]):
        h=int(secs//3600); m=int((secs%3600)//60); sec=int(secs%60)
        print(f"  {sp:12s} {h:02d}:{m:02d}:{sec:02d}")

    preview = f"{base}_PREVIEW.txt"
    with open(preview, "w", encoding="utf-8") as f:
        for seg in final["segments"][:200]:
            sp = seg.get("speaker", "")
            line = (seg.get("text","") or "").strip()
            f.write(f"[{sp}] {line}\n" if sp else (line + "\n"))
    print(f"[preview] wrote first 200 lines to {preview}")
    print("[preview] run again without --pause-before-write to emit SRT/VTT/TXT/JSON.")
    sys.exit(0)

# =============== Writers ============================
def _fmt_ts(t: float) -> str:
    if t is None: t = 0.0
    t = max(0.0, float(t))
    h = int(t // 3600); t -= h * 3600
    m = int(t // 60);   t -= m * 60
    s = int(t);         ms = int(round((t - s) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _fmt_ts_vtt(t: float) -> str:
    if t is None: t = 0.0
    t = max(0.0, float(t))
    h = int(t // 3600); t -= h * 3600
    m = int(t // 60);   t -= m * 60
    s = int(t);         ms = int(round((t - s) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def write_srt(segments, path):
    from pathlib import Path
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start=_fmt_ts(seg.get("start",0.0)); end=_fmt_ts(seg.get("end",0.0))
            text=(seg.get("text","") or "").strip(); sp=seg.get("speaker")
            if sp: text=f"[{sp}] {text}"
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def write_vtt(segments, path):
    from pathlib import Path
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start=_fmt_ts_vtt(seg.get("start",0.0)); end=_fmt_ts_vtt(seg.get("end",0.0))
            text=(seg.get("text","") or "").strip(); sp=seg.get("speaker")
            if sp: text=f"[{sp}] {text}"
            f.write(f"{start} --> {end}\n{text}\n\n")

base_noext = base  # for clarity
SRT = f"{base_noext}.srt";  VTT = f"{base_noext}.vtt";  TXT = f"{base_noext}.txt";  JSON_OUT = f"{base_noext}.json"

print("[WRITE] generating SRT/VTT/TXT/JSON…")
write_srt(final["segments"], SRT)
write_vtt(final["segments"], VTT)

with tqdm(total=len(final["segments"]), unit="seg", desc="TXT", leave=False) as pbar, \
     open(TXT, "w", encoding="utf-8") as f:
    for seg in final["segments"]:
        sp = seg.get("speaker",""); line=(seg.get("text","") or "").strip()
        f.write(f"[{sp}] {line}\n" if sp else (line + "\n")); pbar.update(1)

atomic_write(JSON_OUT, final, mode="json")

try:
    srt_sz=os.path.getsize(SRT); vtt_sz=os.path.getsize(VTT)
    txt_sz=os.path.getsize(TXT); jsn_sz=os.path.getsize(JSON_OUT)
    print(f"[write] sizes → srt:{srt_sz:,} vtt:{vtt_sz:,} txt:{txt_sz:,} json:{jsn_sz:,} bytes")
except Exception:
    pass

mark_done(state, "write"); save_state(base, state)
print("Done →", OUTDIR)
print("Tip: resume later with the same command and --resume (reuses latest textN).")

