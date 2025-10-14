"""High level orchestration for auto-tune preflight analysis."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from hashlib import sha1
from pathlib import Path
from typing import Dict, Tuple

from ..constants.autotune_defaults import AUTO_TUNE_VERSION
from .audio_stats import AudioDiagnostics, analyze_audio
from .auto_tune import AutoTuneSuggestion, suggest_config

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "dnd_session_transcribe" / "autotune"
_PRE_NORM_SUFFIX = ".autonorm.wav"


class PreflightError(RuntimeError):
    """Raised when the preflight pipeline cannot be executed."""


def preflight_analyze_and_suggest(
    audio_path: str | Path,
    user_overrides: Dict[str, object] | None = None,
    mode: str = "apply",
    pre_norm_mode: str = "suggest",
    cache_ttl: int = 86400,
    no_cache: bool = False,
    artifact_dir: str | Path | None = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Run audio analysis, derive suggestions, and return runtime overrides."""

    if mode not in {"apply", "suggest"}:
        raise ValueError("mode must be 'apply' or 'suggest'")
    if pre_norm_mode not in {"off", "suggest", "apply"}:
        raise ValueError("pre_norm_mode must be 'off', 'suggest', or 'apply'")

    audio_path = Path(audio_path).resolve()
    artifact_dir_path = Path(artifact_dir).resolve() if artifact_dir else None

    overrides = dict(user_overrides or {})

    diag_obj, suggestion, cache_hit = _load_or_compute(audio_path, cache_ttl, no_cache)

    applied_cfg, final_snapshot, pre_norm_state, clipped = _merge_with_overrides(
        suggestion,
        overrides,
        mode,
        pre_norm_mode,
        audio_path,
        artifact_dir_path,
    )

    if clipped:
        logger.warning("Auto-tune suggestions skipped due to user overrides: %s", clipped)

    diagnostics_payload = {
        "metrics": diag_obj.to_dict(),
        "suggestion": suggestion.to_dict(),
        "final_config": final_snapshot,
        "pre_norm": pre_norm_state,
        "cache": {
            "hit": cache_hit,
        },
        "clipped_keys": clipped,
    }

    return applied_cfg, diagnostics_payload


def _load_or_compute(
    audio_path: Path,
    cache_ttl: int,
    no_cache: bool,
) -> Tuple[AudioDiagnostics, AutoTuneSuggestion, bool]:
    cache_key = _hash_audio(audio_path)
    cache_file = _CACHE_DIR / f"{cache_key}_{AUTO_TUNE_VERSION}.json"

    if not no_cache:
        diag, suggestion = _read_cache(cache_file, cache_ttl)
        if diag and suggestion:
            return diag, suggestion, True

    diag_obj = analyze_audio(audio_path)
    suggestion = suggest_config(diag_obj)
    if not no_cache:
        _write_cache(cache_file, diag_obj, suggestion)
    return diag_obj, suggestion, False


def _read_cache(
    cache_file: Path,
    cache_ttl: int,
) -> Tuple[AudioDiagnostics | None, AutoTuneSuggestion | None]:
    if not cache_file.exists():
        return None, None
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    timestamp = payload.get("timestamp")
    if timestamp is None or (time.time() - float(timestamp)) > cache_ttl:
        return None, None
    diagnostics = payload.get("diagnostics")
    suggestion_payload = payload.get("suggestion")
    if not diagnostics or not suggestion_payload:
        return None, None
    try:
        diag_obj = AudioDiagnostics(**diagnostics)
        suggestion = AutoTuneSuggestion(
            cfg=suggestion_payload.get("cfg", {}),
            rationale=suggestion_payload.get("rationale", {}),
        )
    except TypeError:
        return None, None
    return diag_obj, suggestion


def _write_cache(cache_file: Path, diag: AudioDiagnostics, suggestion: AutoTuneSuggestion) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.time(),
        "diagnostics": diag.to_dict(),
        "suggestion": suggestion.to_dict(),
    }
    cache_file.write_text(json.dumps(payload), encoding="utf-8")


def _merge_with_overrides(
    suggestion: AutoTuneSuggestion,
    overrides: Dict[str, object],
    mode: str,
    pre_norm_mode: str,
    audio_path: Path,
    artifact_dir: Path | None,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object], list[str]]:
    applied: Dict[str, object] = {}
    final_snapshot: Dict[str, object] = dict(overrides)
    clipped: list[str] = []

    suggested_cfg = dict(suggestion.cfg)
    suggested_pre_norm = str(suggested_cfg.get("pre_norm", "off"))

    if mode == "apply":
        for key, value in suggested_cfg.items():
            if key == "pre_norm":
                continue
            if key in overrides:
                clipped.append(key)
                continue
            applied[key] = value
            final_snapshot[key] = value
    else:
        final_snapshot.update(overrides)

    for key, value in overrides.items():
        final_snapshot[key] = value

    final_snapshot.setdefault(
        "condition_on_previous_text", suggested_cfg.get("condition_on_previous_text", False)
    )

    final_pre_norm = _resolve_pre_norm_state(suggested_pre_norm, pre_norm_mode, mode)
    final_snapshot["pre_norm"] = final_pre_norm

    normalized_path = None
    ffmpeg_status = "skipped"
    if final_pre_norm == "apply" and mode == "apply":
        normalized_path, ffmpeg_status = _maybe_run_pre_norm(audio_path, artifact_dir)
    elif final_pre_norm == "apply" and mode == "suggest":
        ffmpeg_status = "deferred"

    pre_norm_state = {
        "suggested": suggested_pre_norm,
        "final": final_pre_norm,
        "applied_path": str(normalized_path) if normalized_path else None,
        "ffmpeg": ffmpeg_status,
    }

    if mode == "apply":
        applied["pre_norm"] = final_pre_norm
    else:
        applied = {}

    return applied, final_snapshot, pre_norm_state, clipped


def _resolve_pre_norm_state(suggested: str, pre_norm_mode: str, mode: str) -> str:
    if suggested != "apply":
        return "off"
    if pre_norm_mode == "off":
        return "off"
    if pre_norm_mode == "apply" and mode == "apply":
        return "apply"
    return "suggest"


def _maybe_run_pre_norm(audio_path: Path, artifact_dir: Path | None) -> Tuple[Path | None, str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.warning("FFmpeg not available; cannot apply pre-normalisation")
        return None, "missing_ffmpeg"

    dest_dir = artifact_dir or audio_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"{audio_path.stem}{_PRE_NORM_SUFFIX}"

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        "highpass=f=100,lowpass=f=8000,dynaudnorm=f=75:s=10",
        "-c:a",
        "pcm_s16le",
        str(target),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        logger.warning("Pre-normalisation failed: %s", exc)
        return None, "failed"
    return target, "applied"


def _hash_audio(path: Path) -> str:
    hasher = sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


__all__ = ["preflight_analyze_and_suggest", "PreflightError"]
