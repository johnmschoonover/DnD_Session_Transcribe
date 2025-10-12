"""Job management utilities for the DnD Session Transcribe web UI."""

from __future__ import annotations

import argparse
import asyncio
import functools
import itertools
import json
import logging
import os
import random
import re
import secrets
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from urllib.parse import quote, quote_plus

from fastapi import HTTPException, UploadFile

from ... import cli

LOGGER = logging.getLogger(__name__)

__all__ = [
    "JobRunner",
    "JobService",
    "ScheduleResult",
    "build_cli_args",
    "safe_filename",
]


# ---------------------------------------------------------------------------
# Utility helpers lifted from the legacy web module.
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_filename(filename: str) -> str:
    """Return a filesystem-safe representation of *filename*."""

    name = Path(filename or "").name
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return sanitized or "upload"


def build_cli_args(
    audio: Path,
    *,
    outdir: Path,
    ram: bool = False,
    resume: bool = False,
    num_speakers: Optional[int] = None,
    hotwords_file: Optional[Path | str] = None,
    initial_prompt_file: Optional[Path | str] = None,
    spelling_map: Optional[Path | str] = None,
    asr_model: Optional[str] = None,
    asr_device: Optional[str] = None,
    asr_compute_type: Optional[str] = None,
    precise_rerun: bool = False,
    precise_model: Optional[str] = None,
    precise_device: Optional[str] = None,
    precise_compute_type: Optional[str] = None,
    vocal_extract: Optional[str] = None,
    log_level: str = cli.LOG.level,
    preview_start: Optional[float | int | str] = None,
    preview_duration: Optional[float | int | str] = None,
    preview_output: Optional[Path | str] = None,
) -> argparse.Namespace:
    """Construct an ``argparse.Namespace`` compatible with the CLI entry point."""

    return argparse.Namespace(
        audio=str(audio),
        outdir=str(outdir),
        ram=ram,
        resume=resume,
        num_speakers=num_speakers,
        hotwords_file=str(hotwords_file) if hotwords_file is not None else None,
        initial_prompt_file=
        str(initial_prompt_file) if initial_prompt_file is not None else None,
        spelling_map=str(spelling_map) if spelling_map is not None else None,
        precise_rerun=precise_rerun,
        asr_model=asr_model,
        asr_device=asr_device,
        asr_compute_type=asr_compute_type,
        precise_model=precise_model,
        precise_device=precise_device,
        precise_compute_type=precise_compute_type,
        vocal_extract=vocal_extract,
        log_level=log_level,
        preview_start=preview_start,
        preview_duration=preview_duration,
        preview_output=str(preview_output) if preview_output is not None else None,
    )


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    serialized = json.dumps(data, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as tmp_file:
        tmp_file.write(serialized)
        temp_name = tmp_file.name
    os.replace(temp_name, path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse JSON from %s: %s", path, exc)
        return {}


def _job_status_template(job_id: str, created_at: str) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": "running",
        "created_at": created_at,
        "updated_at": created_at,
        "output_dir": None,
        "error": None,
    }


def _generate_job_id(prefix: str = "job") -> str:
    token = secrets.token_hex(2)
    return f"{prefix}-{datetime.utcnow():%Y%m%d-%H%M%S}-{token}"


_ASR_MODEL_SUGGESTIONS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v2",
    "large-v3",
    "turbo",
    "distil-large-v2",
    "distil-large-v3",
    "distil-medium.en",
    "distil-small.en",
]

_COMPUTE_TYPE_SUGGESTIONS = [
    "auto",
    "float16",
    "float32",
    "int8",
    "int8_float16",
    "int8_float32",
    "int8_float16_float32",
]

_DEVICE_OPTIONS = [
    ("", "Config default"),
    ("cpu", "cpu"),
    ("cuda", "cuda"),
    ("mps", "mps"),
]
_DEVICE_VALUES = [value for value, _ in _DEVICE_OPTIONS]

_VOCAL_OPTIONS = [
    ("", "Config default"),
    ("off", "Off"),
    ("bandpass", "Band-pass filter"),
]
_VOCAL_VALUES = [value for value, _ in _VOCAL_OPTIONS]


def _resolve_selection(value: str, allowed: Sequence[str]) -> tuple[str, list[str]]:
    normalized = value.strip()
    sentinel = normalized.casefold()

    if sentinel == "all":
        return "all", list(allowed)
    if sentinel == "random":
        return "random", list(allowed)

    for option in allowed:
        if sentinel == option.casefold():
            return "single", [option]

    raise ValueError(f"Invalid selection: {value}")


def _checkbox_to_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() not in {"0", "false", "off"}


@dataclass(slots=True)
class ScheduleResult:
    job_ids: list[str]
    skipped_ids: list[str]
    message: str


class JobRunner:
    """Execute the CLI pipeline in the background using the event loop."""

    def __init__(
        self,
        *,
        loop_factory: Callable[[], asyncio.AbstractEventLoop] | None = None,
        transcription_runner: Callable[..., Optional[Path]] | None = None,
    ) -> None:
        self._loop_factory = loop_factory or asyncio.get_running_loop
        self._transcription_runner = (
            transcription_runner or cli.run_transcription
        )

    def _loop(self) -> asyncio.AbstractEventLoop:
        try:
            return self._loop_factory()
        except RuntimeError:
            return asyncio.get_event_loop()

    def submit(self, args: argparse.Namespace, job_id: str, job_dir: Path, created_at: str) -> asyncio.Future[Any]:
        loop = self._loop()
        future = loop.run_in_executor(
            None,
            self._run_job,
            args,
            job_id,
            job_dir,
            created_at,
        )
        future.add_done_callback(functools.partial(self._consume_future, job_id))
        return future

    def _consume_future(self, job_label: str, fut: asyncio.Future[Any]) -> None:
        try:
            fut.result()
        except SystemExit:
            LOGGER.info("Job %s exited early", job_label)
        except Exception:  # pylint: disable=broad-except
            LOGGER.exception("Job %s raised an exception", job_label)

    def _run_job(self, args: argparse.Namespace, job_id: str, job_dir: Path, created_at: str) -> None:
        log_path = job_dir / "job.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )

        status = _job_status_template(job_id, created_at)
        _write_json(job_dir / "status.json", status)

        original_settings = {
            "ASR": {
                "hotwords_file": cli.ASR.hotwords_file,
                "initial_prompt_file": cli.ASR.initial_prompt_file,
                "model": cli.ASR.model,
                "device": cli.ASR.device,
                "compute_type": cli.ASR.compute_type,
            },
            "PRE": {"vocal_extract": cli.PRE.vocal_extract},
            "DIA": {"num_speakers": cli.DIA.num_speakers},
            "PREC": {
                "enabled": cli.PREC.enabled,
                "model": cli.PREC.model,
                "device": cli.PREC.device,
                "compute_type": cli.PREC.compute_type,
            },
        }

        resolved_outdir: Path | None = None

        try:
            resolved_outdir = self._transcription_runner(
                args, configure_logging=False, log_handlers=[handler]
            )
            status["status"] = "completed"
            if resolved_outdir is not None:
                resolved_outdir_str = str(resolved_outdir)
                status["output_dir"] = resolved_outdir_str

                metadata_path = job_dir / "metadata.json"
                metadata = _read_json(metadata_path)
                if metadata.get("output_dir") != resolved_outdir_str:
                    metadata["output_dir"] = resolved_outdir_str
                    _write_json(metadata_path, metadata)
            status["updated_at"] = _utc_now()
            _write_json(job_dir / "status.json", status)
        except BaseException as exc:  # pylint: disable=broad-except
            status["status"] = "failed"
            status["error"] = str(exc)
            status["updated_at"] = _utc_now()
            _write_json(job_dir / "status.json", status)
            LOGGER.exception("Job %s failed", job_id)
        finally:
            cli.ASR.hotwords_file = original_settings["ASR"]["hotwords_file"]
            cli.ASR.initial_prompt_file = original_settings["ASR"]["initial_prompt_file"]
            cli.ASR.model = original_settings["ASR"]["model"]
            cli.ASR.device = original_settings["ASR"]["device"]
            cli.ASR.compute_type = original_settings["ASR"]["compute_type"]
            cli.PRE.vocal_extract = original_settings["PRE"]["vocal_extract"]
            cli.DIA.num_speakers = original_settings["DIA"]["num_speakers"]
            cli.PREC.enabled = original_settings["PREC"]["enabled"]
            cli.PREC.model = original_settings["PREC"]["model"]
            cli.PREC.device = original_settings["PREC"]["device"]
            cli.PREC.compute_type = original_settings["PREC"]["compute_type"]


class JobService:
    """Business logic facade consumed by the FastAPI routes."""

    def __init__(self, base_dir: Path, runner: JobRunner | None = None) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._runner = runner or JobRunner()

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def job_dir(self, job_id: str) -> Path:
        return self._base_dir / job_id

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for path in sorted(self._base_dir.iterdir(), reverse=True):
            if not path.is_dir():
                continue
            status = _read_json(path / "status.json")
            if not status:
                continue
            meta = _read_json(path / "metadata.json")
            status.setdefault("audio_filename", meta.get("audio_filename", ""))
            jobs.append(status)
        jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
        return jobs

    def load_job(self, job_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        job_dir = self.job_dir(job_id)
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        status = _read_json(job_dir / "status.json")
        meta = _read_json(job_dir / "metadata.json")
        status.setdefault("job_id", job_id)
        status.setdefault("created_at", meta.get("created_at", ""))
        status.setdefault("audio_filename", meta.get("audio_filename", ""))
        return status, meta

    def collect_outputs(self, job_dir: Path, status: Mapping[str, Any]) -> tuple[list[tuple[str, str]], str | None]:
        outputs: list[tuple[str, str]] = []
        preview_link: str | None = None
        raw_job_id = str(status.get("job_id") or job_dir.name)
        quoted_job_id = quote(raw_job_id, safe="")

        output_dir = status.get("output_dir")
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = job_dir / "outputs"
        if out_path.exists():
            for child in sorted(out_path.glob("*")):
                if child.is_file():
                    rel = child.relative_to(job_dir)
                    rel_url = quote(rel.as_posix(), safe="/")
                    url = f"/runs/{quoted_job_id}/files/{rel_url}"
                    outputs.append((child.name, url))
                    if preview_link is None and child.name.endswith("_preview.wav"):
                        preview_link = url
        return outputs, preview_link

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    async def schedule_jobs(self, form_items: Iterable[tuple[str, Any]], audio_file: UploadFile) -> ScheduleResult:
        form_values: dict[str, Any] = {}
        for key, value in form_items:
            if key == "audio_file":
                continue
            form_values[key] = value

        indices: list[str | None] = []
        seen_indices: set[str] = set()
        for key in form_values:
            if not key.startswith("job-"):
                continue
            parts = key.split("-", 2)
            if len(parts) < 3:
                continue
            idx = parts[1]
            if idx and idx not in seen_indices and idx.isdigit():
                seen_indices.add(idx)
                indices.append(idx)

        indices.sort(key=lambda value: int(value))
        if not indices:
            indices = [None]

        temp_audio: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("wb", delete=False) as tmp_file:
                while chunk := await audio_file.read(1024 * 1024):
                    tmp_file.write(chunk)
                temp_audio = Path(tmp_file.name)
        finally:
            await audio_file.close()

        if temp_audio is None:
            raise HTTPException(status_code=400, detail="Failed to read audio file")

        clean_name = safe_filename(audio_file.filename)

        def _field_key(job_idx: str | None, name: str) -> str:
            return f"job-{job_idx}-{name}" if job_idx is not None else name

        def _get_value(job_idx: str | None, name: str, default: str = "") -> str:
            key = _field_key(job_idx, name)
            raw = form_values.get(key, default)
            if isinstance(raw, str):
                return raw
            return default if raw is None else str(raw)

        def _get_checkbox(job_idx: str | None, name: str) -> bool:
            key = _field_key(job_idx, name)
            if key not in form_values:
                return False
            raw = form_values[key]
            return _checkbox_to_bool(str(raw))

        job_ids: list[str] = []
        skipped_jobs: list[str] = []
        job_plans: list[dict[str, Any]] = []
        rng = random.Random(secrets.randbits(64))
        seen_signatures: set[tuple[Any, ...]] = set()
        batch_counter = 0

        def _signature(base: Mapping[str, Any], selections: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
            normalized: dict[str, Any] = {}
            for key, value in base.items():
                if isinstance(value, (dict, list)):
                    normalized[key] = json.dumps(value, sort_keys=True)
                else:
                    normalized[key] = value
            normalized.update(selections)
            return tuple(sorted(normalized.items()))

        def _detail(message: str) -> str:
            return f"{message} for {audio_file.filename or 'upload'}"

        try:
            for order, index in enumerate(indices):
                log_level_text = _get_value(index, "log_level", cli.LOG.level).strip()
                if log_level_text:
                    log_level_requested = log_level_text.upper()
                else:
                    log_level_requested = cli.LOG.level

                try:
                    log_level_mode, log_level_choices = _resolve_selection(
                        log_level_requested, tuple(cli.LOG_LEVELS.keys())
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=_detail("Invalid log level")) from exc

                asr_model_value = _get_value(index, "asr_model").strip()
                num_speakers_text = _get_value(index, "num_speakers").strip()
                asr_device_value = _get_value(index, "asr_device").strip()
                asr_compute_type_value = _get_value(index, "asr_compute_type").strip()
                precise_model_value = _get_value(index, "precise_model").strip()
                precise_device_value = _get_value(index, "precise_device").strip()
                precise_compute_type_value = _get_value(index, "precise_compute_type").strip()
                hotwords_text = _get_value(index, "hotwords").strip()
                initial_prompt_text = _get_value(index, "initial_prompt").strip()
                spelling_map_text = _get_value(index, "spelling_map").strip()
                preview_output_text = _get_value(index, "preview_output").strip()
                vocal_extract_value = _get_value(index, "vocal_extract").strip()

                try:
                    asr_device_mode, asr_device_choices = _resolve_selection(
                        asr_device_value, _DEVICE_VALUES
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=_detail("Invalid asr_device value")) from exc

                try:
                    precise_device_mode, precise_device_choices = _resolve_selection(
                        precise_device_value, _DEVICE_VALUES
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=_detail("Invalid precise_device value")) from exc

                try:
                    vocal_mode, vocal_choices = _resolve_selection(
                        vocal_extract_value, _VOCAL_VALUES
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=_detail("Invalid vocal_extract value")) from exc

                if num_speakers_text:
                    try:
                        parsed_num_speakers = int(num_speakers_text)
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail=_detail("num_speakers must be an integer")) from exc
                else:
                    parsed_num_speakers = None

                ram_enabled = _get_checkbox(index, "ram")
                resume_enabled = _get_checkbox(index, "resume")
                precise_rerun_enabled = _get_checkbox(index, "precise_rerun")
                preview_enabled_flag = _get_checkbox(index, "preview_enabled")
                preview_start_text = _get_value(index, "preview_start").strip()
                preview_duration_text = _get_value(index, "preview_duration").strip()

                preview_fields_supplied = bool(preview_start_text or preview_duration_text)
                preview_requested = preview_enabled_flag or preview_fields_supplied
                preview_meta: dict[str, Any] = {"requested": preview_requested}

                preview_start_arg: Optional[float]
                preview_duration_arg: Optional[float]
                if preview_requested:
                    try:
                        start_value = (
                            cli.parse_time_spec(preview_start_text)
                            if preview_start_text
                            else 0.0
                        )
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail=_detail("Invalid preview_start value")) from exc

                    try:
                        duration_value = (
                            cli.parse_time_spec(preview_duration_text)
                            if preview_duration_text
                            else 10.0
                        )
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail=_detail("Invalid preview_duration value")) from exc

                    if duration_value <= 0:
                        raise HTTPException(status_code=400, detail=_detail("Preview duration must be positive"))

                    preview_start_arg = start_value
                    preview_duration_arg = duration_value
                    preview_meta.update({"start": start_value, "duration": duration_value})
                else:
                    preview_start_arg = None
                    preview_duration_arg = None

                preview_output_name: Optional[str] = None
                if preview_output_text:
                    preview_output_name = safe_filename(preview_output_text)
                    if not preview_output_name.lower().endswith(".wav"):
                        preview_output_name += ".wav"

                selection_modes = {
                    "log_level": (log_level_mode, log_level_choices),
                    "asr_device": (asr_device_mode, asr_device_choices),
                    "precise_device": (precise_device_mode, precise_device_choices),
                    "vocal_extract": (vocal_mode, vocal_choices),
                }

                base_config = {
                    "order": order,
                    "index": index,
                    "num_speakers": parsed_num_speakers,
                    "ram": ram_enabled,
                    "resume": resume_enabled,
                    "precise_rerun": precise_rerun_enabled,
                    "asr_model": asr_model_value or None,
                    "asr_compute_type": asr_compute_type_value or None,
                    "precise_model": precise_model_value or None,
                    "precise_compute_type": precise_compute_type_value or None,
                    "hotwords_text": hotwords_text,
                    "initial_prompt_text": initial_prompt_text,
                    "spelling_map_text": spelling_map_text,
                    "preview_requested": preview_requested,
                    "preview_start": preview_start_arg,
                    "preview_duration": preview_duration_arg,
                    "preview_output_name": preview_output_name,
                    "preview_meta": preview_meta,
                    "log_level_requested": log_level_requested,
                    "asr_device_requested": asr_device_value,
                    "precise_device_requested": precise_device_value,
                    "vocal_extract_requested": vocal_extract_value,
                }

                base_values = {
                    name: choices[0]
                    for name, (mode, choices) in selection_modes.items()
                    if mode == "single"
                }
                all_field_names = [
                    name for name, (mode, _) in selection_modes.items() if mode == "all"
                ]
                all_combinations = (
                    list(
                        itertools.product(
                            *(selection_modes[name][1] for name in all_field_names)
                        )
                    )
                    if all_field_names
                    else [()]
                )
                random_fields = {
                    name: choices
                    for name, (mode, choices) in selection_modes.items()
                    if mode == "random"
                }
                random_field_names = list(random_fields)

                for combo in all_combinations:
                    resolved_values = base_values.copy()
                    for field_name, chosen in zip(all_field_names, combo):
                        resolved_values[field_name] = chosen

                    random_combos = [()]
                    if random_field_names:
                        random_combos = list(
                            itertools.product(
                                *(random_fields[name] for name in random_field_names)
                            )
                        )
                        rng.shuffle(random_combos)

                    assigned = False
                    attempted_candidate: dict[str, str] | None = None
                    for random_choice in random_combos:
                        candidate_values = resolved_values.copy()
                        for field_name, chosen in zip(random_field_names, random_choice):
                            candidate_values[field_name] = chosen
                        signature = _signature(base_config, candidate_values)
                        if signature not in seen_signatures:
                            seen_signatures.add(signature)
                            assigned = True
                            final_values = candidate_values
                            break
                        attempted_candidate = candidate_values

                    if not assigned:
                        reason = "Skipped duplicate configuration"
                        if random_field_names:
                            reason = "Skipped duplicate configuration (random exhausted)"
                        skip_settings = {
                            "log_level": base_config["log_level_requested"],
                            "num_speakers": base_config["num_speakers"],
                            "ram": base_config["ram"],
                            "resume": base_config["resume"],
                            "precise_rerun": base_config["precise_rerun"],
                            "asr_model": base_config["asr_model"],
                            "asr_device": base_config["asr_device_requested"],
                            "asr_compute_type": base_config["asr_compute_type"],
                            "precise_model": base_config["precise_model"],
                            "precise_device": base_config["precise_device_requested"],
                            "precise_compute_type": base_config["precise_compute_type"],
                            "vocal_extract": base_config["vocal_extract_requested"],
                            "preview_requested": base_config["preview_requested"],
                            "preview_start": base_config["preview_start"],
                            "preview_duration": base_config["preview_duration"],
                            "preview_output": base_config["preview_output_name"],
                            "hotwords": base_config["hotwords_text"],
                            "initial_prompt": base_config["initial_prompt_text"],
                            "spelling_map": base_config["spelling_map_text"],
                            "batch_index": None,
                            "job_index": index,
                        }
                        if attempted_candidate:
                            skip_settings["resolved_log_level"] = attempted_candidate.get("log_level")
                            skip_settings["resolved_asr_device"] = attempted_candidate.get("asr_device")
                            skip_settings["resolved_precise_device"] = attempted_candidate.get(
                                "precise_device"
                            )
                            skip_settings["resolved_vocal_extract"] = attempted_candidate.get(
                                "vocal_extract"
                            )

                        skip_job_id = _generate_job_id()
                        skip_dir = self.job_dir(skip_job_id)
                        created_at = _utc_now()
                        status_snapshot = _job_status_template(skip_job_id, created_at)
                        status_snapshot["status"] = "skipped"
                        status_snapshot["error"] = reason
                        metadata = {
                            "job_id": skip_job_id,
                            "audio_filename": audio_file.filename,
                            "created_at": created_at,
                            "preview": dict(base_config["preview_meta"]),
                            "settings": skip_settings,
                            "batch_index": None,
                            "job_index": index,
                            "skip_reason": reason,
                        }
                        skipped_jobs.append(skip_job_id)
                        skip_dir.mkdir(parents=True, exist_ok=True)
                        _write_json(skip_dir / "metadata.json", metadata)
                        _write_json(skip_dir / "status.json", status_snapshot)
                        continue

                    job_id = _generate_job_id()
                    job_dir = self.job_dir(job_id)
                    inputs_dir = job_dir / "inputs"
                    outputs_dir = job_dir / "outputs"
                    audio_target = inputs_dir / clean_name

                    hotwords_path = inputs_dir / "hotwords.txt" if hotwords_text else None
                    initial_prompt_path = (
                        inputs_dir / "initial_prompt.txt" if initial_prompt_text else None
                    )
                    spelling_map_path = (
                        inputs_dir / "spelling_map.csv" if spelling_map_text else None
                    )
                    preview_output_path = (
                        outputs_dir / preview_output_name if preview_output_name else None
                    )

                    file_payloads: list[tuple[Path, str]] = []
                    if hotwords_path:
                        file_payloads.append((hotwords_path, hotwords_text))
                    if initial_prompt_path:
                        file_payloads.append((initial_prompt_path, initial_prompt_text))
                    if spelling_map_path:
                        file_payloads.append((spelling_map_path, spelling_map_text))

                    args = build_cli_args(
                        audio_target,
                        outdir=outputs_dir,
                        ram=ram_enabled,
                        resume=resume_enabled,
                        num_speakers=parsed_num_speakers,
                        hotwords_file=hotwords_path,
                        initial_prompt_file=initial_prompt_path,
                        spelling_map=spelling_map_path,
                        asr_model=asr_model_value or None,
                        asr_device=final_values["asr_device"] or None,
                        asr_compute_type=asr_compute_type_value or None,
                        precise_rerun=precise_rerun_enabled,
                        precise_model=precise_model_value or None,
                        precise_device=final_values["precise_device"] or None,
                        precise_compute_type=precise_compute_type_value or None,
                        vocal_extract=final_values["vocal_extract"] or None,
                        log_level=final_values["log_level"],
                        preview_start=preview_start_arg,
                        preview_duration=preview_duration_arg,
                        preview_output=preview_output_path,
                    )

                    settings_snapshot = {
                        key: (str(val) if isinstance(val, Path) else val)
                        for key, val in vars(args).items()
                    }
                    settings_snapshot["preview_requested"] = preview_requested
                    settings_snapshot["batch_index"] = batch_counter
                    settings_snapshot["job_index"] = index
                    if random_field_names:
                        settings_snapshot["randomized_fields"] = sorted(random_field_names)

                    created_at = _utc_now()
                    status_snapshot = _job_status_template(job_id, created_at)
                    status_snapshot["audio_filename"] = audio_file.filename

                    metadata = {
                        "job_id": job_id,
                        "audio_filename": audio_file.filename,
                        "created_at": created_at,
                        "preview": dict(base_config["preview_meta"]),
                        "settings": settings_snapshot,
                        "batch_index": batch_counter,
                        "job_index": index,
                    }

                    job_plans.append(
                        {
                            "job_id": job_id,
                            "job_dir": job_dir,
                            "inputs_dir": inputs_dir,
                            "outputs_dir": outputs_dir,
                            "audio_path": audio_target,
                            "args": args,
                            "metadata": metadata,
                            "status": status_snapshot,
                            "created_at": created_at,
                            "file_payloads": file_payloads,
                        }
                    )
                    batch_counter += 1

            if not job_plans and not skipped_jobs:
                raise HTTPException(status_code=400, detail="No job configurations provided")

            for plan in job_plans:
                job_dir = plan["job_dir"]
                inputs_dir = plan["inputs_dir"]
                outputs_dir = plan["outputs_dir"]
                audio_path = plan["audio_path"]

                job_dir.mkdir(parents=True, exist_ok=True)
                inputs_dir.mkdir(exist_ok=True)
                outputs_dir.mkdir(exist_ok=True)
                shutil.copyfile(temp_audio, audio_path)

                for payload_path, payload_content in plan.get("file_payloads", []):
                    payload_path.parent.mkdir(parents=True, exist_ok=True)
                    payload_path.write_text(payload_content, encoding="utf-8")

                _write_json(job_dir / "metadata.json", plan["metadata"])
                _write_json(job_dir / "status.json", plan["status"])

                self._runner.submit(plan["args"], plan["job_id"], job_dir, plan["created_at"])
                job_ids.append(plan["job_id"])

            parts: list[str] = []
            if job_ids:
                if len(job_ids) == 1:
                    parts.append(f"Started job {job_ids[0]}")
                else:
                    parts.append("Started jobs " + ", ".join(job_ids))
            if skipped_jobs:
                parts.append("Skipped duplicate configs " + ", ".join(skipped_jobs))
            message = quote_plus("; ".join(parts) if parts else "No jobs were scheduled")

            return ScheduleResult(job_ids=job_ids, skipped_ids=skipped_jobs, message=message)
        finally:
            if temp_audio is not None:
                temp_audio.unlink(missing_ok=True)

    def delete_job(self, job_id: str) -> None:
        job_dir = self.job_dir(job_id)
        if not job_dir.exists() or not job_dir.is_dir():
            raise HTTPException(status_code=404, detail="Job not found")
        try:
            shutil.rmtree(job_dir)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Failed to delete job %s", job_id)
            raise HTTPException(status_code=500, detail="Failed to delete job") from exc


__all__ += [
    "_ASR_MODEL_SUGGESTIONS",
    "_COMPUTE_TYPE_SUGGESTIONS",
    "_DEVICE_OPTIONS",
    "_VOCAL_OPTIONS",
]
