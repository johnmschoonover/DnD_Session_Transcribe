"""HTTP route definitions for the web UI."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from ..services.jobs import JobService
from ..templates import render_home, render_job_detail

router = APIRouter()


def get_job_service(request: Request) -> JobService:
    service = getattr(request.app.state, "job_service", None)
    if service is None:
        raise RuntimeError("JobService has not been configured on the FastAPI app")
    return service


@router.get("/", response_class=HTMLResponse)
async def home(request: Request, service: JobService = Depends(get_job_service)) -> str:
    message = request.query_params.get("message")
    load_job_id = request.query_params.get("load_job")
    prefill_jobs: list[dict[str, Any]] | None = None
    load_message: str | None = None

    if load_job_id:
        try:
            prefill_jobs = [service.export_job_settings(load_job_id)]
            load_message = f"Loaded settings from {load_job_id}"
        except HTTPException as exc:
            if exc.status_code == 404:
                load_message = f"Job {load_job_id} not found"
            else:
                raise

    combined_message = message
    if load_message:
        combined_message = (
            f"{message} | {load_message}" if message else load_message
        )

    jobs = service.list_jobs()
    return render_home(jobs, combined_message, prefill_jobs=prefill_jobs or [])


@router.post("/transcribe")
async def transcribe(
    request: Request,
    audio_file: UploadFile = File(...),
    service: JobService = Depends(get_job_service),
) -> RedirectResponse:
    form = await request.form()
    result = await service.schedule_jobs(form.multi_items(), audio_file)
    return RedirectResponse(url=f"/?message={result.message}", status_code=303)


@router.get("/runs/{job_id}", response_class=HTMLResponse)
async def show_job(job_id: str, service: JobService = Depends(get_job_service)) -> str:
    status, metadata = service.load_job(job_id)
    job_dir = service.job_dir(job_id)
    files, preview_link = service.collect_outputs(job_dir, status)
    log_available = (job_dir / "job.log").exists()
    return render_job_detail(
        status,
        files,
        log_available,
        preview=metadata.get("preview"),
        preview_url=preview_link,
        settings=metadata.get("settings"),
    )


@router.get("/runs/{job_id}/log")
async def download_log(job_id: str, service: JobService = Depends(get_job_service)) -> FileResponse:
    job_dir = service.job_dir(job_id)
    log_path = job_dir / "job.log"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log not found")
    return FileResponse(log_path)


@router.get("/runs/{job_id}/files/{file_path:path}")
async def download_file(
    job_id: str,
    file_path: str,
    service: JobService = Depends(get_job_service),
) -> FileResponse:
    job_dir = service.job_dir(job_id)
    target = (job_dir / file_path).resolve()
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        target.relative_to(job_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid file path") from exc
    return FileResponse(target)


@router.post("/runs/{job_id}/delete")
async def delete_job(job_id: str, service: JobService = Depends(get_job_service)) -> RedirectResponse:
    service.delete_job(job_id)
    message = f"Deleted job {job_id}"
    return RedirectResponse(
        url=f"/?message={quote_plus(message)}#run-console", status_code=303
    )


@router.post("/runs/batch-delete")
async def batch_delete(
    request: Request, service: JobService = Depends(get_job_service)
) -> RedirectResponse:
    form = await request.form()
    job_ids = [value for value in form.getlist("job_ids") if value]
    if not job_ids:
        raise HTTPException(status_code=400, detail="No jobs selected")

    deleted, missing = service.delete_jobs(job_ids)
    parts: list[str] = []
    if deleted:
        if len(deleted) == 1:
            parts.append(f"Deleted job {deleted[0]}")
        else:
            parts.append("Deleted jobs " + ", ".join(deleted))
    if missing:
        if len(missing) == 1:
            parts.append(f"Job {missing[0]} was not found")
        else:
            parts.append("Jobs " + ", ".join(missing) + " were not found")
    message = "; ".join(parts) if parts else "No jobs deleted"
    return RedirectResponse(
        url=f"/?message={quote_plus(message)}#run-console", status_code=303
    )
