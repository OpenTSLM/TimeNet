"""Jobs HTTP routes."""

# ruff: noqa: DOC201
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from timenet.viewer.jobs import JobManager
from timenet.viewer.schemas import WindowQuery
from timenet.viewer.windows import window_result
from timenet.viewer.workers import ReaderWorker


def router(worker: ReaderWorker, jobs: JobManager) -> APIRouter:
    """Create the jobs transport adapter."""
    router = APIRouter()

    @router.post("/api/v1/jobs/windows", status_code=202)
    async def window_job(body: WindowQuery) -> JSONResponse:
        """Schedule a bounded window operation without occupying an HTTP request."""
        job = jobs.submit(lambda: worker.run(window_result, body), operation="window")
        return JSONResponse({"api_version": 1, "job_id": job.job_id, "state": job.state}, status_code=202)

    @router.get("/api/v1/jobs/{job_id}")
    async def job_status(job_id: str) -> JSONResponse:
        """Return sanitized status for one background operation."""
        job = jobs.get(job_id)
        if job is None:
            return JSONResponse({"code": "job_not_found", "message": "job not found"}, status_code=404)
        return JSONResponse(
            {"api_version": 1, "job_id": job.job_id, "state": job.state, "error": job.error, "progress": job.progress}
        )

    @router.get("/api/v1/jobs/{job_id}/result")
    async def job_result(job_id: str) -> JSONResponse:
        """Return a completed background result exactly once it is ready."""
        job = jobs.get(job_id)
        if job is None:
            return JSONResponse({"code": "job_not_found", "message": "job not found"}, status_code=404)
        if job.state not in {"succeeded", "cancelled"} or job.result is None:
            return JSONResponse({"code": "result_not_ready", "message": "job result is not ready"}, status_code=409)
        return JSONResponse({key: value for key, value in job.result.items() if key != "findings"})

    @router.delete("/api/v1/jobs/{job_id}")
    async def cancel_job(job_id: str) -> JSONResponse:
        """Cancel or discard a viewer-only job."""
        job = jobs.cancel(job_id)
        if job is None:
            return JSONResponse({"code": "job_not_found", "message": "job not found"}, status_code=404)
        return JSONResponse({"api_version": 1, "job_id": job.job_id, "state": job.state})

    return router
