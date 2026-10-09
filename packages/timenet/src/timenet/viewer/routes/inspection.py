"""Inspection HTTP routes."""

# ruff: noqa: DOC201
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from timenet.errors import TimeFValidationError
from timenet.viewer import projections
from timenet.viewer.inspection import ViewerInspection
from timenet.viewer.schemas import (
    AnnotationQuery,
    AnnotationWindowQuery,
    OwnerRecordQuery,
    RecordDetailQuery,
    RecordQuery,
    SignalQuery,
    SourceQuery,
    TaskDetailQuery,
    TaskQuery,
    WindowQuery,
)
from timenet.viewer.windows import window_result
from timenet.viewer.workers import ReaderWorker


def router(worker: ReaderWorker, registry: str) -> APIRouter:
    """Create the inspection transport adapter."""
    router = APIRouter()

    @router.get("/api/v1/session")
    async def session() -> JSONResponse:
        """Return the pinned session's public capabilities."""
        result = await worker.call(projections.session)
        result["registry"] = registry
        return JSONResponse(result)

    @router.get("/api/v1/overview")
    async def overview() -> JSONResponse:
        """Return manifest-backed overview facts without reading signal values."""
        result = await worker.call(ViewerInspection.overview)
        return JSONResponse(result)

    @router.post("/api/v1/records/query")
    async def records(body: RecordQuery) -> JSONResponse:
        """Return a bounded record-ID page."""
        result = await worker.call(projections.records, body)
        return JSONResponse(result)

    @router.post("/api/v1/sources/query")
    async def sources(body: SourceQuery) -> JSONResponse:
        """Return immediate sources for a selected record."""
        result = await worker.call(projections.sources, body)
        return JSONResponse(result)

    @router.post("/api/v1/records/detail")
    async def record_detail(body: RecordDetailQuery) -> JSONResponse:
        """Return persisted clock, span provenance and bounded metadata."""
        result = await worker.call(ViewerInspection.record_detail, body)
        return JSONResponse(result)

    @router.post("/api/v1/signals/query")
    async def signals(body: SignalQuery) -> JSONResponse:
        """Return direct signals for a selected source."""
        result = await worker.call(projections.signals, body)
        return JSONResponse(result)

    @router.post("/api/v1/owners/record")
    async def owner_record(body: OwnerRecordQuery) -> JSONResponse:
        """Resolve a Source or Signal owner to its containing Record for viewer navigation."""
        result = await worker.call(ViewerInspection.owner_record, body)
        if result is None:
            return JSONResponse({"code": "not_found", "message": "object not found"}, status_code=404)
        return JSONResponse(result)

    @router.post("/api/v1/tasks/query")
    async def tasks(body: TaskQuery) -> JSONResponse:
        """Return a bounded, lightweight task page."""
        result = await worker.call(ViewerInspection.tasks, body)
        return JSONResponse(result)

    @router.post("/api/v1/tasks/detail")
    async def task_detail(body: TaskDetailQuery) -> JSONResponse:
        """Return bounded task relationships for the selected task."""
        result = await worker.call(ViewerInspection.task_detail, body)
        if result is None:
            return JSONResponse({"code": "not_found", "message": "object not found"}, status_code=404)
        return JSONResponse(result)

    @router.post("/api/v1/annotations/query")
    async def annotations(body: AnnotationQuery) -> JSONResponse:
        """Return a bounded annotation occurrence page."""
        result = await worker.call(ViewerInspection.browse_annotations, body)
        return JSONResponse(result)

    @router.post("/api/v1/annotations/window")
    async def annotation_window(body: AnnotationWindowQuery) -> JSONResponse:
        """Return direct signal annotations overlapping a selected time window."""
        result = await worker.call(ViewerInspection.window_annotations, body)
        return JSONResponse(result)

    @router.post("/api/v1/windows")
    async def window(body: WindowQuery) -> JSONResponse:
        """Return a bounded scalar raw table or reduced plot window."""
        try:
            return JSONResponse(await worker.call(window_result, body))
        except TimeFValidationError as exc:
            return JSONResponse({"code": "invalid_window", "message": str(exc)}, status_code=422)

    return router
