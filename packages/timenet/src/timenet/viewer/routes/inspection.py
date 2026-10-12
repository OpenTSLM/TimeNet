"""Inspection HTTP routes."""

# ruff: noqa: DOC201
from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from timenet.errors import TimeFValidationError
from timenet.reader.inspection import ReaderInspection
from timenet.viewer import projections
from timenet.viewer.contracts import endpoint
from timenet.viewer.schemas import (
    AnnotationQuery,
    AnnotationWindowQuery,
    OwnerRecordQuery,
    RecordDetailQuery,
    RecordQuery,
    SessionQuery,
    SignalQuery,
    SourceQuery,
    TaskDetailQuery,
    TaskQuery,
    WindowQuery,
)
from timenet.viewer.windows import window_result
from timenet.viewer.workers import ReaderWorker


def router(
    worker: ReaderWorker | Callable[[], ReaderWorker],
    registry: str,
    *,
    include_session: bool = True,
) -> APIRouter:
    """Create the inspection transport adapter."""
    router = APIRouter()

    def active_worker() -> ReaderWorker:
        """Return the reader worker serving this request."""
        return worker if isinstance(worker, ReaderWorker) else worker()

    if include_session:

        @endpoint("GET", "/api/v1/session").route(router)
        async def session(_query: Annotated[SessionQuery, Query()]) -> JSONResponse:
            """Return the pinned session's public capabilities."""
            result = await active_worker().call(projections.session)
            result["registry"] = registry
            return JSONResponse(result)

    @endpoint("GET", "/api/v1/overview").route(router)
    async def overview() -> JSONResponse:
        """Return manifest-backed overview facts without reading signal values."""
        result = await active_worker().call(ReaderInspection.overview)
        return JSONResponse(result)

    @endpoint("POST", "/api/v1/records/query").route(router)
    async def records(body: RecordQuery) -> JSONResponse:
        """Return a bounded record-ID page."""
        result = await active_worker().call(projections.records, body)
        return JSONResponse(result)

    @endpoint("POST", "/api/v1/sources/query").route(router)
    async def sources(body: SourceQuery) -> JSONResponse:
        """Return immediate sources for a selected record."""
        result = await active_worker().call(projections.sources, body)
        return JSONResponse(result)

    @endpoint("POST", "/api/v1/records/detail").route(router)
    async def record_detail(body: RecordDetailQuery) -> JSONResponse:
        """Return persisted clock, span provenance and bounded metadata."""
        result = await active_worker().call(ReaderInspection.record_detail, body)
        return JSONResponse(result)

    @endpoint("POST", "/api/v1/signals/query").route(router)
    async def signals(body: SignalQuery) -> JSONResponse:
        """Return direct signals for a selected source."""
        result = await active_worker().call(projections.signals, body)
        return JSONResponse(result)

    @endpoint("POST", "/api/v1/owners/record").route(router)
    async def owner_record(body: OwnerRecordQuery) -> JSONResponse:
        """Resolve a Source or Signal owner to its containing Record for viewer navigation."""
        result = await active_worker().call(ReaderInspection.owner_record, body)
        if result is None:
            return JSONResponse({"code": "not_found", "message": "object not found"}, status_code=404)
        return JSONResponse(result)

    @endpoint("POST", "/api/v1/tasks/query").route(router)
    async def tasks(body: TaskQuery) -> JSONResponse:
        """Return a bounded, lightweight task page."""
        result = await active_worker().call(ReaderInspection.tasks, body)
        return JSONResponse(result)

    @endpoint("POST", "/api/v1/tasks/detail").route(router)
    async def task_detail(body: TaskDetailQuery) -> JSONResponse:
        """Return bounded task relationships for the selected task."""
        result = await active_worker().call(ReaderInspection.task_detail, body)
        if result is None:
            return JSONResponse({"code": "not_found", "message": "object not found"}, status_code=404)
        return JSONResponse(result)

    @endpoint("POST", "/api/v1/annotations/query").route(router)
    async def annotations(body: AnnotationQuery) -> JSONResponse:
        """Return a bounded annotation occurrence page."""
        result = await active_worker().call(ReaderInspection.browse_annotations, body)
        return JSONResponse(result)

    @endpoint("POST", "/api/v1/annotations/window").route(router)
    async def annotation_window(body: AnnotationWindowQuery) -> JSONResponse:
        """Return signal, ancestor-source, and record annotations visible in the selected window."""
        result = await active_worker().call(ReaderInspection.window_annotations, body)
        return JSONResponse(result)

    @endpoint("POST", "/api/v1/windows").route(router)
    async def window(body: WindowQuery) -> JSONResponse:
        """Return a bounded scalar raw table or reduced plot window."""
        try:
            return JSONResponse(await active_worker().call(window_result, body))
        except TimeFValidationError as exc:
            return JSONResponse({"code": "invalid_window", "message": str(exc)}, status_code=422)

    return router
