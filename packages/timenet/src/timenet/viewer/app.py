"""Composition and resource lifetime for the authenticated local inspector."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from functools import partial
from importlib.resources import files
from typing import Annotated, Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse

from timenet.errors import TimeFFormatError, TimeFValidationError, TimeNetDatasetNotFoundError, TimeNetRegistryError
from timenet.reader import TimeFReader
from timenet.registry import BaseRegistry
from timenet.viewer import projections
from timenet.viewer.catalog import catalog_page
from timenet.viewer.contracts import endpoint
from timenet.viewer.routes import inspection
from timenet.viewer.schemas import CatalogQuery, DatasetSelection, SessionQuery, StrictQuery
from timenet.viewer.security import RequestBoundary, security_headers
from timenet.viewer.workers import ReaderWorker


class _RegistryViewerSession:
    """Own the one dataset reader selected from a browsable registry."""

    def __init__(self, registry: BaseRegistry) -> None:
        """Store the registry until a dataset is selected."""
        self.transitioning = False
        self._registry = registry
        self._reader: TimeFReader | None = None
        self._worker: ReaderWorker | None = None

    def open(self, dataset_id: str, version: str | None) -> None:
        """Replace the active reader with a pinned registry version."""
        reader = TimeFReader(self._registry.open_version(dataset_id, version))
        try:
            worker = ReaderWorker(reader, "viewer-interactive")
        except BaseException:
            reader.close()
            raise
        self.close()
        self._reader = reader
        self._worker = worker

    def worker(self) -> ReaderWorker:
        """Return the selected dataset worker.

        Raises:
            TimeFValidationError: If no dataset has been selected.
        """
        if self.transitioning or self._worker is None:
            raise TimeFValidationError("select a dataset from the registry first")
        return self._worker

    @property
    def selected(self) -> bool:
        """Whether this session has an active dataset reader."""
        return self._worker is not None

    def close(self) -> None:
        """Release the selected dataset, if any."""
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        if self._reader is not None:
            self._reader.close()
            self._reader = None


def create_app(reader: TimeFReader, *, token: str, port: int, registry: str = "unknown") -> FastAPI:
    """Return local HTTP adapters composed around one independently owned reader."""
    worker = ReaderWorker(reader, "viewer-interactive")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:
        """Close the reader on its owning thread.

        Yields:
            The active application lifetime.
        """
        try:
            yield
        finally:
            await asyncio.to_thread(worker.close)

    app = _base_app(token, port, lifespan)

    app.include_router(inspection.router(worker, registry))
    return app


def create_registry_app(registry: BaseRegistry, *, token: str, port: int, registry_name: str) -> FastAPI:
    """Return local HTTP adapters for a registry catalog."""
    session_state = _RegistryViewerSession(registry)
    lifecycle_lock = asyncio.Lock()
    registry_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="viewer-registry")

    async def registry_call(operation: Callable, *args: object) -> object:
        return await asyncio.wrap_future(registry_executor.submit(partial(operation, *args)))

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:
        """Release a selected dataset reader when the catalog server stops.

        Yields:
            The active application lifetime.
        """
        try:
            yield
        finally:
            await registry_call(session_state.close)
            registry_executor.shutdown(wait=True)

    app = _base_app(token, port, lifespan)

    @endpoint("GET", "/api/v1/session").route(app)
    async def session(_query: Annotated[SessionQuery, Query()]) -> JSONResponse:
        """Return catalog or selected-dataset session capabilities."""
        if session_state.selected:
            result = await session_state.worker().call(projections.session)
            return JSONResponse({"mode": "dataset", "registry": registry_name, **result})
        return JSONResponse({"api_version": 1, "mode": "registry", "registry": registry_name})

    @endpoint("GET", "/api/v1/registry/datasets").route(app)
    async def datasets(query: Annotated[CatalogQuery, Query()]) -> JSONResponse:
        """Return every available dataset's searchable metadata."""
        return JSONResponse(await registry_call(catalog_page, registry, query))

    @endpoint("POST", "/api/v1/registry/open").route(app)
    async def open_dataset(selection: DatasetSelection) -> JSONResponse:
        """Select one registry dataset for inspection.

        Returns:
            The selected viewer mode.

        Raises:
            TimeFValidationError: If the requested dataset cannot be opened.
        """
        try:
            async with lifecycle_lock:
                session_state.transitioning = True
                try:
                    await registry_call(session_state.open, selection.dataset_id, selection.version)
                finally:
                    session_state.transitioning = False
        except (OSError, ValueError) as exc:
            raise TimeFValidationError(f"cannot open {selection.dataset_id}: {exc}") from exc
        return JSONResponse({"api_version": 1, "mode": "dataset"})

    @endpoint("POST", "/api/v1/registry/close").route(app)
    async def close_dataset(_body: StrictQuery) -> JSONResponse:
        """Return to catalog browsing and release the selected reader."""
        async with lifecycle_lock:
            session_state.transitioning = True
            try:
                await registry_call(session_state.close)
            finally:
                session_state.transitioning = False
        return JSONResponse({"api_version": 1, "mode": "registry"})

    app.include_router(inspection.router(session_state.worker, registry_name, include_session=False))

    return app


def _base_app(token: str, port: int, lifespan: Callable[[FastAPI], AbstractAsyncContextManager[Any]]) -> FastAPI:
    """Return an app with shared security, error handling, and packaged asset routes."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.add_middleware(RequestBoundary, token=token, authority=f"127.0.0.1:{port}")
    app.middleware("http")(security_headers)

    @app.exception_handler(TimeFValidationError)
    def invalid_request(_request: Request, exc: TimeFValidationError) -> JSONResponse:
        """Return sanitized caller errors."""
        return JSONResponse(
            {"code": "invalid_request", "message": str(exc)},
            status_code=422,
        )

    @app.exception_handler(RequestValidationError)
    def invalid_fields(_request: Request, exc: RequestValidationError) -> JSONResponse:
        """Return safe field errors without echoing submitted values."""
        fields = [{"field": ".".join(map(str, error["loc"])), "message": error["msg"]} for error in exc.errors()]
        return JSONResponse(
            {
                "code": "invalid_request",
                "message": "; ".join(f"{item['field']}: {item['message']}" for item in fields),
                "fields": fields,
            },
            status_code=422,
        )

    @app.exception_handler(TimeFFormatError)
    def invalid_format(_request: Request, _exc: TimeFFormatError) -> JSONResponse:
        """Return an actionable error for incompatible data."""
        return JSONResponse(
            {
                "code": "invalid_artifact",
                "message": "Dataset is corrupt or incompatible; rebuild it with this TimeNet version.",
            },
            status_code=422,
        )

    @app.exception_handler(TimeNetDatasetNotFoundError)
    def missing_dataset(_request: Request, _exc: TimeNetDatasetNotFoundError) -> JSONResponse:
        """Return an error explaining an unavailable selection."""
        return JSONResponse(
            {"code": "dataset_not_found", "message": "Dataset version is unavailable; refresh the catalog."},
            status_code=404,
        )

    @app.exception_handler(TimeNetRegistryError)
    def registry_failure(_request: Request, _exc: TimeNetRegistryError) -> JSONResponse:
        """Return a retryable registry error."""
        return JSONResponse(
            {"code": "registry_unavailable", "message": "Registry is unavailable; check its connection and retry."},
            status_code=503,
        )

    @app.get("/", response_class=HTMLResponse)
    async def shell() -> str:
        """Return the frontend-owned fact-free shell."""
        return files("timenet.viewer").joinpath("static/index.html").read_text()

    @app.get("/static/{asset}")
    async def static_asset(asset: str) -> HTMLResponse:
        """Return only known locally packaged browser assets."""
        media = {
            "app.js": "application/javascript",
            "app.css": "text/css",
            "plotly-basic.min.js": "application/javascript",
        }
        if asset not in media:
            return HTMLResponse("Not found", status_code=404)
        return HTMLResponse(files("timenet.viewer").joinpath("static", asset).read_text(), media_type=media[asset])

    return app
