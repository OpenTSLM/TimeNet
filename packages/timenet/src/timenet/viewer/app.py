"""Composition and resource lifetime for the authenticated local inspector."""

from __future__ import annotations

from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.viewer.jobs import JobManager
from timenet.viewer.routes import inspection, jobs
from timenet.viewer.security import RequestBoundary, security_headers
from timenet.viewer.workers import ReaderWorker


def create_app(reader: TimeFReader, *, token: str, port: int, registry: str = "unknown") -> FastAPI:
    """Return local HTTP adapters composed around one independently owned reader."""
    worker = ReaderWorker(reader, "viewer-interactive")
    interactive_jobs = JobManager(executor=worker.executor)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:
        """Release jobs before closing readers on their owning threads.

        Yields:
            The active application lifetime.
        """
        try:
            yield
        finally:
            interactive_jobs.close()
            worker.close()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.add_middleware(RequestBoundary, token=token, authority=f"127.0.0.1:{port}")
    app.middleware("http")(security_headers)

    @app.exception_handler(TimeFValidationError)
    def invalid_request(_request: Request, exc: TimeFValidationError) -> JSONResponse:
        """Return sanitized caller errors."""
        return JSONResponse({"message": str(exc)}, status_code=409 if str(exc) == "job queue is full" else 422)

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

    app.include_router(inspection.router(worker, registry))
    app.include_router(jobs.router(worker, interactive_jobs))
    return app
