"""Error responses and request validation shared by the viewer's API routes."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import NoReturn

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from timenet.errors import (
    TimeFFormatError,
    TimeNetAccessError,
    TimeNetDatasetNotFoundError,
    TimeNetDownloadError,
    TimeNetRegistryError,
)
from timenet.manifest import Manifest
from timenet.types import Version
from timenet.types.metadata_fields import validate_dataset_id


class ViewerError(BaseModel):
    """Safe error facts with a stable operation and optional configured cap."""

    code: str
    message: str
    operation: str
    limit: int | None = None


def fail(status: int, code: str, message: str, operation: str, limit: int | None = None) -> NoReturn:
    """Abort the request with a structured error under the response's ``detail`` key.

    Raises:
        HTTPException: Always, carrying the serialized error.
    """
    raise HTTPException(status, ViewerError(code=code, message=message, operation=operation, limit=limit).model_dump())


@contextmanager
def registry_errors(operation: str) -> Generator[None]:
    """Map registry and storage failures to safe responses without exposing their messages.

    Yields:
        Control to the guarded registry or reader calls.
    """
    try:
        yield
    except TimeNetDatasetNotFoundError:
        fail(404, "not_found", "Dataset version not found.", operation)
    except TimeFFormatError:
        fail(500, "corrupt_artifact", "The dataset artifact is corrupt or unsupported.", operation)
    except TimeNetAccessError:
        fail(403, "restricted_access", "Dataset access is restricted.", operation)
    except (TimeNetRegistryError, TimeNetDownloadError, OSError):
        fail(503, "temporarily_unavailable", "Dataset data is temporarily unavailable. Retry later.", operation)
    except NotImplementedError:
        fail(501, "unsupported_operation", "The registry does not support this operation.", operation)


def resolve_version(dataset_id: str, version: str, initial_manifest: Manifest | None, operation: str) -> str:
    """Validate an explicit dataset reference before any registry access.

    Args:
        dataset_id: Requested dataset ID.
        version: Requested exact version, in any spelling ``Version.parse`` accepts.
        initial_manifest: Pinned manifest of a direct launch, which is the only version it serves.
        operation: Operation named in an error response.

    Returns:
        The canonical version string, so "01.0.0" and "1.0.0" name the same release everywhere.
    """
    try:
        validate_dataset_id(dataset_id)
        canonical = str(Version.parse(version))
    except ValueError:
        fail(422, "invalid_parameters", "Supply a valid dataset ID and exact version.", operation)
    if initial_manifest is not None and (
        dataset_id != initial_manifest.dataset_id or canonical != str(initial_manifest.metadata.dataset_version)
    ):
        fail(404, "not_found", "Dataset version not available in this viewer.", operation)
    return canonical


def install_error_handlers(app: FastAPI) -> None:
    """Report invalid API parameters in the same structured form as every other API error."""

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        if not request.url.path.startswith("/api/"):
            return await request_validation_exception_handler(request, exc)
        error = ViewerError(
            code="invalid_parameters",
            message="Invalid or missing request parameters.",
            operation=request.url.path.rsplit("/", 1)[-1],
        )
        return JSONResponse(status_code=422, content={"detail": error.model_dump()})
