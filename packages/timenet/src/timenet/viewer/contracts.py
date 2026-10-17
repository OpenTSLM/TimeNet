"""Endpoint request and response types used to generate the browser client."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter, FastAPI

from timenet.reader.inspection import results
from timenet.viewer import responses, schemas


@dataclass(frozen=True)
class Endpoint:
    """One public viewer operation."""

    path: str
    request: type | None
    response: type
    method: Literal["GET", "POST"] = "POST"
    status: int = 200
    query: type | None = None

    @property
    def key(self) -> str:
        """The method-qualified client operation identifier."""
        return f"{self.method} {self.path}"

    def route(self, router: APIRouter | FastAPI) -> Callable:
        """Register an implementation with this operation's method and response contract.

        Returns:
            The framework's route decorator.
        """
        return router.api_route(self.path, methods=[self.method], status_code=self.status, response_model=self.response)


ENDPOINTS = (
    Endpoint("/api/v1/session", None, responses.Session, method="GET", query=schemas.SessionQuery),
    Endpoint("/api/v1/overview", None, results.Overview, method="GET"),
    Endpoint("/api/v1/registry/datasets", None, responses.CatalogPage, method="GET", query=schemas.CatalogQuery),
    Endpoint("/api/v1/registry/open", schemas.DatasetSelection, responses.ModeResult),
    Endpoint("/api/v1/registry/close", schemas.StrictQuery, responses.ModeResult),
    Endpoint("/api/v1/records/query", schemas.RecordQuery, results.RecordPage),
    Endpoint("/api/v1/records/detail", schemas.RecordDetailQuery, results.RecordDetail),
    Endpoint("/api/v1/sources/query", schemas.SourceQuery, results.SourcePage),
    Endpoint("/api/v1/signals/query", schemas.SignalQuery, results.SignalPage),
    Endpoint("/api/v1/owners/record", schemas.OwnerRecordQuery, results.OwnerRecord),
    Endpoint("/api/v1/tasks/query", schemas.TaskQuery, results.TaskPage),
    Endpoint("/api/v1/tasks/detail", schemas.TaskDetailQuery, results.TaskDetail),
    Endpoint("/api/v1/annotations/query", schemas.AnnotationQuery, results.AnnotationPage),
    Endpoint("/api/v1/annotations/window", schemas.AnnotationWindowQuery, results.AnnotationPage),
    Endpoint("/api/v1/windows", schemas.WindowQuery, results.WindowResult),
)


def endpoint(method: str, path: str) -> Endpoint:
    """Find the shared operation definition for route registration.

    Returns:
        The matching registered operation.
    """
    return next(item for item in ENDPOINTS if item.method == method and item.path == path)
