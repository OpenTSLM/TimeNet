"""Transport-independent structural inputs for bounded inspection queries."""

from __future__ import annotations

from typing import Literal, Protocol


class RecordQuery(Protocol):
    """Structural input to RecordQuery inspection."""

    query: str | None
    after: str | None
    limit: int


class RecordDetailQuery(Protocol):
    """Structural input to RecordDetailQuery inspection."""

    record_id: str
    signal_id: str | None


class SourceQuery(Protocol):
    """Structural input to SourceQuery inspection."""

    record_id: str
    parent_id: str | None
    after: str | None
    limit: int


class SignalQuery(Protocol):
    """Structural input to SignalQuery inspection."""

    record_id: str
    source_id: str
    after: str | None
    limit: int


class WindowQuery(Protocol):
    """Structural input to WindowQuery inspection."""

    record_id: str
    signal_id: str
    start: int
    stop: int | None
    mode: Literal["plot", "raw"]
    width: int
    component: tuple[int, ...]
    start_us: int | None
    end_us: int | None
    full: bool


class TaskQuery(Protocol):
    """Structural input to TaskQuery inspection."""

    query: str | None
    split: str | None
    task_type: str | None
    after: str | None
    limit: int


class AnnotationQuery(Protocol):
    """Structural input to AnnotationQuery inspection."""

    object_type: Literal["Dataset", "Record", "Source", "Signal", "Task"] | None
    object_id: str | None
    span_type: Literal["static", "point", "interval"] | None
    value_query: str | None
    after: str | None
    limit: int


class TaskDetailQuery(Protocol):
    """Structural input to TaskDetailQuery inspection."""

    task_id: str
    after: int
    limit: int


class OwnerRecordQuery(Protocol):
    """Structural input to OwnerRecordQuery inspection."""

    object_type: Literal["Source", "Signal"]
    object_id: str


class AnnotationWindowQuery(Protocol):
    """Structural input to AnnotationWindowQuery inspection."""

    record_id: str
    after: str | None
    signal_id: str
    start_us: int | None
    end_us: int | None
    limit: int
