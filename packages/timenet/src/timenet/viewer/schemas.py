"""Validated request bodies for the inspection HTTP API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictQuery(BaseModel):
    """Reject misspelled and unsupported request fields."""

    model_config = ConfigDict(extra="forbid")


class RecordQuery(StrictQuery):
    """The bounded record-list request body."""

    query: str | None = Field(default=None, max_length=512)
    after: str | None = Field(default=None, max_length=4096)
    limit: int = Field(default=50, ge=1, le=200)


class RecordDetailQuery(StrictQuery):
    """One record's persisted metadata and optional selected signal metadata."""

    record_id: str = Field(min_length=1, max_length=4096)
    signal_id: str | None = Field(default=None, max_length=4096)


class SourceQuery(StrictQuery):
    """The bounded source-tree request body."""

    record_id: str = Field(min_length=1, max_length=4096)
    parent_id: str | None = Field(default=None, max_length=4096)
    after: str | None = Field(default=None, max_length=4096)
    limit: int = Field(default=50, ge=1, le=200)


class SignalQuery(StrictQuery):
    """The bounded signal-list request body."""

    record_id: str = Field(min_length=1, max_length=4096)
    source_id: str = Field(min_length=1, max_length=4096)
    after: str | None = Field(default=None, max_length=4096)
    limit: int = Field(default=50, ge=1, le=200)


class WindowQuery(StrictQuery):
    """A bounded scalar signal window request."""

    record_id: str = Field(min_length=1, max_length=4096)
    signal_id: str = Field(min_length=1, max_length=4096)
    start: int = Field(default=0, ge=0)
    stop: int | None = Field(default=None, ge=0)
    mode: Literal["plot", "raw"] = "plot"
    width: int = Field(default=800, ge=256, le=2000)
    component: tuple[int, ...] = ()
    start_us: int | None = None
    end_us: int | None = None
    full: bool = False


class TaskQuery(StrictQuery):
    """A bounded task-list request body."""

    query: str | None = Field(default=None, max_length=512)
    split: str | None = Field(default=None, max_length=64)
    task_type: str | None = Field(default=None, max_length=64)
    after: str | None = Field(default=None, max_length=4096)
    limit: int = Field(default=50, ge=1, le=200)


class AnnotationQuery(StrictQuery):
    """A bounded annotation-list request body."""

    object_type: Literal["Dataset", "Record", "Source", "Signal", "Task"] | None = None
    object_id: str | None = Field(default=None, max_length=4096)
    span_type: Literal["static", "point", "interval"] | None = None
    value_query: str | None = Field(default=None, max_length=512)
    after: str | None = Field(default=None, max_length=4096)
    limit: int = Field(default=50, ge=1, le=200)


class TaskDetailQuery(StrictQuery):
    """The selected task identifier."""

    task_id: str = Field(min_length=1, max_length=4096)
    after: int = Field(default=0, ge=0)
    limit: int = Field(default=200, ge=1, le=200)


class OwnerRecordQuery(StrictQuery):
    """A Source or Signal owner whose containing Record is needed for navigation."""

    object_type: Literal["Source", "Signal"]
    object_id: str = Field(min_length=1, max_length=4096)


class AnnotationWindowQuery(StrictQuery):
    """A selected signal step/time window for annotation overlays."""

    record_id: str = Field(min_length=1, max_length=4096)
    after: str | None = Field(default=None, max_length=16384)
    signal_id: str = Field(min_length=1, max_length=4096)
    start_us: int | None = None
    end_us: int | None = None
    limit: int = Field(default=50, ge=1, le=200)
