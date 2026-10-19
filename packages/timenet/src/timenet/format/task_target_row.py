"""Typed value columns and references exchanged by task-target codecs."""

from typing import TypedDict


class TaskTargetValues(TypedDict):
    """Describe the value columns shared by public and stored target rows."""

    target_kind: str
    text_value: str | None
    integer_value: int | None
    float_value: float | None
    boolean_value: bool | None
    span_start: int | None
    span_end: int | None


class TaskTargetRow(TaskTargetValues):
    """Describe one target using public object IDs and nullable value columns."""

    record_id: str | None
    signal_id: str | None
    span_signal_ids: tuple[str, ...] | list[str] | None


class StoredTaskTargetRow(TaskTargetValues):
    """Describe a catalog target using internal object keys and its task position."""

    task_key: int
    position: int
    record_key: int | None
    signal_key: int | None
    signal_keys: list[int] | None
