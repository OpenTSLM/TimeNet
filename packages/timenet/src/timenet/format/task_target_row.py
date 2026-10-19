"""The canonical value columns exchanged by task-target codecs."""

from typing import TypedDict


class TaskTargetRow(TypedDict):
    """Describe one target using public object IDs and nullable value columns."""

    target_kind: str
    text_value: str | None
    integer_value: int | None
    float_value: float | None
    boolean_value: bool | None
    record_id: str | None
    signal_id: str | None
    span_start: int | None
    span_end: int | None
    span_signal_ids: tuple[str, ...] | list[str] | None
