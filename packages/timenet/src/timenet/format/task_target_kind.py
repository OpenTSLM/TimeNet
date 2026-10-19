"""Tags for the scalar, object, and span targets stored in the control catalog."""

from enum import StrEnum, unique


@unique
class TaskTargetKind(StrEnum):
    """Identify the value columns used by one stored task target."""

    TEXT = "text"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    RECORD = "record"
    SIGNAL = "signal"
    TIME_POINT = "time_point"
    TIME_INTERVAL = "time_interval"
    STEP_POINT = "step_point"
    STEP_INTERVAL = "step_interval"


def is_step_span(kind: str) -> bool:
    """Return whether a stored target kind uses a single step-indexed Signal."""
    return kind in {TaskTargetKind.STEP_POINT, TaskTargetKind.STEP_INTERVAL}
