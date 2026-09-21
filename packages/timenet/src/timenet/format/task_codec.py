"""Canonical codecs for Task configuration, scopes, and typed target rows."""

from collections.abc import Iterable
from typing import Any, cast

from timenet.dataset.record import Record
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFFormatError
from timenet.types import (
    ClassificationTask,
    LocalizationMode,
    ScalarPredictionTask,
    Span,
    StepInterval,
    StepPoint,
    Task,
    TaskType,
    TemporalLocalizationTask,
    TimeInterval,
    TimePoint,
)


TARGET_VALUE_COLUMNS = (
    "text_value",
    "integer_value",
    "float_value",
    "boolean_value",
    "record_id",
    "signal_id",
    "span_start",
    "span_end",
    "span_signal_ids",
)


def encode_span(span: Span | None) -> dict[str, Any] | None:
    """Encode a concrete span without losing its frame or shape.

    Returns:
        The canonical span mapping, or ``None``.

    Raises:
        TimeFFormatError: If the Span subtype is unsupported.
    """
    if span is None:
        return None
    if isinstance(span, TimePoint):
        return {"type": "time_point", "start": span.start_us, "signal_ids": span.time_series_ids}
    if isinstance(span, TimeInterval):
        return {
            "type": "time_interval",
            "start": span.start_us,
            "end": span.end_us,
            "signal_ids": span.time_series_ids,
        }
    if isinstance(span, StepPoint):
        return {"type": "step_point", "start": span.start, "signal_id": span.time_series_id}
    if isinstance(span, StepInterval):
        return {
            "type": "step_interval",
            "start": span.start,
            "end": span.stop,
            "signal_id": span.time_series_id,
        }
    raise TimeFFormatError(f"cannot encode unsupported span type {type(span).__name__}")


def decode_span(data: dict[str, Any] | None) -> Span | None:
    """Decode a canonical span mapping.

    Returns:
        The decoded Span, or ``None``.

    Raises:
        TimeFFormatError: If the stored span type is unknown.
    """
    if data is None:
        return None
    span_type = data.get("type")
    if span_type == "time_point":
        return TimePoint(start_us=data["start"], time_series_ids=_tuple_or_none(data.get("signal_ids")))
    if span_type == "time_interval":
        return TimeInterval(
            start_us=data["start"],
            end_us=data["end"],
            time_series_ids=_tuple_or_none(data.get("signal_ids")),
        )
    if span_type == "step_point":
        return StepPoint(start=data["start"], time_series_id=data["signal_id"])
    if span_type == "step_interval":
        return StepInterval(start=data["start"], stop=data["end"], time_series_id=data["signal_id"])
    raise TimeFFormatError(f"task payload has unknown span type {span_type!r}")


def encode_task_payload(task: Task) -> dict[str, Any]:
    """Encode only configuration owned by the concrete Task class.

    Returns:
        The task configuration mapping, without targets or relationships.
    """
    if isinstance(task, ClassificationTask):
        return {"target_schema": task.target_schema}
    if isinstance(task, ScalarPredictionTask):
        return {"unit": task.unit, "target_name": task.target_name}
    if isinstance(task, TemporalLocalizationTask):
        return {"mode": str(task.mode)}
    return {}


def decode_task_payload(task_type: TaskType, payload: dict[str, Any]) -> dict[str, Any]:
    """Decode concrete Task configuration from its canonical mapping.

    Returns:
        Keyword arguments for the concrete Task constructor.
    """
    if task_type is TaskType.CLASSIFICATION:
        return {"target_schema": payload.get("target_schema")}
    if task_type is TaskType.SCALAR_PREDICTION:
        return {"unit": payload.get("unit"), "target_name": payload.get("target_name")}
    if task_type is TaskType.TEMPORAL_LOCALIZATION:
        return {"mode": LocalizationMode(payload.get("mode", LocalizationMode.SPARSE))}
    return {}


def encode_target(target: object) -> dict[str, object | None]:
    """Encode one public target item as a typed table row.

    Returns:
        A row with one target kind and its corresponding value column.

    Raises:
        TimeFFormatError: If the target type is unsupported.
    """
    row: dict[str, object | None] = dict.fromkeys(TARGET_VALUE_COLUMNS)
    if isinstance(target, bool):
        row.update(target_kind="boolean", boolean_value=target)
    elif isinstance(target, str):
        row.update(target_kind="text", text_value=target)
    elif isinstance(target, int):
        row.update(target_kind="integer", integer_value=target)
    elif isinstance(target, float):
        row.update(target_kind="float", float_value=target)
    elif isinstance(target, Record):
        row.update(target_kind="record", record_id=target.id)
    elif isinstance(target, Signal):
        row.update(target_kind="signal", signal_id=target.id)
    elif isinstance(target, TimePoint):
        row.update(
            target_kind="time_point",
            span_start=target.start_us,
            span_signal_ids=target.time_series_ids,
        )
    elif isinstance(target, TimeInterval):
        row.update(
            target_kind="time_interval",
            span_start=target.start_us,
            span_end=target.end_us,
            span_signal_ids=target.time_series_ids,
        )
    elif isinstance(target, StepPoint):
        row.update(target_kind="step_point", span_start=target.start, signal_id=target.time_series_id)
    elif isinstance(target, StepInterval):
        row.update(
            target_kind="step_interval",
            span_start=target.start,
            span_end=target.stop,
            signal_id=target.time_series_id,
        )
    else:
        raise TimeFFormatError(f"cannot encode unsupported target type {type(target).__name__}")
    return row


def decode_target(  # noqa: PLR0911 - each target kind has one direct decoding branch
    row: dict[str, object],
    *,
    records: dict[str, Record],
    signals: dict[str, Signal],
) -> object:
    """Decode one typed target row and resolve stored object references.

    Returns:
        The native scalar, Record, Signal, or Span target.

    Raises:
        TimeFFormatError: If a required value or referenced object is missing, or the kind is unknown.
    """
    kind = row["target_kind"]
    if kind == "text":
        return _required(row, "text_value")
    if kind == "integer":
        return _required(row, "integer_value")
    if kind == "float":
        return _required(row, "float_value")
    if kind == "boolean":
        return _required(row, "boolean_value")
    if kind == "record":
        return _resolve(records, row, "record_id", "Record")
    if kind == "signal":
        return _resolve(signals, row, "signal_id", "Signal")
    start = _required(row, "span_start")
    signal_ids = _tuple_or_none(row["span_signal_ids"])
    if kind == "time_point":
        return TimePoint(start_us=start, time_series_ids=signal_ids)
    if kind == "time_interval":
        return TimeInterval(start_us=start, end_us=_required(row, "span_end"), time_series_ids=signal_ids)
    signal_id = _required(row, "signal_id")
    if kind == "step_point":
        return StepPoint(start=start, time_series_id=signal_id)
    if kind == "step_interval":
        return StepInterval(start=start, stop=_required(row, "span_end"), time_series_id=signal_id)
    raise TimeFFormatError(f"task target has unknown kind {kind!r}")


def _required(row: dict[str, object], name: str) -> Any:
    value = row.get(name)
    if value is None:
        raise TimeFFormatError(f"task target kind {row.get('target_kind')!r} requires {name}")
    return value


def _resolve(objects: dict[str, Any], row: dict[str, object], name: str, label: str) -> Any:
    public_id = _required(row, name)
    try:
        return objects[public_id]
    except KeyError as exc:
        raise TimeFFormatError(f"task target refers to missing {label} {public_id!r}") from exc


def _tuple_or_none(value: object) -> tuple[str, ...] | None:
    """Normalize a stored signal-ID sequence.

    Returns:
        The immutable sequence, or ``None``.
    """
    return None if value is None else tuple(cast("Iterable[str]", value))
