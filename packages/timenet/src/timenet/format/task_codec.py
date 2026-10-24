"""Canonical codecs between Task fields and the typed columns of the ``tasks`` and ``task_targets`` tables.

Spans, task configuration, and targets are all stored as plain columns rather than JSON, so DuckDB can
filter on them and the reader builds objects straight from row values. Object references travel as
public IDs here; the writer and reader swap them for internal keys at the table boundary. A span
holds Signal objects, so encoding reads their ids and decoding looks the read-back Signals up by id.
"""

from collections.abc import Iterable, Mapping
from typing import Any, TypeVar, cast

from pydantic import ValidationError

from timenet.dataset.record import Record
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFFormatError
from timenet.format.task_target_kind import TaskTargetKind
from timenet.format.task_target_row import TargetValueRow
from timenet.types import (
    ClassificationTask,
    InputModality,
    LocalizationMode,
    ScalarPredictionTask,
    Span,
    Split,
    StepInterval,
    StepPoint,
    Task,
    TaskType,
    TemporalLocalizationTask,
    TimeInterval,
    TimePoint,
)


_T = TypeVar("_T")


PAYLOAD_COLUMNS = ("target_schema", "unit", "target_name", "mode")
"""The ``tasks`` columns that hold subclass-specific configuration. Unused ones are NULL."""


def decode_input_modalities(values: list[str] | None) -> frozenset[InputModality]:
    """Decode a required stored input modality set.

    Returns:
        The declared input kinds.

    Raises:
        TimeFFormatError: If a stored input kind is unknown or repeated.
    """
    if not values:
        raise TimeFFormatError("task has no stored input modalities")
    try:
        result = frozenset(InputModality(value) for value in values)
    except ValueError as exc:
        raise TimeFFormatError(f"task has an unknown stored input modality: {values!r}") from exc
    if len(result) != len(values):
        raise TimeFFormatError(f"task has repeated stored input modalities: {values!r}")
    return result


def decode_split(value: str | None) -> Split | None:
    """Decode a stored split.

    Returns:
        The task's partition, or ``None`` for a task its dataset does not partition.

    Raises:
        TimeFFormatError: If the stored split is unknown.
    """
    if value is None:
        return None
    try:
        return Split(value)
    except ValueError as exc:
        raise TimeFFormatError(f"task has an unknown stored split: {value!r}") from exc


def encode_span(span: Span | None) -> dict[str, Any] | None:
    """Encode a concrete span as the four span columns without losing its frame or shape.

    Returns:
        ``{"span_type", "span_start", "span_end", "signal_ids"}``, or ``None``. ``signal_ids`` is the
        span's series scope: ``None`` for every series, a tuple of IDs for a time span, and a
        one-element tuple for a step span.

    Raises:
        TimeFFormatError: If the Span subtype is unsupported.
    """
    if span is None:
        return None
    if not isinstance(span, (TimePoint, TimeInterval, StepPoint, StepInterval)):
        raise TimeFFormatError(f"cannot encode unsupported span type {type(span).__name__}")
    row = TargetValueRow.from_target(span)
    return {
        "span_type": row.target_kind.value,
        "span_start": row.span_start,
        "span_end": row.span_end,
        "signal_ids": row.span_signal_ids,
    }


def decode_span(
    span_type: str | None,
    span_start: int | None,
    span_end: int | None,
    signal_ids: Iterable[str] | None,
    *,
    signals: Mapping[str, Signal],
) -> Span | None:
    """Decode the four span columns.

    Args:
        span_type: The stored span shape, or ``None`` for no span.
        span_start: The stored start bound.
        span_end: The stored end bound.
        signal_ids: The ids of the Signals the span is scoped to.
        signals: The read-back Signals by id, which take the place of ``signal_ids``.

    Returns:
        The decoded Span, or ``None`` when ``span_type`` is NULL.

    Raises:
        TimeFFormatError: If the stored span type is unknown, a required bound is NULL, or a
            Signal id names no read-back Signal.
    """
    if span_type is None:
        return None
    if span_start is None:
        raise TimeFFormatError(f"stored {span_type} span has no start")
    series = _signals(signal_ids, signals)
    if span_type == "time_point":
        return TimePoint(start_us=span_start, time_series=series)
    if span_type == "time_interval":
        if span_end is None:
            raise TimeFFormatError("stored time_interval span has no end")
        return TimeInterval(start_us=span_start, end_us=span_end, time_series=series)
    if not series:
        raise TimeFFormatError(f"stored {span_type} span names no series")
    if span_type == "step_point":
        return StepPoint(start=span_start, time_series=series[0])
    if span_type == "step_interval":
        if span_end is None:
            raise TimeFFormatError("stored step_interval span has no stop")
        return StepInterval(start=span_start, stop=span_end, time_series=series[0])
    raise TimeFFormatError(f"stored span has unknown type {span_type!r}")


def encode_task_payload(task: Task) -> dict[str, str | None]:
    """Encode the configuration owned by the concrete Task class as the payload columns.

    Returns:
        One value per :data:`PAYLOAD_COLUMNS`, ``None`` where the task type has no such field.
    """
    columns: dict[str, str | None] = dict.fromkeys(PAYLOAD_COLUMNS)
    if isinstance(task, ClassificationTask):
        columns["target_schema"] = task.target_schema
    elif isinstance(task, ScalarPredictionTask):
        columns["unit"] = None if task.unit is None else str(task.unit)
        columns["target_name"] = task.target_name
    elif isinstance(task, TemporalLocalizationTask):
        columns["mode"] = str(task.mode)
    return columns


def decode_task_payload(task_type: TaskType, columns: dict[str, str | None]) -> dict[str, Any]:
    """Decode concrete Task configuration from the payload columns.

    Returns:
        Keyword arguments for the concrete Task constructor.
    """
    if task_type is TaskType.CLASSIFICATION:
        return {"target_schema": columns.get("target_schema")}
    if task_type is TaskType.SCALAR_PREDICTION:
        return {"unit": columns.get("unit"), "target_name": columns.get("target_name")}
    if task_type is TaskType.TEMPORAL_LOCALIZATION:
        return {"mode": LocalizationMode(columns.get("mode") or LocalizationMode.SPARSE)}
    return {}


def decode_target(  # noqa: PLR0911 - each target kind has one direct decoding branch
    row: Mapping[str, object] | TargetValueRow,
    *,
    records: Mapping[str, Record],
    signals: Mapping[str, Signal],
) -> object:
    """Decode one typed target row and resolve stored object references.

    Returns:
        The native scalar, Record, Signal, or Span target.

    Raises:
        TimeFFormatError: If a required value or referenced object is missing, or the kind is unknown.
    """
    try:
        values = TargetValueRow.model_validate(row)
    except ValidationError as exc:
        raise TimeFFormatError(f"invalid stored task target: {exc}") from exc
    kind = values.target_kind
    if kind == TaskTargetKind.TEXT:
        return _required(values.text_value, kind, "text_value")
    if kind == TaskTargetKind.INTEGER:
        return _required(values.integer_value, kind, "integer_value")
    if kind == TaskTargetKind.FLOAT:
        return _required(values.float_value, kind, "float_value")
    if kind == TaskTargetKind.BOOLEAN:
        return _required(values.boolean_value, kind, "boolean_value")
    if kind == TaskTargetKind.RECORD:
        return _resolve(records, _required(values.record_id, kind, "record_id"), "Record")
    if kind == TaskTargetKind.SIGNAL:
        return _resolve(signals, _required(values.signal_id, kind, "signal_id"), "Signal")
    start = _required(values.span_start, kind, "span_start")
    if kind == TaskTargetKind.TIME_POINT:
        return TimePoint(start_us=start, time_series=_signals(values.span_signal_ids, signals))
    if kind == TaskTargetKind.TIME_INTERVAL:
        return TimeInterval(
            start_us=start,
            end_us=_required(values.span_end, kind, "span_end"),
            time_series=_signals(values.span_signal_ids, signals),
        )
    (signal_id,) = _required(values.span_signal_ids, kind, "span_signal_ids")
    signal = _resolve(signals, signal_id, "Signal")
    if kind == TaskTargetKind.STEP_POINT:
        return StepPoint(start=start, time_series=signal)
    if kind == TaskTargetKind.STEP_INTERVAL:
        return StepInterval(start=start, stop=_required(values.span_end, kind, "span_end"), time_series=signal)
    raise TimeFFormatError(f"task target has unknown kind {kind!r}")


def _required(value: _T | None, kind: TaskTargetKind, name: str) -> _T:
    if value is None:
        raise TimeFFormatError(f"task target kind {kind.value!r} requires {name}")
    return value


def _resolve(objects: Mapping[str, _T], public_id: str, label: str) -> _T:
    try:
        return objects[public_id]
    except KeyError as exc:
        raise TimeFFormatError(f"task target refers to missing {label} {public_id!r}") from exc


def _signals(signal_ids: object, signals: Mapping[str, Signal]) -> tuple[Signal, ...] | None:
    """Resolve a stored Signal id sequence to the read-back Signals.

    Returns:
        The Signals in stored order, or ``None`` when no ids are stored.
    """
    if signal_ids is None:
        return None
    return tuple(_resolve(signals, signal_id, "Signal") for signal_id in cast("Iterable[str]", signal_ids))
