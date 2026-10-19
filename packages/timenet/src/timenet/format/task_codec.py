"""Canonical codecs between Task fields and the typed columns of the ``tasks`` and ``task_targets`` tables.

Spans, task configuration, and targets are all stored as plain columns rather than JSON, so DuckDB can
filter on them and the reader builds objects straight from row values. Object references travel as
public IDs here; the writer and reader swap them for internal keys at the table boundary.
"""

from collections.abc import Iterable
from typing import Any, Literal, cast

from timenet.dataset.record import Record
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFFormatError
from timenet.format.task_target_kind import TaskTargetKind
from timenet.format.task_target_row import TaskTargetRow
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
    if isinstance(span, TimePoint):
        return {
            "span_type": "time_point",
            "span_start": span.start_us,
            "span_end": None,
            "signal_ids": span.time_series_ids,
        }
    if isinstance(span, TimeInterval):
        return {
            "span_type": "time_interval",
            "span_start": span.start_us,
            "span_end": span.end_us,
            "signal_ids": span.time_series_ids,
        }
    if isinstance(span, StepPoint):
        return {
            "span_type": "step_point",
            "span_start": span.start,
            "span_end": None,
            "signal_ids": (span.time_series_id,),
        }
    if isinstance(span, StepInterval):
        return {
            "span_type": "step_interval",
            "span_start": span.start,
            "span_end": span.stop,
            "signal_ids": (span.time_series_id,),
        }
    raise TimeFFormatError(f"cannot encode unsupported span type {type(span).__name__}")


def decode_span(
    span_type: str | None,
    span_start: int | None,
    span_end: int | None,
    signal_ids: Iterable[str] | None,
) -> Span | None:
    """Decode the four span columns.

    Returns:
        The decoded Span, or ``None`` when ``span_type`` is NULL.

    Raises:
        TimeFFormatError: If the stored span type is unknown or a required bound is NULL.
    """
    if span_type is None:
        return None
    if span_start is None:
        raise TimeFFormatError(f"stored {span_type} span has no start")
    ids = _tuple_or_none(signal_ids)
    if span_type == "time_point":
        return TimePoint(start_us=span_start, time_series_ids=ids)
    if span_type == "time_interval":
        if span_end is None:
            raise TimeFFormatError("stored time_interval span has no end")
        return TimeInterval(start_us=span_start, end_us=span_end, time_series_ids=ids)
    if not ids:
        raise TimeFFormatError(f"stored {span_type} span names no series")
    if span_type == "step_point":
        return StepPoint(start=span_start, time_series_id=ids[0])
    if span_type == "step_interval":
        if span_end is None:
            raise TimeFFormatError("stored step_interval span has no stop")
        return StepInterval(start=span_start, stop=span_end, time_series_id=ids[0])
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


def _target_row(  # noqa: PLR0913 - the canonical target columns
    kind: TaskTargetKind,
    *,
    text_value: str | None = None,
    integer_value: int | None = None,
    float_value: float | None = None,
    boolean_value: bool | None = None,
    record_id: str | None = None,
    signal_id: str | None = None,
    span_start: int | None = None,
    span_end: int | None = None,
    span_signal_ids: tuple[str, ...] | list[str] | None = None,
) -> TaskTargetRow:
    """Fill the canonical value columns, leaving unused ones NULL.

    Returns:
        The stored kind tag and every target value column.
    """
    return TaskTargetRow(
        target_kind=kind.value,
        text_value=text_value,
        integer_value=integer_value,
        float_value=float_value,
        boolean_value=boolean_value,
        record_id=record_id,
        signal_id=signal_id,
        span_start=span_start,
        span_end=span_end,
        span_signal_ids=span_signal_ids,
    )


def encode_target(target: object) -> TaskTargetRow:  # noqa: PLR0911 - one return per target kind
    """Encode one public target item as a typed table row.

    Returns:
        A row with one target kind and its corresponding value column.

    Raises:
        TimeFFormatError: If the target type is unsupported.
    """
    if isinstance(target, bool):
        return _target_row(TaskTargetKind.BOOLEAN, boolean_value=target)
    elif isinstance(target, str):
        return _target_row(TaskTargetKind.TEXT, text_value=target)
    elif isinstance(target, int):
        return _target_row(TaskTargetKind.INTEGER, integer_value=target)
    elif isinstance(target, float):
        return _target_row(TaskTargetKind.FLOAT, float_value=target)
    elif isinstance(target, Record):
        return _target_row(TaskTargetKind.RECORD, record_id=target.id)
    elif isinstance(target, Signal):
        return _target_row(TaskTargetKind.SIGNAL, signal_id=target.id)
    elif isinstance(target, TimePoint):
        return _target_row(
            TaskTargetKind.TIME_POINT,
            span_start=target.start_us,
            span_signal_ids=target.time_series_ids,
        )
    elif isinstance(target, TimeInterval):
        return _target_row(
            TaskTargetKind.TIME_INTERVAL,
            span_start=target.start_us,
            span_end=target.end_us,
            span_signal_ids=target.time_series_ids,
        )
    elif isinstance(target, StepPoint):
        return _target_row(TaskTargetKind.STEP_POINT, span_start=target.start, signal_id=target.time_series_id)
    elif isinstance(target, StepInterval):
        return _target_row(
            TaskTargetKind.STEP_INTERVAL,
            span_start=target.start,
            span_end=target.stop,
            signal_id=target.time_series_id,
        )
    else:
        raise TimeFFormatError(f"cannot encode unsupported target type {type(target).__name__}")


def decode_target(  # noqa: PLR0911 - each target kind has one direct decoding branch
    row: TaskTargetRow,
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
    try:
        kind = TaskTargetKind(row["target_kind"])
    except ValueError as exc:
        raise TimeFFormatError(f"task target has unknown kind {row['target_kind']!r}") from exc
    if kind == TaskTargetKind.TEXT:
        return _required(row, "text_value")
    if kind == TaskTargetKind.INTEGER:
        return _required(row, "integer_value")
    if kind == TaskTargetKind.FLOAT:
        return _required(row, "float_value")
    if kind == TaskTargetKind.BOOLEAN:
        return _required(row, "boolean_value")
    if kind == TaskTargetKind.RECORD:
        return _resolve(records, row, "record_id", "Record")
    if kind == TaskTargetKind.SIGNAL:
        return _resolve(signals, row, "signal_id", "Signal")
    start = _required(row, "span_start")
    signal_ids = _tuple_or_none(row["span_signal_ids"])
    if kind == TaskTargetKind.TIME_POINT:
        return TimePoint(start_us=start, time_series_ids=signal_ids)
    if kind == TaskTargetKind.TIME_INTERVAL:
        return TimeInterval(start_us=start, end_us=_required(row, "span_end"), time_series_ids=signal_ids)
    signal_id = _required(row, "signal_id")
    if kind == TaskTargetKind.STEP_POINT:
        return StepPoint(start=start, time_series_id=signal_id)
    if kind == TaskTargetKind.STEP_INTERVAL:
        return StepInterval(start=start, stop=_required(row, "span_end"), time_series_id=signal_id)
    raise TimeFFormatError(f"task target has unknown kind {kind!r}")


def _required(
    row: TaskTargetRow,
    name: Literal[
        "text_value",
        "integer_value",
        "float_value",
        "boolean_value",
        "record_id",
        "signal_id",
        "span_start",
        "span_end",
    ],
) -> Any:
    value = row.get(name)
    if value is None:
        raise TimeFFormatError(f"task target kind {row.get('target_kind')!r} requires {name}")
    return value


def _resolve(objects: dict[str, Any], row: TaskTargetRow, name: Literal["record_id", "signal_id"], label: str) -> Any:
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
