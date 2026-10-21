"""Canonical codecs between Task fields and the typed columns of the ``tasks`` and ``task_targets`` tables.

Spans, task configuration, and targets are all stored as plain columns rather than JSON, so DuckDB can
filter on them and the reader builds objects straight from row values. Object references are public
IDs, and the target rows use the ``task_targets`` column names, so they are stored and read back as
they are. A span holds Signal objects, so encoding reads their ids and decoding looks the read-back
Signals up by id.
"""

from collections.abc import Iterable, Mapping
from enum import StrEnum, unique
from typing import Any, TypedDict, cast

from timenet.dataset.record import Record
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFFormatError
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
from timenet.types.ids import EntityId


@unique
class SpanKind(StrEnum):
    """Stored shape of a span, in ``tasks.scope_type`` and ``task_targets.target_kind``."""

    TIME_POINT = "time_point"
    TIME_INTERVAL = "time_interval"
    STEP_POINT = "step_point"
    STEP_INTERVAL = "step_interval"


TIME_SPAN_KINDS = frozenset({SpanKind.TIME_POINT, SpanKind.TIME_INTERVAL})
STEP_SPAN_KINDS = frozenset({SpanKind.STEP_POINT, SpanKind.STEP_INTERVAL})


@unique
class TargetKind(StrEnum):
    """Stored kind of one inline target, in ``task_targets.target_kind``."""

    TEXT = "text"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    RECORD = "record"
    SIGNAL = "signal"
    TIME_POINT = SpanKind.TIME_POINT.value
    TIME_INTERVAL = SpanKind.TIME_INTERVAL.value
    STEP_POINT = SpanKind.STEP_POINT.value
    STEP_INTERVAL = SpanKind.STEP_INTERVAL.value


@unique
class TaskRelationshipField(StrEnum):
    """Task field that a ``task_record_refs`` or ``task_annotation_refs`` row belongs to."""

    INPUTS = "inputs"
    CANDIDATE_RECORDS = "candidate_records"
    INPUT_ANNOTATIONS = "input_annotations"
    TARGET_ANNOTATIONS = "target_annotations"


RECORD_FIELDS = frozenset({TaskRelationshipField.INPUTS, TaskRelationshipField.CANDIDATE_RECORDS})
"""The fields stored in ``task_record_refs``."""
ANNOTATION_FIELDS = frozenset({TaskRelationshipField.INPUT_ANNOTATIONS, TaskRelationshipField.TARGET_ANNOTATIONS})
"""The fields stored in ``task_annotation_refs``."""


class EncodedTarget(TypedDict):
    """One ``task_targets`` row without its task and position, with object references as public IDs."""

    target_kind: TargetKind
    text_value: str | None
    integer_value: int | None
    float_value: float | None
    boolean_value: bool | None
    record_id: EntityId | None
    signal_id: EntityId | None
    span_start: int | None
    span_end: int | None
    span_signal_ids: tuple[EntityId, ...] | None


def empty_target(kind: TargetKind) -> EncodedTarget:
    """Return a target row of ``kind`` with every value column NULL.

    Returns:
        The row, ready for the caller to fill the columns of its kind.
    """
    return {
        "target_kind": kind,
        "text_value": None,
        "integer_value": None,
        "float_value": None,
        "boolean_value": None,
        "record_id": None,
        "signal_id": None,
        "span_start": None,
        "span_end": None,
        "span_signal_ids": None,
    }


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
            "span_type": SpanKind.TIME_POINT,
            "span_start": span.start_us,
            "span_end": None,
            "signal_ids": _signal_ids(span.time_series),
        }
    if isinstance(span, TimeInterval):
        return {
            "span_type": SpanKind.TIME_INTERVAL,
            "span_start": span.start_us,
            "span_end": span.end_us,
            "signal_ids": _signal_ids(span.time_series),
        }
    if isinstance(span, StepPoint):
        return {
            "span_type": SpanKind.STEP_POINT,
            "span_start": span.start,
            "span_end": None,
            "signal_ids": (span.time_series.id,),
        }
    if isinstance(span, StepInterval):
        return {
            "span_type": SpanKind.STEP_INTERVAL,
            "span_start": span.start,
            "span_end": span.stop,
            "signal_ids": (span.time_series.id,),
        }
    raise TimeFFormatError(f"cannot encode unsupported span type {type(span).__name__}")


def decode_span(
    span_type: str | None,
    span_start: int | None,
    span_end: int | None,
    signal_ids: Iterable[int] | None,
    *,
    signals: Mapping[int, Signal],
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
    try:
        kind = SpanKind(span_type)
    except ValueError as exc:
        raise TimeFFormatError(f"stored span has unknown type {span_type!r}") from exc
    if span_start is None:
        raise TimeFFormatError(f"stored {kind} span has no start")
    series = _signals(signal_ids, signals)
    if kind is SpanKind.TIME_POINT:
        return TimePoint(start_us=span_start, time_series=series)
    if kind is SpanKind.TIME_INTERVAL:
        if span_end is None:
            raise TimeFFormatError("stored time_interval span has no end")
        return TimeInterval(start_us=span_start, end_us=span_end, time_series=series)
    if not series:
        raise TimeFFormatError(f"stored {kind} span names no series")
    if kind is SpanKind.STEP_POINT:
        return StepPoint(start=span_start, time_series=series[0])
    if span_end is None:
        raise TimeFFormatError("stored step_interval span has no stop")
    return StepInterval(start=span_start, stop=span_end, time_series=series[0])


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


def encode_target(target: object) -> EncodedTarget:  # noqa: PLR0911 - one direct branch per target type
    """Encode one public target item as a typed table row.

    Returns:
        A row with one target kind and its corresponding value columns.

    Raises:
        TimeFFormatError: If the target type is unsupported.
    """
    if isinstance(target, bool):
        row = empty_target(TargetKind.BOOLEAN)
        row["boolean_value"] = target
        return row
    if isinstance(target, str):
        row = empty_target(TargetKind.TEXT)
        row["text_value"] = target
        return row
    if isinstance(target, int):
        row = empty_target(TargetKind.INTEGER)
        row["integer_value"] = target
        return row
    if isinstance(target, float):
        row = empty_target(TargetKind.FLOAT)
        row["float_value"] = target
        return row
    if isinstance(target, Record):
        row = empty_target(TargetKind.RECORD)
        row["record_id"] = target.id
        return row
    if isinstance(target, Signal):
        row = empty_target(TargetKind.SIGNAL)
        row["signal_id"] = target.id
        return row
    if isinstance(target, TimePoint):
        row = empty_target(TargetKind.TIME_POINT)
        row["span_start"] = target.start_us
        row["span_signal_ids"] = _signal_ids(target.time_series)
        return row
    if isinstance(target, TimeInterval):
        row = empty_target(TargetKind.TIME_INTERVAL)
        row["span_start"] = target.start_us
        row["span_end"] = target.end_us
        row["span_signal_ids"] = _signal_ids(target.time_series)
        return row
    if isinstance(target, StepPoint):
        row = empty_target(TargetKind.STEP_POINT)
        row["span_start"] = target.start
        row["signal_id"] = target.time_series.id
        return row
    if isinstance(target, StepInterval):
        row = empty_target(TargetKind.STEP_INTERVAL)
        row["span_start"] = target.start
        row["span_end"] = target.stop
        row["signal_id"] = target.time_series.id
        return row
    raise TimeFFormatError(f"cannot encode unsupported target type {type(target).__name__}")


def decode_target(  # noqa: PLR0911 - each target kind has one direct decoding branch
    row: dict[str, object],
    *,
    records: Mapping[int, Record],
    signals: Mapping[int, Signal],
) -> object:
    """Decode one typed target row and resolve stored object references.

    Returns:
        The native scalar, Record, Signal, or Span target.

    Raises:
        TimeFFormatError: If a required value or referenced object is missing, or the kind is unknown.
    """
    try:
        kind = TargetKind(cast("str", row["target_kind"]))
    except ValueError as exc:
        raise TimeFFormatError(f"task target has unknown kind {row['target_kind']!r}") from exc
    if kind is TargetKind.TEXT:
        return _required(row, "text_value")
    if kind is TargetKind.INTEGER:
        return _required(row, "integer_value")
    if kind is TargetKind.FLOAT:
        return _required(row, "float_value")
    if kind is TargetKind.BOOLEAN:
        return _required(row, "boolean_value")
    if kind is TargetKind.RECORD:
        return _resolve(records, row, "record_id", "Record")
    if kind is TargetKind.SIGNAL:
        return _resolve(signals, row, "signal_id", "Signal")
    start = _required(row, "span_start")
    if kind is TargetKind.TIME_POINT:
        return TimePoint(start_us=start, time_series=_signals(row["span_signal_ids"], signals))
    if kind is TargetKind.TIME_INTERVAL:
        return TimeInterval(
            start_us=start, end_us=_required(row, "span_end"), time_series=_signals(row["span_signal_ids"], signals)
        )
    if kind is TargetKind.STEP_POINT:
        return StepPoint(start=start, time_series=_resolve(signals, row, "signal_id", "Signal"))
    return StepInterval(
        start=start, stop=_required(row, "span_end"), time_series=_resolve(signals, row, "signal_id", "Signal")
    )


def _required(row: dict[str, object], name: str) -> Any:
    value = row.get(name)
    if value is None:
        raise TimeFFormatError(f"task target kind {row.get('target_kind')!r} requires {name}")
    return value


def _resolve(objects: Mapping[int, Any], row: dict[str, object], name: str, label: str) -> Any:
    public_id = _required(row, name)
    try:
        return objects[public_id]
    except KeyError as exc:
        raise TimeFFormatError(f"task target refers to missing {label} {public_id!r}") from exc


def _signal_ids(series: tuple[Signal, ...] | None) -> tuple[int, ...] | None:
    """Return the ids of a time span's Signals, or ``None`` for a span over the whole record."""
    return None if series is None else tuple(signal.id for signal in series)


def _signals(signal_ids: object, signals: Mapping[int, Signal]) -> tuple[Signal, ...] | None:
    """Resolve a stored Signal id sequence to the read-back Signals.

    Returns:
        The Signals in stored order, or ``None`` when no ids are stored.

    Raises:
        TimeFFormatError: If an id names no read-back Signal.
    """
    if signal_ids is None:
        return None
    resolved: list[Signal] = []
    for signal_id in cast("Iterable[int]", signal_ids):
        signal = signals.get(signal_id)
        if signal is None:
            raise TimeFFormatError(f"stored span refers to missing Signal {signal_id!r}")
        resolved.append(signal)
    return tuple(resolved)
