"""Canonical JSON codec for task scalar and span payloads."""

from typing import Any

from timenet.errors import TimeFFormatError
from timenet.types import (
    AnswerTask,
    ClassificationTask,
    ForecastingTask,
    ScalarPredictionTask,
    Span,
    StepInterval,
    StepPoint,
    Task,
    TaskType,
    TemporalLocalizationTask,
    TimeInterval,
    TimePoint,
    TSCorrespondenceTask,
    TSEditingTask,
    TSGenerationTask,
)


def encode_span(span: Span | None) -> dict[str, Any] | None:
    """Encode a concrete span without losing its frame or shape.

    Returns:
        A JSON-compatible tagged mapping, or ``None``.

    Raises:
        TimeFFormatError: If ``span`` is not a built-in concrete span.
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
    """Decode a span from canonical task JSON.

    Returns:
        The concrete span, or ``None``.

    Raises:
        TimeFFormatError: If the stored type tag is unknown.
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
    """Encode only the concrete task's scalar and span payload fields.

    Returns:
        A JSON-compatible mapping with no object relationships.

    Raises:
        TimeFFormatError: If ``task`` is not a built-in concrete task.
    """
    if isinstance(task, ClassificationTask):
        return {"target": task.target, "target_schema": task.target_schema}
    if isinstance(task, AnswerTask):
        return {"target": task.target}
    if isinstance(task, ScalarPredictionTask):
        return {"target": task.target, "unit": task.unit, "target_name": task.target_name}
    if isinstance(task, TemporalLocalizationTask):
        target = None if task.target is None else [encode_span(span) for span in task.target]
        return {"target": target, "mode": str(task.mode)}
    if isinstance(task, ForecastingTask):
        return {"target_span": encode_span(task.target_span)}
    if isinstance(task, TSEditingTask | TSGenerationTask | TSCorrespondenceTask):
        return {}
    raise TimeFFormatError(f"cannot encode unsupported task type {type(task).__name__}")


def decode_task_payload(task_type: TaskType, payload: dict[str, Any]) -> dict[str, Any]:
    """Decode subclass-specific constructor arguments from stored JSON.

    Returns:
        Keyword arguments for the concrete task constructor.
    """
    if task_type is TaskType.TEMPORAL_LOCALIZATION:
        target = payload.get("target")
        return {
            "target": None if target is None else tuple(decode_span(span) for span in target),
            "mode": payload["mode"],
        }
    if task_type is TaskType.FORECASTING:
        return {"target_span": decode_span(payload.get("target_span"))}
    return dict(payload)


def _tuple_or_none(value: list[str] | None) -> tuple[str, ...] | None:
    """Normalize a JSON list scope.

    Returns:
        A tuple, or ``None``.
    """
    return None if value is None else tuple(value)
