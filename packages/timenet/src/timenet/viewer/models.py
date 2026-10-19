"""HTTP projections for tasks, without hydrated record hierarchies or signal values."""

from typing import Annotated, Literal, overload

from pydantic import BaseModel, Field, JsonValue, StrictBool, StrictFloat, StrictInt, StrictStr

from timenet.dataset import Record, Signal
from timenet.errors import TimeFFormatError
from timenet.json import JsonMapping
from timenet.types import Annotation, Task
from timenet.types.annotations import value_type_of
from timenet.types.spans import Span, StepSpan, TimeSpan
from timenet.types.splits import Split
from timenet.types.tasks import (
    ClassificationTask,
    ScalarPredictionTask,
    TaskType,
    TemporalLocalizationTask,
    TSCorrespondenceTask,
)
from timenet.types.wire import ValueType


class SpanDetail(BaseModel):
    """A point or half-open interval with explicit coordinate units and signal IDs."""

    kind: Literal["point", "interval"]
    unit: Literal["microseconds", "steps"]
    start: int
    stop: int | None
    signal_ids: list[str] | None


class ScalarTarget(BaseModel):
    """An inline target, retaining its native scalar type."""

    kind: Literal["str", "int", "float", "bool"]
    value: StrictStr | StrictInt | StrictFloat | StrictBool


class ReferenceTarget(BaseModel):
    """A record or signal target represented only by its public ID."""

    kind: Literal["record", "signal"]
    id: str


class SpanTarget(BaseModel):
    """A localized target in time or step coordinates."""

    kind: Literal["span"] = "span"
    value: SpanDetail


TaskTarget = Annotated[ScalarTarget | ReferenceTarget | SpanTarget, Field(discriminator="kind")]


class AnnotationDetail(BaseModel):
    """Typed annotation content and occurrence facts attached to a task."""

    id: str
    occurrence_id: str | None
    key: str
    value_type: ValueType | None
    value: JsonValue
    span: SpanDetail | None
    unit: str | None
    description: str | None
    source: str | None
    confidence: float | None
    metadata: JsonMapping
    occurrence_metadata: JsonMapping


class TaskSummary(BaseModel):
    """One list row with a bounded prompt preview."""

    id: str
    task_type: TaskType
    split: Split | None
    prompt_preview: str | None


class TaskPage(BaseModel):
    """A response-sized page with lookahead instead of an exact filtered total."""

    items: list[TaskSummary]
    offset: int
    limit: int
    has_more: bool


class TaskDetail(BaseModel):
    """Task fields with references projected to IDs and values left lazy."""

    id: str
    task_type: TaskType
    split: Split | None
    prompt: str | None
    rationale: str | None
    input_record_ids: list[str]
    parent_task_ids: list[str]
    input_modalities: list[str]
    scope: SpanDetail | None
    targets: list[TaskTarget] | None
    configuration: JsonMapping
    metadata: JsonMapping
    annotations: list[AnnotationDetail]
    input_annotations: list[AnnotationDetail]
    target_annotations: list[AnnotationDetail]


@overload
def project_span(span: Span) -> SpanDetail: ...


@overload
def project_span(span: None) -> None: ...


def project_span(span: Span | None) -> SpanDetail | None:
    """Return explicit span coordinates without resolving any referenced signal.

    Raises:
        TimeFFormatError: If the span has an unsupported coordinate frame.
    """
    if span is None:
        return None
    unit: Literal["microseconds", "steps"]
    if isinstance(span, TimeSpan):
        start = span.start_us
        signal_ids = None if span.time_series_ids is None else list(span.time_series_ids)
        unit = "microseconds"
    elif isinstance(span, StepSpan):
        start = span.start
        signal_ids = [span.time_series_id]
        unit = "steps"
    else:
        raise TimeFFormatError("unsupported task span")
    return SpanDetail(
        kind="point" if span.is_point else "interval",
        unit=unit,
        start=start,
        stop=None if span.is_point else span.exclusive_end,
        signal_ids=signal_ids,
    )


def project_annotation(annotation: Annotation) -> AnnotationDetail:
    """Return annotation facts without resolving its owner hierarchy."""
    return AnnotationDetail(
        id=annotation.id,
        occurrence_id=annotation.occurrence_id,
        key=annotation.key,
        value_type=value_type_of(annotation.value),
        value=annotation.value,
        span=project_span(annotation.span),
        unit=None if annotation.unit is None else str(annotation.unit),
        description=annotation.description,
        source=annotation.source,
        confidence=annotation.confidence,
        metadata=annotation.metadata,
        occurrence_metadata=annotation.occurrence_metadata,
    )


def project_target(target: object) -> ScalarTarget | ReferenceTarget | SpanTarget:
    """Return a typed scalar, span, or public object reference.

    Raises:
        TimeFFormatError: If the hydrated task contains an unsupported target.
    """
    if isinstance(target, (Record, Signal)):
        return ReferenceTarget(kind="record" if isinstance(target, Record) else "signal", id=target.id)
    if isinstance(target, Span):
        return SpanTarget(value=project_span(target))
    if isinstance(target, bool):
        return ScalarTarget(kind="bool", value=target)
    if isinstance(target, int):
        return ScalarTarget(kind="int", value=target)
    if isinstance(target, float):
        return ScalarTarget(kind="float", value=target)
    if isinstance(target, str):
        return ScalarTarget(kind="str", value=target)

    raise TimeFFormatError("unsupported task target")


def project_task(task: Task) -> TaskDetail:
    """Return all task facts while retaining only IDs for referenced objects."""
    configuration: JsonMapping = {}
    if isinstance(task, ClassificationTask):
        configuration["target_schema"] = task.target_schema
    if isinstance(task, ScalarPredictionTask):
        configuration.update(unit=None if task.unit is None else str(task.unit), target_name=task.target_name)
    if isinstance(task, TemporalLocalizationTask):
        configuration["mode"] = str(task.mode)
    if isinstance(task, TSCorrespondenceTask):
        configuration["candidate_record_ids"] = [record.id for record in task.candidate_records]
    return TaskDetail(
        id=task.id,
        task_type=task.task_type,
        split=task.split,
        prompt=task.prompt,
        rationale=task.rationale,
        input_record_ids=[record.id for record in task.inputs],
        parent_task_ids=[parent.id for parent in task.from_tasks],
        input_modalities=sorted(str(value) for value in task.resolved_input_modalities),
        scope=project_span(task.scope),
        targets=None if task.targets is None else [project_target(target) for target in task.targets],
        configuration=configuration,
        metadata=task.metadata,
        annotations=[project_annotation(item) for item in task.annotations],
        input_annotations=[project_annotation(item) for item in task.input_annotations],
        target_annotations=[project_annotation(item) for item in task.target_annotations],
    )
