"""Named wire-safe results shared by reader projections and generated clients."""

from typing import Annotated, Literal, NotRequired

from pydantic import Field
from typing_extensions import TypedDict


class Page(TypedDict):
    """Common continuation fields."""

    api_version: int
    next_cursor: str | None


class RecordItem(TypedDict):
    """One record's public identity and temporal envelope."""

    record_id: str
    start_us: str | None
    end_us: str | None


class RecordPage(Page):
    """A bounded record page."""

    items: list[RecordItem]


class SourceItem(TypedDict):
    """One source in the record hierarchy."""

    source_id: str
    parent_id: str | None
    name: str


class SourcePage(Page):
    """A bounded source page."""

    items: list[SourceItem]


class SignalItem(TypedDict):
    """One signal's inexpensive structural facts."""

    signal_id: str
    name: str
    axis_kind: Literal["regular", "irregular", "ordinal"]
    n_values: int
    dtype: str
    unit: str | None
    value_shape: list[int]


class SignalPage(Page):
    """A bounded signal page."""

    items: list[SignalItem]


class AnnotationItem(TypedDict):
    """A bounded annotation value and its public owner."""

    occurrence_id: str
    object_type: Literal["Dataset", "Record", "Source", "Signal", "Task"]
    object_id: str
    name: str
    value_kind: str | None
    value: str | None
    value_truncated: bool
    span_type: Literal["static", "point", "interval"]
    start_us: str | None
    end_us: str | None
    field: NotRequired[str]
    dataset_id: NotRequired[str]
    version: NotRequired[str]
    content_id: NotRequired[str]


class AnnotationPage(Page):
    """A bounded annotation page."""

    items: list[AnnotationItem]
    more: NotRequired[bool]


class TargetPreview(TypedDict):
    """Preview flags and bounded signal references shared by target variants."""

    value_truncated: bool
    signal_ids: list[str]
    signal_ids_truncated: bool


class ScalarTarget(TargetPreview):
    """Required facts for a scalar target."""

    kind: Literal["text", "integer", "float"]
    value: str
    record_id: None
    signal_id: None
    start: None
    end: None


class BooleanTarget(TargetPreview):
    """Required facts for a boolean target."""

    kind: Literal["boolean"]
    value: bool
    record_id: None
    signal_id: None
    start: None
    end: None


class RecordTarget(TargetPreview):
    """Required facts for a record target."""

    kind: Literal["record"]
    value: None
    record_id: str
    signal_id: None
    start: None
    end: None


class SignalTarget(TargetPreview):
    """Required facts for a signal target."""

    kind: Literal["signal"]
    value: None
    record_id: None
    signal_id: str
    start: None
    end: None


class PointTarget(TargetPreview):
    """Required facts for a point target."""

    kind: Literal["time_point", "step_point"]
    value: None
    record_id: None
    signal_id: None
    start: str
    end: None


class IntervalTarget(TargetPreview):
    """Required facts for a interval target."""

    kind: Literal["time_interval", "step_interval"]
    value: None
    record_id: None
    signal_id: None
    start: str
    end: str


TargetItem = Annotated[
    ScalarTarget | BooleanTarget | RecordTarget | SignalTarget | PointTarget | IntervalTarget,
    Field(discriminator="kind"),
]


class TaskItem(TypedDict):
    """One task's searchable fields."""

    task_id: str
    task_type: str
    prompt: str | None
    split: str | None


class TaskPage(Page):
    """A bounded task page."""

    items: list[TaskItem]


class ScopeSignals(TypedDict):
    """The current bounded page of a scope's signal references."""

    signal_ids: list[str]
    more_signal_ids: bool


class NoScope(ScopeSignals):
    """A task without an explicit scope."""

    kind: Literal[None]
    start: None
    end: None


class PointScope(ScopeSignals):
    """A scope at one temporal or step coordinate."""

    kind: Literal["time_point", "step_point"]
    start: str
    end: None


class IntervalScope(ScopeSignals):
    """A scope with both exclusive interval bounds."""

    kind: Literal["time_interval", "step_interval"]
    start: str
    end: str


TaskScope = Annotated[NoScope | PointScope | IntervalScope, Field(discriminator="kind")]


class TaskConfiguration(TypedDict):
    """Optional configuration fields shared across task kinds."""

    target_schema: str | None
    unit: str | None
    target_name: str | None
    mode: str | None


class TaskDetail(TaskItem):
    """Bounded task facts and independently truncated relationships."""

    api_version: int
    rationale: str | None
    input_modalities: list[str]
    metadata: str
    metadata_truncated: bool
    scope: TaskScope
    configuration: TaskConfiguration
    has_inline_targets: bool
    prompt_truncated: bool
    rationale_truncated: bool
    input_record_ids: list[str]
    targets: list[TargetItem]
    more_targets: bool
    annotation_refs: list[AnnotationItem]
    more_annotation_refs: bool
    parent_task_ids: list[str]
    candidate_record_ids: list[str]
    next_after: int | None


class SignalSpec(TypedDict):
    """Signal fields needed for component selection."""

    name: str
    dtype: str
    value_shape: list[int]


class SignalDetail(TypedDict):
    """Signal metadata preview."""

    signal_id: str
    name: str
    axis_kind: Literal["regular", "irregular", "ordinal"]
    n_values: int
    spec: SignalSpec
    metadata: str
    metadata_truncated: bool


class TimeSpan(TypedDict):
    """Exact half-open temporal bounds in microseconds."""

    start_us: str
    end_us: str


class RecordDetail(TypedDict):
    """Record clock, provenance and metadata preview."""

    record_id: str
    dataset_id: str
    version: str
    start_time_us: str | None
    span: TimeSpan | None
    span_provenance: str
    metadata: str
    metadata_truncated: bool
    signal: NotRequired[SignalDetail]
    source_path: NotRequired[list[str]]


class OwnerRecord(TypedDict):
    """The record containing a signal or source."""

    api_version: int
    object_type: str
    record_id: str
    dataset_id: str


class Identity(TypedDict):
    """Human-facing dataset identity."""

    dataset_id: str
    version: str
    name: str
    description: str
    license: str
    access: str
    domains: list[str]
    tags: list[str]
    source_url: str | None


class TensorSpec(TypedDict):
    """A tensor component shape."""

    spec_type: str
    value_shape: list[int]
    dimension_names: list[str]


class SchemaOverview(TypedDict):
    """Available task and signal families."""

    spec_types: list[str]
    task_types: list[str]
    tensor_specs: list[TensorSpec]


class FormatOverview(TypedDict):
    """Physical format identifiers."""

    timef_version: int
    values_backend: str


class Overview(TypedDict):
    """Manifest-backed dataset overview."""

    api_version: int
    identity: Identity
    format: FormatOverview
    counts: dict[str, str]
    schema: SchemaOverview
    dependencies: list[str]


class WindowSample(TypedDict):
    """One exact scalar display value."""

    index: str
    x: str
    value: str | None
    kind: str
    display: str | None
    original_bytes: str | None
    value_truncated: bool


class StepBounds(TypedDict):
    """Exact half-open step bounds."""

    start: str
    stop: str


class WindowBucket(StepBounds):
    """Coverage of one reduction bucket."""

    coverage: dict[str, str]


class WindowResult(TypedDict):
    """A bounded raw page or reduced numeric window."""

    api_version: int
    record_id: str
    signal_id: str
    signal_name: str
    spec_name: str
    unit: str | None
    dtype: str
    component: list[int]
    axis_kind: Literal["regular", "irregular", "ordinal"]
    mode: Literal["plot", "raw"]
    requested: StepBounds
    scanned_count: str
    coverage: dict[str, str]
    reduced: bool
    buckets: list[WindowBucket]
    next_start: str | None
    items: list[WindowSample]
