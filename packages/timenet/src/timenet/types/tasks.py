"""Declarative modelling tasks over records and signals."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import TYPE_CHECKING, ClassVar, TypeAlias, TypeGuard, cast

import pint

from timenet._json import JsonMapping, validate_json_mapping
from timenet.errors import TimeFValidationError
from timenet.types.annotations import Annotation, SupportsAnnotate, value_type_of
from timenet.types.ids import new_id
from timenet.types.modalities import InputModality
from timenet.types.spans import Span, StepSpan, TimeSpan
from timenet.types.splits import Split, parse_split
from timenet.types.units import normalize_unit


if TYPE_CHECKING:
    # Runtime imports would create a cycle: dataset objects import these task types too.
    from timenet.dataset.record import Record
    from timenet.dataset.time_series import Signal

    TargetItem: TypeAlias = str | int | float | bool | Record | Signal | Span
else:
    TargetItem = object


@unique
class TaskType(StrEnum):
    """Stable storage tags for the built-in task classes."""

    CLASSIFICATION = "classification"
    ANSWER = "answer"
    SCALAR_PREDICTION = "scalar_prediction"
    TEMPORAL_LOCALIZATION = "temporal_localization"
    FORECASTING = "forecasting"
    TS_EDITING = "ts_editing"
    TS_GENERATION = "ts_generation"
    TS_CORRESPONDENCE = "ts_correspondence"


def validate_input_modalities(
    modalities: Iterable[InputModality] | None,
    *,
    task_id: str,
    prompt: str | None,
    has_record_inputs: bool,
) -> frozenset[InputModality]:
    """Normalize and validate the input kinds declared by a task.

    Args:
        modalities: The task's declared input kinds.
        task_id: The task identifier used in validation errors.
        prompt: The task's optional text prompt.
        has_record_inputs: Whether the task supplies input Records.

    Returns:
        The validated input kinds as a frozen set.

    Raises:
        TimeFValidationError: If the declaration is missing, invalid, or inconsistent with the inputs.
    """
    if modalities is None or isinstance(modalities, (str, bytes)):
        raise TimeFValidationError("input_modalities must be a non-empty collection of InputModality values")
    try:
        declared = frozenset(InputModality(value) for value in modalities)
    except (TypeError, ValueError) as exc:
        raise TimeFValidationError(f"invalid input_modalities for task {task_id!r}") from exc
    if not declared:
        raise TimeFValidationError(f"task {task_id!r} must declare an input modality")
    if prompt and InputModality.TEXT not in declared:
        raise TimeFValidationError(f"task {task_id!r} has a prompt but does not declare text input")
    if InputModality.NO_INPUT in declared:
        if has_record_inputs or declared - {InputModality.NO_INPUT, InputModality.TEXT}:
            raise TimeFValidationError(f"task {task_id!r} cannot combine no_input with record or media inputs")
    elif not has_record_inputs and declared <= {InputModality.TEXT}:
        raise TimeFValidationError(f"task {task_id!r} needs no_input when it has only a text prompt")
    return declared


@unique
class LocalizationMode(StrEnum):
    """Whether a localization target is sparse or exhaustive."""

    SPARSE = "sparse"
    EXHAUSTIVE = "exhaustive"


@dataclass(kw_only=True)
class Task(SupportsAnnotate, ABC):
    """One modelling problem with ordered inputs and ordered output items.

    Targets deliberately accept a mixture of supported values. Concrete task classes describe the
    modelling operation, but they do not impose a second hierarchy of target wrapper classes. None
    means that the answer is supplied through target annotations. An empty tuple is an explicit
    empty answer.
    """

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """Return the concrete task's stable storage tag."""

    id: str = field(default_factory=new_id)
    """Unique public task identifier."""
    inputs: tuple["Record", ...] = field(default=(), compare=False)
    """Records supplied to the model, in semantic order."""
    targets: tuple[TargetItem, ...] | None = field(default=None, compare=False)
    """Ordered native values and stored objects that form the expected output."""
    prompt: str | None = None
    """What the model is asked, or None for an unprompted task."""
    split: Split | None = None
    """The partition the task belongs to: train, validation, or test. None when the dataset does not
    partition its tasks."""
    input_modalities: frozenset[InputModality] | None = None
    """Optional override. None infers the input kinds when resolved or written."""
    scope: Span | None = None
    """Optional region of the inputs to which the task applies."""
    input_annotations: tuple[Annotation, ...] = field(default=(), compare=False)
    """Existing annotation occurrences supplied to the model as context."""
    target_annotations: tuple[Annotation, ...] = field(default=(), compare=False)
    """Existing annotation occurrences that form the expected output."""
    rationale: str | None = None
    """Optional explanation associated with the answer."""
    from_tasks: tuple["Task", ...] = field(default=(), compare=False)
    """Tasks from which this task was derived."""
    annotations: tuple[Annotation, ...] = ()
    """Annotations describing this task itself."""
    metadata: JsonMapping = field(default_factory=dict)
    """Optional JSON-compatible task metadata."""

    def __post_init__(self) -> None:
        """Normalize caller-provided sequences and validate target item types and the split.

        Raises:
            TimeFValidationError: If one target item has an unsupported type, or the split is unknown.
        """
        self.inputs = tuple(self.inputs)
        self.targets = None if self.targets is None else tuple(self.targets)
        self.input_annotations = tuple(self.input_annotations)
        self.target_annotations = tuple(self.target_annotations)
        self.from_tasks = tuple(self.from_tasks)
        self.annotations = tuple(self.annotations)
        self.metadata = validate_json_mapping(self.metadata)
        if self.split is not None:
            self.split = parse_split(self.split)
        if self.input_modalities is not None:
            self.input_modalities = self.resolved_input_modalities
        for target in self.targets or ():
            if not _is_target_item(target):
                raise TimeFValidationError(f"{type(self).__name__} target has unsupported type {type(target).__name__}")

    @property
    def resolved_input_modalities(self) -> frozenset[InputModality]:
        """Return the validated override, or infer kinds from the current inputs.

        Inference reads Signal specs within the scope, the prompt, and text input annotations.
        It does not load values or inspect targets. NO_INPUT means there are no input Records.
        The result is not cached: connectors can assign inputs after construction.
        """
        modalities = self.input_modalities
        if modalities is None:
            modalities = self._infer_input_modalities()
        return validate_input_modalities(
            modalities, task_id=self.id, prompt=self.prompt, has_record_inputs=bool(self.inputs)
        )

    def _infer_input_modalities(self) -> frozenset[InputModality]:
        signal_ids = None
        if isinstance(self.scope, StepSpan):
            signal_ids = {self.scope.time_series_id}
        elif isinstance(self.scope, TimeSpan) and self.scope.time_series_ids is not None:
            signal_ids = set(self.scope.time_series_ids)
        signals = {signal.id: signal for record in self.inputs for signal in record.walk_signals()}
        if signal_ids is not None and signal_ids - signals.keys():
            raise TimeFValidationError(
                f"task {self.id!r} scope names unknown input Signals: {signal_ids - signals.keys()}"
            )
        kinds = {
            signal.spec.modality
            for signal_id, signal in signals.items()
            if signal_ids is None or signal_id in signal_ids
        }
        if self.prompt or any(
            value_type_of(annotation.value) in {"str", "list"} for annotation in self.input_annotations
        ):
            kinds.add(InputModality.TEXT)
        if not self.inputs:
            kinds.add(InputModality.NO_INPUT)
        return frozenset(kinds)

    def annotate(self, annotation: Annotation) -> Annotation:
        """Attach one annotation occurrence and return it.

        Returns:
            The attached occurrence.
        """
        attached = annotation._new_occurrence()
        self.annotations = (*self.annotations, attached)
        return attached

    def spans(self) -> tuple[Span, ...]:
        """Return the task scope followed by every span-valued target."""
        spans = [] if self.scope is None else [self.scope]
        spans.extend(target for target in self.targets or () if isinstance(target, Span))
        return tuple(spans)

    def check_against_scope(self) -> None:
        """Validate configuration that depends on the finalized scope."""


@dataclass(kw_only=True)
class ClassificationTask(Task):
    """Classify the inputs, optionally using a named label vocabulary."""

    task_type: ClassVar[TaskType] = TaskType.CLASSIFICATION
    target_schema: str | None = None
    """Optional name of the label vocabulary."""


@dataclass(kw_only=True)
class AnswerTask(Task):
    """Answer a question or caption the inputs."""

    task_type: ClassVar[TaskType] = TaskType.ANSWER


@dataclass(kw_only=True)
class ScalarPredictionTask(Task):
    """Predict one or more quantities described by the task configuration."""

    task_type: ClassVar[TaskType] = TaskType.SCALAR_PREDICTION
    unit: str | pint.Unit | None = None
    """Optional physical unit shared by numerical target items."""
    target_name: str | None = None
    """Optional name of the predicted quantity."""

    def __post_init__(self) -> None:
        """Normalize the common fields and physical unit."""
        super().__post_init__()
        self.unit = normalize_unit(self.unit)


@dataclass(kw_only=True)
class TemporalLocalizationTask(Task):
    """Locate events or regions in the inputs."""

    task_type: ClassVar[TaskType] = TaskType.TEMPORAL_LOCALIZATION
    mode: LocalizationMode = LocalizationMode.SPARSE
    """Whether the localization is sparse or exhaustive."""

    def __post_init__(self) -> None:
        """Normalize common fields and the localization mode.

        Raises:
            TimeFValidationError: If the localization mode is unknown.
        """
        super().__post_init__()
        try:
            self.mode = LocalizationMode(self.mode)
        except ValueError as exc:
            raise TimeFValidationError(
                f"unknown localization mode {self.mode!r}. Expected one of {[mode.value for mode in LocalizationMode]}"
            ) from exc


@dataclass(kw_only=True)
class ForecastingTask(Task):
    """Predict future values from the supplied inputs."""

    task_type: ClassVar[TaskType] = TaskType.FORECASTING


@dataclass(kw_only=True)
class TSEditingTask(Task):
    """Transform the input recording into the output described by targets."""

    task_type: ClassVar[TaskType] = TaskType.TS_EDITING


@dataclass(kw_only=True)
class TSGenerationTask(Task):
    """Generate output described by the prompt, optionally without record inputs."""

    task_type: ClassVar[TaskType] = TaskType.TS_GENERATION


@dataclass(kw_only=True)
class TSCorrespondenceTask(Task):
    """Relate input recordings to an optional candidate pool."""

    task_type: ClassVar[TaskType] = TaskType.TS_CORRESPONDENCE
    candidate_records: tuple["Record", ...] = field(default=(), compare=False)
    """Optional candidate pool. An empty tuple means the dataset-wide pool."""

    def __post_init__(self) -> None:
        """Normalize common fields and validate Record targets against the candidate pool.

        Raises:
            TimeFValidationError: If a Record target is outside the declared candidate pool.
        """
        super().__post_init__()
        self.candidate_records = tuple(self.candidate_records)
        if not self.candidate_records:
            return
        candidate_ids = {record.id for record in self.candidate_records}
        outside = [target.id for target in self.targets or () if _is_record(target) and target.id not in candidate_ids]
        if outside:
            raise TimeFValidationError(f"TSCorrespondenceTask Record targets {outside} are not in candidate_records")


def _is_record(value: object) -> TypeGuard["Record"]:
    """Return whether a target is a Record without importing Record at module load time."""
    from timenet.dataset.record import Record  # noqa: PLC0415

    return isinstance(value, Record)


def _is_target_item(value: object) -> bool:
    """Return whether value is supported in the public mixed target collection."""
    from timenet.dataset.record import Record  # noqa: PLC0415
    from timenet.dataset.time_series import Signal  # noqa: PLC0415

    return isinstance(value, (str, int, float, bool, Record, Signal, Span))


def _concrete_task_classes() -> list[type[Task]]:
    """Collect every concrete task class across the hierarchy.

    Returns:
        The discovered concrete task classes.
    """
    concrete: list[type[Task]] = []
    stack = list(Task.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if "task_type" in vars(cls):
            concrete.append(cls)
    return concrete


def _build_task_registry(classes: Iterable[type[Task]] | None = None) -> dict[TaskType, type[Task]]:
    """Map each concrete task's storage tag to its class.

    Returns:
        Concrete task classes keyed by their stable task type.

    Raises:
        TimeFValidationError: If two classes claim the same task type.
    """
    registry: dict[TaskType, type[Task]] = {}
    for cls in classes if classes is not None else _concrete_task_classes():
        task_type = cast("TaskType", cls.task_type)
        if task_type in registry:
            raise TimeFValidationError(
                f"both {registry[task_type].__name__} and {cls.__name__} claim task_type {task_type!r}"
            )
        registry[task_type] = cls
    return registry


TASKS: dict[TaskType, type[Task]] = _build_task_registry()
"""Concrete task classes keyed by their stable storage tag."""
