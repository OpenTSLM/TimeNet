"""Declarative modelling tasks over records and signals."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum, unique
from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar, Self, TypeAlias, TypeGuard, TypeVar, final

import pint

from timenet.errors import TimeFValidationError
from timenet.json import JsonMapping, validate_json_mapping
from timenet.types._fields import ValueField
from timenet.types.annotations import Annotation, SupportsAnnotate, value_type_of
from timenet.types.ids import EntityId, assigned_id
from timenet.types.modalities import InputModality
from timenet.types.spans import Span, StepSpan, TimeSpan
from timenet.types.splits import Split, parse_split
from timenet.types.text import InlineText, TextHandle, optional_text
from timenet.types.units import normalize_unit


if TYPE_CHECKING:
    # Runtime imports would create a cycle: dataset objects import these task types too.
    from timenet.dataset.record import Record
    from timenet.dataset.time_series import Signal

    TargetItem: TypeAlias = TextHandle | int | float | bool | Record | Signal | Span
    TargetInput: TypeAlias = str | TargetItem
    # A target as read_value returns it: text handles read to strings.
    TargetValue: TypeAlias = str | int | float | bool | Record | Signal | Span
else:
    TargetItem = object
    TargetInput = object
    TargetValue = object

_Item = TypeVar("_Item")


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
    task_id: EntityId | None,
    prompt: str | TextHandle | None,
    has_record_inputs: bool,
) -> frozenset[InputModality]:
    """Normalize and validate the input kinds declared by a task.

    Args:
        modalities: The task's declared input kinds.
        task_id: The task id used in validation errors, or ``None`` before the task has one.
        prompt: The task's optional text prompt.
        has_record_inputs: Whether the task supplies input Records.

    Returns:
        The validated input kinds as a frozen set.

    Raises:
        TimeFValidationError: If the declaration is missing, invalid, or inconsistent with the inputs.
    """
    if modalities is None or isinstance(modalities, (str, bytes)):
        raise TimeFValidationError("input_modalities must be a non-empty collection of InputModality values")
    task = "task" if task_id is None else f"task {task_id!r}"
    try:
        return _validated_input_modalities(tuple(modalities), bool(prompt), has_record_inputs)
    except TimeFValidationError as exc:
        raise TimeFValidationError(f"{task} {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise TimeFValidationError(f"invalid input_modalities for {task}") from exc


@lru_cache(maxsize=256)
def _validated_input_modalities(
    modalities: tuple[InputModality, ...], has_prompt: bool, has_record_inputs: bool
) -> frozenset[InputModality]:
    """Cache validation of the small immutable modality vocabulary.

    The messages name no task, so the caller prefixes them.

    Returns:
        The validated declaration.

    Raises:
        TimeFValidationError: If the declaration disagrees with the input shape.
    """
    declared = frozenset(InputModality(value) for value in modalities)
    if not declared:
        raise TimeFValidationError("must declare an input modality")
    if has_prompt and InputModality.TEXT not in declared:
        raise TimeFValidationError("has a prompt but does not declare text input")
    if InputModality.NO_INPUT in declared:
        if has_record_inputs or declared - {InputModality.NO_INPUT, InputModality.TEXT}:
            raise TimeFValidationError("cannot combine no_input with record or media inputs")
    elif not has_record_inputs and declared <= {InputModality.TEXT}:
        raise TimeFValidationError("needs no_input when it has only a text prompt")
    return declared


@unique
class LocalizationMode(StrEnum):
    """Whether a localization target is sparse or exhaustive."""

    SPARSE = "sparse"
    EXHAUSTIVE = "exhaustive"


def _target_values(values: Iterable[TargetInput] | None) -> tuple[TargetItem, ...] | None:
    """Wrap string targets while preserving native scalar and object targets.

    Returns:
        Ordered targets with text handles, or ``None`` for annotation-based answers.

    Raises:
        TimeFValidationError: If ``values`` is a bare string, which would split into characters.
    """
    if values is None:
        return None
    if isinstance(values, str):
        raise TimeFValidationError("targets take a sequence of items; wrap a single string as (value,)")
    return tuple(InlineText(value) if isinstance(value, str) else value for value in values)


@dataclass(kw_only=True)
class Task(SupportsAnnotate):
    """One modelling problem with ordered inputs and ordered output items.

    Targets deliberately accept a mixture of supported values. Concrete task classes describe the
    modelling operation, but they do not impose a second hierarchy of target wrapper classes. None
    means that the answer is supplied through target annotations. An empty tuple is an explicit
    empty answer.

    Only a concrete task class, one that sets ``task_type``, can be built; ``Task`` itself cannot.

    The task classes form a closed set. TimeF stores a task's ``task_type`` tag, and the reader maps
    it back to a built-in class, so only this module defines subclasses, and each concrete class is
    final. To reuse a prompt, write a function that builds a built-in task::

        def age_question(record: Record, age: int) -> AnswerTask:
            return AnswerTask(inputs=(record,), prompt="How old is the subject?", targets=(str(age),))
    """

    task_type: ClassVar[TaskType]
    """The concrete task class's stable storage tag."""

    inputs: tuple["Record", ...] = field(default=(), compare=False)
    """Records supplied to the model, in semantic order."""
    targets: ValueField[Iterable[TargetInput] | None, tuple[TargetItem, ...] | None] = field(
        default=ValueField(_target_values, None), compare=False
    )
    """Ordered native values and stored objects that form the expected output."""
    prompt: ValueField[str | TextHandle | None, TextHandle | None] = field(default=ValueField(optional_text, None))
    """Call ``read()`` for the prompt; ``None`` identifies an unprompted task."""
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
    rationale: ValueField[str | TextHandle | None, TextHandle | None] = field(default=ValueField(optional_text, None))
    """Call ``read()`` for the explanation; ``None`` means no rationale is present."""
    from_tasks: tuple["Task", ...] = field(default=(), compare=False)
    """Tasks from which this task was derived."""
    annotations: tuple[Annotation, ...] = ()
    """Annotations describing this task itself."""
    metadata: JsonMapping = field(default_factory=dict)
    """Optional JSON-compatible task metadata."""
    _id: EntityId | None = field(default=None, init=False, repr=False, compare=False)

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Accept subclasses from this module only, since stored task types form a closed set.

        Raises:
            TypeError: If the subclass is defined outside ``timenet.types.tasks``.
        """
        if cls.__module__ != Task.__module__:
            raise TypeError(
                f"{cls.__name__} can't subclass a task class: TimeF stores a closed set of task "
                "types, and the reader maps each one back to a built-in class. To reuse a prompt or "
                "other settings, write a builder function that returns a built-in task"
            )
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        """Normalize caller-provided sequences and validate target item types and the split.

        Raises:
            TimeFValidationError: If the class sets no ``task_type``, one target item has an
                unsupported type, or the split is unknown.
        """
        if not isinstance(getattr(type(self), "task_type", None), TaskType):
            raise TimeFValidationError(
                f"{type(self).__name__} sets no task_type; build a concrete task class such as AnswerTask"
            )
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
    def id(self) -> EntityId:
        """The task's id, unique among the tasks of its dataset.

        :meth:`~timenet.dataset.TimeFDataset.add_task` sets it. The reader sets it when it reads the task back.

        Raises:
            TimeFValidationError: If the task is not in a dataset yet.
        """  # noqa: DOC502 (raised by assigned_id)
        return assigned_id(self._id, type(self).__name__)

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
            modalities, task_id=self._id, prompt=self.prompt, has_record_inputs=bool(self.inputs)
        )

    def _infer_input_modalities(self) -> frozenset[InputModality]:
        scoped: tuple[Signal, ...] | None = None
        if isinstance(self.scope, StepSpan):
            scoped = (self.scope.time_series,)
        elif isinstance(self.scope, TimeSpan):
            scoped = self.scope.time_series
        signals = {id(signal): signal for record in self.inputs for signal in record.walk_signals()}
        unknown = [signal.name for signal in scoped or () if id(signal) not in signals]
        if unknown:
            raise TimeFValidationError(f"task scope names Signals that are not in its inputs: {unknown}")
        kinds = {signal.spec.modality for signal in (signals.values() if scoped is None else scoped)}
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

    def named_records(self) -> tuple["Record", ...]:
        """Return the Records that the task names, each once: its inputs, then its Record targets.

        A task class with candidate Records returns them too, after the inputs.
        """
        return _each_once((*self.inputs, *(target for target in self.targets or () if _is_record(target))))

    def referenced_signals(self) -> tuple["Signal", ...]:
        """Return the Signals that the task references, each once.

        They are its Signal targets, then the Signals that its scope and its span targets name, in the
        order of :meth:`spans`.
        """
        signals: list[Signal] = [target for target in self.targets or () if _is_signal(target)]
        for span in self.spans():
            if isinstance(span, StepSpan):
                signals.append(span.time_series)
            elif isinstance(span, TimeSpan):
                signals.extend(span.time_series or ())
        return _each_once(signals)

    def with_references(self, records: Mapping[EntityId, "Record"], signals: Mapping[EntityId, "Signal"]) -> Self:
        """Return a copy of the task that references the given Records and Signals, by ID.

        The copy has the class of the task. Each input, candidate Record, Record target, and Signal
        target of the copy is the object of the same ID in ``records`` or ``signals``. The scope and
        the span targets of the copy name the Signals of the same IDs in ``signals``. Each span keeps
        its class. The copy has no ID.

        Every other field keeps its value: the annotations of the task, the annotations that it
        references, ``from_tasks``, the text, and the metadata. A caller that needs other values for
        them sets them on the copy.

        Args:
            records: The Records for the copy, by ID.
            signals: The Signals for the copy, by ID.

        Returns:
            The copy.

        Raises:
            TimeFValidationError: If a Record or Signal that the task references has no ID, or if
                ``records`` or ``signals`` has no object for its ID.
        """  # noqa: DOC502 (raised by _by_id and assigned_id)
        return replace(self, **self._references_by_id(records, signals))

    def _references_by_id(
        self, records: Mapping[EntityId, "Record"], signals: Mapping[EntityId, "Signal"]
    ) -> dict[str, object]:
        """Return the value for the copy of each field that references Records or Signals.

        :meth:`with_references` builds the copy from these values. A task class with more such fields
        adds them.

        Returns:
            The values, by field name.
        """
        targets = (
            None if self.targets is None else tuple(_target_by_id(item, records, signals) for item in self.targets)
        )
        return {
            "inputs": tuple(_by_id(records, record) for record in self.inputs),
            "targets": targets,
            "scope": None if self.scope is None else _span_by_id(self.scope, signals),
        }

    def check_against_scope(self) -> None:
        """Validate configuration that depends on the finalized scope."""


@final
@dataclass(kw_only=True)
class ClassificationTask(Task):
    """Classify the inputs, optionally using a named label vocabulary."""

    task_type: ClassVar[TaskType] = TaskType.CLASSIFICATION
    target_schema: str | None = None
    """Optional name of the label vocabulary."""


@final
@dataclass(kw_only=True)
class AnswerTask(Task):
    """Answer a question or caption the inputs."""

    task_type: ClassVar[TaskType] = TaskType.ANSWER


@final
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


@final
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


@final
@dataclass(kw_only=True)
class ForecastingTask(Task):
    """Predict future values from the supplied inputs."""

    task_type: ClassVar[TaskType] = TaskType.FORECASTING


@final
@dataclass(kw_only=True)
class TSEditingTask(Task):
    """Transform the input recording into the output described by targets."""

    task_type: ClassVar[TaskType] = TaskType.TS_EDITING


@final
@dataclass(kw_only=True)
class TSGenerationTask(Task):
    """Generate output described by the prompt, optionally without record inputs."""

    task_type: ClassVar[TaskType] = TaskType.TS_GENERATION


@final
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
        candidates = {id(record) for record in self.candidate_records}
        outside = [
            position
            for position, target in enumerate(self.targets or ())
            if _is_record(target) and id(target) not in candidates
        ]
        if outside:
            raise TimeFValidationError(
                f"TSCorrespondenceTask Record targets at positions {outside} are not in candidate_records"
            )

    def named_records(self) -> tuple["Record", ...]:
        """Return the Records that the task names, each once.

        They are its inputs, its candidate Records, then its Record targets.
        """
        targets = (target for target in self.targets or () if _is_record(target))
        return _each_once((*self.inputs, *self.candidate_records, *targets))

    def _references_by_id(
        self, records: Mapping[EntityId, "Record"], signals: Mapping[EntityId, "Signal"]
    ) -> dict[str, object]:
        """Return the values of :meth:`Task._references_by_id`, and the candidate Records for the copy.

        Returns:
            The values, by field name.
        """
        candidates = tuple(_by_id(records, record) for record in self.candidate_records)
        return {**super()._references_by_id(records, signals), "candidate_records": candidates}


def _is_record(value: object) -> TypeGuard["Record"]:
    """Return whether a target is a Record without importing Record at module load time."""
    from timenet.dataset.record import Record  # noqa: PLC0415

    return isinstance(value, Record)


def _is_signal(value: object) -> TypeGuard["Signal"]:
    """Return whether a target is a Signal without importing Signal at module load time."""
    from timenet.dataset.time_series import Signal  # noqa: PLC0415

    return isinstance(value, Signal)


def _each_once(items: Iterable[_Item]) -> tuple[_Item, ...]:
    """Return the items in their order, without a second occurrence of one object."""
    return tuple({id(item): item for item in items}.values())


def _by_id(objects: Mapping[EntityId, _Item], reference: "Record | Signal") -> _Item:
    """Return the object in ``objects`` with the ID of a Record or Signal that a task references.

    Raises:
        TimeFValidationError: If ``objects`` has no object for the ID.
    """
    found = objects.get(reference.id)
    if found is None:
        raise TimeFValidationError(
            f"with_references() got no {type(reference).__name__} for ID {reference.id}, which the task references"
        )
    return found


def _span_by_id(span: Span, signals: Mapping[EntityId, "Signal"]) -> Span:
    """Return a span of the class of ``span`` that names the Signals of the same IDs in ``signals``.

    Returns:
        The new span, or ``span`` itself if it names no Signal.
    """
    if isinstance(span, StepSpan):
        return replace(span, time_series=_by_id(signals, span.time_series))
    if isinstance(span, TimeSpan) and span.time_series is not None:
        return replace(span, time_series=tuple(_by_id(signals, signal) for signal in span.time_series))
    return span


def _target_by_id(
    target: TargetItem, records: Mapping[EntityId, "Record"], signals: Mapping[EntityId, "Signal"]
) -> TargetItem:
    """Return the target for a copy: the Record, Signal, or span of the same IDs, or the same value.

    Returns:
        The target.
    """
    if _is_record(target):
        return _by_id(records, target)
    if _is_signal(target):
        return _by_id(signals, target)
    if isinstance(target, Span):
        return _span_by_id(target, signals)
    return target


def _is_target_item(value: object) -> bool:
    """Return whether value can form one item in a task's expected answer."""
    from timenet.dataset.record import Record  # noqa: PLC0415
    from timenet.dataset.time_series import Signal  # noqa: PLC0415

    return isinstance(value, (TextHandle, int, float, bool, Record, Signal, Span))


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
        task_type = cls.task_type
        if task_type in registry:
            raise TimeFValidationError(
                f"both {registry[task_type].__name__} and {cls.__name__} claim task_type {task_type!r}"
            )
        registry[task_type] = cls
    return registry


TASKS: dict[TaskType, type[Task]] = _build_task_registry()
"""Concrete task classes keyed by their stable storage tag."""
