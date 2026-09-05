"""Tasks: labeled training targets that reference one or more records.

Every task has the same shape: inputs give one typed answer. The shared frame lives on the
:class:`Task` base. The base holds the records the task is about, an optional ``prompt``, and an
optional ``scope`` that narrows the input to a region. It also holds the annotations given as
context, the answer (``target``, inline or by reference to stored annotations), and an optional
``rationale`` chain of thought. A subclass adds only what makes its answer a different kind of
thing. That is a category, free text, a number, a set of regions, or a produced series. Prompt and
scope live on the base. So a whole-recording category and a category over a given window are the
same task type, with ``scope`` unset or set. A caption is an :class:`AnswerTask` with no prompt.

The class is the type tag (for example, a filter such as ``search(task=ClassificationTask)``). The
instance carries the payload. Task payload shapes are fixed in code, unlike specs and annotations.
So the reader resolves tasks against the built-in :data:`TASKS` registry. It does not rebuild them
from the manifest. Tasks are mutable, so :meth:`~timenet.dataset.TimeFDataset.add_task` can set
``record_ids`` after construction.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import ClassVar

import pint

from timenet.errors import TimeFValidationError
from timenet.types.ids import new_id
from timenet.types.spans import (
    Span,
    StepInterval,
    StepSpan,
    TimeInterval,
    TimePoint,
    TimeSpan,
)
from timenet.types.units import normalize_unit


@unique
class TaskType(StrEnum):
    """Stable type tags for the built-in task classes. They are also the on-disk task partition names."""

    CLASSIFICATION = "classification"
    ANSWER = "answer"
    SCALAR_PREDICTION = "scalar_prediction"
    TEMPORAL_LOCALIZATION = "temporal_localization"
    FORECASTING = "forecasting"
    TS_EDITING = "ts_editing"
    TS_GENERATION = "ts_generation"
    TS_CORRESPONDENCE = "ts_correspondence"


@unique
class LocalizationMode(StrEnum):
    """Whether a localization target covers the whole recording or can leave some time unmarked."""

    SPARSE = "sparse"
    """Only the marked spans are claimed. Unmarked time stays unlabeled (for example, R-peaks)."""
    EXHAUSTIVE = "exhaustive"
    """The spans intend to tile the whole region of interest (for example, sleep staging)."""


@dataclass(frozen=True)
class TaskRefs:
    """Which of a task's payload fields hold references. Each task declares this next to its class.

    The writer, the reader, and the copy-on-write editor must know that
    ``ForecastingTask.target_record_id`` is a record id, and that ``ScalarPredictionTask.target`` is a
    plain number. The declaration on the class keeps that knowledge next to the field. The other place
    for it is a lookup table in the format layer and an ``isinstance`` chain in the editor. Those two
    drift the moment someone adds a task.

    The base fields (``record_ids``, ``scope``, the annotation id tuples) are common to every task. The
    format layer handles them directly. This declaration covers only the type-specific payload.
    """

    record_id_fields: tuple[str, ...] = ()
    """Payload fields that hold a record id or a tuple of them. If one id is lost, the task is not valid."""
    time_series_id_fields: tuple[str, ...] = ()
    """Payload fields that hold a time-series id or a tuple of them, resolved against the task's records."""
    span_fields: tuple[str, ...] = ()
    """Payload fields that hold a :class:`~timenet.types.spans.Span` or a tuple of them."""


@dataclass(kw_only=True)
class Task:
    """Base for all tasks. You do not construct it directly. Each subclass declares ``task_type`` and an answer type.

    It holds everything that is the same across task types. So generic code (training loops, the writer,
    the editor) can read a task without its concrete type.
    """

    task_type: ClassVar[TaskType]
    """The subclass's stable type tag. Deliberately absent here: the base is not a task."""
    refs: ClassVar[TaskRefs] = TaskRefs()
    """Which payload fields hold record ids or spans (see :class:`TaskRefs`)."""
    answer_fields: ClassVar[tuple[str, ...]] = ("target",)
    """The payload fields holding the inline answer. A subclass whose answer is not (only) ``target``
    overrides this, so the answer-exclusivity check stays generic instead of hardcoding ``target``."""
    answer_is_record: ClassVar[bool] = False
    """True when the answer is a produced series, found by a payload record id and not ``target``."""
    target_is_scalar: ClassVar[bool] = False
    """True when ``to_features_and_targets`` can return ``target`` as a scalar."""

    id: str = field(default_factory=new_id)
    """Unique task identifier, a UUIDv7 string by default."""
    record_ids: tuple[str, ...] = ()
    """Ids of the records this task is about. ``add_task`` sets them."""
    prompt: str | None = None
    """What the model is asked, when the task is prompted. ``None`` for an unprompted task."""
    scope: Span | None = None
    """The region of the input the task is about. ``None`` means the whole record."""
    input_annotation_ids: tuple[str, ...] = ()
    """Annotations given to the model as context, not ones it must produce."""
    target: object | None = None
    """The answer, typed by the subclass. ``None`` when the answer is stored by reference, or produced
    as a series (see ``answer_is_record``)."""
    target_annotation_ids: tuple[str, ...] = ()
    """The answer by reference: it is these stored annotations, not an inline copy of them. It is
    exclusive with ``target``. :meth:`~timenet.dataset.TimeFDataset.add_task` enforces that."""
    rationale: str | None = None
    """Chain of thought to train on. Any task can carry one. ``None`` when the source stores none."""
    from_tasks: tuple["Task", ...] = ()
    """Source tasks this one was derived from."""

    @property
    def from_task_ids(self) -> tuple[str, ...]:
        """Return the ids of the tasks this one was derived from.

        Returns:
            The ``id`` of every task in ``from_tasks``.
        """
        return tuple(task.id for task in self.from_tasks)

    def spans(self) -> tuple[Span, ...]:
        """Return every span the task carries: its ``scope`` and any span-valued payload field.

        This lets :meth:`~timenet.dataset.TimeFDataset.add_task` check a task's spans without knowledge
        of its concrete type.

        Returns:
            The task's spans, ``scope`` first.
        """
        found: list[Span] = [] if self.scope is None else [self.scope]
        for name in type(self).refs.span_fields:
            value = getattr(self, name)
            if isinstance(value, Span):
                found.append(value)
            elif value is not None:
                found.extend(value)
        return tuple(found)

    def check_against_scope(self) -> None:
        """Validate payload that depends on the task's finalized ``scope``.

        :meth:`~timenet.dataset.TimeFDataset.add_task` calls this after stamping any ``scope=`` passed
        there, so a scope supplied at registration is in force. The base task has nothing scope-dependent
        to check. :class:`ForecastingTask` overrides it.
        """


@dataclass(kw_only=True)
class ClassificationTask(Task):
    """One categorical label: over the whole record, or over ``scope`` when one is set.

    A whole-recording class ("this ECG shows atrial fibrillation") and a scoped label ("this 30 s
    epoch is sleep stage N2") differ in one way. The only difference is whether ``scope`` narrows
    the input. So both are this type.
    """

    task_type: ClassVar[TaskType] = TaskType.CLASSIFICATION
    target_is_scalar: ClassVar[bool] = True
    target: str | None = None
    """The label."""
    target_schema: str | None = None
    """Name of the label vocabulary the target belongs to."""


@dataclass(kw_only=True)
class AnswerTask(Task):
    """Free-form text out. With no prompt it is a caption. With a ``prompt`` it answers a question.

    ``rationale`` (on the base) carries the chain of thought. So a plain answer and a reasoned answer are
    the same type, with the field unset or set.
    """

    task_type: ClassVar[TaskType] = TaskType.ANSWER
    target_is_scalar: ClassVar[bool] = True
    target: str | None = None
    """The answer, or the caption when the task is unprompted."""


@dataclass(kw_only=True)
class ScalarPredictionTask(Task):
    """One number out, with the quantity it measures and its physical unit kept as data.

    An answer task stores a regression target as a string, which loses its type. A ``float`` with a
    ``unit`` keeps the type. It makes regression metrics, batching, and unit-aware conversion simple.
    """

    task_type: ClassVar[TaskType] = TaskType.SCALAR_PREDICTION
    target_is_scalar: ClassVar[bool] = True
    target: float | None = None
    """The predicted value."""
    unit: str | pint.Unit | None = None
    """Optional physical unit of ``target``. Give a unit string (for example, ``"bpm"``) or a
    :class:`pint.Unit`. The constructor checks it against the shared registry. It stores a
    ``pint.Unit`` as its name."""
    target_name: str | None = None
    """Name of the quantity to predict (for example, ``"mean_heart_rate"``)."""

    def __post_init__(self) -> None:
        """Normalize ``unit`` against the shared registry so it is always a plain string or ``None``."""
        self.unit = normalize_unit(self.unit)


@dataclass(kw_only=True)
class TemporalLocalizationTask(Task):
    """Regions out: find where something happens, from a description of it.

    This is the inverse of a scoped :class:`ClassificationTask`, which gives the region and asks for its
    label. One type covers event detection, segmentation, and change-point detection, because they share
    one target: a tuple of :class:`~timenet.types.spans.TimePoint` or
    :class:`~timenet.types.spans.TimeInterval` spans, on the recording timeline. An optional label from a
    referenced annotation marks each one, and each one is scoped to particular series.
    """

    task_type: ClassVar[TaskType] = TaskType.TEMPORAL_LOCALIZATION
    refs: ClassVar[TaskRefs] = TaskRefs(span_fields=("target",))
    target: tuple[TimePoint | TimeInterval, ...] | None = None
    """The regions to find. ``None`` means the answer is stored by reference in
    ``target_annotation_ids``; ``()`` is a positive answer that nothing was found in scope."""
    mode: LocalizationMode = LocalizationMode.SPARSE
    """Whether the spans must cover the region of interest (see :class:`LocalizationMode`)."""

    def __post_init__(self) -> None:
        """Coerce ``mode`` to the enum and reject a step-framed target.

        Raises:
            TimeFValidationError: If ``mode`` is unknown. If a target span counts in steps, which has
                no place on the recording timeline localization reports against.
        """
        try:
            self.mode = LocalizationMode(self.mode)
        except ValueError as exc:
            raise TimeFValidationError(
                f"unknown localization mode {self.mode!r}. Expected one of {[mode.value for mode in LocalizationMode]}"
            ) from exc
        for span in self.target or ():
            if isinstance(span, StepSpan):
                raise TimeFValidationError(
                    f"TemporalLocalizationTask localizes on the recording timeline, so its targets are "
                    f"TimePoint or TimeInterval spans, but got a step span {span!r}"
                )


@dataclass(kw_only=True)
class ForecastingTask(Task):
    """A series out: continue the context into the future.

    The future is either a whole separate record (``target_record_id``) or a region of the attached
    record (``target_span``). Exactly one of the two is set, never both and never neither. The second
    shape lets one unsplit series carry a horizon. So the dataset can ship the raw recording, not a
    context/target pair. ``target_span`` also needs ``scope``. Without it, the base ``Task.scope``
    default of ``None`` means the whole record, which then covers the region that ``target_span``
    predicts.
    """

    task_type: ClassVar[TaskType] = TaskType.FORECASTING
    refs: ClassVar[TaskRefs] = TaskRefs(
        record_id_fields=("context_record_ids", "target_record_id"),
        span_fields=("target_span",),
    )
    answer_is_record: ClassVar[bool] = True
    context_record_ids: tuple[str, ...] = ()
    """Ids of the records that provide forecasting context."""
    target_record_id: str | None = None
    """Id of the record whose future values the task predicts. ``None`` when ``target_span`` names the
    region to predict in the attached record instead."""
    target_span: TimeInterval | StepInterval | None = None
    """The region to predict, inside the record the task is attached to. It is an interval, not a point
    (the type says so), and it is exclusive with ``target_record_id``. It is in the same frame as
    ``scope``. For a series with a timeline, that is a :class:`~timenet.types.spans.TimeInterval` in
    microseconds. For a series that counts in steps, it is a :class:`~timenet.types.spans.StepInterval`,
    the only frame an ordinal series can carry. :meth:`~timenet.dataset.TimeFDataset.add_task` checks that
    it falls inside the record, because that method has the record. It needs an explicit ``scope`` for
    the context region. A ``scope`` of ``None`` means the whole record, which covers the region to
    predict."""

    def __post_init__(self) -> None:
        """Reject a forecasting task with a bad target or a self-referential context.

        Raises:
            TimeFValidationError: If the task sets neither ``target_record_id`` nor ``target_span``.
                If it sets ``target_record_id`` with an empty ``context_record_ids``, which is a
                forecast with no input. If it sets ``target_record_id`` and that same id also appears
                in ``context_record_ids``, its own answer as input. If ``target_span`` and
                ``target_record_id`` are both set. If ``target_span`` carries ``context_record_ids``.
                Its context is already ``scope``, so a context record would re-expose the target
                region. If ``target_span`` is a point, which spans no values. :meth:`check_against_scope`
                checks for a missing scope, a frame mismatch, or a context that leaks the target. It
                runs once ``add_task`` has stamped any ``scope=``, and also here when the task is built
                with its own ``scope``.
        """
        if self.target_span is None:
            if self.target_record_id is None:
                raise TimeFValidationError(
                    "ForecastingTask requires either target_record_id or target_span to name the future "
                    "to predict. Got neither"
                )
            if not self.context_record_ids:
                raise TimeFValidationError(
                    "ForecastingTask got a target_record_id with an empty context_record_ids, which is a "
                    "forecast with no input. The separate-record form forecasts from context_record_ids, so "
                    "give at least one id. The target_span form takes its context from scope instead"
                )
            if self.target_record_id in self.context_record_ids:
                raise TimeFValidationError(
                    f"ForecastingTask target_record_id {self.target_record_id!r} also appears in "
                    f"context_record_ids, so the forecast would read its own answer as input. Drop it from "
                    f"the context"
                )
            return
        if self.target_record_id is not None:
            raise TimeFValidationError(
                f"ForecastingTask target_span names a region of the attached record, so it cannot be "
                f"combined with target_record_id={self.target_record_id!r}. Use one or the other"
            )
        if self.context_record_ids:
            raise TimeFValidationError(
                f"ForecastingTask target_span takes its context from scope, so context_record_ids must be "
                f"empty in this form, got {list(self.context_record_ids)}. A whole context record would "
                f"re-expose the target region; put covariate context on another series of scope with "
                f"time_series_ids instead"
            )
        if self.target_span.is_point:
            raise TimeFValidationError(
                f"ForecastingTask target_span is the region to predict, so it must be an interval with a "
                f"duration, not a point: got {self.target_span!r}"
            )
        if self.scope is not None:  # early check. add_task re-checks after it stamps any late scope=
            self._check_scope_against_target(self.scope, self.target_span)

    def check_against_scope(self) -> None:
        """Reject a ``target_span`` forecast whose context scope is missing, misframed, or leaks the target.

        Raises:
            TimeFValidationError: If ``target_span`` is set with no ``scope``, the whole-record default
                would include the region to predict. If ``scope`` and ``target_span`` are in different
                frames. If ``scope`` reaches into or past ``target_span`` on a series they share.
        """
        if self.target_span is None:
            return
        if self.scope is None:
            raise TimeFValidationError(
                "ForecastingTask target_span needs an explicit scope for the context region. A scope of "
                "None means the whole record (see Task.scope), which covers the region that target_span "
                "predicts. Pass scope= on the task or to add_task"
            )
        self._check_scope_against_target(self.scope, self.target_span)

    @staticmethod
    def _check_scope_against_target(scope: Span, target: TimeInterval | StepInterval) -> None:
        """Reject a context ``scope`` in the wrong frame, or one that leaks the region to predict.

        A context is safe when it ends at or before the target starts on every series they share. A
        point context covers a single position, so its exclusive end is one unit past its start.

        Args:
            scope: The context region.
            target: The region to predict.

        Raises:
            TimeFValidationError: If ``scope`` and ``target`` are in different frames, or ``scope``
                reaches into or past ``target`` on a series they share.
        """
        if isinstance(target, StepInterval):
            if not isinstance(scope, StepSpan):
                raise TimeFValidationError(
                    f"ForecastingTask target_span counts in steps, so scope must too; they must share a "
                    f"frame. Got scope={scope!r}, target_span={target!r}"
                )
            if scope.time_series_id == target.time_series_id and scope.exclusive_end > target.start:
                raise TimeFValidationError(
                    f"ForecastingTask context overlaps the region to predict on series "
                    f"{target.time_series_id!r}: the context (scope={scope!r}) must end at or before "
                    f"the target_span={target!r} starts, or the target leaks into the input"
                )
            return
        if not isinstance(scope, TimeSpan):
            raise TimeFValidationError(
                f"ForecastingTask target_span is in seconds, so scope must too; they must share a frame. "
                f"Got scope={scope!r}, target_span={target!r}"
            )
        # A time span with time_series_ids=None covers every series, so it shares one with any target.
        s_ids, t_ids = scope.time_series_ids, target.time_series_ids
        shared = s_ids is None or t_ids is None or not set(s_ids).isdisjoint(t_ids)
        if shared and scope.exclusive_end > target.start_us:
            raise TimeFValidationError(
                f"ForecastingTask context overlaps the region to predict. On a shared series, the context "
                f"(scope={scope!r}) must end at or before the target_span={target!r} starts. Otherwise the "
                f"target leaks into the input, or the forecast predicts the past from the future. If the "
                f"context is a future-known covariate, put it on a different series with time_series_ids."
            )


@dataclass(kw_only=True)
class TSEditingTask(Task):
    """A series out: transform the source record into the target record, as the ``prompt`` instructs.

    This covers denoising, filtering, and deliberate corruption ("add baseline wander"). The ``prompt``
    is the instruction, and both sides of the edit are stored records.
    """

    task_type: ClassVar[TaskType] = TaskType.TS_EDITING
    refs: ClassVar[TaskRefs] = TaskRefs(record_id_fields=("source_record_id", "target_record_id"))
    answer_is_record: ClassVar[bool] = True
    source_record_id: str
    """Id of the record to edit."""
    target_record_id: str
    """Id of the record that holds the edited result."""


@dataclass(kw_only=True)
class TSGenerationTask(Task):
    """A series out from a text specification alone: the ``prompt`` describes what to synthesize."""

    task_type: ClassVar[TaskType] = TaskType.TS_GENERATION
    refs: ClassVar[TaskRefs] = TaskRefs(record_id_fields=("target_record_id",))
    answer_is_record: ClassVar[bool] = True
    target_record_id: str
    """Id of the record that holds the series to generate."""


@dataclass(kw_only=True)
class TSCorrespondenceTask(Task):
    """Relate one series to others: which candidate record corresponds to the records in ``record_ids``.

    This covers retrieval, nearest-neighbor, and match questions. The base ``record_ids`` are the query.
    ``candidate_record_ids`` is the pool for the answer, and ``target`` names the correct one or ones. An
    empty pool keeps the answer open. Then any record in the dataset can be the answer.
    """

    task_type: ClassVar[TaskType] = TaskType.TS_CORRESPONDENCE
    refs: ClassVar[TaskRefs] = TaskRefs(
        record_id_fields=("candidate_record_ids", "target"),
        time_series_id_fields=("target_time_series_ids",),
    )
    answer_fields: ClassVar[tuple[str, ...]] = ("target", "target_time_series_ids")
    candidate_record_ids: tuple[str, ...] = ()
    """Ids of the records that supply the answer. Empty means the pool has no limit."""
    target: tuple[str, ...] | None = None
    """Ids of the corresponding record(s), which must be in ``candidate_record_ids`` when it is set."""
    target_time_series_ids: tuple[str, ...] | None = None
    """Ids of the corresponding series, when the answer names signals rather than whole records (for
    example "which signals correlate with X"). Each id must resolve to a series on the task's records."""

    def __post_init__(self) -> None:
        """Reject an empty ``target``/``target_time_series_ids``, or a ``target`` outside the pool.

        Raises:
            TimeFValidationError: If ``target`` or ``target_time_series_ids`` is ``()`` rather than
                ``None`` or non-empty. Also if ``target`` names a record the candidate pool lacks.
        """
        if self.target is not None and not self.target:
            raise TimeFValidationError(
                "TSCorrespondenceTask target must be None (answer stored by reference) or non-empty, got ()"
            )
        if self.target_time_series_ids is not None and not self.target_time_series_ids:
            raise TimeFValidationError("TSCorrespondenceTask target_time_series_ids must be None or non-empty, got ()")
        if not self.candidate_record_ids:  # no limit on the pool: any record can be the answer
            return
        outside = tuple(sid for sid in self.target or () if sid not in self.candidate_record_ids)
        if outside:
            raise TimeFValidationError(
                f"TSCorrespondenceTask target {list(outside)} is not in candidate_record_ids "
                f"{list(self.candidate_record_ids)}. The answer must be one of the candidates"
            )


def _concrete_task_classes() -> list[type[Task]]:
    """Collect every concrete task class across the hierarchy.

    A concrete class declares a ``task_type``. This walks the subclass tree, so it skips any
    intermediate base.

    Returns:
        The concrete task classes.
    """
    concrete: list[type[Task]] = []
    stack = list(Task.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if "task_type" in vars(cls):  # a concrete leaf assigns its own task_type
            concrete.append(cls)
    return concrete


def _build_task_registry(classes: Iterable[type[Task]] | None = None) -> dict[TaskType, type[Task]]:
    """Map each concrete task's ``task_type`` to its class.

    Args:
        classes: The classes to register. The default is :func:`_concrete_task_classes`. You can
            override it, so the collision check in this function is testable without a throwaway subclass of
            ``Task``. A throwaway subclass leaks into every other caller of ``__subclasses__()``, so
            the override avoids one.

    Returns:
        Each class keyed by its ``task_type``.

    Raises:
        TimeFValidationError: If a reference declaration names an unknown field, omits a record-id
            payload field, or two classes declare the same ``task_type``.
    """
    registry: dict[TaskType, type[Task]] = {}
    for cls in classes if classes is not None else _concrete_task_classes():
        fields = set(cls.__dataclass_fields__)
        declared = set(cls.refs.record_id_fields) | set(cls.refs.span_fields)
        unknown = declared - fields
        if unknown:
            raise TimeFValidationError(f"{cls.__name__}.refs names unknown fields {sorted(unknown)}")
        record_id_fields = {name for name in fields if name.endswith(("_sample_id", "_sample_ids"))}
        missing = record_id_fields - set(cls.refs.record_id_fields)
        if missing:
            raise TimeFValidationError(f"{cls.__name__}.refs.record_id_fields omits record-id fields {sorted(missing)}")
        if cls.task_type in registry:
            raise TimeFValidationError(
                f"both {registry[cls.task_type].__name__} and {cls.__name__} claim task_type {cls.task_type!r}"
            )
        registry[cls.task_type] = cls
    return registry


TASKS: dict[TaskType, type[Task]] = _build_task_registry()
