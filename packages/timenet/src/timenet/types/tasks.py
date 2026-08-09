"""Tasks: labeled training targets that reference one or more samples.

Every task is the same shape — *inputs -> one typed answer* — so the shared frame lives on the
:class:`Task` base: the samples it is about, an optional ``prompt``, an optional ``scope`` narrowing the
input to a region, the annotations handed in as context, the answer (``target``, inline or by reference to
stored annotations), and an optional ``rationale`` chain of thought. A subclass adds only what makes its
answer a different *kind* of thing: a category, free text, a number, a set of regions, or a produced
series. Because prompt and scope are on the base, a whole-recording category and a category over a given
window are the same task type with ``scope`` unset or set, and a caption is an
:class:`AnswerTask` with no prompt.

The class is the type tag (used as a filter, e.g. ``search(task=ClassificationTask)``) and the instance
carries the payload. Unlike specs and annotations, task payload shapes are fixed in code, so tasks are
resolved on read against the built-in :data:`TASKS` registry rather than reconstructed from the manifest.
Tasks are mutable so :meth:`~timenet.dataset.TimeFDataset.add_task` can populate ``sample_ids`` after
construction.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import ClassVar

import pint

from timenet.errors import TimeFValidationError
from timenet.types.ids import new_id
from timenet.types.spans import Span
from timenet.types.units import normalize_unit


@unique
class TaskType(StrEnum):
    """Stable type tags for the built-in task classes; also the on-disk task partition names."""

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
    """Whether a localization target intends to cover the recording or may leave time unmarked."""

    SPARSE = "sparse"
    """Only the marked spans are claimed; unmarked time is simply unlabeled (e.g. R-peaks)."""
    EXHAUSTIVE = "exhaustive"
    """Declares that spans intend to tile the region of interest (e.g. sleep staging)."""


@dataclass(frozen=True)
class TaskRefs:
    """Which of a task's *payload* fields hold references, declared beside the class that owns them.

    The writer, the reader, and the copy-on-write editor all have to know that
    ``ForecastingTask.target_sample_id`` is a sample id while ``ScalarPredictionTask.target`` is a plain
    number. Declaring it on the class keeps that knowledge next to the field instead of in a lookup table
    in the format layer and an ``isinstance`` chain in the editor, which drift the moment a task is added.

    The base fields (``sample_ids``, ``scope``, the annotation id tuples) are common to every task and are
    handled directly; this covers only the type-specific payload.
    """

    sample_id_fields: tuple[str, ...] = ()
    """Payload fields holding a sample id or a tuple of them. Losing one invalidates the task."""
    span_fields: tuple[str, ...] = ()
    """Payload fields holding a :class:`~timenet.types.spans.Span` or a tuple of them."""


@dataclass(kw_only=True)
class Task:
    """Base for all tasks. Not instantiated directly; subclasses declare ``task_type`` and an answer type.

    Holds everything that is the same across task types, so generic code (training loops, the writer, the
    editor) can read a task without knowing its concrete type.
    """

    task_type: ClassVar[TaskType]
    """The subclass's stable type tag. Deliberately absent here: the base is not a task."""
    refs: ClassVar[TaskRefs] = TaskRefs()
    """Which payload fields hold sample ids or spans (see :class:`TaskRefs`)."""
    answer_is_sample: ClassVar[bool] = False
    """True when the answer is a *produced series*, located by a payload sample id rather than ``target``."""
    target_is_scalar: ClassVar[bool] = False
    """True when ``target`` can be returned as a scalar by ``to_features_and_targets``."""

    id: str = field(default_factory=new_id)
    """Unique task identifier, a UUIDv7 string by default."""
    sample_ids: tuple[str, ...] = ()
    """Ids of the samples this task is about, populated by ``add_task``."""
    prompt: str | None = None
    """What the model is asked, when the task is prompted; ``None`` for an unprompted task."""
    scope: Span | None = None
    """The region of the input the task is about; ``None`` means the whole sample."""
    input_annotation_ids: tuple[str, ...] = ()
    """Annotations handed to the model as context, as opposed to ones it has to produce."""
    target: object | None = None
    """The answer, typed by the subclass. ``None`` when the answer is stored by reference, or produced
    as a series (see ``answer_is_sample``)."""
    target_annotation_ids: tuple[str, ...] = ()
    """The answer *by reference*: it is these stored annotations rather than an inline copy of them.
    Exclusive with ``target``; :meth:`~timenet.dataset.TimeFDataset.add_task` enforces that."""
    rationale: str | None = None
    """Chain of thought to train on. Any task may carry one; ``None`` when the source stores none."""
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
        """Return every span the task carries: its ``scope`` plus any span-valued payload field.

        Lets :meth:`~timenet.dataset.TimeFDataset.add_task` bounds-check a task's geometry without
        knowing which concrete type it is looking at.

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


@dataclass(kw_only=True)
class ClassificationTask(Task):
    """One categorical label: over the whole sample, or over ``scope`` when one is set.

    A whole-recording class ("this ECG shows atrial fibrillation") and a label on a supplied region ("this
    30 s epoch is sleep stage N2") differ only in whether the input is narrowed, so both are this type.
    """

    task_type: ClassVar[TaskType] = TaskType.CLASSIFICATION
    target_is_scalar: ClassVar[bool] = True
    target: str | None = None
    """The label."""
    target_schema: str | None = None
    """Name of the label vocabulary the target belongs to."""


@dataclass(kw_only=True)
class AnswerTask(Task):
    """Free-form text out. Unprompted it is a caption; with a ``prompt`` it is a question answered.

    ``rationale`` (on the base) carries the chain of thought, so a plain answer and a reasoned answer are
    the same type with the field unset or set.
    """

    task_type: ClassVar[TaskType] = TaskType.ANSWER
    target_is_scalar: ClassVar[bool] = True
    target: str | None = None
    """The answer, or the caption when the task is unprompted."""


@dataclass(kw_only=True)
class ScalarPredictionTask(Task):
    """One number out, with the quantity it measures and its physical unit kept as data.

    Encoding a regression target as a string in an answer task loses its type. Keeping it a ``float``
    with a ``unit`` makes regression metrics, batching, and unit-aware conversion straightforward.
    """

    task_type: ClassVar[TaskType] = TaskType.SCALAR_PREDICTION
    target_is_scalar: ClassVar[bool] = True
    target: float | None = None
    """The predicted value."""
    unit: str | pint.Unit | None = None
    """Optional physical unit of ``target`` — a unit string (e.g. ``"bpm"``) or a :class:`pint.Unit`.
    Validated against the shared registry on construction; a ``pint.Unit`` is stored as its name."""
    target_name: str | None = None
    """Name of the quantity being predicted (e.g. ``"mean_heart_rate"``)."""

    def __post_init__(self) -> None:
        """Normalize ``unit`` against the shared registry so it is always a plain string or ``None``."""
        self.unit = normalize_unit(self.unit)


@dataclass(kw_only=True)
class TemporalLocalizationTask(Task):
    """Regions out: find where something happens, given a description of it.

    The inverse of a scoped :class:`ClassificationTask`, which supplies the region and asks for its label.
    One type covers event detection, segmentation, and change-point detection because they share this
    target: a point (a :class:`~timenet.types.spans.Span` with no ``end_s``) or an interval, each
    optionally labeled by the annotation it references and scoped to particular series.
    """

    task_type: ClassVar[TaskType] = TaskType.TEMPORAL_LOCALIZATION
    refs: ClassVar[TaskRefs] = TaskRefs(span_fields=("target",))
    target: tuple[Span, ...] | None = None
    """The regions to find, or ``None`` when they are stored as ``target_annotation_ids``."""
    mode: LocalizationMode = LocalizationMode.SPARSE
    """Whether the spans must cover the region of interest (see :class:`LocalizationMode`)."""

    def __post_init__(self) -> None:
        """Coerce ``mode`` to the enum and reject an explicitly empty ``target``.

        Raises:
            TimeFValidationError: If ``mode`` is unknown, or ``target`` is ``()`` rather than ``None``
                or non-empty.
        """
        try:
            self.mode = LocalizationMode(self.mode)
        except ValueError as exc:
            raise TimeFValidationError(
                f"unknown localization mode {self.mode!r}; expected one of {[mode.value for mode in LocalizationMode]}"
            ) from exc
        if self.target is not None and not self.target:
            raise TimeFValidationError(
                "TemporalLocalizationTask target must be None (answer stored by reference) or non-empty, got ()"
            )


@dataclass(kw_only=True)
class ForecastingTask(Task):
    """A series out: continue the context into the future.

    The future is either a whole separate sample (``target_sample_id``) or a region of the sample the
    task is attached to (``target_span``): exactly one, never both and never neither. The second shape
    lets a single unsplit series carry a horizon, so the dataset can ship the raw recording rather than a
    context/target pair. When ``target_span`` is used, ``scope`` must be set too: the base ``Task.scope``
    default of ``None`` means the whole sample, which would include the region ``target_span`` predicts.
    The two must share a frame: seconds on a series with a timeline, or steps on one that counts in steps,
    which is the only frame an ordinal series (no cadence, no timeline) can carry.
    """

    task_type: ClassVar[TaskType] = TaskType.FORECASTING
    refs: ClassVar[TaskRefs] = TaskRefs(
        sample_id_fields=("context_sample_ids", "target_sample_id"),
        span_fields=("target_span",),
    )
    answer_is_sample: ClassVar[bool] = True
    context_sample_ids: tuple[str, ...] = ()
    """Ids of the samples that provide forecasting context."""
    target_sample_id: str | None = None
    """Id of the sample whose future values are predicted; ``None`` when ``target_span`` names the region
    to predict within the attached sample instead."""
    target_span: Span | None = None
    """The region to predict, within the sample the task is attached to. An interval, not a point,
    exclusive with ``target_sample_id``, and in the same frame as ``scope``: microseconds on the
    **source recording timeline** for a series with one, or steps for a series that counts in steps (the
    only frame an ordinal series can carry). That it falls inside the sample is checked by
    :meth:`~timenet.dataset.TimeFDataset.add_task`, which has the sample to check against. Requires an
    explicit ``scope`` naming the context region, since the base ``scope=None`` default of "the whole
    sample" would otherwise include the region to predict."""

    def __post_init__(self) -> None:
        """Reject a target_span that is a point, paired, unset, unscoped, or in a different frame from scope.

        Raises:
            TimeFValidationError: If ``target_span`` is a point rather than an interval, if it is set
                alongside ``target_sample_id``, if neither ``target_span`` nor ``target_sample_id`` is
                set, if ``target_span`` is set without ``scope``, or if its frame differs from
                ``scope``'s.
        """
        if self.target_span is None:
            if self.target_sample_id is None:
                raise TimeFValidationError(
                    "ForecastingTask requires either target_sample_id or target_span to name the future "
                    "to predict; got neither"
                )
            return
        if self.target_span.is_point:
            raise TimeFValidationError(
                f"ForecastingTask target_span must be an interval, not a point: a point has no extent "
                f"and so names no values to predict. A one-step horizon is the interval covering that "
                f"step: IntervalSpan.seconds(t, t + step) on a series with a timeline, or "
                f"IntervalSpan.steps(k, k + 1, time_series_ids=...) on one that counts in steps. "
                f"Got {self.target_span!r}"
            )
        if self.target_sample_id is not None:
            raise TimeFValidationError(
                f"ForecastingTask target_span names a region of the attached sample, so it cannot be "
                f"combined with target_sample_id={self.target_sample_id!r}; use one or the other"
            )
        if self.scope is None:
            raise TimeFValidationError(
                "ForecastingTask target_span needs an explicit scope naming the context region; "
                "scope=None would mean the whole sample (see Task.scope), which would include the region "
                "target_span names to predict. Pass scope= to this constructor; add_task's scope= is "
                "stamped on after this check runs and so cannot satisfy it"
            )
        if self.scope.frame is not self.target_span.frame:
            raise TimeFValidationError(
                f"ForecastingTask scope and target_span must share a frame: the context is "
                f"{self.scope.frame.value!r} and the horizon is {self.target_span.frame.value!r}, which "
                f"do not lie on one axis. Give both in seconds, or both in steps on the same series"
            )


@dataclass(kw_only=True)
class TSEditingTask(Task):
    """A series out: transform the source sample into the target sample, as the ``prompt`` instructs.

    Covers denoising, filtering, and deliberate corruption ("add baseline wander"); the instruction is
    the ``prompt`` and both sides of the edit are stored samples.
    """

    task_type: ClassVar[TaskType] = TaskType.TS_EDITING
    refs: ClassVar[TaskRefs] = TaskRefs(sample_id_fields=("source_sample_id", "target_sample_id"))
    answer_is_sample: ClassVar[bool] = True
    source_sample_id: str
    """Id of the sample to be edited."""
    target_sample_id: str
    """Id of the sample holding the edited result."""


@dataclass(kw_only=True)
class TSGenerationTask(Task):
    """A series out from a text specification alone: the ``prompt`` describes what to synthesize."""

    task_type: ClassVar[TaskType] = TaskType.TS_GENERATION
    refs: ClassVar[TaskRefs] = TaskRefs(sample_id_fields=("target_sample_id",))
    answer_is_sample: ClassVar[bool] = True
    target_sample_id: str
    """Id of the sample holding the series to generate."""


@dataclass(kw_only=True)
class TSCorrespondenceTask(Task):
    """Relate one series to others: which candidate sample corresponds to the sample(s) in ``sample_ids``.

    Covers retrieval, nearest-neighbour, and matching questions. The base ``sample_ids`` are the query;
    ``candidate_sample_ids`` is the pool the answer is chosen from, and ``target`` names the correct
    one(s). An empty pool leaves it open-ended (any sample in the dataset may be the answer).
    """

    task_type: ClassVar[TaskType] = TaskType.TS_CORRESPONDENCE
    refs: ClassVar[TaskRefs] = TaskRefs(sample_id_fields=("candidate_sample_ids", "target"))
    candidate_sample_ids: tuple[str, ...] = ()
    """Ids of the samples the answer is chosen from; empty means the pool is unconstrained."""
    target: tuple[str, ...] | None = None
    """Ids of the corresponding sample(s), which must come from ``candidate_sample_ids`` when it is set."""

    def __post_init__(self) -> None:
        """Reject an empty ``target`` or one naming a sample outside the candidate pool.

        Raises:
            TimeFValidationError: If ``target`` is ``()`` rather than ``None`` or non-empty, or if it
                names a sample the (non-empty) candidate pool does not contain.
        """
        if self.target is not None and not self.target:
            raise TimeFValidationError(
                "TSCorrespondenceTask target must be None (answer stored by reference) or non-empty, got ()"
            )
        if not self.candidate_sample_ids:  # an unconstrained pool: any sample may be the answer
            return
        outside = tuple(sid for sid in self.target or () if sid not in self.candidate_sample_ids)
        if outside:
            raise TimeFValidationError(
                f"TSCorrespondenceTask target {list(outside)} is not in candidate_sample_ids "
                f"{list(self.candidate_sample_ids)}; the answer must be one of the candidates"
            )


def _concrete_task_classes() -> list[type[Task]]:
    """Collect every concrete task class (those declaring a ``task_type``) across the hierarchy.

    Walks the subclass tree so any intermediate base is skipped.

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
        classes: The classes to register. Defaults to :func:`_concrete_task_classes`; overridable so
            the collision check below is unit-testable without registering throwaway subclasses of
            ``Task`` itself, which would leak into every other caller of ``__subclasses__()``.

    Returns:
        Each class keyed by its ``task_type``.

    Raises:
        TimeFValidationError: If a reference declaration names an unknown field, omits a sample-id
            payload field, or two classes declare the same ``task_type``.
    """
    registry: dict[TaskType, type[Task]] = {}
    for cls in classes if classes is not None else _concrete_task_classes():
        fields = set(cls.__dataclass_fields__)
        declared = set(cls.refs.sample_id_fields) | set(cls.refs.span_fields)
        unknown = declared - fields
        if unknown:
            raise TimeFValidationError(f"{cls.__name__}.refs names unknown fields {sorted(unknown)}")
        sample_id_fields = {name for name in fields if name.endswith(("_sample_id", "_sample_ids"))}
        missing = sample_id_fields - set(cls.refs.sample_id_fields)
        if missing:
            raise TimeFValidationError(f"{cls.__name__}.refs.sample_id_fields omits sample-id fields {sorted(missing)}")
        if cls.task_type in registry:
            raise TimeFValidationError(
                f"task_type {cls.task_type!r} is claimed by both {registry[cls.task_type].__name__} and {cls.__name__}"
            )
        registry[cls.task_type] = cls
    return registry


TASKS: dict[TaskType, type[Task]] = _build_task_registry()
