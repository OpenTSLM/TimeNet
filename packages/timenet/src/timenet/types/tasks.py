"""Tasks: labeled training targets that reference one or more samples.

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
import uuid

from timenet.errors import TimeFValidationError


@unique
class TaskType(StrEnum):
    """Stable type tags for the built-in task classes; also the on-disk task partition names."""

    CLASSIFICATION = "classification"
    LABELING = "labeling"
    CAPTIONING = "captioning"
    QUESTION_AND_ANSWER = "question_and_answer"
    FORECASTING = "forecasting"
    REASONING = "reasoning"


@dataclass(kw_only=True)
class Task:
    """Base for all tasks. Not instantiated directly; subclasses declare ``task_type`` and a payload."""

    task_type: ClassVar[TaskType]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sample_ids: tuple[str, ...] = ()
    from_tasks: tuple["Task", ...] = ()

    @property
    def from_task_ids(self) -> tuple[str, ...]:
        """Return the ids of the tasks this one was derived from.

        Returns:
            The ``id`` of every task in ``from_tasks``.
        """
        return tuple(task.id for task in self.from_tasks)


@dataclass(kw_only=True)
class ClassificationTask(Task):
    """One discrete label applied to the whole sample."""

    task_type: ClassVar[TaskType] = TaskType.CLASSIFICATION
    label: str
    label_schema: str | None = None


@dataclass(kw_only=True)
class LabelingTask(Task):
    """Time-localized labels within a sample, optionally targeting specific series and windows."""

    task_type: ClassVar[TaskType] = TaskType.LABELING
    label: str
    label_schema: str | None = None
    time_series_ids: tuple[str, ...] | None = None
    windows_s: tuple[tuple[float, float], ...] | None = None
    """Spans in the **source recording timeline**, the same frame as ``TimeSeries.t_start_s``.
    ``None`` means the sample's full duration. That a window actually falls inside the target
    sample's span is checked by :meth:`~timenet.dataset.TimeFDataset.add_task`, which has the
    sample to check against."""

    def __post_init__(self) -> None:
        """Reject an explicitly empty or malformed ``time_series_ids`` / ``windows_s``.

        Raises:
            TimeFValidationError: If either field is ``()`` rather than ``None``, or a window's end is
                not strictly greater than its start.
        """
        if self.time_series_ids is not None and not self.time_series_ids:
            raise TimeFValidationError("LabelingTask time_series_ids must be None (all channels) or non-empty, got ()")
        if self.windows_s is not None and not self.windows_s:
            raise TimeFValidationError("LabelingTask windows_s must be None (full duration) or non-empty, got ()")
        for start_s, end_s in self.windows_s or ():
            if end_s <= start_s:
                raise TimeFValidationError(f"LabelingTask window end ({end_s}) must be > start ({start_s})")


@dataclass(kw_only=True)
class CaptioningTask(Task):
    """Free-form text describing the sample."""

    task_type: ClassVar[TaskType] = TaskType.CAPTIONING
    answer: str


@dataclass(kw_only=True)
class QATask(Task):
    """A question and answer pair."""

    task_type: ClassVar[TaskType] = TaskType.QUESTION_AND_ANSWER
    question: str
    answer: str


@dataclass(kw_only=True)
class ForecastingTask(Task):
    """Predict future values of a series, conditioned on context samples."""

    task_type: ClassVar[TaskType] = TaskType.FORECASTING
    context_sample_ids: tuple[str, ...]
    target_sample_id: str


@dataclass(kw_only=True)
class ReasoningTask(Task):
    """A question answered by reasoning to a final answer. Often composed via ``from_tasks``.

    Unlike :class:`QATask` (single-label answer), a reasoning task carries the chain of thought in
    ``rationale``. The ``answer`` is the evaluation target; the ``rationale`` is the reasoning trace to
    train / fine-tune on (``None`` when the source has no stored rationale).
    """

    task_type: ClassVar[TaskType] = TaskType.REASONING
    question: str
    rationale: str | None = None
    answer: str


def _concrete_task_classes() -> list[type[Task]]:
    """Collect every concrete task class (those declaring their own ``task_type``).

    Walks the subclass tree rather than reading ``Task.__subclasses__()`` directly, so an intermediate
    base that groups tasks without claiming a ``task_type`` is skipped instead of raising.

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
        ValueError: If two classes declare the same ``task_type``
    """
    registry: dict[TaskType, type[Task]] = {}
    for cls in classes if classes is not None else _concrete_task_classes():
        if cls.task_type in registry:
            raise ValueError(
                f"task_type {cls.task_type!r} is claimed by both {registry[cls.task_type].__name__} and {cls.__name__}"
            )
        registry[cls.task_type] = cls
    return registry


TASKS: dict[TaskType, type[Task]] = _build_task_registry()
