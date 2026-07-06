"""Tasks: labeled training targets that reference one or more samples.

The class is the type tag (used as a filter, e.g. ``search(task=ClassificationTask)``) and the instance
carries the payload. Unlike specs and annotations, task payload shapes are fixed in code, so tasks are
resolved on read against the built-in :data:`TASKS` registry rather than reconstructed from the manifest.
Tasks are mutable so :meth:`~timenet.dataset.TimeFDataset.add_task` can populate ``sample_ids`` after
construction.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

from timenet.types.ids import new_id


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
    id: str = field(default_factory=new_id)
    """Unique task identifier, a UUIDv7 string by default."""
    sample_ids: tuple[str, ...] = ()
    """Ids of the samples this task targets, populated by ``add_task``."""
    from_tasks: tuple["Task", ...] = ()
    """Source tasks this one was derived from."""

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
    """The discrete class label for the sample."""
    label_schema: str | None = None
    """Name of the label vocabulary the label belongs to."""


@dataclass(kw_only=True)
class LabelingTask(Task):
    """Time-localized labels within a sample, optionally targeting specific series and windows."""

    task_type: ClassVar[TaskType] = TaskType.LABELING
    label: str
    """The label applied to the targeted region."""
    label_schema: str | None = None
    """Name of the label vocabulary the label belongs to."""
    time_series_ids: tuple[str, ...] | None = None
    """Series the label targets, or None for all series in the sample."""
    windows_s: tuple[tuple[float, float], ...] | None = None
    """Labeled windows as (start, end) times in seconds, or None for the whole sample."""


@dataclass(kw_only=True)
class CaptioningTask(Task):
    """Free-form text describing the sample."""

    task_type: ClassVar[TaskType] = TaskType.CAPTIONING
    answer: str
    """Free-form text describing the sample."""


@dataclass(kw_only=True)
class QATask(Task):
    """A question and answer pair."""

    task_type: ClassVar[TaskType] = TaskType.QUESTION_AND_ANSWER
    question: str
    """The question posed about the sample."""
    answer: str
    """The answer to the question."""


@dataclass(kw_only=True)
class ForecastingTask(Task):
    """Predict future values of a series, conditioned on context samples."""

    task_type: ClassVar[TaskType] = TaskType.FORECASTING
    context_sample_ids: tuple[str, ...]
    """Ids of the samples that provide forecasting context."""
    target_sample_id: str
    """Id of the sample whose future values are predicted."""


@dataclass(kw_only=True)
class ReasoningTask(Task):
    """A question answered by reasoning to a final answer. Often composed via ``from_tasks``.

    Unlike :class:`QATask` (single-label answer), a reasoning task carries the chain of thought in
    ``rationale``. The ``answer`` is the evaluation target; the ``rationale`` is the reasoning trace to
    train / fine-tune on (``None`` when the source has no stored rationale).
    """

    task_type: ClassVar[TaskType] = TaskType.REASONING
    question: str
    """The question to be answered by reasoning."""
    rationale: str | None = None
    """The reasoning trace to train on, or None when the source stores none."""
    answer: str
    """The final answer used as the evaluation target."""


TASKS: dict[TaskType, type[Task]] = {cls.task_type: cls for cls in Task.__subclasses__()}
