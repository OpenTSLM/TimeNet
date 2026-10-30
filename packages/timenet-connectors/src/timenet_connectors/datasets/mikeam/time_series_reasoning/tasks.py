"""The authors' three tasks, each built as the item their loader gives a model.

Each task is an ``AnswerTask``: its prompt is the authors' context, the options lettered inside it, and
its target the letter of the right option. The README explains the format and the order of the options.
"""

from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

from timenet.dataset import Record
from timenet.json import JsonValue
from timenet.types import AnswerTask, Split, Task
from timenet_connectors.bases.loader import DatasetTask
from timenet_connectors.datasets.mikeam.time_series_reasoning.keys import TaskKey
from timenet_connectors.datasets.mikeam.time_series_reasoning.request import description_item, question_item
from timenet_connectors.datasets.mikeam.time_series_reasoning.rows import (
    CounterfactualRow,
    DescriptionRow,
    QuestionRow,
    Row,
)


R = TypeVar("R", bound=Row)


@dataclass(frozen=True)
class TimeSeriesReasoningTask(DatasetTask, Generic[R]):
    """A task of one row of one of the authors' task files. Every kind becomes an ``AnswerTask``."""

    TIMENET_TYPE: ClassVar[type[Task]] = AnswerTask

    row: R
    """The row the task is built from."""

    @property
    def split(self) -> Split | None:
        """Return the authors' split of the row's file: a description file's, ``None`` for a question file."""
        return self.row.ref.file.timenet_split


def _provenance(row: Row) -> dict[str, JsonValue]:
    return {TaskKey.FILE: row.ref.file.value, TaskKey.LINE: row.ref.line}


def _question_metadata(row: QuestionRow) -> dict[str, JsonValue]:
    return _provenance(row) | {
        TaskKey.ANSWER_INDEX: row.answer_index,
        TaskKey.ANSWER_TEXT: row.options[row.answer_index],
        TaskKey.SOURCE_ISSUES: [issue.value for issue in row.source_issues],
    }


@dataclass(frozen=True)
class DescriptionTask(TimeSeriesReasoningTask[DescriptionRow]):
    """Pick the series' description among four (the authors' ``ts2desc_mcq``), under their split."""

    record: Record
    """The Record that holds the values of the row's own file."""

    def to_timenet_task(self) -> AnswerTask:
        """Return the task, its options in the row's seeded order and its target the description's letter."""
        prompt, label = description_item(self.row)
        metadata = _provenance(self.row) | {TaskKey.SCENARIO_CATEGORY: self.row.scenario_category.value}
        return AnswerTask(inputs=(self.record,), prompt=prompt, targets=(label,), split=self.split, metadata=metadata)


@dataclass(frozen=True)
class QuestionTask(TimeSeriesReasoningTask[QuestionRow]):
    """Answer a question about one series (the authors' ``qa_mcq``). The release gives it no split."""

    record: Record

    def to_timenet_task(self) -> AnswerTask:
        """Return the task, its options in published order and its target the letter of ``answer_index``."""
        prompt, label = question_item(self.row)
        return AnswerTask(inputs=(self.record,), prompt=prompt, targets=(label,), metadata=_question_metadata(self.row))


@dataclass(frozen=True)
class CounterfactualTask(TimeSeriesReasoningTask[CounterfactualRow]):
    """Answer a question that compares a series with its changed version (``qa_counterfactual``).

    The release gives it no split.
    """

    record: Record
    changed: Record
    """The changed series' Record. Its ``counterfactual`` annotation is never input: the authors'
    prompt doesn't include it."""

    def to_timenet_task(self) -> AnswerTask:
        """Return the task on the series and its changed version, in the order the authors' loader joins them."""
        prompt, label = question_item(self.row)
        metadata = _question_metadata(self.row) | {TaskKey.CATEGORY: self.row.category.value}
        return AnswerTask(inputs=(self.record, self.changed), prompt=prompt, targets=(label,), metadata=metadata)
