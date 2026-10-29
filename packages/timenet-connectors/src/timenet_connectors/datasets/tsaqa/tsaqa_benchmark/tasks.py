"""TSAQA's one task per row, of one kind per question format, built as the TimeNet task it becomes.

The paper crosses six tasks with three question formats (Table 3). A source task kind is a format,
since the format decides the TimeNet type, and the row's paper task is in its metadata:

- A true-or-false question (:class:`TrueOrFalseTask`) is a ``ClassificationTask``. ``T`` and ``F``
  mean the same in every question, so they are one vocabulary, the ``true_or_false_vocabulary``
  dataset annotation.
- A multiple-choice question (:class:`MultipleChoiceTask`) and a puzzle (:class:`PuzzleTask`) are
  each an ``AnswerTask``. Their letters name choices that differ from one question to the next.

The target is the published answer, as the paper scores it (Section 3.5). The release has no rationales.
"""

from dataclasses import dataclass
from typing import ClassVar

from timenet.dataset import Record
from timenet.json import JsonValue
from timenet.types import Annotation, AnswerTask, ClassificationTask, Split, Task
from timenet_connectors.bases.loader import DatasetTask
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.issues import SourceIssue
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.keys import CONTEXT_KEYS, AnnotationKey, TaskKey
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.request import task_prompt
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.rows import Row


@dataclass(frozen=True)
class TSAQATask(DatasetTask):
    """The task of one row: its question, about the row's Records, under the authors' split."""

    row: Row
    inputs: tuple[Record, ...]
    """The row's Records in the order the prompt numbers them: input series, then printed series."""
    issues: frozenset[SourceIssue]
    """The row's source issues, as the index found them."""

    @property
    def split(self) -> Split:
        """Return the task's split: the authors' split file of the row."""
        return self.row.ref.split.timenet_split

    def _input_annotations(self) -> tuple[Annotation, ...]:
        """Return the context the input Records carry, which the model gets with the question."""
        return tuple(
            annotation for record in self.inputs for annotation in record.annotations if annotation.key in CONTEXT_KEYS
        )

    def _metadata(self) -> dict[str, JsonValue]:
        """Return the task's provenance and source issues. Metadata is never model input."""
        row = self.row
        metadata: dict[str, JsonValue] = {
            TaskKey.TASK: row.task.value,
            TaskKey.QUESTION_TYPE: row.question_type.value,
            TaskKey.QUESTION_FORMAT: row.format.value,
            TaskKey.FILE: row.ref.split.source_file,
            TaskKey.ROW: row.ref.row,
        }
        if row.question.transform is not None:
            metadata[TaskKey.TRANSFORM] = row.question.transform.value
        metadata[TaskKey.SOURCE_ISSUES] = [issue.value for issue in sorted(self.issues)]
        return metadata


@dataclass(frozen=True)
class TrueOrFalseTask(TSAQATask):
    """A true-or-false question: whether a claim about the series holds, answered ``T`` or ``F``."""

    TIMENET_TYPE: ClassVar[type[Task]] = ClassificationTask

    def to_timenet_task(self) -> ClassificationTask:
        """Return the classification task, its target the published ``T`` or ``F``."""
        return ClassificationTask(
            inputs=self.inputs,
            prompt=task_prompt(self.row),
            targets=(self.row.answer,),
            split=self.split,
            input_annotations=self._input_annotations(),
            target_schema=AnnotationKey.TRUE_OR_FALSE_VOCABULARY,
            metadata=self._metadata(),
        )


@dataclass(frozen=True)
class MultipleChoiceTask(TSAQATask):
    """A multiple-choice question, answered with the letter of one of its two to four choices."""

    TIMENET_TYPE: ClassVar[type[Task]] = AnswerTask

    def to_timenet_task(self) -> AnswerTask:
        """Return the answer task, its target the published letter.

        When the choices are text, such as a classification question's class names, the metadata adds
        the right choice's text.
        """
        metadata = self._metadata()
        choices = dict(self.row.question.choices)
        if choices:
            metadata[TaskKey.ANSWER_TEXT] = choices[self.row.answer]
        return AnswerTask(
            inputs=self.inputs,
            prompt=task_prompt(self.row),
            targets=(self.row.answer,),
            split=self.split,
            input_annotations=self._input_annotations(),
            metadata=metadata,
        )


@dataclass(frozen=True)
class PuzzleTask(TSAQATask):
    """A puzzle: order the four patches that follow the input series, answered as ``C,B,D,A``.

    The answer lists the patch letters in chronological order. The paper scores it by the share of
    positions that match (Section 3.5).
    """

    TIMENET_TYPE: ClassVar[type[Task]] = AnswerTask

    def to_timenet_task(self) -> AnswerTask:
        """Return the answer task, its target the published order of the four letters."""
        return AnswerTask(
            inputs=self.inputs,
            prompt=task_prompt(self.row),
            targets=(self.row.answer,),
            split=self.split,
            input_annotations=self._input_annotations(),
            metadata=self._metadata(),
        )
