"""TSAQA's one task per row, built as a TimeNet task of the type its question's format decides.

The paper crosses six tasks with three question formats (Table 3). The format decides the TimeNet
type, so each format has its own builder, and the row's paper task is in its metadata:

- A true-or-false question (:func:`true_or_false_task`) is a ``ClassificationTask``. ``T`` and ``F``
  mean the same in every question, so they are one vocabulary, the ``true_or_false_vocabulary``
  dataset annotation.
- A multiple-choice question (:func:`multiple_choice_task`) and a puzzle (:func:`puzzling_task`) are
  each an ``AnswerTask``. Their letters name choices that differ from one question to the next.

The target is the published answer, as the paper scores it (Section 3.5). The release has no rationales.
"""

from collections.abc import Callable

from timenet.dataset import Record
from timenet.json import JsonValue
from timenet.types import Annotation, AnswerTask, ClassificationTask, Task
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.issues import SourceIssue
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.keys import CONTEXT_KEYS, AnnotationKey, TaskKey
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.request import task_prompt
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.rows import Row
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.vocabulary import QuestionFormat


def _input_annotations(inputs: tuple[Record, ...]) -> tuple[Annotation, ...]:
    """Return the context the input Records carry, which the model gets with the question."""
    return tuple(annotation for record in inputs for annotation in record.annotations if annotation.key in CONTEXT_KEYS)


def _metadata(row: Row, issues: frozenset[SourceIssue]) -> dict[str, JsonValue]:
    """Return the task's provenance and source issues. Metadata is never model input."""
    metadata: dict[str, JsonValue] = {
        TaskKey.TASK: row.task.value,
        TaskKey.QUESTION_TYPE: row.question_type.value,
        TaskKey.QUESTION_FORMAT: row.format.value,
        TaskKey.FILE: row.ref.split.source_file,
        TaskKey.ROW: row.ref.row,
    }
    if row.question.transform is not None:
        metadata[TaskKey.TRANSFORM] = row.question.transform.value
    metadata[TaskKey.SOURCE_ISSUES] = [issue.value for issue in sorted(issues)]
    return metadata


def true_or_false_task(row: Row, inputs: tuple[Record, ...], issues: frozenset[SourceIssue]) -> ClassificationTask:
    """Return a true-or-false question's task: whether a claim about the series holds, answered ``T`` or ``F``.

    Args:
        row: The row, whose split file gives the split.
        inputs: The row's Records in the order the prompt numbers them: input series, then printed series.
        issues: The row's source issues, as the index found them.

    Returns:
        The classification task, its target the published ``T`` or ``F``.
    """
    return ClassificationTask(
        inputs=inputs,
        prompt=task_prompt(row),
        targets=(row.answer,),
        split=row.ref.split.timenet_split,
        input_annotations=_input_annotations(inputs),
        target_schema=AnnotationKey.TRUE_OR_FALSE_VOCABULARY,
        metadata=_metadata(row, issues),
    )


def multiple_choice_task(row: Row, inputs: tuple[Record, ...], issues: frozenset[SourceIssue]) -> AnswerTask:
    """Return a multiple-choice question's task, answered with the letter of one of its two to four choices.

    When the choices are text, such as a classification question's class names, the metadata adds the
    right choice's text.

    Args:
        row: The row, whose split file gives the split.
        inputs: The row's Records in the order the prompt numbers them: input series, then printed series.
        issues: The row's source issues, as the index found them.

    Returns:
        The answer task, its target the published letter.
    """
    metadata = _metadata(row, issues)
    choices = dict(row.question.choices)
    if choices:
        metadata[TaskKey.ANSWER_TEXT] = choices[row.answer]
    return AnswerTask(
        inputs=inputs,
        prompt=task_prompt(row),
        targets=(row.answer,),
        split=row.ref.split.timenet_split,
        input_annotations=_input_annotations(inputs),
        metadata=metadata,
    )


def puzzling_task(row: Row, inputs: tuple[Record, ...], issues: frozenset[SourceIssue]) -> AnswerTask:
    """Return a puzzle's task: order the four patches that follow the input series, answered as ``C,B,D,A``.

    The answer lists the patch letters in chronological order. The paper scores it by the share of
    positions that match (Section 3.5).

    Args:
        row: The row, whose split file gives the split.
        inputs: The row's Records in the order the prompt numbers them: input series, then the patches.
        issues: The row's source issues, as the index found them.

    Returns:
        The answer task, its target the published order of the four letters.
    """
    return AnswerTask(
        inputs=inputs,
        prompt=task_prompt(row),
        targets=(row.answer,),
        split=row.ref.split.timenet_split,
        input_annotations=_input_annotations(inputs),
        metadata=_metadata(row, issues),
    )


BUILDERS: dict[QuestionFormat, Callable[[Row, tuple[Record, ...], frozenset[SourceIssue]], Task]] = {
    QuestionFormat.TRUE_OR_FALSE: true_or_false_task,
    QuestionFormat.MULTIPLE_CHOICE: multiple_choice_task,
    QuestionFormat.PUZZLING: puzzling_task,
}
"""The builder of each question format."""
