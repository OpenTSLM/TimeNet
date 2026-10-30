"""The authors' three tasks, each built as the item their loader gives a model.

Each task is an ``AnswerTask``: its prompt is the authors' context, the options lettered inside it, and
its target the letter of the right option. The README explains the format and the order of the options.
"""

from timenet.dataset import Record
from timenet.json import JsonValue
from timenet.types import AnswerTask
from timenet_connectors.datasets.mikeam.time_series_reasoning.keys import TaskKey
from timenet_connectors.datasets.mikeam.time_series_reasoning.request import description_item, question_item
from timenet_connectors.datasets.mikeam.time_series_reasoning.rows import (
    CounterfactualRow,
    DescriptionRow,
    QuestionRow,
    Row,
)


def _provenance(row: Row) -> dict[str, JsonValue]:
    return {TaskKey.FILE: row.ref.file.value, TaskKey.LINE: row.ref.line}


def _question_metadata(row: QuestionRow) -> dict[str, JsonValue]:
    return _provenance(row) | {
        TaskKey.ANSWER_INDEX: row.answer_index,
        TaskKey.ANSWER_TEXT: row.options[row.answer_index],
        TaskKey.SOURCE_ISSUES: [issue.value for issue in row.source_issues],
    }


def description_task(row: DescriptionRow, record: Record) -> AnswerTask:
    """Return the task that picks the series' description among four (the authors' ``ts2desc_mcq``).

    Args:
        row: The row, whose file gives the authors' split.
        record: The Record that holds the values of the row's own file.

    Returns:
        The task, its options in the row's seeded order and its target the description's letter.
    """
    prompt, label = description_item(row)
    metadata = _provenance(row) | {TaskKey.SCENARIO_CATEGORY: row.scenario_category.value}
    return AnswerTask(
        inputs=(record,), prompt=prompt, targets=(label,), split=row.ref.file.timenet_split, metadata=metadata
    )


def question_task(row: QuestionRow, record: Record) -> AnswerTask:
    """Return the task that answers a question about one series (the authors' ``qa_mcq``).

    The release gives it no split.

    Args:
        row: The row.
        record: The Record of the row's series.

    Returns:
        The task, its options in published order and its target the letter of ``answer_index``.
    """
    prompt, label = question_item(row)
    return AnswerTask(inputs=(record,), prompt=prompt, targets=(label,), metadata=_question_metadata(row))


def counterfactual_task(row: CounterfactualRow, record: Record, changed: Record) -> AnswerTask:
    """Return the task that compares a series with its changed version (``qa_counterfactual``).

    The release gives it no split.

    Args:
        row: The row.
        record: The Record of the row's series.
        changed: The changed series' Record. Its ``counterfactual`` annotation is never input: the
            authors' prompt doesn't include it.

    Returns:
        The task on the series and its changed version, in the order the authors' loader joins them.
    """
    prompt, label = question_item(row)
    metadata = _question_metadata(row) | {TaskKey.CATEGORY: row.category.value}
    return AnswerTask(inputs=(record, changed), prompt=prompt, targets=(label,), metadata=metadata)
