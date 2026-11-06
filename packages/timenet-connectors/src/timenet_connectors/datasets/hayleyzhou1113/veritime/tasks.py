"""Build tasks with the published prompts, labels and rationales."""

from typing import TypedDict

from timenet.dataset import Record
from timenet.json import JsonValue
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    ScalarPredictionTask,
    Split,
    Task,
)
from timenet_connectors.datasets.hayleyzhou1113.veritime.source import Row
from timenet_connectors.datasets.hayleyzhou1113.veritime.vocabulary import (
    ANNOTATION_DESCRIPTIONS,
    CLASSES,
    LABEL_SETS,
    SourceTask,
)


class _TaskFields(TypedDict):
    """The arguments every VeriTime task shares."""

    inputs: tuple[Record, ...]
    prompt: str
    rationale: str
    split: Split
    metadata: dict[str, JsonValue]


def build_task(row: Row, record: Record) -> Task:
    """Build the row's task. The type follows the answer.

    A fixed class gives a ClassificationTask, an option's text an AnswerTask and a number a
    ScalarPredictionTask.

    Returns:
        The task, with the process labels and, on a scenario row, the ChatTS answer as annotations.
    """
    fields = _TaskFields(
        inputs=(record,),
        prompt=row.question,
        rationale=row.cot,
        split=row.location.part.timef_split,
        metadata={
            "source_task": row.task.value,
            "part": row.location.part.value,
            "row": row.location.location.line - 1,
        },
    )
    task: Task
    if row.task in CLASSES:
        task = ClassificationTask(targets=(row.label,), label_set=LABEL_SETS[row.task], **fields)
    elif row.task is SourceTask.INFERENTIAL_CALCULATION:
        # A count, except one row whose two-part question also asks an amplitude: its label is a float.
        quantity = int(row.label) if row.label.isdecimal() else float(row.label)
        task = ScalarPredictionTask(targets=(quantity,), **fields)
    else:
        # The options differ by question, so there is no shared vocabulary. The target is the label as published.
        task = AnswerTask(targets=(row.label,), **fields)
    for label in row.annotations:
        task.annotate(Annotation(key=label.key, value=label.value, description=ANNOTATION_DESCRIPTIONS[label.key]))
    return task
