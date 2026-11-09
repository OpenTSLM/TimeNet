"""Build the TSAQA tasks from checked rows and their Records."""

from collections.abc import Iterator, Mapping, Sequence
import itertools

from timenet.dataset import Record, TimeFDataset
from timenet.json import JsonValue
from timenet.types import Annotation, AnswerTask, ClassificationTask, Task
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.labels import (
    CONTEXT_KEYS,
    PUBLISHED_LISTS,
    SOURCE_ISSUES,
    SPLITS,
    TRUE_OR_FALSE_LABEL_SET,
    QuestionFormat,
    SourceIssue,
)
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.source import Row, RowKey


def _task(row: Row, inputs: tuple[Record, ...], issues: frozenset[SourceIssue], dataset: TimeFDataset) -> Task:
    """Build a row's task: a true-or-false row is a classification task, any other an answer task.

    The model reads what Appendix E.1 lists: the published context, then the published ``task`` and
    ``question_type``, then the question. The context is on the first input Record: published texts
    under their column names, or published lists under their ``published_*`` keys. The question format
    read from the question stays in the metadata, for routing.

    Args:
        row: The row.
        inputs: The row's input Records.
        issues: The row's source issues.
        dataset: The dataset that shares each published task and question-type annotation.

    Returns:
        The task, its target the published answer.
    """
    annotations = itertools.chain.from_iterable(record.annotations for record in inputs)
    input_annotations = (
        *(a for a in annotations if a.key in CONTEXT_KEYS or a.key in PUBLISHED_LISTS.values()),
        *(dataset.annotate_shared(Annotation(key=key, value=value)) for key, value in _published_labels(row)),
    )
    metadata: dict[str, JsonValue] = {
        "task": row.task.value,
        "question_type": row.question_type,
        "question_format": row.question_format.value,
        "file": row.file,
        "row": row.index,
    }
    if row.transform is not None:
        metadata["transform"] = row.transform
    metadata[SOURCE_ISSUES] = [issue.value for issue in sorted(issues)]
    if row.question_format is QuestionFormat.TRUE_OR_FALSE:
        return ClassificationTask(
            inputs=inputs,
            prompt=row.prompt,
            targets=(row.answer,),
            split=SPLITS[row.split],
            input_annotations=input_annotations,
            label_set=TRUE_OR_FALSE_LABEL_SET,
            metadata=metadata,
        )
    if row.choices:
        metadata["answer_text"] = row.choices[row.answer]
    return AnswerTask(
        inputs=inputs,
        prompt=row.prompt,
        targets=(row.answer,),
        split=SPLITS[row.split],
        input_annotations=input_annotations,
        metadata=metadata,
    )


def iter_tasks(
    dataset: TimeFDataset,
    rows: Sequence[Row],
    records: Sequence[tuple[Record, ...]],
    issues: Mapping[RowKey, frozenset[SourceIssue]],
) -> Iterator[Task]:
    """Yield the task of each row, in row order, for the dataset's task stream.

    Yields:
        Each row's task.
    """
    for row, inputs in zip(rows, records, strict=True):
        yield _task(row, inputs, issues.get(row.key, frozenset()), dataset)


def _published_labels(row: Row) -> tuple[tuple[str, str], tuple[str, str]]:
    """Return the published ``task`` and ``question_type`` of a row, as annotation keys and values."""
    return ("task", row.task.value), ("question_type", row.question_type)
