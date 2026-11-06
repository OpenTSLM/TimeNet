"""Build repeatable EngineMT-QA tasks from checked pairs and records."""

from collections.abc import Iterator, Mapping, Sequence

from timenet.dataset import Record
from timenet.errors import TimeNetConnectorError
from timenet.json import JsonValue
from timenet.types import (
    AnswerTask,
    ClassificationTask,
    Task,
)
from timenet_connectors.datasets.pandalin98.enginemt_qa.source import Pair, QAFile, QAInventory
from timenet_connectors.datasets.pandalin98.enginemt_qa.vocabulary import STEPS_PER_CYCLE, MetadataKey, SourceIssue


def _inputs(pair: Pair, records: Mapping[int, Record]) -> tuple[Record, ...]:
    """Return the Records of the pair's rows, in published order.

    Raises:
        TimeNetConnectorError: If a row has no Record, as when a QA file changed since conversion.
    """
    missing = [row for row in pair.rows if row not in records]
    if missing:
        raise TimeNetConnectorError(
            f"{pair.place.part.source_file} line {pair.place.line_index + 1}: no Record for rows {missing}"
        )
    return tuple(records[row] for row in pair.rows)


def _task(pair: Pair, inputs: tuple[Record, ...], issues: Sequence[SourceIssue]) -> AnswerTask | ClassificationTask:
    """Build the task of a pair that has a label.

    Args:
        inputs: The Records of the pair's rows, in published order.
        issues: The pair's source issues, in ``SourceIssue`` order.

    Returns:
        An answer task for an open question, with the published answer as target. A classification task
        for a closed question, with the published letters as targets over its label set.
    """
    issue_names: list[JsonValue] = [issue.value for issue in issues]
    metadata = pair.place.metadata | {MetadataKey.STAGE: pair.stage.value, MetadataKey.SOURCE_ISSUES: issue_names}
    if pair.stage.is_multi_cycle:
        metadata[MetadataKey.STEPS_PER_CYCLE] = STEPS_PER_CYCLE
    split = pair.place.part.timef_split
    if pair.closed is None:
        return AnswerTask(inputs=inputs, prompt=pair.question, targets=(pair.answer,), split=split, metadata=metadata)
    return ClassificationTask(
        inputs=inputs,
        prompt=pair.question,
        targets=pair.closed.parse_answer(pair.answer),
        label_set=pair.closed.label_set,
        split=split,
        metadata=metadata | {MetadataKey.CLOSED_QUESTION: pair.closed.value},
    )


def iter_tasks(qa_files: Sequence[QAFile], records: Mapping[int, Record], scan: QAInventory) -> Iterator[Task]:
    """Yield labelled QA tasks in source order on each pass."""
    for qa in qa_files:
        for pair in qa.pairs():
            if pair.has_label:
                yield _task(pair, _inputs(pair, records), scan.issues(pair))
