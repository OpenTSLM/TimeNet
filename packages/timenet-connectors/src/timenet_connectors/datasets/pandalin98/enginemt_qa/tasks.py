"""EngineMT-QA's tasks, one per stage, each QA pair built as a TimeNet task.

Every task keeps the published question as its prompt, ``<ts>`` marker included, and its file as its
split. Its metadata names the pair as the authors' evaluation does: the file, the 0-based line
(``source_line``) and the 0-based pair in the line (``qa_offset``).
"""

from typing import Any

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.types import AnswerTask, ClassificationTask
from timenet_connectors.datasets.pandalin98.enginemt_qa.keys import vocabulary_key
from timenet_connectors.datasets.pandalin98.enginemt_qa.qa import QAPair
from timenet_connectors.datasets.pandalin98.enginemt_qa.release import MULTI_CYCLE_ROWS, N_STEPS


STEPS_PER_CYCLE = N_STEPS // MULTI_CYCLE_ROWS
"""The steps of each cycle the authors' model reads in a multi-cycle task: the first 60 of 600.

``TsQaDataset.__getitem__`` concatenates ``seq_data[id - 1][: 600 // 10]`` over the 10 ids, so its input
keeps the single-cycle shape (600, 33). The task's inputs are the 10 whole Records; ``model_input``
rebuilds the authors' input from them.
"""


def _metadata(pair: QAPair) -> dict[str, Any]:
    ref = pair.ref
    metadata: dict[str, Any] = {
        "source_file": ref.part.source_file,
        "source_line": ref.line,
        "qa_offset": ref.qa_offset,
    }
    if pair.stage.is_multi_cycle:
        metadata["steps_per_cycle"] = STEPS_PER_CYCLE
    return metadata


def open_task(pair: QAPair, records: tuple[Record, ...]) -> AnswerTask:
    """Return an open question's task, which the authors score with BLEU and ROUGE against the published answer.

    Understanding (stage 1) and decision-making (stage 4) ask open questions.

    Args:
        pair: The QA pair, whose file gives the split.
        records: The Records of the rows the pair names, in published order.

    Returns:
        The answer task, with the published answer as its target.
    """
    return AnswerTask(
        inputs=records,
        prompt=pair.question,
        targets=(pair.answer,),
        split=pair.ref.split,
        metadata=_metadata(pair),
    )


def closed_task(pair: QAPair, records: tuple[Record, ...]) -> ClassificationTask:
    """Return a closed question's task, whose classes are its option letters, which the authors score as a set.

    Perception (stage 2) and reasoning (stage 3) ask closed questions.

    Args:
        pair: The QA pair, whose file gives the split.
        records: The Records of the rows the pair names, in published order.

    Returns:
        The classification task, with the published letters as targets.

    Raises:
        TimeFValidationError: If the pair is open, or its answer is empty.
    """
    closed, letters = pair.closed, pair.letters
    if closed is None or not letters:
        raise TimeFValidationError(f"{pair.ref.context}: a closed task needs a published answer")
    return ClassificationTask(
        inputs=records,
        prompt=pair.question,
        targets=letters,
        target_schema=vocabulary_key(closed.vocabulary),
        split=pair.ref.split,
        metadata=_metadata(pair) | {"closed_question": closed.value},
    )


def stage_task(pair: QAPair, records: tuple[Record, ...]) -> AnswerTask | ClassificationTask:
    """Return the task of a pair of any stage: closed for perception and reasoning, open otherwise.

    Args:
        pair: The QA pair.
        records: The Records of the rows the pair names, in published order.

    Returns:
        The pair's task.
    """
    return closed_task(pair, records) if pair.stage.is_closed else open_task(pair, records)
