"""Convert HARESPOD ranking and pairing answers into tasks."""

import json

from timenet.dataset import Record
from timenet.errors import TimeFFormatError
from timenet.types import AnswerTask, Task, TSCorrespondenceTask
from timenet_connectors.datasets.yang_ai_lab.hearts_core.cases import Case
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import ANSWER_KEY
from timenet_connectors.datasets.yang_ai_lab.hearts_core.tasks import task_fields
from timenet_connectors.datasets.yang_ai_lab.hearts_harespod.records import CaseRecords
from timenet_connectors.datasets.yang_ai_lab.hearts_harespod.release import PROMPTS


def convert_case(case: Case, records: CaseRecords) -> Task:
    """Build a HARESPOD task and validate its record references.

    Returns:
        The task, not yet registered.

    Raises:
        TimeFFormatError: If a ranking does not order all input records.
    """
    inputs = tuple(built.record for built in records.inputs)
    shared = task_fields(case, inputs, PROMPTS[case.directory].substitute())
    if case.definition.task_type is TSCorrespondenceTask:
        pool = tuple(built.record for built in records.candidates)
        return TSCorrespondenceTask(candidate_records=pool, targets=_matches(case, inputs, pool), **shared)
    answer = case.payload[ANSWER_KEY]
    names = sorted(record.id.rsplit("-", 1)[1] for record in inputs)
    if sorted(str(label) for label in answer) != names:
        raise TimeFFormatError(f"HEARTS {case.id} answer {answer!r} does not order the records {names}")
    return AnswerTask(targets=(json.dumps([str(label) for label in answer]),), **shared)


def _matches(case: Case, inputs: tuple[Record, ...], pool: tuple[Record, ...]) -> tuple[Record, ...]:
    """Resolve a pairing answer, a map of input suffix to candidate suffix, to candidates in input order.

    Returns:
        The matched candidates.

    Raises:
        TimeFFormatError: If the answer does not pair every input with a candidate of its own.
    """
    answer = case.payload[ANSWER_KEY]
    candidates = {record.id.rsplit("_", 1)[1]: record for record in pool}
    try:
        matched = tuple(candidates[str(answer[record.id.rsplit("_", 1)[1]])] for record in inputs)
    except KeyError as exc:
        raise TimeFFormatError(f"HEARTS {case.id} answer {answer!r} names no candidate for every input") from exc
    if len({record.id for record in matched}) != len(matched):
        raise TimeFFormatError(f"HEARTS {case.id} answer {answer!r} pairs two inputs with one candidate")
    return matched
