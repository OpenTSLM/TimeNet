"""Build HARESPOD's ranking windows and pairing candidates."""

from typing import NamedTuple

from timenet_connectors.datasets.yang_ai_lab.hearts_core.cases import Case
from timenet_connectors.datasets.yang_ai_lab.hearts_core.records import Built, FrameLayout, frame_record
from timenet_connectors.datasets.yang_ai_lab.hearts_harespod.release import CANDIDATE_KEYS, COLUMN_SPECS


_LAYOUT = FrameLayout(COLUMN_SPECS, ("timestamp",), ("timestamp_min",))


class CaseRecords(NamedTuple):
    """The input windows and pairing candidates of a HARESPOD case."""

    inputs: tuple[Built, ...]
    candidates: tuple[Built, ...]


def case_records(case: Case) -> CaseRecords:
    """Expand the case's input and candidate frame dictionaries.

    Returns:
        Input and candidate records in key order.
    """
    input_key = str(case.definition.inputs[0])
    candidate_key = CANDIDATE_KEYS.get(case.task)
    return CaseRecords(_records(case, input_key), () if candidate_key is None else _records(case, candidate_key))


def _records(case: Case, key: str) -> tuple[Built, ...]:
    """Build the named windows in one frame dictionary.

    Returns:
        Records with their source origins.
    """
    return tuple(
        frame_record(case, name, {(key, name): frame}, _LAYOUT, subject=case.payload.get("subject_id"))
        for name, frame in sorted(case.payload[key].items())
    )
