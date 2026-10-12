from dataclasses import dataclass, field

import pytest

from timenet.dataset.composition import inherited_state
from timenet.testing import make_dataset


def test_inherited_state_detects_a_change_and_ignores_tasks_and_overlays():
    record = make_dataset().records[0]
    inherited = frozenset(a.occurrence_id for a in record.annotations if a.occurrence_id is not None)
    before = inherited_state(record, inherited)

    record.task_ids = (*record.task_ids, "new-task")
    record.annotations = (*record.annotations, record.annotations[0]._new_occurrence())
    assert inherited_state(record, inherited) == before

    record.sources[0].metadata["changed"] = True
    assert inherited_state(record, inherited) != before


def test_inherited_state_rejects_an_unknown_leaf_type():
    @dataclass
    class Odd:
        id: str = "odd"
        annotations: tuple = ()
        task_ids: tuple = ()
        payload: set = field(default_factory=set)

    with pytest.raises(TypeError, match="cannot snapshot a set"):
        inherited_state(Odd(), frozenset())  # ty: ignore[invalid-argument-type]
