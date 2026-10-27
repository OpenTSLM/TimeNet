from fractions import Fraction
from pathlib import Path
import sys
from typing import NamedTuple

import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.format.stored_values import PackedValueReader
from timenet.reader import TimeFReader
from timenet.types import Annotation, AnswerTask, Task, TSCorrespondenceTask, TSEditingTask


# Make the sibling helper module ``_task_release`` importable by name under pytest's importlib
# import mode, which does not add test dirs to ``sys.path``.
sys.path.insert(0, str(Path(__file__).parent))

from _task_release import _SPEC, _TEXT, _metadata, _write


def _record(name: str) -> Record:
    """Return a Record whose irregular Signal sits in a Source nested below the first one's Source."""
    signals = (
        Signal(spec=_SPEC, name="I", time_axis=RegularAxis.from_rate_hz(Fraction(500)), data=[0.0, 1.0, 2.0]),
        Signal.from_irregular([1.0, 2.0, 3.0], time_offsets_us=[0, 2, 5], spec=_SPEC, name="II"),
    )
    nested = Source(name="Nested", signals=signals[1:])
    record = Record(sources=(Source(name="Source", signals=signals[:1], sources=(nested,)),), metadata={"name": name})
    record.annotate(Annotation(key="report", value=_TEXT + name))
    signals[0].annotate(Annotation(key="labels", value=[_TEXT, name]))
    return record


class _Release(NamedTuple):
    path: Path
    records: dict[str, int]
    """Record IDs by name."""
    tasks: dict[str, int]
    """Task IDs by name, in ID order."""

    def named(self, tasks) -> dict[str, Task]:
        """Return tasks by name, in the order given."""
        names = {task_id: name for name, task_id in self.tasks.items()}
        return {names[task.id]: task for task in tasks}


@pytest.fixture
def release(tmp_path) -> _Release:
    """Write a release whose tasks reach records as inputs, candidates, targets, and parents.

    ``a-child``, ``b-edit``, and ``d-mark`` take record ``r1`` as input. ``a-child`` also takes
    ``r3`` as its candidate and target and ``r4`` as a candidate only, references a dataset
    annotation, and derives from ``p-parent``, which with ``c-other`` takes ``r2`` and references a
    third dataset annotation. ``b-edit`` targets a Signal of ``r1``, and ``d-mark`` targets another
    dataset annotation. No task references the dataset annotation ``unused``.
    The tasks' IDs follow their names, so ``p-parent`` sorts after the task that derives from it.
    Every text is long enough to be packed.
    """
    dataset = TimeFDataset(metadata=_metadata())
    records = {name: dataset.add_record(record=_record(name)) for name in ("r1", "r3", "r2", "r4")}
    held, built, parent_only = records["r1"], records["r3"], records["r2"]
    note = dataset.annotate(Annotation(key="note", value=_TEXT + "dataset"))
    dataset.annotate(Annotation(key="unused", value=_TEXT + "unused"))
    mark = dataset.annotate(Annotation(key="mark", value=_TEXT + "mark"))
    shared = dataset.annotate(Annotation(key="shared", value=_TEXT + "shared"))
    parent = AnswerTask(inputs=(parent_only,), prompt=_TEXT + "parent", targets=("yes",), input_annotations=(shared,))
    other = AnswerTask(inputs=(parent_only,), prompt=_TEXT + "other", targets=("no",), input_annotations=(shared,))
    marked = AnswerTask(inputs=(held,), prompt=_TEXT + "mark", target_annotations=(mark,))
    child = TSCorrespondenceTask(
        inputs=(held,),
        candidate_records=(built, records["r4"]),
        targets=(built,),
        prompt=_TEXT + "child",
        rationale=_TEXT + "why",
        input_annotations=(note,),
        from_tasks=(parent,),
    )
    child.annotate(Annotation(key="note", value=_TEXT + "task"))
    edit = TSEditingTask(inputs=(held,), targets=(held.signals[0], _TEXT + "target"), prompt=_TEXT)
    tasks = {"a-child": child, "b-edit": edit, "c-other": other, "d-mark": marked, "p-parent": parent}
    dataset.add_tasks(tasks=tasks.values())
    release = _Release(
        _write(tmp_path, dataset),
        {name: record.id for name, record in records.items()},
        {name: task.id for name, task in tasks.items()},
    )
    assert list(release.tasks.values()) == sorted(release.tasks.values())
    return release


@pytest.fixture
def reads(monkeypatch):
    """Record every payload and array read."""
    reads = []
    for owner, name in ((PackedValueReader, "read_scalar"), (TimeFReader, "_read_array")):
        original = getattr(owner, name)

        def counting(self, location, *args, original=original):
            reads.append(location)
            return original(self, location, *args)

        monkeypatch.setattr(owner, name, counting)
    return reads
