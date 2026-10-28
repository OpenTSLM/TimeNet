from collections.abc import Generator
from contextlib import contextmanager
from fractions import Fraction
import gc
from pathlib import Path
import re
import sys
import weakref

from _task_release import _SPEC, _metadata, _streamed, _write
import pyarrow as pa
import pytest

from timenet.dataset import IrregularAxis, Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
import timenet.format.control_reader as control_reader_module
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.duckdb import connect_control
from timenet.format.stored_values import StoredOffsets
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.types import AnswerTask, Split, Task


def test_the_tasks_of_one_stream_batch_share_its_objects(release):
    tasks = _streamed(release)
    child, edit, other, marked, parent = (tasks[i] for i in ("a-child", "b-edit", "c-other", "d-mark", "p-parent"))
    assert isinstance(edit.targets, tuple)
    # One batch holds every task: a record, Signal, annotation, or parent that tasks share is one object.
    assert child.inputs[0] is edit.inputs[0] is marked.inputs[0]
    assert edit.targets[0] is child.inputs[0].signals[0]
    assert child.from_tasks[0] is parent
    assert other.inputs[0] is parent.inputs[0]
    assert other.input_annotations[0] is parent.input_annotations[0]


def _values(record: Record) -> list[list[float]]:
    return [signal.to_numpy().tolist() for signal in record.signals]


def test_stream_batches_build_separate_objects_for_what_they_share(release):
    # Batches of two: a-child and b-edit, then c-other and d-mark, then p-parent.
    tasks = _streamed(release, batch_size=2)
    child, edit, marked, parent = (tasks[i] for i in ("a-child", "b-edit", "d-mark", "p-parent"))
    assert child.inputs[0] is edit.inputs[0]
    # d-mark uses r1 as well, from a batch of its own, so it holds a separate copy of the record.
    (record,) = marked.inputs
    assert record is not child.inputs[0]
    assert record.id == child.inputs[0].id
    assert _values(record) == _values(child.inputs[0]) == [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]
    # The first batch builds the parent of a-child; the last yields a second copy of it.
    (built,) = child.from_tasks
    assert built is not parent
    assert built.id == parent.id
    assert built.prompt == parent.prompt
    # One task read shares them across its batches: the parent it builds for a-child is the one it returns.
    with TimeFReader(open_local_version(release.path)) as reader:
        records = tuple(reader.iter_records())
        read = release.named(reader.read_tasks(records))
    assert read["a-child"].from_tasks[0] is read["p-parent"]


def _reached(task: Task) -> list[object]:
    """Return a task and the parents, records, Signals, and annotations it reaches."""
    tasks = (task, *task.from_tasks)
    records = [
        item
        for each in tasks
        for item in (*each.inputs, *getattr(each, "candidate_records", ()), *(each.targets or ()))
        if isinstance(item, Record)
    ]
    signals = [signal for record in records for signal in record.signals]
    signals.extend(item for item in task.targets or () if isinstance(item, Signal))
    annotations = [annotation for each in tasks for annotation in (*each.input_annotations, *each.target_annotations)]
    return [*tasks, *records, *signals, *annotations]


def test_a_task_stream_holds_nothing_of_a_batch_it_has_yielded(release, monkeypatch):
    refs: list[weakref.ref] = []
    alive_while_building: list[int] = []
    original = DuckDBControlReader._task_rows

    def task_rows(self, task_ids):
        if refs:
            gc.collect()
            alive_while_building.append(sum(ref() is not None for ref in refs))
        return original(self, task_ids)

    monkeypatch.setattr(DuckDBControlReader, "_task_rows", task_rows)
    ids = []
    with TimeFReader(open_local_version(release.path)) as reader:
        stream = reader.iter_tasks(batch_size=1)
        # Each task is dropped before the next batch is built; b-edit's batch walks its record's Signals.
        while (task := next(stream, None)) is not None:
            ids.append(task.id)
            refs.extend(weakref.ref(item) for item in _reached(task))
            del task
    assert ids == list(release.tasks.values())
    # Each later batch is built after the stream let go of every object of the batches before it.
    assert alive_while_building == [0, 0, 0, 0]


def test_a_stream_selects_its_task_ids_one_page_at_a_time(release, monkeypatch):
    pages: list[tuple[int | None, int, int]] = []
    selected = DuckDBControlReader._selected_task_ids

    def recording(self, *args, after, limit):
        ids = selected(self, *args, after=after, limit=limit)
        pages.append((after, limit, len(ids)))
        return ids

    monkeypatch.setattr(DuckDBControlReader, "_selected_task_ids", recording)
    with TimeFReader(open_local_version(release.path)) as reader:
        stream = reader.iter_tasks(batch_size=2)
        first = next(stream)
        # The first batch needs one page of two IDs, not the IDs of every task.
        assert pages == [(None, 2, 2)]
        tasks = [first, *stream]
    ids = list(release.tasks.values())
    assert [task.id for task in tasks] == ids
    assert pages == [(None, 2, 2), (ids[1], 2, 2), (ids[3], 2, 1)]


@pytest.mark.parametrize("batch_size", [0, -1, True, 2.0, "5"])
def test_a_stream_takes_only_a_positive_int_batch_size(release, batch_size):
    with (
        TimeFReader(open_local_version(release.path)) as reader,
        pytest.raises(TimeFValidationError, match="batch_size must be a positive int"),
    ):
        next(reader.iter_tasks(batch_size=batch_size))


def test_signals_share_one_time_offsets_handle_per_axis_per_task_read_and_stream_batch(tmp_path, monkeypatch):
    """Each task takes a selector record and a record on one shared irregular axis."""
    dataset = TimeFDataset(metadata=_metadata())
    axis = IrregularAxis.spanning([0, 2, 5])
    tasks = []
    for _ in range(4):
        regular = Signal(spec=_SPEC, name="I", time_axis=RegularAxis.from_rate_hz(Fraction(500)), data=[0.0])
        selector = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(regular,)),)))
        signal = Signal.from_loader(
            spec=_SPEC,
            name="II",
            time_axis=axis,
            n_values=3,
            loader=lambda: pa.array([1.0, 2.0, 3.0], type=pa.float32()),
            time_offsets_loader=lambda: pa.array([0, 2, 5], type=pa.int64()),
        )
        shared = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(signal,)),)))
        tasks.append(AnswerTask(inputs=(selector, shared), targets=("yes",), split=Split.TRAIN))
    dataset.add_tasks(tasks=tasks)
    monkeypatch.setattr(control_reader_module, "TASK_BATCH_SIZE", 2)
    with TimeFReader(open_local_version(_write(tmp_path, dataset))) as reader:
        selectors = tuple(reader.iter_records(task.inputs[0].id for task in tasks))
        # Both reads run two batches of two tasks, and each batch reads its tasks' shared records.
        by_records = reader.read_tasks(selectors)
        by_split = list(reader.iter_tasks(split=Split.TRAIN, batch_size=2))

    def offsets(task):
        return task.inputs[1].signals[0]._time_offsets

    assert all(isinstance(offsets(task), StoredOffsets) for task in (*by_records, *by_split))
    # One task read shares one handle per axis across its batches.
    assert len({id(offsets(task)) for task in by_records}) == 1
    # A stream shares one within each batch, and each batch has its own.
    first, second, third, fourth = by_split
    assert offsets(first) is offsets(second)
    assert offsets(third) is offsets(fourth)
    assert offsets(first) is not offsets(third)


def _chain_release(tmp_path, length: int) -> tuple[Path, list[int]]:
    """Write ``length`` tasks, each deriving from the one before, and return the release and the task IDs."""
    dataset = TimeFDataset(metadata=_metadata())
    signal = Signal(spec=_SPEC, name="I", time_axis=RegularAxis.from_rate_hz(Fraction(500)), data=[0.0])
    record = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(signal,)),)))
    chain = [AnswerTask(inputs=(record,), targets=("0",))]
    for index in range(1, length):
        chain.append(AnswerTask(inputs=(record,), targets=(str(index),), from_tasks=(chain[-1],)))
    dataset.add_tasks(tasks=chain)
    ids = [task.id for task in chain]
    assert ids == sorted(ids)
    return _write(tmp_path, dataset), ids


def _stack_depth() -> int:
    """Return the number of frames on the stack of the caller."""
    frame, depth = sys._getframe(1), 0
    while frame is not None:
        frame, depth = frame.f_back, depth + 1
    return depth


@contextmanager
def _recursion_limit(limit: int) -> Generator[None]:
    """Set the recursion limit for the block, and restore the old limit after it."""
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(limit)
    try:
        yield
    finally:
        sys.setrecursionlimit(old)


def test_a_later_batch_builds_a_parent_chain_longer_than_the_recursion_limit(tmp_path, monkeypatch):
    length = 300
    path, ids = _chain_release(tmp_path, length)
    builds: list[list[int]] = []
    original = DuckDBControlReader._task_rows

    def task_rows(self, task_ids):
        builds.append(sorted(task_ids))
        return original(self, task_ids)

    monkeypatch.setattr(DuckDBControlReader, "_task_rows", task_rows)
    # The first batch holds every task but the last. The second batch builds the whole chain again,
    # which is longer than the lowered recursion limit allows a recursive build to go.
    with TimeFReader(open_local_version(path)) as reader, _recursion_limit(_stack_depth() + length // 2):
        tasks = list(reader.iter_tasks(batch_size=length - 1))
    assert [task.id for task in tasks] == ids
    # Each batch reads its tasks and the parents they need in one build.
    assert builds == [ids[:-1], ids]
    parents = []
    task = tasks[-1]
    while task.from_tasks:
        (task,) = task.from_tasks
        parents.append(task)
    assert [parent.id for parent in parents] == ids[-2::-1]
    # The second batch built its own copies of the parents.
    by_id = {task.id: task for task in tasks}
    assert all(parent is not by_id[parent.id] for parent in parents)
    assert [parent.targets for parent in parents] == [by_id[parent.id].targets for parent in parents]


@pytest.mark.parametrize(
    "read",
    [
        lambda reader: list(reader.iter_tasks(batch_size=1)),
        lambda reader: list(reader.iter_tasks()),
        lambda reader: reader.read_tasks(),
    ],
    ids=["one-task-a-batch", "one-batch", "whole-table"],
)
def test_a_task_read_rejects_a_stored_derivation_cycle(tmp_path, read):
    path, ids = _chain_release(tmp_path, 3)
    version = open_local_version(path)
    # The first task derives from the last one too, which closes a cycle through all three.
    with connect_control(path / "control.duckdb") as connection:
        connection.execute(
            "INSERT INTO task_dependencies (task_id, position, parent_task_id) VALUES (?, 0, ?)", [ids[0], ids[-1]]
        )
    with (
        TimeFReader(version) as reader,
        pytest.raises(TimeFFormatError, match=re.escape(f"task dependencies contain a cycle through tasks {ids}")),
    ):
        read(reader)
