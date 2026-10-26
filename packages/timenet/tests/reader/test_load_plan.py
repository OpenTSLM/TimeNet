from fractions import Fraction
import gc
import pickle
import shutil
import weakref

import pyarrow as pa
import pytest

from timenet.dataset import IrregularAxis, Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
import timenet.format.control_reader as control_reader_module
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.stored_array import StoredArray
from timenet.format.stored_values import PackedValueReader, StoredOffsets, StoredText, StoredTextList
import timenet.reader
from timenet.reader import Loading, LoadPlan, TimeFReader
from timenet.registry import open_local_version
from timenet.types import (
    Annotation,
    AnswerTask,
    DatasetMetadata,
    Domain,
    License,
    Split,
    Task,
    TimeSeriesSpec,
    TSCorrespondenceTask,
    TSEditingTask,
    Version,
    ureg,
)
from timenet.writer import TimeFWriter


_TEXT = "A detailed clinical report. " * 300
_KINDS = ("signal_values", "time_offsets", "text")
_SPEC = TimeSeriesSpec(spec_type="ecg", name="lead", unit_value=ureg.millivolt)


def _record(record_id: str) -> Record:
    signals = (
        Signal(
            spec=_SPEC,
            name="I",
            time_axis=RegularAxis.from_rate_hz(Fraction(500)),
            data=[0.0, 1.0, 2.0],
            id=f"{record_id}-I",
        ),
        Signal.from_irregular([1.0, 2.0, 3.0], time_offsets_us=[0, 2, 5], spec=_SPEC, name="II", id=f"{record_id}-II"),
    )
    record = Record(record_id=record_id, sources=(Source(id=f"{record_id}-source", name="Source", signals=signals),))
    record.annotate(Annotation(id=f"{record_id}-report", key="report", value=_TEXT + record_id))
    signals[0].annotate(Annotation(id=f"{record_id}-labels", key="labels", value=[_TEXT, record_id]))
    return record


def _write(tmp_path, dataset: TimeFDataset):
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _metadata() -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id="test/plans",
        dataset_version=Version(1, 0, 0),
        name="Plans",
        description="Packed text, irregular axes, and tasks that reach records in every way.",
        license=License.CC_BY_4_0,
        domains=(Domain.GENERAL,),
    )


@pytest.fixture
def release(tmp_path):
    """Write a release whose tasks reach records as inputs, candidates, targets, and parents.

    ``a-child``, ``b-edit``, and ``d-mark`` take record ``r1`` as input. ``a-child`` also takes
    ``r3`` as its candidate and target, references a dataset annotation, and derives from
    ``p-parent``, which with ``c-other`` takes ``r2`` and references a third dataset annotation.
    ``b-edit`` targets a Signal of ``r1``, and ``d-mark`` targets another dataset annotation.
    Every text is long enough to be packed.
    """
    dataset = TimeFDataset(metadata=_metadata())
    held, built, parent_only = (dataset.add_record(record=_record(name)) for name in ("r1", "r3", "r2"))
    note = dataset.annotate(Annotation(id="note", key="note", value=_TEXT + "dataset"))
    mark = dataset.annotate(Annotation(id="mark", key="mark", value=_TEXT + "mark"))
    shared = dataset.annotate(Annotation(id="shared", key="shared", value=_TEXT + "shared"))
    parent = AnswerTask(
        id="p-parent", inputs=(parent_only,), prompt=_TEXT + "parent", targets=("yes",), input_annotations=(shared,)
    )
    other = AnswerTask(
        id="c-other", inputs=(parent_only,), prompt=_TEXT + "other", targets=("no",), input_annotations=(shared,)
    )
    marked = AnswerTask(id="d-mark", inputs=(held,), prompt=_TEXT + "mark", target_annotations=(mark,))
    child = TSCorrespondenceTask(
        id="a-child",
        inputs=(held,),
        candidate_records=(built,),
        targets=(built,),
        prompt=_TEXT + "child",
        rationale=_TEXT + "why",
        input_annotations=(note,),
        from_tasks=(parent,),
    )
    child.annotate(Annotation(id="child-note", key="note", value=_TEXT + "task"))
    edit = TSEditingTask(id="b-edit", inputs=(held,), targets=(held.signals[0], _TEXT + "target"), prompt=_TEXT)
    dataset.add_tasks(tasks=[parent, child, edit, other, marked])
    return _write(tmp_path, dataset)


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


def _handles(*objects) -> dict[str, list]:
    """Return the stored value handles on records, signals, annotations, and tasks, by plan field."""
    found: dict[str, dict[int, object]] = {kind: {} for kind in _KINDS}

    def add(handle):
        for kind, types in zip(_KINDS, (StoredArray, StoredOffsets, StoredText | StoredTextList), strict=True):
            if isinstance(handle, types):
                found[kind][id(handle)] = handle

    def signal(item):
        add(item._values)
        add(item._time_offsets)
        for annotation in item.annotations:
            add(annotation.value)

    for item in objects:
        if isinstance(item, Record):
            for annotation in item.annotations:
                add(annotation.value)
            for source in item.walk_sources():
                for annotation in source.annotations:
                    add(annotation.value)
                for each in source.signals:
                    signal(each)
        elif isinstance(item, Signal):
            signal(item)
        elif isinstance(item, Annotation):
            add(item.value)
        else:
            for value in (item.prompt, item.rationale, *(item.targets or ())):
                add(value)
            for annotation in item.annotations:
                add(annotation.value)
    return {kind: list(handles.values()) for kind, handles in found.items()}


def _loaded(handles: dict[str, list]) -> dict[str, list[bool]]:
    return {kind: [handle.is_loaded for handle in values] for kind, values in handles.items()}


def _all(handles, *, loaded: bool) -> bool:
    return all(handle.is_loaded is loaded for values in handles.values() for handle in values)


def _everything(dataset: TimeFDataset) -> dict[str, list]:
    return _handles(*dataset.records, *dataset.tasks, *dataset.annotations)


def test_the_reader_package_exports_the_plan():
    assert {"LoadPlan", "Loading", "TimeFReader"} <= set(timenet.reader.__all__)


def test_a_plan_defaults_to_lazy_and_names_the_two_complete_plans():
    assert LoadPlan() == LoadPlan.ALL_LAZY
    assert all(getattr(LoadPlan.ALL_LAZY, kind) is Loading.LAZY for kind in _KINDS)
    assert all(getattr(LoadPlan.ALL_EAGER, kind) is Loading.EAGER for kind in _KINDS)


@pytest.mark.parametrize("mode", ["eager", "lazy", True, None])
def test_a_plan_is_frozen_keyword_only_and_takes_only_loading_modes(mode):
    plan = LoadPlan(text=Loading.EAGER)
    with pytest.raises(AttributeError):
        plan.text = Loading.LAZY  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        LoadPlan(Loading.EAGER)  # ty: ignore[too-many-positional-arguments]
    with pytest.raises(TimeFValidationError, match=r"LoadPlan.signal_values takes Loading.LAZY or Loading.EAGER"):
        LoadPlan(signal_values=mode)


def test_reads_load_nothing_without_a_plan(release, reads):
    with TimeFReader(open_local_version(release)) as reader:
        dataset = reader.read()
        tasks = reader.read_tasks()
        streamed = list(reader.iter_tasks())
        handles = _everything(dataset)
        assert all(handles.values())
        assert _all(handles, loaded=False)
        assert _all(_handles(*tasks, *(record for task in tasks for record in task.inputs)), loaded=False)
        assert _all(_handles(*streamed), loaded=False)
    assert not reads


@pytest.mark.parametrize("kind", _KINDS)
def test_a_plan_loads_only_the_kinds_it_names(release, reads, kind):
    with TimeFReader(open_local_version(release)) as reader:
        handles = _everything(reader.read(load=LoadPlan(**{kind: Loading.EAGER})))
    assert all(handles.values())
    loaded = _loaded(handles)
    assert all(loaded[kind])
    assert not any(state for other in _KINDS if other != kind for state in loaded[other])
    # Every value loads once, and signals on one irregular axis share its time offsets.
    assert len(reads) == len(handles[kind])


def test_a_fully_loaded_dataset_needs_no_storage(release, reads):
    with TimeFReader(open_local_version(release)) as reader:
        dataset = reader.read(load=LoadPlan.ALL_EAGER)
    loaded = len(reads)
    shutil.rmtree(release)
    # One read builds one graph: tasks reference the dataset's own records and annotations.
    note = next(annotation for annotation in dataset.annotations if annotation.id == "note")
    child = next(task for task in dataset.tasks if task.id == "a-child")
    assert isinstance(child, TSCorrespondenceTask)
    assert child.input_annotations[0] is note
    assert child.candidate_records[0] is next(record for record in dataset.records if record.id == "r3")
    assert _all(_everything(dataset), loaded=True)
    for record in dataset.records:
        for signal in record.signals:
            assert signal.to_arrow().to_pylist() == signal.read_steps(0, 3).to_pylist()
        assert record.signals[1].time_offsets_us().tolist() == [0, 2, 5]
        assert record.annotations[0].value == _TEXT + record.id
    assert note.value == _TEXT + "dataset"
    assert len(reads) == loaded


def test_a_task_read_loads_what_its_tasks_reach_and_leaves_held_and_parent_objects_alone(release, reads):
    with TimeFReader(open_local_version(release)) as reader:
        (held,) = reader.iter_records(("r1",))
        child, edit, marked = reader.read_tasks((held,), load=LoadPlan.ALL_EAGER)
    assert (child.id, edit.id, marked.id) == ("a-child", "b-edit", "d-mark")
    assert isinstance(child, TSCorrespondenceTask)
    assert child.targets is not None
    assert edit.targets is not None
    assert child.inputs == (held,)
    assert child.inputs[0] is held
    # The records passed in stay as they are, including the Signal that one task targets.
    assert _all(_handles(held), loaded=False)
    assert edit.targets[0] is held.signals[0]
    # The tasks' text, their candidate and target record, and the dataset annotation they reference load.
    (built,) = child.candidate_records
    assert child.targets == (built,)
    assert child.targets[0] is built
    assert _all(_handles(child, edit, built, *child.input_annotations), loaded=True)
    assert all(_handles(child, edit, built, *child.input_annotations).values())
    # So does a dataset annotation that a task names as its target.
    (mark,) = marked.target_annotations
    assert _all(_handles(marked, mark), loaded=True)
    assert _handles(mark)["text"]
    # A parent reached only through from_tasks is built but not loaded, and nor is its record.
    (parent,) = child.from_tasks
    assert _all(_handles(parent, *parent.inputs), loaded=False)


def _streamed(release, monkeypatch, batch_size: int | None = None) -> dict[str, Task]:
    """Stream every task of the release with every value loaded, ``batch_size`` tasks a batch."""
    if batch_size is not None:
        monkeypatch.setattr(control_reader_module, "_TASK_BATCH_SIZE", batch_size)
    with TimeFReader(open_local_version(release)) as reader:
        return {task.id: task for task in reader.iter_tasks(load=LoadPlan.ALL_EAGER)}


def test_the_tasks_of_one_stream_batch_share_its_objects(release, monkeypatch):
    tasks = _streamed(release, monkeypatch)
    child, edit, other, marked, parent = (tasks[i] for i in ("a-child", "b-edit", "c-other", "d-mark", "p-parent"))
    assert isinstance(edit.targets, tuple)
    # One batch holds every task: a record, Signal, annotation, or parent that tasks share is one object.
    assert child.inputs[0] is edit.inputs[0] is marked.inputs[0]
    assert edit.targets[0] is child.inputs[0].signals[0]
    assert child.from_tasks[0] is parent
    assert other.inputs[0] is parent.inputs[0]
    assert other.input_annotations[0] is parent.input_annotations[0]
    # The parent is a task of the batch, so it loads with its record and annotation.
    reached = _handles(*tasks.values(), *(record for task in tasks.values() for record in task.inputs))
    assert _all(reached, loaded=True)
    assert _all(_handles(*other.input_annotations), loaded=True)


def _values(record: Record) -> list[list[float]]:
    return [signal.to_numpy().tolist() for signal in record.signals]


def test_stream_batches_build_separate_objects_for_what_they_share(release, monkeypatch):
    # Batches of two: a-child and b-edit, then c-other and d-mark, then p-parent.
    tasks = _streamed(release, monkeypatch, batch_size=2)
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
    with TimeFReader(open_local_version(release)) as reader:
        records = tuple(reader.iter_records())
        read = {task.id: task for task in reader.read_tasks(records, load=LoadPlan.ALL_EAGER)}
    assert read["a-child"].from_tasks[0] is read["p-parent"]


def test_a_stream_batch_loads_what_it_reaches_through_an_earlier_batch_built_it_unloaded(release, monkeypatch):
    # One task a batch: a-child, b-edit, c-other, d-mark, p-parent.
    tasks = _streamed(release, monkeypatch, batch_size=1)
    child, other, parent = tasks["a-child"], tasks["c-other"], tasks["p-parent"]
    (built,) = child.from_tasks
    (through_parent,) = built.inputs
    # The first batch reaches p-parent, its record r2, and its annotation only through from_tasks.
    assert _all(_handles(built, through_parent, *built.input_annotations), loaded=False)
    # The third uses r2 and the annotation directly, so it builds them anew and loads them.
    assert other.inputs[0] is not through_parent
    assert other.inputs[0].id == through_parent.id
    assert other.input_annotations[0] is not built.input_annotations[0]
    assert other.input_annotations[0].occurrence_id == built.input_annotations[0].occurrence_id
    reached = _handles(other, *other.inputs, *other.input_annotations)
    assert all(reached.values())
    assert _all(reached, loaded=True)
    # The last yields p-parent itself, built and loaded once more.
    assert parent is not built
    assert parent.id == built.id
    assert _all(_handles(parent, *parent.inputs), loaded=True)


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
    monkeypatch.setattr(control_reader_module, "_TASK_BATCH_SIZE", 1)
    refs: list[weakref.ref] = []
    alive_while_building: list[int] = []
    original = DuckDBControlReader._task_rows

    def task_rows(self, task_keys):
        if refs:
            gc.collect()
            alive_while_building.append(sum(ref() is not None for ref in refs))
        return original(self, task_keys)

    monkeypatch.setattr(DuckDBControlReader, "_task_rows", task_rows)
    ids = []
    with TimeFReader(open_local_version(release)) as reader:
        stream = reader.iter_tasks(load=LoadPlan.ALL_EAGER)
        # Each task is dropped before the next batch is built; b-edit's batch walks its record's Signals.
        while (task := next(stream, None)) is not None:
            ids.append(task.id)
            refs.extend(weakref.ref(item) for item in _reached(task))
            del task
    assert ids == ["a-child", "b-edit", "c-other", "d-mark", "p-parent"]
    # Each later batch is built after the stream let go of every object of the batches before it.
    assert alive_while_building == [0, 0, 0, 0]


def test_a_task_stream_loads_its_tasks_and_leaves_the_records_it_is_given_alone(release, reads, monkeypatch):
    monkeypatch.setattr(control_reader_module, "_TASK_BATCH_SIZE", 1)
    with TimeFReader(open_local_version(release)) as reader:
        (held,) = reader.iter_records(("r1",))
        tasks = list(reader.iter_tasks((held,), load=LoadPlan.ALL_EAGER))
    assert [task.id for task in tasks] == ["a-child", "b-edit", "d-mark"]
    # Every batch uses the records it is given, down to the Signal that b-edit targets.
    assert all(task.inputs[0] is held for task in tasks)
    assert tasks[1].targets is not None
    assert tasks[1].targets[0] is held.signals[0]
    assert _all(_handles(*tasks), loaded=True)
    assert _all(_handles(held), loaded=False)
    assert reads


def test_signals_share_one_time_offsets_handle_per_axis_per_task_read_and_stream_batch(tmp_path, monkeypatch, reads):
    """Each task takes a selector record and a record on one shared irregular axis."""
    dataset = TimeFDataset(metadata=_metadata())
    axis = IrregularAxis.spanning([0, 2, 5])
    tasks = []
    for name in ("1", "2", "3", "4"):
        regular = Signal(spec=_SPEC, name="I", time_axis=RegularAxis.from_rate_hz(Fraction(500)), data=[0.0])
        selector = dataset.add_record(
            record=Record(record_id=f"selector-{name}", sources=(Source(name="Source", signals=(regular,)),))
        )
        signal = Signal.from_loader(
            spec=_SPEC,
            name="II",
            id=f"shared-{name}-II",
            time_axis=axis,
            n_values=3,
            loader=lambda: pa.array([1.0, 2.0, 3.0], type=pa.float32()),
            time_offsets_loader=lambda: pa.array([0, 2, 5], type=pa.int64()),
        )
        shared = Record(record_id=f"shared-{name}", sources=(Source(name="Source", signals=(signal,)),))
        dataset.add_record(record=shared)
        tasks.append(AnswerTask(id=f"q-{name}", inputs=(selector, shared), targets=("yes",), split=Split.TRAIN))
    dataset.add_tasks(tasks=tasks)
    monkeypatch.setattr(control_reader_module, "_TASK_BATCH_SIZE", 2)
    plan = LoadPlan(time_offsets=Loading.EAGER)
    with TimeFReader(open_local_version(_write(tmp_path, dataset))) as reader:
        selectors = tuple(reader.iter_records(tuple(f"selector-{name}" for name in "1234")))
        reads.clear()
        # Both reads run two batches of two tasks, and each batch reads its tasks' shared records.
        by_records = reader.read_tasks(selectors, load=plan)
        by_split = list(reader.iter_tasks(split=Split.TRAIN, load=plan))

    def offsets(task):
        return task.inputs[1].signals[0]._time_offsets

    assert all(
        isinstance(offsets(task), StoredOffsets) and offsets(task).is_loaded for task in (*by_records, *by_split)
    )
    # One task read shares one handle per axis across its batches.
    assert len({id(offsets(task)) for task in by_records}) == 1
    # A stream shares one within each batch, and each batch has its own.
    first, second, third, fourth = by_split
    assert offsets(first) is offsets(second)
    assert offsets(third) is offsets(fourth)
    assert offsets(first) is not offsets(third)
    # The task read loads the shared axis once, and the stream once per batch.
    assert len(reads) == 3


def test_a_failed_read_returns_nothing_and_leaves_held_objects_unchanged(release):
    version = open_local_version(release)
    with TimeFReader(version) as reader:
        held = reader.read()
    (r1,) = (record for record in held.records if record.id == "r1")
    before = _loaded(_everything(held))
    # Only the values files go, so a read loads its text and time offsets before its arrays fail.
    for part in version.manifest.files.time_series:
        (release / part.path).unlink()

    with TimeFReader(version) as reader:
        with pytest.raises(TimeFFormatError):
            reader.read(load=LoadPlan.ALL_EAGER)
        # The tasks of r1 also use r3, which the read builds and fails to load.
        with pytest.raises(TimeFFormatError):
            reader.read_tasks((r1,), load=LoadPlan.ALL_EAGER)
        with pytest.raises(TimeFFormatError):
            list(reader.iter_tasks((r1,), load=LoadPlan.ALL_EAGER))
        # A lazy read opens no payload, so it still succeeds.
        assert reader.read().records
    assert _loaded(_everything(held)) == before
    assert not any(state for states in before.values() for state in states)


def test_loaded_handles_pickle_their_values_and_unpickle_without_storage(release, reads, monkeypatch):
    with TimeFReader(open_local_version(release)) as reader:
        lazy = reader.read()
        loaded = reader.read(load=LoadPlan.ALL_EAGER)
    lazy_handles, loaded_handles = _everything(lazy), _everything(loaded)
    for kind in _KINDS:
        for unloaded, filled in zip(lazy_handles[kind], loaded_handles[kind], strict=True):
            # A handle pickles its location; a loaded one pickles its value too.
            assert len(pickle.dumps(unloaded)) < len(pickle.dumps(filled))
    reads.clear()

    def unexpected(*args):
        pytest.fail("a loaded handle needs no catalog query")

    monkeypatch.setattr(DuckDBControlReader, "chunk_locations_by_keys", unexpected)
    restored = pickle.loads(pickle.dumps(loaded))
    assert _all(_everything(restored), loaded=True)
    for record in restored.records:
        assert [signal.to_arrow().to_pylist() for signal in record.signals] == [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]
        assert [signal._values._location for signal in record.signals]
        assert record.signals[1].time_offsets_us().tolist() == [0, 2, 5]
    assert next(task for task in restored.tasks if task.id == "a-child").prompt == _TEXT + "child"
    assert not reads
