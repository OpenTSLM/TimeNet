import pickle
import shutil

from _task_release import _KINDS, _TEXT, _all, _handles, _streamed
import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.duckdb import connect_control
import timenet.reader
from timenet.reader import Loading, LoadPlan, TimeFReader
from timenet.registry import open_local_version
from timenet.types import ObjectKind, TSCorrespondenceTask


def _loaded(handles: dict[str, list]) -> dict[str, list[bool]]:
    return {kind: [handle.is_loaded for handle in values] for kind, values in handles.items()}


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
    with TimeFReader(open_local_version(release.path)) as reader:
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
    with TimeFReader(open_local_version(release.path)) as reader:
        handles = _everything(reader.read(load=LoadPlan(**{kind: Loading.EAGER})))
    assert all(handles.values())
    loaded = _loaded(handles)
    assert all(loaded[kind])
    assert not any(state for other in _KINDS if other != kind for state in loaded[other])
    # Every value loads once, and signals on one irregular axis share its time offsets.
    assert len(reads) == len(handles[kind])


def test_a_fully_loaded_dataset_needs_no_storage(release, reads):
    with TimeFReader(open_local_version(release.path)) as reader:
        dataset = reader.read(load=LoadPlan.ALL_EAGER)
    loaded = len(reads)
    shutil.rmtree(release.path)
    # One read builds one graph: tasks reference the dataset's own records and annotations.
    note = next(annotation for annotation in dataset.annotations if annotation.key == "note")
    child = release.named(dataset.tasks)["a-child"]
    assert isinstance(child, TSCorrespondenceTask)
    assert child.input_annotations[0] is note
    built = next(record for record in dataset.records if record.id == release.records["r3"])
    assert child.candidate_records[0] is built
    # The dataset annotation that no task references loads with the dataset too.
    unused = next(annotation for annotation in dataset.annotations if annotation.key == "unused")
    assert _handles(unused)["text"]
    assert _all(_everything(dataset), loaded=True)
    for record in dataset.records:
        for signal in record.signals:
            assert signal.to_arrow().to_pylist() == signal.read_steps(0, 3).to_pylist()
        assert record.signals[1].time_offsets_us().tolist() == [0, 2, 5]
        assert record.annotations[0].value == f"{_TEXT}{record.metadata['name']}"
    assert note.value == _TEXT + "dataset"
    assert len(reads) == loaded


def test_a_task_read_loads_what_its_tasks_reach_and_leaves_held_and_parent_objects_alone(release, reads):
    with TimeFReader(open_local_version(release.path)) as reader:
        (held,) = reader.iter_records((release.records["r1"],))
        tasks = reader.read_tasks((held,), load=LoadPlan.ALL_EAGER)
    assert list(release.named(tasks)) == ["a-child", "b-edit", "d-mark"]
    child, edit, marked = tasks
    assert isinstance(child, TSCorrespondenceTask)
    assert child.targets is not None
    assert edit.targets is not None
    assert child.inputs == (held,)
    assert child.inputs[0] is held
    # The records passed in stay as they are, including the Signal that one task targets.
    assert _all(_handles(held), loaded=False)
    assert edit.targets[0] is held.signals[0]
    # The tasks' text, their candidate and target records, and the dataset annotation they reference
    # load; so does a candidate that is no target.
    built, candidate = child.candidate_records
    assert child.targets == (built,)
    assert child.targets[0] is built
    assert candidate.id == release.records["r4"]
    assert _all(_handles(child, edit, built, candidate, *child.input_annotations), loaded=True)
    assert all(_handles(child, edit, built, candidate, *child.input_annotations).values())
    # So does a dataset annotation that a task names as its target.
    (mark,) = marked.target_annotations
    assert _all(_handles(marked, mark), loaded=True)
    assert _handles(mark)["text"]
    # A parent reached only through from_tasks is built but not loaded, and nor is its record.
    (parent,) = child.from_tasks
    assert _all(_handles(parent, *parent.inputs), loaded=False)


def test_a_stream_batch_loads_what_it_reaches_through_an_earlier_batch_built_it_unloaded(release):
    # One task a batch: a-child, b-edit, c-other, d-mark, p-parent.
    tasks = _streamed(release, batch_size=1)
    child, other, parent = tasks["a-child"], tasks["c-other"], tasks["p-parent"]
    (built,) = child.from_tasks
    (through_parent,) = built.inputs
    # The first batch reaches p-parent, its record r2, and its annotation only through from_tasks.
    assert _all(_handles(built, through_parent, *built.input_annotations), loaded=False)
    # The third uses r2 and the annotation directly, so it builds them anew and loads them.
    assert other.inputs[0] is not through_parent
    assert other.inputs[0].id == through_parent.id
    assert other.input_annotations[0] is not built.input_annotations[0]
    assert other.input_annotations[0] == built.input_annotations[0]
    reached = _handles(other, *other.inputs, *other.input_annotations)
    assert all(reached.values())
    assert _all(reached, loaded=True)
    # The last yields p-parent itself, built and loaded once more.
    assert parent is not built
    assert parent.id == built.id
    assert _all(_handles(parent, *parent.inputs), loaded=True)


def test_a_task_stream_loads_its_tasks_and_leaves_the_records_it_is_given_alone(release, reads):
    with TimeFReader(open_local_version(release.path)) as reader:
        (held,) = reader.iter_records((release.records["r1"],))
        tasks = list(reader.iter_tasks((held,), load=LoadPlan.ALL_EAGER, batch_size=1))
    assert list(release.named(tasks)) == ["a-child", "b-edit", "d-mark"]
    # Every batch uses the records it is given, down to the Signal that b-edit targets.
    assert all(task.inputs[0] is held for task in tasks)
    assert tasks[1].targets is not None
    assert tasks[1].targets[0] is held.signals[0]
    assert _all(_handles(*tasks), loaded=True)
    assert _all(_handles(held), loaded=False)
    assert reads


def _occurrence_ids(release, *keys) -> dict[str, int]:
    """Return the occurrence ID of the dataset annotation of each key."""
    with connect_control(release.path / "control.duckdb", read_only=True) as connection:
        rows = connection.execute(
            """SELECT c.name, o.occurrence_id FROM annotation_occurrences o JOIN annotation_contents c USING (content_id)
               WHERE o.object_type = 'Dataset' AND c.name IN (SELECT unnest(?))""",
            [list(keys)],
        ).fetchall()
    return dict(rows)


def test_task_reads_read_only_the_dataset_annotations_their_tasks_reference(release, monkeypatch):
    occurrences = _occurrence_ids(release, "note", "mark", "shared", "unused")
    requested: list[list[int]] = []
    read_annotations = DuckDBControlReader._read_annotations

    def recording(self, owners, by_occurrence=None):
        if ObjectKind.DATASET in owners:
            requested.append(sorted(owners[ObjectKind.DATASET]))
        return read_annotations(self, owners, by_occurrence)

    monkeypatch.setattr(DuckDBControlReader, "_read_annotations", recording)
    with TimeFReader(open_local_version(release.path)) as reader:
        reader.read_tasks()
        one_pass, requested[:] = list(requested), []
        list(reader.iter_tasks(batch_size=1))
    referenced = sorted(occurrences[key] for key in ("note", "mark", "shared"))
    # One read asks for the referenced occurrences together, never for every one of the Dataset's.
    assert one_pass == [referenced]
    # A stream's batches each ask for their own: a-child's batch builds p-parent first, then c-other,
    # d-mark, and p-parent again; b-edit references none.
    assert requested == [[occurrences[key]] for key in ("shared", "note", "shared", "mark", "shared")]
    assert occurrences["unused"] not in {identity for batch in (*one_pass, *requested) for identity in batch}


def test_a_failed_read_returns_nothing_and_leaves_held_objects_unchanged(release):
    version = open_local_version(release.path)
    with TimeFReader(version) as reader:
        held = reader.read()
    (r1,) = (record for record in held.records if record.id == release.records["r1"])
    before = _loaded(_everything(held))
    # Only the values files go, so a read loads its text and time offsets before its arrays fail.
    for part in version.manifest.files.time_series:
        (release.path / part.path).unlink()

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
    with TimeFReader(open_local_version(release.path)) as reader:
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

    monkeypatch.setattr(DuckDBControlReader, "chunk_locations_by_ids", unexpected)
    restored = pickle.loads(pickle.dumps(loaded))
    assert _all(_everything(restored), loaded=True)
    for record in restored.records:
        assert [signal.to_arrow().to_pylist() for signal in record.signals] == [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]
        assert [signal._values._location for signal in record.signals]
        assert record.signals[1].time_offsets_us().tolist() == [0, 2, 5]
    assert release.named(restored.tasks)["a-child"].prompt == _TEXT + "child"
    assert not reads
