from fractions import Fraction

import pyarrow as pa
import pytest

from timenet.composition import BuildContext
from timenet.dataset import IrregularAxis, Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.duckdb import connect_control
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.testing import CountingLoader
from timenet.types import (
    Annotation,
    AnswerTask,
    DatasetMetadata,
    DatasetRef,
    License,
    TimeSeriesSpec,
    Version,
)
from timenet.writer import TimeFWriter


def _parent(dataset_id="first/recordings", version=None, value=1, irregular=False):
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id=dataset_id,
            dataset_version=Version(1, 0, 0) if version is None else version,
            name="Recordings",
            description="Test parent",
            license=License.MIT,
        )
    )
    loader = CountingLoader([value] * 3)
    axis = RegularAxis(period_us=Fraction(1_000_000), axis_id="axis")
    spec = TimeSeriesSpec(spec_type="test", name="Test", unit_value=None)
    signals = [
        Signal.from_loader(
            id=identifier,
            name=identifier,
            spec=spec,
            time_axis=axis,
            loader=loader,
            n_values=3,
            source_id="raw-recording" if index else None,
            annotations=(Annotation(id="label", occurrence_id=f"signal-label-{index}", key="label", value=value),),
            metadata={"raw_id": "record-001", "nested": [value]},
        )
        for index, identifier in enumerate(("signal-a", "signal-b"))
    ]
    if irregular:
        signals.append(
            Signal.from_loader(
                id="signal-c",
                name="signal-c",
                spec=spec,
                time_axis=IrregularAxis(first_us=0, last_us=2_000_000, axis_id="irregular-axis"),
                loader=loader,
                time_offsets_loader=lambda: pa.array([0, 1_000_000, 2_000_000], type=pa.int64()),
                n_values=3,
            )
        )
    record = dataset.add_record(
        record=Record(
            record_id="record-001",
            task_ids=("parent-task",),
            sources=(
                Source(id="root", name="root", sources=(Source(id="source", name="source", signals=tuple(signals)),)),
            ),
            annotations=(
                Annotation(
                    id="subjects",
                    occurrence_id="record-subjects",
                    key="subject_ids",
                    value=["subject-001"],
                    metadata={"description": "metadata", "nested": [value]},
                ),
            ),
            metadata={"raw_id": "record-001", "nested": [value]},
        )
    )
    return dataset, record, loader


def _child(parents, dataset_id="test/child"):
    return TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id=dataset_id,
            dataset_version=Version(1, 0, 0),
            name="Child",
            description="Test child",
            license=License.MIT,
            parents=tuple(
                DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
                for parent in parents
            ),
        )
    )


def _store(root, dataset, backend="parquet"):
    dataset.derive_schema()
    with TimeFWriter(root, dataset, values_backend=backend) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _forbid_value_loads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("imported structure must not load values or offsets")

    monkeypatch.setattr(TimeFReader, "_load_signal", forbidden)
    monkeypatch.setattr(TimeFReader, "_load_offsets", forbidden)


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_identical_parent_ids_remain_separate_in_memory_and_on_disk(tmp_path, backend):
    parents = [_parent(value=1)[0], _parent("second/recordings", value=2)[0]]
    for parent in parents:
        _store(tmp_path, parent, backend)
    child = _child(parents)
    native = child.add_record(record=Record(record_id="record-001", sources=(Source(id="root", name="native"),)))
    with BuildContext.open(child.metadata, LocalRegistry(tmp_path)) as context:
        for parent in context.parents.values():
            assert next(parent.iter_records()).id == "record-001"
            (imported,) = parent.import_records(child, ["record-001"])
            assert imported.id == f"{parent.reference}::record-001"
            assert imported.sources[0].id == f"{parent.reference}::root"
            assert imported.sources[0].sources[0].id == f"{parent.reference}::source"
            assert imported.signals[0].time_axis is imported.signals[1].time_axis
            assert imported.signals[0].time_axis.axis_id == f"{parent.reference}::axis"
            assert imported.signals[0].source_id is None
            assert imported.signals[1].source_id == "raw-recording"
            assert imported.annotations[0].id == f"{parent.reference}::subjects"
            assert imported.annotations[0].occurrence_id == f"{parent.reference}::record-subjects"
            assert imported.annotations[0].value == ["subject-001"]
            assert imported.metadata["raw_id"] == "record-001"
            child.add_task(
                task=AnswerTask(
                    id=parent.dataset_id,
                    inputs=(imported,),
                    input_annotations=(imported.annotations[0],),
                    targets=(imported.signals[0],),
                )
            )
        context.verify_imports(child)
        child.set_dependencies(context.dependency_lock())
        directory = _store(tmp_path, child, backend)
    assert native.id == "record-001"
    with connect_control(directory / "control.duckdb", read_only=True) as connection:
        assert connection.execute("SELECT parent_record_id FROM record_imports").fetchall() == [
            ("record-001",),
            ("record-001",),
        ]
        assert connection.execute("SELECT count(*) FROM signals").fetchone() == (0,)
    with LocalRegistry(tmp_path).open_reader("test/child") as reader:
        ids = [f"{parent.metadata.dataset_id}@1.0.0::record-001" for parent in reversed(parents)]
        records = tuple(reader.iter_records(ids))
        assert [record.id for record in records] == ids
        assert [record.signals[0].to_arrow().to_pylist() for record in records] == [[2.0] * 3, [1.0] * 3]
        assert reader._manifest.counts.signal_chunks == 0
        assert all(part.path == "time_series.zarr/zarr.json" for part in reader._manifest.files.time_series)
        assert reader._manifest.counts.axes == 2
        assert reader.task_table((ids[0],))["task_id"].to_pylist() == ["second/recordings"]
        tasks = tuple(reader.iter_tasks(records))
        assert {task.inputs[0].id for task in tasks} == set(ids)
        for task in tasks:
            assert task.targets is not None
            assert task.targets[0] is task.inputs[0].signals[0]
            assert task.input_annotations[0] is task.inputs[0].annotations[0]


def test_imports_stay_lazy_and_reject_repeated_or_unqualified_records(tmp_path, monkeypatch):
    parent, _, _ = _parent(irregular=True)
    _store(tmp_path, parent)
    child = _child((parent,))
    _forbid_value_loads(monkeypatch)
    with BuildContext.open(child.metadata, LocalRegistry(tmp_path)) as context:
        view = context.parent(parent.metadata.dataset_id)
        (imported,) = view.import_records(child)
        assert imported.task_ids == ()
        assert isinstance(imported.signals[2].time_axis, IrregularAxis)
        assert imported.signals[2].time_axis.axis_id == f"{view.reference}::irregular-axis"
        with pytest.raises(TimeFValidationError, match="already registered"):
            view.import_records(child)
        with pytest.raises(TimeFValidationError, match="not qualified"):
            child.import_record(next(view.iter_records()), parent=view.dataset_id)
        child.set_dependencies(context.dependency_lock())
        _store(tmp_path, child)
    imported.annotate(Annotation(key="reviewed", value=True))
    child.check_records()
    nested = imported.metadata["nested"]
    assert isinstance(nested, list)
    nested.append(2)
    with pytest.raises(TimeFValidationError, match="parent-owned"):
        child.check_records()


def test_different_transitive_versions_with_identical_ids_round_trip(tmp_path):
    branches = []
    for major, name in ((1, "left"), (2, "right")):
        parent, _, _ = _parent(version=Version(major, 0, 0), value=major)
        _store(tmp_path, parent)
        branch = _child((parent,), dataset_id=f"test/{name}")
        with BuildContext.open(branch.metadata, LocalRegistry(tmp_path)) as context:
            context.parent(parent.metadata.dataset_id).import_records(branch)
            branch.set_dependencies(context.dependency_lock())
            _store(tmp_path, branch)
        branches.append(branch)
    child = _child(branches)
    with BuildContext.open(child.metadata, LocalRegistry(tmp_path)) as context:
        for view in context.parents.values():
            view.import_records(child)
        child.set_dependencies(context.dependency_lock())
        _store(tmp_path, child)
    with LocalRegistry(tmp_path).open_reader("test/child") as reader:
        records = tuple(reader.iter_records())
        assert [record.id for record in records] == [
            "test/left@1.0.0::first/recordings@1.0.0::record-001",
            "test/right@1.0.0::first/recordings@2.0.0::record-001",
        ]
        assert [record.signals[0].to_arrow().to_pylist() for record in records] == [[1.0] * 3, [2.0] * 3]


def test_nested_reads_attach_each_layer_overlay(tmp_path):
    parent, _, _ = _parent()
    _store(tmp_path, parent)
    current = parent
    for dataset_id in ("test/middle", "test/child"):
        child = _child((current,), dataset_id=dataset_id)
        with BuildContext.open(child.metadata, LocalRegistry(tmp_path)) as context:
            (record,) = context.parent(current.metadata.dataset_id).import_records(child)
            record.annotate(Annotation(key="layer", value=dataset_id))
            child.set_dependencies(context.dependency_lock())
            _store(tmp_path, child)
        current = child
    with LocalRegistry(tmp_path).open_reader("test/child") as reader:
        restored = reader.read()
        record = restored.records[0]
        assert record.id == "test/middle@1.0.0::first/recordings@1.0.0::record-001"
        assert record.signals[0].id == "test/middle@1.0.0::first/recordings@1.0.0::signal-a"
        assert record.annotations[0].occurrence_id == "test/middle@1.0.0::first/recordings@1.0.0::record-subjects"
        assert [annotation.value for annotation in record.annotations if annotation.key == "layer"] == [
            "test/middle",
            "test/child",
        ]
        restored.check_records()
        assert record.signals[0].to_arrow().to_pylist() == [1.0] * 3


@pytest.mark.parametrize(
    "field",
    [
        "record_metadata",
        "annotation_value",
        "annotation_metadata",
        "signal_metadata",
        "source_metadata",
        "remove_annotation",
    ],
)
def test_rewrite_after_reading_rejects_parent_changes(tmp_path, field, monkeypatch):
    parent, _, _ = _parent(irregular=True)
    _store(tmp_path, parent)
    child = _child((parent,))
    with BuildContext.open(child.metadata, LocalRegistry(tmp_path)) as context:
        context.parent(parent.metadata.dataset_id).import_records(child)
        child.set_dependencies(context.dependency_lock())
        directory = _store(tmp_path, child)
    with LocalRegistry(tmp_path).open_reader("test/child") as reader:
        restored = reader.read()
    # Validating inherited fields needs neither the child artifact nor any signal values.
    directory.joinpath("control.duckdb").unlink()
    _forbid_value_loads(monkeypatch)
    record = restored.records[0]
    record.annotate(Annotation(key="reviewed", value=True))
    _store(tmp_path / "copy", restored)
    if field == "record_metadata":
        nested = record.metadata["nested"]
        assert isinstance(nested, list)
        nested.append(2)
    elif field == "annotation_value":
        record.annotations[0].value.append("subject-002")
    elif field == "annotation_metadata":
        record.annotations[0].metadata["changed"] = True
    elif field == "signal_metadata":
        nested = record.signals[0].metadata["nested"]
        assert isinstance(nested, list)
        nested.append(2)
    elif field == "source_metadata":
        record.sources[0].metadata["changed"] = True
    else:
        record.annotations = record.annotations[1:]
    with pytest.raises(TimeFValidationError, match="parent-owned"):
        _store(tmp_path / "rejected", restored)


@pytest.mark.parametrize("corruption", ["undeclared_parent", "wrong_child_id", "missing_parent_record"])
def test_invalid_import_provenance_is_a_format_error(tmp_path, corruption):
    parent, _, _ = _parent()
    _store(tmp_path, parent)
    child = _child((parent,))
    with BuildContext.open(child.metadata, LocalRegistry(tmp_path)) as context:
        context.parent(parent.metadata.dataset_id).import_records(child)
        child.set_dependencies(context.dependency_lock())
        directory = _store(tmp_path, child)
    with connect_control(directory / "control.duckdb") as connection:
        if corruption == "undeclared_parent":
            connection.execute("UPDATE record_imports SET parent_dataset_id = 'other/recordings'")
        elif corruption == "wrong_child_id":
            connection.execute("UPDATE records SET record_id = 'wrong'")
        else:
            connection.execute("UPDATE record_imports SET parent_record_id = 'missing'")
            connection.execute("UPDATE records SET record_id = 'first/recordings@1.0.0::missing'")
    with (
        LocalRegistry(tmp_path).open_reader(parent.metadata.dataset_id) as reader,
        DuckDBControlReader(
            directory / "control.duckdb",
            parent_records=lambda _parent, ids, annotations, prefix: reader.iter_records(
                ids, with_annotations=annotations, prefix=prefix
            ),
            parent_references={parent.metadata.dataset_id: child.metadata.parents[0]},
        ) as control,
        pytest.raises(TimeFFormatError),
    ):
        control.read_records()
