import pickle

import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.format.checksums import file_checksum
from timenet.format.control_audit import audit_control_database
from timenet.format.duckdb import connect_control
from timenet.manifest import LockedDependency
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.testing import make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    DatasetRef,
    TemporalLocalizationTask,
    TimeInterval,
    TSEditingTask,
    Version,
)
from timenet.writer import TimeFWriter


def store(root, dataset):
    dataset.derive_schema()
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def child_of(parent: TimeFDataset, parent_dir) -> TimeFDataset:
    """Return a child dataset with the exact parent release locked."""
    metadata = DatasetMetadata.model_validate(
        {
            **parent.metadata.model_dump(),
            "dataset_id": "test/child",
            "parents": [DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)],
        },
    )
    child = TimeFDataset(metadata=metadata)
    child.set_dependencies(
        (
            LockedDependency(
                dataset_id=parent.metadata.dataset_id,
                version=parent.metadata.dataset_version,
                manifest_checksum=file_checksum(parent_dir / "manifest.json"),
            ),
        )
    )
    return child


def grandchild_of(child: TimeFDataset, child_dir) -> TimeFDataset:
    """Return a grandchild dataset with the child and its parent locked."""
    grandchild = TimeFDataset(
        metadata=DatasetMetadata.model_validate(
            {
                **child.metadata.model_dump(),
                "dataset_id": "test/grandchild",
                "parents": [DatasetRef(dataset_id="test/child", version=Version(1, 0, 0))],
            },
        )
    )
    grandchild.set_dependencies(
        (
            *child.dependencies,
            LockedDependency(
                dataset_id="test/child",
                version=Version(1, 0, 0),
                manifest_checksum=file_checksum(child_dir / "manifest.json"),
            ),
        )
    )
    return grandchild


_CHANGES = [
    "metadata",
    "source_metadata",
    "signal_annotation",
    "inherited_annotation",
    "remove_annotation",
    "record_id",
]


def _change(record, change):
    if change == "record_id":
        record.record_id = "renamed"
    elif change == "metadata":
        record.metadata["new"] = {"nested": [1]}
    elif change == "source_metadata":
        record.sources[0].metadata["new"] = "value"
    elif change == "signal_annotation":
        record.signals[0].annotate(Annotation(key="new", value="value"))
    elif change == "inherited_annotation":
        record.annotations[0].metadata["new"] = "value"
    else:
        record.annotations = record.annotations[1:]


@pytest.fixture
def composition(tmp_path):
    parent = make_dataset()
    child = child_of(parent, store(tmp_path, parent))
    with LocalRegistry(tmp_path).open_reader(parent.metadata.dataset_id) as reader:
        imported = child.import_record(next(reader.iter_records(["record-0"])), parent="timenet/hello-world")
        yield tmp_path, child, imported


@pytest.mark.parametrize("kind", ["scope", "annotation", "signal", "span_target"])
def test_tasks_reference_inherited_objects(composition, kind):
    root, child, record = composition
    signal = record.signals[0]
    scope = TimeInterval.seconds(0, 0.25, time_series_ids=(signal.id,))
    if kind == "scope":
        task = ClassificationTask(inputs=(record,), targets=("x",), scope=scope)
    elif kind == "annotation":
        task = AnswerTask(
            inputs=(record,),
            input_annotations=(record.annotations[0],),
            target_annotations=(signal.annotations[0],),
        )
    elif kind == "signal":
        task = TSEditingTask(inputs=(record,), targets=(signal,))
    else:
        task = TemporalLocalizationTask(inputs=(record,), targets=(scope,))
    child.add_task(task=task)
    directory = store(root, child)
    with connect_control(directory / "control.duckdb", read_only=True) as connection:
        audit_control_database(connection)
    with LocalRegistry(root).open_reader("test/child") as reader:
        restored = reader.read()
        result = restored.tasks[0]
        assert result.scope == task.scope
        if kind == "annotation":
            assert result.input_annotations[0] is restored.records[0].annotations[0]
            assert result.target_annotations[0] is restored.records[0].signals[0].annotations[0]
        elif kind == "signal":
            assert result.targets is not None
            assert result.targets[0] is restored.records[0].signals[0]
            assert restored.records[0].signals[0].to_arrow().equals(signal.to_arrow())
        assert reader.task_table().num_rows == 1
        assert reader.target_table().num_rows == (0 if kind == "annotation" else 1)
        assert reader._manifest.files.time_series == ()


@pytest.mark.parametrize("change", _CHANGES)
def test_rejects_changes_to_parent_owned_data(composition, change):
    root, child, record = composition
    _change(record, change)
    with pytest.raises(TimeFValidationError, match=r"parent-owned fields|registered ID"):
        store(root, child)
    assert not (root / "test/child/1.0.0/manifest.json").exists()


@pytest.mark.parametrize("change", _CHANGES)
def test_rewrite_after_reading_rejects_parent_changes(composition, change, monkeypatch):
    root, child, _ = composition
    directory = store(root, child)
    with LocalRegistry(root).open_reader("test/child") as reader:
        restored = reader.read()
    # Validating inherited fields needs neither the child artifact nor any signal values.
    directory.joinpath("control.duckdb").unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("validating inherited fields must not load values or offsets")

    monkeypatch.setattr(TimeFReader, "_load_signal", forbidden)
    monkeypatch.setattr(TimeFReader, "_load_offsets", forbidden)
    record = restored.records[0]
    record.annotate(Annotation(key="reviewed", value=True))
    store(root / "copy", restored)
    _change(record, change)
    with pytest.raises(TimeFValidationError, match=r"parent-owned fields|registered ID"):
        store(root / "rejected", restored)


def test_selected_reads_resolve_imports_before_and_after_a_full_read(tmp_path):
    parent = make_dataset()
    child = child_of(parent, store(tmp_path, parent))
    with LocalRegistry(tmp_path).open_reader(parent.metadata.dataset_id) as reader:
        for record in reader.iter_records(["record-0", "record-1"]):
            child.import_record(record, parent="timenet/hello-world")
    store(tmp_path, child)
    with LocalRegistry(tmp_path).open_reader("test/child") as reader:
        ids = reader.record_ids()
        assert ids == ("record-0", "record-1")
        (first,) = reader.iter_records(ids[:1])
        assert first.signals[0].to_arrow().equals(parent.records[0].signals[0].to_arrow())
        (second,) = reader.iter_records(ids[1:])
        assert second.id == "record-1"
        assert [record.id for record in reader.iter_records()] == list(ids)
        (again,) = reader.iter_records(ids[:1])
        assert again.id == ids[0]


def test_parent_graph_reopens_after_closing_a_parent_or_pickling(composition):
    root, child, record = composition
    grandchild = grandchild_of(child, store(root, child))
    with LocalRegistry(root).open_reader("test/child") as reader:
        grandchild.import_record(next(reader.iter_records()), parent="test/child")
        store(root, grandchild)
    with LocalRegistry(root).open_reader("test/grandchild") as reader:
        restored = next(reader.iter_records())
        assert restored.signals[0].to_arrow().equals(record.signals[0].to_arrow())
        middle = reader._parents["test/child"]
        middle._parents["timenet/hello-world"].close()
        assert middle.record_ids() == (record.id,)
        assert restored.signals[0].to_arrow().equals(record.signals[0].to_arrow())
        with pickle.loads(pickle.dumps(reader)) as copy:
            again = next(copy.iter_records())
            assert again.id == restored.id
            assert again.signals[0].to_arrow().equals(record.signals[0].to_arrow())
        reader.close()
        assert next(reader.iter_records()).id == restored.id


def test_inherited_signal_annotation_equal_to_a_record_annotation_stays_referenceable(tmp_path):
    parent = make_dataset()
    # Equal annotation content must retain distinct Record and Signal occurrences.
    signal_occurrence = parent.records[0].signals[0].annotate(Annotation(key="cohort", value="A", id="cohort-shared"))
    child = child_of(parent, store(tmp_path, parent))
    with LocalRegistry(tmp_path).open_reader(parent.metadata.dataset_id) as reader:
        imported = child.import_record(next(reader.iter_records(["record-0"])), parent="timenet/hello-world")
        inherited = next(
            a for a in imported.signals[0].annotations if a.occurrence_id == signal_occurrence.occurrence_id
        )
        child.add_task(task=AnswerTask(inputs=(imported,), targets=("x",), input_annotations=(inherited,)))
        store(tmp_path, child)
    with LocalRegistry(tmp_path).open_reader("test/child") as reader:
        restored = reader.read()
        assert restored.tasks[0].input_annotations[0].occurrence_id == signal_occurrence.occurrence_id
        assert restored.tasks[0].input_annotations[0] is restored.records[0].signals[0].annotations[1]


def test_accepts_record_overlays_and_lazy_value_reads(composition):
    root, child, record = composition
    record.signals[0].to_arrow()
    overlay = record.annotate(Annotation(key="new", value="value"))
    child.add_task(task=AnswerTask(inputs=(record,), targets=("x",), input_annotations=(overlay,)))
    store(root, child)
    with LocalRegistry(root).open_reader("test/child") as reader:
        restored = reader.read()
        assert restored.tasks[0].input_annotations[0].id == overlay.id


def test_read_write_preserves_parent_ownership_and_overlays(composition):
    root, child, record = composition
    overlay = record.annotate(Annotation(key="reviewed", value=True))
    child.add_task(task=AnswerTask(inputs=(record,), targets=("x",), input_annotations=(overlay,)))
    store(root, child)
    with LocalRegistry(root).open_reader("test/child") as reader:
        restored = reader.read()
        assert restored.dependencies == child.dependencies
        assert restored.owned_records == ()
        assert (
            restored.record_imports[record.id].inherited_annotation_ids
            == child.record_imports[record.id].inherited_annotation_ids
        )
        copy = store(root / "copy", restored)
        assert not tuple(copy.glob("time_series/*.parquet"))
    # Keep parents in their original location while reopening the copied child.

    class Registry(LocalRegistry):
        def get_manifest(self, dataset_id, version=None):
            if dataset_id == "test/child":
                return LocalRegistry(root / "copy").get_manifest(dataset_id, version)
            return super().get_manifest(dataset_id, version)

        def open_version(self, dataset_id, version=None):
            if dataset_id == "test/child":
                return LocalRegistry(root / "copy").open_version(dataset_id, version)
            return super().open_version(dataset_id, version)

    with Registry(root).open_reader("test/child") as reader:
        again = reader.read()
        assert [a.id for a in again.records[0].annotations].count(overlay.id) == 1
        assert again.tasks[0].input_annotations[0].id == overlay.id
        assert again.records[0].signals[0].to_arrow().equals(record.signals[0].to_arrow())


def test_proxy_row_stores_only_the_record_id(composition):
    root, child, record = composition
    directory = store(root, child)
    with connect_control(directory / "control.duckdb", read_only=True) as connection:
        rows = connection.execute(
            """SELECT record_id, clock_id, time_span_start_us, time_span_end_us, metadata
               FROM records JOIN record_imports USING (record_key)"""
        ).fetchall()
        assert rows == [(record.id, None, None, None, None)]
        assert connection.execute("SELECT parent_dataset_id FROM record_imports").fetchall() == [
            ("timenet/hello-world",)
        ]
        assert connection.execute("SELECT count(*) FROM clocks").fetchone() == (0,)
        audit_control_database(connection)


def test_counts_include_inherited_hierarchy_and_overlays(composition):
    root, child, record = composition
    record.annotate(Annotation(key="reviewed", value=True))
    store(root, child)
    counts = LocalRegistry(root).get_manifest("test/child").counts
    assert counts.records == 1
    assert counts.sources == len(tuple(record.walk_sources()))
    assert counts.signals == len(record.signals)
    assert counts.axes == len({signal.time_axis.axis_id for signal in record.signals})
    assert counts.annotation_contents == len({a.content_id for a in record.walk_annotations()})
    assert counts.annotation_occurrences == len(tuple(record.walk_annotations()))
    assert sum(counts.signals_by_spec.values()) == len(record.signals)
    assert counts.signal_chunks == 0


def test_annotation_option_reaches_every_parent_layer(composition):
    root, child, record = composition
    record.annotate(Annotation(key="reviewed", value=True))
    grandchild = grandchild_of(child, store(root, child))
    with LocalRegistry(root).open_reader("test/child") as reader:
        grandchild.import_record(next(reader.iter_records()), parent="test/child")
        store(root, grandchild)
    with LocalRegistry(root).open_reader("test/grandchild") as reader:
        without = next(reader.iter_records(with_annotations=False))
        assert not tuple(without.walk_annotations())
        with_annotations = next(reader.iter_records())
        assert len(tuple(with_annotations.walk_annotations())) == len(tuple(record.walk_annotations()))
