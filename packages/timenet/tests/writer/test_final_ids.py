"""The writer stores dense final ids through a mapping and never changes the caller's objects."""

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import duckdb
import pyarrow as pa
import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis, RegularAxis
from timenet.dataset.edit import edit_version, remove_records
from timenet.dataset.storage_order import iter_record_signals, iter_record_sources
from timenet.errors import TimeFValidationError
from timenet.format import control_writer
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.control_writer import DuckDBControlWriter
from timenet.format.final_ids import content_digest
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import assert_datasets_equal, make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    ForecastingTask,
    License,
    StepInterval,
    Task,
    TimeInterval,
    TimeSeriesSpec,
    Version,
    ids as ids_module,
    new_id,
    ureg,
)
from timenet.types.ids import FINAL_ID_LIMIT, is_provisional
from timenet.writer import FinalIds, TimeFWriter


_PRIMARY_IDS = {
    "records": "record_id",
    "sources": "source_id",
    "signals": "signal_id",
    "axes": "axis_id",
    "annotation_contents": "content_id",
    "annotation_occurrences": "occurrence_id",
    "tasks": "task_id",
}
_TABLES = (
    "control_metadata",
    "datasets",
    "records",
    "clocks",
    "sources",
    "axes",
    "axis_offsets",
    "signals",
    "signal_chunks",
    "annotation_contents",
    "annotation_occurrences",
    "tasks",
    "task_targets",
    "task_record_refs",
    "task_annotation_refs",
    "task_dependencies",
)
_SPEC = TimeSeriesSpec(spec_type="level", name="Level", unit_value=ureg.dimensionless)
_OTHER_SPEC = TimeSeriesSpec(spec_type="rate", name="Rate", unit_value=ureg.dimensionless)
_SHARED_AXIS = RegularAxis.from_rate_hz(4)


def _metadata(version: Version | None = None) -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id="timenet/final-ids",
        dataset_version=version or Version(1, 0, 0),
        name="Final ids",
        description="d",
        license=License.MIT,
    )


def _signal(name: str, n: int = 4, *, spec: TimeSeriesSpec = _SPEC, axis=_SHARED_AXIS, **kwargs) -> Signal:
    array = pa.array([float(i) for i in range(n)], pa.float32())
    return Signal.from_loader(spec=spec, name=name, time_axis=axis, loader=lambda: array, n_values=n, **kwargs)


def _record(*signals: Signal, **kwargs) -> Record:
    return Record(sources=(Source(name="s", signals=signals),), **kwargs)


def _tree_dataset() -> TimeFDataset:
    """Two records whose sources and signals sit in non-alphabetical tuple order, annotated everywhere."""
    dataset = TimeFDataset(metadata=_metadata())
    dataset.annotate(Annotation(key="site", value="lab"))
    shared = Annotation(key="cohort", value="A")
    records = []
    for index in range(2):
        child_z = Source(name="z-child", signals=(_signal("y"), _signal("x")))
        child_a = Source(name="a-child", signals=(_signal("w"),))
        root_b = Source(name="b-root", sources=(child_z, child_a), signals=(_signal("v"),))
        root_a = Source(name="a-root", signals=(_signal("u", spec=_OTHER_SPEC),))
        record = dataset.add_record(record=Record(sources=(root_b, root_a), metadata={"index": index}))
        record.annotate(Annotation(key="note", value=f"record {index}"))
        record.annotate(shared)
        child_z.annotate(Annotation(key="placement", value="chest"))
        child_z.signals[1].annotate(Annotation(key="artifact", span=TimeInterval.seconds(0.0, 0.25)))
        records.append(record)
    first = ClassificationTask(inputs=(records[0],), targets=("normal",))
    first.annotate(Annotation(key="difficulty", value="easy"))
    x_signal = records[1].sources[0].sources[0].signals[1]
    dataset.add_tasks(
        tasks=[
            first,
            AnswerTask(
                inputs=(records[0],),
                prompt="Which cohort?",
                targets=("A",),
                input_annotations=(records[0].annotations[1],),
                from_tasks=(first,),
            ),
            ClassificationTask(
                inputs=(records[1],),
                targets=("onset",),
                scope=TimeInterval.seconds(0.0, 0.5, time_series_ids=(x_signal.id,)),
            ),
        ]
    )
    return dataset


def _ordinal_dataset() -> TimeFDataset:
    dataset = TimeFDataset(metadata=_metadata())
    ordinal = _signal("steps", 6, axis=OrdinalAxis())
    record = dataset.add_record(record=_record(ordinal))
    dataset.add_task(
        task=ForecastingTask(
            inputs=(record,),
            targets=(StepInterval(time_series_id=ordinal.id, start=4, stop=6),),
            scope=StepInterval(time_series_id=ordinal.id, start=0, stop=4),
        )
    )
    return dataset


def _records_dataset(ids: list[int | None]) -> TimeFDataset:
    dataset = TimeFDataset(metadata=_metadata())
    for index, record_id in enumerate(ids):
        signal = _signal(f"s{index}")
        dataset.add_record(record=_record(signal) if record_id is None else _record(signal, id=record_id))
    return dataset


def _write(root: Path, dataset: TimeFDataset, **kwargs: object) -> tuple[Path, FinalIds]:
    with TimeFWriter(root, dataset, **kwargs) as writer:  # ty: ignore[invalid-argument-type]
        ids = writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version), ids


def _read(version_dir: Path) -> TimeFDataset:
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        return reader.read()


def _column(version_dir: Path, sql: str) -> list:
    with duckdb.connect(str(version_dir / "control.duckdb"), read_only=True) as connection:
        return [row[0] if len(row) == 1 else row for row in connection.execute(sql).fetchall()]


def _count(version_dir: Path, table: str, where: str = "true") -> int:
    return _column(version_dir, f"SELECT count(*) FROM {table} WHERE {where}")[0]  # noqa: S608


def _tables(version_dir: Path) -> dict[str, list[tuple]]:
    with duckdb.connect(str(version_dir / "control.duckdb"), read_only=True) as connection:
        return {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()  # noqa: S608
            for table in _TABLES
        }


def _value_checksums(version_dir: Path) -> list[str]:
    manifest = Manifest.model_validate_json((version_dir / "manifest.json").read_text())
    return [part.checksum for part in manifest.files.time_series]


def _occurrence(annotation: Annotation) -> int:
    assert annotation.occurrence_id is not None
    return annotation.occurrence_id


def _ids_by_kind(dataset: TimeFDataset) -> dict[str, list[int]]:
    """Every in-memory id the dataset holds, references included, by kind."""
    ids: dict[str, list[int]] = {kind: [] for kind in ("record", "source", "signal", "axis", "content", "occurrence")}
    ids["task"] = []

    def annotations(items: tuple[Annotation, ...]) -> None:
        for annotation in items:
            ids["content"].append(annotation.id)
            if annotation.occurrence_id is not None:
                ids["occurrence"].append(annotation.occurrence_id)

    annotations(dataset.annotations)
    for record in dataset.records:
        ids["record"].append(record.id)
        ids["task"].extend(record.task_ids)
        annotations(record.annotations)
        for source in iter_record_sources(record):
            ids["source"].append(source.id)
            annotations(source.annotations)
        for signal in iter_record_signals(record):
            ids["signal"].append(signal.id)
            ids["axis"].append(signal.time_axis.id)
            annotations(signal.annotations)
    for task in dataset.tasks:
        ids["task"].extend((task.id, *(parent.id for parent in task.from_tasks)))
        ids["record"].extend(record.id for record in task.inputs)
        annotations((*task.annotations, *task.input_annotations, *task.target_annotations))
        for span in (task.scope, *(task.targets or ())):
            ids["signal"].extend(getattr(span, "time_series_ids", None) or ())
            if isinstance(span, StepInterval):
                ids["signal"].append(span.time_series_id)
    return ids


def _snapshot(dataset: TimeFDataset) -> tuple:
    axes = [(id(signal.time_axis), signal.time_axis.id) for record in dataset.records for signal in record.signals]
    spans = [(id(task.scope), task.scope, task.targets) for task in dataset.tasks]
    return _ids_by_kind(dataset), axes, spans


# ---- dense ids in storage order -------------------------------------------------------------------


def test_first_write_numbers_every_kind_densely_in_storage_order(tmp_path):
    dataset = _tree_dataset()
    version_dir, ids = _write(tmp_path, dataset)

    for table, column in _PRIMARY_IDS.items():
        stored = _column(version_dir, f"SELECT {column} FROM {table} ORDER BY rowid")  # noqa: S608
        assert stored == list(range(1, len(stored) + 1)), table

    assert [ids.records[record.id] for record in dataset.records] == [1, 2]
    sources = [source for record in dataset.records for source in iter_record_sources(record)]
    assert [ids.sources[source.id] for source in sources] == list(range(1, 9))
    assert _column(version_dir, "SELECT name FROM sources ORDER BY source_id")[:4] == [
        "b-root",
        "z-child",
        "a-child",
        "a-root",
    ]
    signals = [signal for record in dataset.records for signal in iter_record_signals(record)]
    assert [ids.signals[signal.id] for signal in signals] == list(range(1, 11))
    assert _column(version_dir, "SELECT name FROM signals ORDER BY signal_id")[:5] == ["v", "y", "x", "w", "u"]
    assert [ids.tasks[task.id] for task in dataset.tasks] == [1, 2, 3]
    assert {ids.axes[signal.time_axis.id] for signal in signals} == {1}


def test_read_order_and_value_placement_follow_ids(tmp_path):
    dataset = _tree_dataset()
    version_dir, ids = _write(tmp_path, dataset)
    restored = _read(version_dir)

    assert [record.metadata["index"] for record in restored.records] == [0, 1]
    assert [record.id for record in restored.records] == [1, 2]
    assert [[signal.name for signal in iter_record_signals(record)] for record in restored.records] == [
        ["v", "y", "x", "w", "u"]
    ] * 2
    placements = _column(
        version_dir,
        """SELECT c.signal_id FROM signal_chunks c JOIN signals s USING (signal_id)
           ORDER BY s.spec_type, c.value_path, c.chunk_major_index, c.chunk_minor_index""",
    )
    level_ids = [
        ids.signals[signal.id]
        for record in dataset.records
        for signal in iter_record_signals(record)
        if signal.name != "u"
    ]
    assert placements == [*level_ids, 5, 10]


def test_occurrences_are_grouped_by_owner_type_and_contents_by_first_sighting(tmp_path):
    version_dir, _ = _write(tmp_path, _tree_dataset())

    owners = _column(version_dir, "SELECT object_type, object_id FROM annotation_occurrences ORDER BY rowid")
    assert [object_type for object_type, _ in owners] == [
        "Dataset",
        *["Record"] * 4,
        *["Signal"] * 2,
        *["Source"] * 2,
        "Task",
    ]
    assert [object_id for _, object_id in owners] == [None, 1, 1, 2, 2, 3, 8, 2, 6, 1]
    contents = _column(version_dir, "SELECT name FROM annotation_contents ORDER BY content_id")
    assert contents == ["site", "note", "cohort", "note", "artifact", "placement", "difficulty"]


def test_two_independent_builds_write_identical_catalogs_and_values(tmp_path):
    first = make_dataset()
    ids_module._state.redraw()
    second = make_dataset()
    assert first.records[0].id != second.records[0].id

    first_dir, first_ids = _write(tmp_path / "first", first)
    second_dir, second_ids = _write(tmp_path / "second", second)

    assert _tables(first_dir) == _tables(second_dir)
    assert _value_checksums(first_dir) == _value_checksums(second_dir)
    assert [first_ids.records[record.id] for record in first.records] == [
        second_ids.records[record.id] for record in second.records
    ]


def test_schema_order_does_not_depend_on_provisional_ids():
    def build(*, reverse_ids: bool) -> TimeFDataset:
        names = ["sine", "cosine"]
        ids = sorted((new_id(), new_id()), reverse=reverse_ids)
        dataset = TimeFDataset(metadata=_metadata())
        sources = []
        for name, source_id in zip(names, ids, strict=True):
            spec = TimeSeriesSpec(spec_type=name, name=name, unit_value=ureg.dimensionless)
            source = Source(id=source_id, name="same", signals=(_signal("s", spec=spec),))
            source.annotate(Annotation(key=f"k{name}", value=1))
            sources.append(source)
        dataset.add_record(record=Record(sources=tuple(sources)))
        return dataset

    schemas = [build(reverse_ids=flag).derive_schema() for flag in (False, True)]
    assert schemas[0] == schemas[1]


def test_same_named_siblings_keep_tuple_order_in_memory_and_after_a_reread(tmp_path):
    later = _signal("x", metadata={"position": 1})
    earlier = _signal("x", metadata={"position": 0})
    assert later.id < earlier.id
    child_later = Source(name="c", metadata={"position": 1})
    child_earlier = Source(name="c", metadata={"position": 0})
    root = Source(name="root", sources=(child_earlier, child_later), signals=(earlier, later))
    dataset = TimeFDataset(metadata=_metadata())
    record = dataset.add_record(record=Record(sources=(root,)))

    (restored,) = _read(_write(tmp_path, dataset)[0]).records

    def positions(item: Record) -> tuple[list[object], list[object]]:
        return (
            [signal.metadata["position"] for signal in item.signals],
            [source.metadata.get("position", -1) for source in item.walk_sources()],
        )

    assert positions(record) == positions(restored) == ([0, 1], [-1, 0, 1])


# ---- the caller's objects and the returned mapping ------------------------------------------------


@pytest.mark.parametrize("build", [make_dataset, _tree_dataset, _ordinal_dataset])
def test_write_leaves_the_callers_objects_unchanged(tmp_path, build):
    dataset = build()
    before = _snapshot(dataset)

    _write(tmp_path, dataset)

    assert _snapshot(dataset) == before
    ids = before[0]
    assert all(is_provisional(value) for kind in ids.values() for value in kind)
    record = dataset.add_record(record=_record(_signal("late")))
    dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("late",)))
    assert dataset.records[-1] is record


def test_write_returns_a_read_only_mapping_of_every_in_memory_id(tmp_path):
    dataset = _tree_dataset()
    with TimeFWriter(tmp_path, dataset) as writer:
        with pytest.raises(RuntimeError, match="before final_ids"):
            _ = writer.final_ids
        ids = writer.write()
        assert writer.final_ids is ids

    for kind, values in _ids_by_kind(dataset).items():
        mapping = ids[kind]  # ty: ignore[invalid-argument-type]
        assert isinstance(mapping, MappingProxyType)
        assert all(not is_provisional(mapping[value]) for value in values), kind
    with pytest.raises(TypeError):
        ids.records[1] = 2  # ty: ignore[invalid-assignment]
    with pytest.raises(KeyError):
        _ = ids.records[new_id()]


def test_kept_ids_map_to_themselves(tmp_path):
    dataset = _records_dataset([None, 7, None])
    version_dir, ids = _write(tmp_path, dataset)

    assert [ids.records[record.id] for record in dataset.records] == [8, 7, 9]
    assert _column(version_dir, "SELECT record_id FROM records ORDER BY rowid") == [8, 7, 9]


def test_ids_copied_before_the_write_resolve_through_the_mapping(tmp_path):
    dataset = _records_dataset([None] * 3)
    record = dataset.records[1]
    record_id, signal_id = record.id, record.signals[0].id

    version_dir, ids = _write(tmp_path, dataset)

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        (back,) = reader.iter_records((ids.records[record_id],))
    assert back.id == ids.records[record_id] == 2
    assert [signal.id for signal in back.signals] == [ids.signals[signal_id]]
    assert back.signals[0].name == "s1"


@pytest.mark.parametrize("build", [make_dataset, _tree_dataset, _ordinal_dataset])
def test_every_reference_resolves_through_the_mapping(tmp_path, build):
    dataset = build()
    version_dir, ids = _write(tmp_path, dataset)
    restored = _read(version_dir)

    assert_datasets_equal(dataset, restored, final_ids=ids)
    by_id = {task.id: task for task in restored.tasks}
    for task in dataset.tasks:
        back = by_id[ids.tasks[task.id]]
        assert [record.id for record in back.inputs] == [ids.records[record.id] for record in task.inputs]
        assert [parent.id for parent in back.from_tasks] == [ids.tasks[parent.id] for parent in task.from_tasks]
        assert [item.occurrence_id for item in back.input_annotations] == [
            ids.occurrences[item.occurrence_id] for item in task.input_annotations if item.occurrence_id is not None
        ]
        scope_ids = getattr(task.scope, "time_series_ids", None) or ()
        assert (getattr(back.scope, "time_series_ids", None) or ()) == tuple(ids.signals[item] for item in scope_ids)


def test_writing_the_same_dataset_twice_and_retrying_after_a_failure_give_identical_catalogs(tmp_path, monkeypatch):
    dataset = make_dataset()
    reference, reference_ids = _write(tmp_path / "reference", dataset)

    def fail(*_args: object) -> None:
        raise RuntimeError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(TimeFWriter, "_write_values", fail)
        with pytest.raises(RuntimeError, match="disk full"):
            _write(tmp_path / "failed", dataset)
    retried, retried_ids = _write(tmp_path / "retried", dataset)
    zarr, _ = _write(tmp_path / "zarr", dataset, values_backend="zarr")

    assert _tables(retried) == _tables(reference)
    assert retried_ids == reference_ids
    zarr_tables, reference_tables = _tables(zarr), _tables(reference)
    assert zarr_tables.pop("signal_chunks") != reference_tables.pop("signal_chunks")
    assert zarr_tables == reference_tables


def test_rewriting_after_adding_records_that_reuse_earlier_objects_stores_them_once(tmp_path):
    axis = RegularAxis.from_rate_hz(4)
    cohort = Annotation(key="cohort", value="A")
    dataset = TimeFDataset(metadata=_metadata())
    first = dataset.add_record(record=_record(_signal("a", axis=axis)))
    first.annotate(cohort)
    v1, _ = _write(tmp_path, dataset)

    second = dataset.add_record(record=_record(_signal("b", axis=axis)))
    second.annotate(cohort)
    second.annotate(Annotation(key="cohort", value="A"))
    dataset._metadata = _metadata(Version(1, 0, 1))
    v2, ids = _write(tmp_path, dataset)

    assert _count(v1, "axes") == _count(v2, "axes") == 1
    assert _count(v1, "annotation_contents") == _count(v2, "annotation_contents") == 1
    assert _count(v2, "annotation_occurrences") == 3
    assert _count(v2, "signals") == 2
    assert_datasets_equal(dataset, _read(v2), final_ids=ids)


# ---- annotation contents and axes are stored once per value ---------------------------------------


def test_equal_annotations_built_separately_share_one_content_row(tmp_path):
    dataset = _records_dataset([None, None])
    first, second = dataset.records
    a = first.annotate(Annotation(key="cohort", value="A", metadata={"x": 1, "y": [1, 2]}))
    b = second.annotate(Annotation(key="cohort", value="A", metadata={"y": [1, 2], "x": 1}))
    assert a.id != b.id

    version_dir, ids = _write(tmp_path, dataset)

    assert ids.contents[a.id] == ids.contents[b.id]
    assert ids.occurrences[_occurrence(a)] != ids.occurrences[_occurrence(b)]
    assert _count(version_dir, "annotation_contents") == 1
    assert _count(version_dir, "annotation_occurrences") == 2


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ({"value": True}, {"value": 1}),
        ({"value": 1}, {"value": 1.0}),
        ({"value": 64, "unit": "years"}, {"value": 64}),
        ({"value": 64, "description": "age"}, {"value": 64}),
        ({"value": 64, "metadata": {"a": 1}}, {"value": 64}),
        ({"value": 64, "metadata": {"a": 0.0}}, {"value": 64, "metadata": {"a": -0.0}}),
        ({"value": "x"}, {"value": ["x"]}),
        ({"value": "x"}, {"value": "x", "key": "other"}),
    ],
)
def test_different_annotation_contents_have_different_digests(left: dict[str, Any], right: dict[str, Any]):
    def build(fields: dict[str, Any]) -> Annotation:
        merged: dict[str, Any] = {"key": "k", **fields}
        return Annotation(**merged)

    assert content_digest(build(left)) != content_digest(build(right))
    assert content_digest(build(left)) == content_digest(build(left))


@pytest.mark.parametrize(("left", "right"), [(0.0, 1.0), (0.5, 0.25)])
def test_different_annotation_contents_stay_separate(tmp_path, left, right):
    dataset = _records_dataset([None, None])
    first, second = dataset.records
    first.annotate(Annotation(key="k", value=left, metadata={"a": 1}))
    second.annotate(Annotation(key="k", value=left, metadata={"a": 2}))
    first.annotate(Annotation(key="j", value=right))
    second.annotate(Annotation(key="j", value=left))

    version_dir, _ = _write(tmp_path, dataset)

    assert _count(version_dir, "annotation_contents") == 4


def test_nan_contents_merge_and_a_reread_writes_again(tmp_path):
    dataset = _records_dataset([None, None])
    for record in dataset.records:
        record.annotate(Annotation(key="k", value=float("nan")))
    v1, _ = _write(tmp_path, dataset)
    restored = _read(v1)
    restored._metadata = _metadata(Version(1, 0, 1))

    v2, _ = _write(tmp_path, restored)

    assert _count(v1, "annotation_contents") == _count(v2, "annotation_contents") == 1
    assert _count(v2, "annotation_occurrences") == 2


def test_zero_and_negative_zero_share_one_content_row_also_in_an_edit(tmp_path):
    dataset = _records_dataset([None, None])
    first, second = dataset.records
    zero = first.annotate(Annotation(key="z", value=0.0))
    negative = second.annotate(Annotation(key="z", value=-0.0))
    assert content_digest(zero) == content_digest(negative)

    v1, ids = _write(tmp_path, dataset)

    assert ids.contents[zero.id] == ids.contents[negative.id]
    assert _column(v1, "SELECT content_id, float_value FROM annotation_contents") == [(1, 0.0)]
    edited = _read(v1)
    edited._metadata = _metadata(Version(1, 0, 1))
    added = edited.add_record(record=_record(_signal("n")))
    again = added.annotate(Annotation(key="z", value=-0.0))
    edited.derive_schema()
    v2, edit_ids = _write(tmp_path, edited)

    assert edit_ids.contents[again.id] == 1
    assert _count(v2, "annotation_contents") == 1
    assert _count(v2, "annotation_occurrences") == 3


def test_a_thousand_records_with_their_own_equal_axes_store_one_axis(tmp_path):
    dataset = TimeFDataset(metadata=_metadata())
    for index in range(1000):
        dataset.add_record(record=_record(_signal(f"s{index}", 2, axis=RegularAxis.from_rate_hz(500))))

    version_dir, ids = _write(tmp_path, dataset)

    assert _count(version_dir, "axes") == 1
    assert {ids.axes[record.signals[0].time_axis.id] for record in dataset.records} == {1}
    assert set(_column(version_dir, "SELECT DISTINCT axis_id FROM signals")) == {1}


def test_different_axis_definitions_stay_separate_and_ordinal_axes_merge(tmp_path):
    axes = [
        RegularAxis.from_rate_hz(500),
        RegularAxis.from_rate_hz(500),
        RegularAxis.from_rate_hz(250),
        RegularAxis.from_rate_hz(500, offset_us=3),
        RegularAxis.from_rate_hz(500).at_index(1),
        OrdinalAxis(),
        OrdinalAxis(),
    ]
    dataset = TimeFDataset(metadata=_metadata())
    record = dataset.add_record(record=_record(*(_signal(f"s{i}", 2, axis=axis) for i, axis in enumerate(axes))))

    version_dir, ids = _write(tmp_path, dataset)

    assert _count(version_dir, "axes") == 5
    finals = [ids.axes[signal.time_axis.id] for signal in iter_record_signals(record)]
    assert finals == [1, 1, 2, 3, 4, 5, 5]


def test_irregular_axes_are_stored_per_id(tmp_path):
    offsets = [0, 5_000, 9_000]
    signals = [
        Signal.from_irregular([1.0, 2.0, 3.0], time_offsets_us=offsets, spec=_SPEC, name=f"i{i}") for i in range(2)
    ]
    assert signals[0].time_axis == signals[1].time_axis
    dataset = TimeFDataset(metadata=_metadata())
    dataset.add_record(record=_record(*signals))

    version_dir, ids = _write(tmp_path, dataset)

    assert _count(version_dir, "axes") == 2
    assert _column(version_dir, "SELECT axis_id, count(*) FROM axis_offsets GROUP BY axis_id ORDER BY axis_id") == [
        (1, 3),
        (2, 3),
    ]
    assert [ids.axes[signal.time_axis.id] for signal in signals] == [1, 2]


def test_signals_sources_records_occurrences_and_tasks_are_never_merged(tmp_path):
    dataset = TimeFDataset(metadata=_metadata())
    records = [dataset.add_record(record=_record(_signal("same"))) for _ in range(2)]
    for record in records:
        record.annotate(Annotation(key="k", value=1))
    dataset.add_tasks(tasks=[ClassificationTask(inputs=(records[0],), targets=("x",)) for _ in range(2)])

    version_dir, _ = _write(tmp_path, dataset)

    for table, expected in (
        ("records", 2),
        ("sources", 2),
        ("signals", 2),
        ("annotation_occurrences", 2),
        ("tasks", 2),
        ("annotation_contents", 1),
        ("axes", 1),
    ):
        assert _count(version_dir, table) == expected, table


# ---- one id with two payloads ---------------------------------------------------------------------


def _assert_rejected_before_values(tmp_path, monkeypatch, dataset: TimeFDataset, message: str) -> None:
    def no_values(*_args: object) -> None:
        pytest.fail("the writer wrote values before it rejected the dataset")

    monkeypatch.setattr(TimeFWriter, "_write_values", no_values)
    with pytest.raises(TimeFValidationError, match=message):
        _write(tmp_path, dataset)
    assert not list((tmp_path / dataset.metadata.dataset_id).iterdir())


def test_one_content_id_with_two_payloads_is_rejected(tmp_path, monkeypatch):
    dataset = _records_dataset([None, None])
    first, second = dataset.records
    annotation = first.annotate(Annotation(key="k", value=1))
    second.annotate(replace(annotation, value=2))

    _assert_rejected_before_values(tmp_path, monkeypatch, dataset, "is reused with different content")


def test_one_regular_axis_id_with_two_definitions_is_rejected(tmp_path, monkeypatch):
    axis = RegularAxis.from_rate_hz(4)
    dataset = TimeFDataset(metadata=_metadata())
    dataset.add_record(record=_record(_signal("a", axis=axis), _signal("b", axis=replace(axis, start_index=1))))

    _assert_rejected_before_values(tmp_path, monkeypatch, dataset, "is reused with different definitions")


def test_one_irregular_axis_id_with_two_endpoints_is_rejected(tmp_path, monkeypatch):
    first = Signal.from_irregular([1.0, 2.0], time_offsets_us=[0, 10], spec=_SPEC, name="a")
    axis = replace(first.time_axis, last_us=20)
    second = Signal.from_irregular([1.0, 2.0], time_offsets_us=[0, 20], spec=_SPEC, name="b")
    second = Signal.from_loader(
        spec=_SPEC,
        name="b",
        time_axis=axis,
        n_values=2,
        loader=second.loader,
        time_offsets_loader=second.time_offsets_loader,
    )
    dataset = TimeFDataset(metadata=_metadata())
    dataset.add_record(record=_record(first, second))

    _assert_rejected_before_values(tmp_path, monkeypatch, dataset, "is reused with different definitions")


# ---- references -----------------------------------------------------------------------------------


def test_annotation_references_point_at_their_own_occurrence_when_contents_merge(tmp_path):
    dataset = _records_dataset([None, None])
    first, second = dataset.records
    first.annotate(Annotation(key="cohort", value="A"))
    target = second.annotate(Annotation(key="cohort", value="A"))
    task = AnswerTask(inputs=(second,), prompt="Cohort?", input_annotations=(target,), target_annotations=(target,))
    dataset.add_task(task=task)

    version_dir, ids = _write(tmp_path, dataset)
    restored = _read(version_dir)

    (back,) = restored.tasks
    (back_first, back_second) = restored.records
    assert back.input_annotations[0] is back_second.annotations[0]
    assert back.target_annotations[0] is back_second.annotations[0]
    assert back.input_annotations[0].occurrence_id == ids.occurrences[_occurrence(target)]
    assert back_first.annotations[0].id == back_second.annotations[0].id
    assert back_first.annotations[0].occurrence_id != back_second.annotations[0].occurrence_id


# ---- copy-on-write edits --------------------------------------------------------------------------


def _edit_without_record_3(tmp_path) -> tuple[Path, Path]:
    v1, _ = _write(tmp_path, make_dataset())
    return v1, edit_version(v1, tmp_path, dataset_version=Version(1, 0, 1), remove_record_ids=(3,), cascade=True)


def _watermarks(version_dir: Path) -> dict[str, int]:
    rows = _column(version_dir, "SELECT key, value FROM control_metadata WHERE starts_with(key, 'max_id.')")
    return {key.removeprefix("max_id."): int(value) for key, value in rows}


def test_an_edit_keeps_ids_and_records_the_watermarks_of_its_base(tmp_path):
    v1, v2 = _edit_without_record_3(tmp_path)
    v1_tables, v2_tables = _tables(v1), _tables(v2)

    assert _column(v2, "SELECT record_id FROM records ORDER BY rowid") == [1, 2]
    for table in ("records", "sources", "signals", "annotation_occurrences"):
        assert {repr(row) for row in v2_tables[table]} <= {repr(row) for row in v1_tables[table]}, table
    assert _column(v2, "SELECT task_id FROM tasks ORDER BY rowid") == [1, 2, 3, 4]
    watermarks = _watermarks(v2)
    for kind, table in {
        "record": "records",
        "source": "sources",
        "occurrence": "annotation_occurrences",
        "content": "annotation_contents",
        "task": "tasks",
    }.items():
        assert watermarks[kind] == max(row[0] for row in v1_tables[table]), kind
    assert watermarks["record"] == 3


def test_new_objects_in_an_edit_continue_above_the_watermark_and_reuse_stored_contents(tmp_path):
    _, v2 = _edit_without_record_3(tmp_path)
    watermarks = _watermarks(v2)
    edited = _read(v2)
    edited._metadata = edited.metadata.model_copy(update={"dataset_version": Version(1, 0, 2)})
    stored = edited.records[0].annotations[0]
    added = edited.add_record(record=_record(_signal("n", axis=RegularAxis.from_rate_hz(3))))
    fresh = added.annotate(Annotation(key="fresh", value=1))
    again = added.annotate(
        Annotation(
            key=stored.key,
            value=stored.value,
            unit=stored.unit,
            description=stored.description,
            metadata=stored.metadata,
        )
    )
    task = edited.add_task(task=ClassificationTask(inputs=(added,), targets=("x",)))
    edited.derive_schema()
    v3, ids = _write(tmp_path, edited)

    assert ids.records[added.id] == watermarks["record"] + 1 == 4
    assert ids.sources[added.sources[0].id] == watermarks["source"] + 1
    assert ids.tasks[task.id] == watermarks["task"] + 1
    assert ids.occurrences[_occurrence(fresh)] == watermarks["occurrence"] + 1
    assert ids.contents[fresh.id] == watermarks["content"] + 1
    assert ids.contents[again.id] == stored.id
    assert ids.records[edited.records[0].id] == edited.records[0].id
    v2_tables, v3_tables = _tables(v2), _tables(v3)
    for table in ("records", "sources", "signals", "tasks"):
        assert {repr(row) for row in v2_tables[table]} <= {repr(row) for row in v3_tables[table]}, table
    for table, column in _PRIMARY_IDS.items():
        stored_ids = _column(v3, f"SELECT {column} FROM {table}")  # noqa: S608
        assert len(stored_ids) == len(set(stored_ids)), table


def test_a_rewrite_keeps_annotation_order_and_occurrence_order(tmp_path):
    dataset = TimeFDataset(metadata=_metadata())
    shared = Annotation(key="x", value=1)
    first = dataset.add_record(record=_record(_signal("a")))
    second = dataset.add_record(record=_record(_signal("b")))
    first.annotate(shared)
    second.annotate(Annotation(key="y", value=2))
    second.annotate(shared)
    v1, _ = _write(tmp_path, dataset)

    restored = _read(v1)
    assert [annotation.key for annotation in restored.records[1].annotations] == ["y", "x"]
    v2 = edit_version(v1, tmp_path, dataset_version=Version(1, 0, 1))
    assert _tables(v2) == _tables(v1)
    assert _column(v2, "SELECT occurrence_id FROM annotation_occurrences ORDER BY rowid") == [1, 2, 3]


# ---- writing one in-memory dataset again ---------------------------------------------------------


def _assert_kept(earlier: FinalIds, later: FinalIds) -> None:
    for kind in ("record", "source", "signal", "axis", "content", "occurrence", "task"):
        assert {key: later[kind][key] for key in earlier[kind]} == dict(earlier[kind]), kind


def test_a_rewrite_after_adding_objects_keeps_every_earlier_id(tmp_path):
    dataset = TimeFDataset(metadata=_metadata())
    a = _signal("a", axis=RegularAxis.from_rate_hz(4))
    first = dataset.add_record(record=_record(a, _signal("b", axis=RegularAxis.from_rate_hz(2))))
    cohort = Annotation(key="cohort", value="A")
    first.annotate(cohort)
    note = first.annotate(Annotation(key="note", value="x"))
    dataset.add_task(task=ClassificationTask(inputs=(first,), targets=("x",)))
    v1, ids1 = _write(tmp_path, dataset)
    watermarks = _watermarks(v1)

    second = dataset.add_record(record=_record(_signal("c", axis=RegularAxis.from_rate_hz(4))))
    second.annotate(cohort)
    second.annotate(Annotation(key="note", value="x"))
    fresh = second.annotate(Annotation(key="fresh", value=1))
    site = dataset.annotate(Annotation(key="site", value="lab"))
    task = dataset.add_task(
        task=ClassificationTask(
            inputs=(first,), targets=("y",), scope=TimeInterval.seconds(0.0, 0.5, time_series_ids=(a.id,))
        )
    )
    dataset._metadata = _metadata(Version(1, 0, 1))
    dataset.derive_schema()
    v2, ids2 = _write(tmp_path, dataset)

    _assert_kept(ids1, ids2)
    for table in _PRIMARY_IDS:
        assert {repr(row) for row in _tables(v1)[table]} <= {repr(row) for row in _tables(v2)[table]}, table
    assert ids2.records[second.id] == watermarks["record"] + 1
    assert ids2.sources[second.sources[0].id] == watermarks["source"] + 1
    assert ids2.signals[second.signals[0].id] == watermarks["signal"] + 1
    assert ids2.tasks[task.id] == watermarks["task"] + 1
    assert {ids2.contents[site.id], ids2.contents[fresh.id]} == {watermarks["content"] + 1, watermarks["content"] + 2}
    assert ids2.occurrences[_occurrence(fresh)] > watermarks["occurrence"]
    assert ids2.occurrences[_occurrence(site)] > watermarks["occurrence"]
    assert ids2.contents[second.annotations[1].id] == ids1.contents[note.id]
    assert ids2.axes[second.signals[0].time_axis.id] == ids1.axes[a.time_axis.id]
    assert _count(v2, "axes") == 2
    assert _count(v2, "annotation_contents") == 4
    scope = _column(v2, f"SELECT scope_signal_ids FROM tasks WHERE task_id = {ids2.tasks[task.id]}")  # noqa: S608
    assert scope == [[ids1.signals[a.id]]]
    assert_datasets_equal(dataset, _read(v2), final_ids=ids2)


def test_removed_records_leave_gaps_that_a_rewrite_does_not_fill(tmp_path):
    dataset = _records_dataset([None, None, None])
    first, removed, last = dataset.records
    _, ids1 = _write(tmp_path, dataset)

    edited = remove_records(dataset, [removed.id])
    edited._metadata = _metadata(Version(1, 0, 1))
    added = edited.add_record(record=_record(_signal("new")))
    edited.derive_schema()
    v2, ids2 = _write(tmp_path, edited)

    assert ids1.records[removed.id] == 2
    assert _column(v2, "SELECT record_id FROM records ORDER BY rowid") == [1, 3, 4]
    assert (ids2.records[first.id], ids2.records[last.id], ids2.records[added.id]) == (1, 3, 4)
    assert ids2.signals[added.signals[0].id] == 4
    assert _watermarks(v2)["record"] == 4


def test_an_unchanged_dataset_written_twice_gives_identical_catalogs(tmp_path):
    dataset = _tree_dataset()
    first, first_ids = _write(tmp_path / "first", dataset)
    second, second_ids = _write(tmp_path / "second", dataset)
    assert _tables(first) == _tables(second)
    assert first_ids == second_ids


def test_an_explicit_id_cannot_take_an_id_an_earlier_write_assigned(tmp_path):
    dataset = _records_dataset([None])
    _write(tmp_path / "first", dataset)
    dataset.add_record(record=_record(_signal("explicit"), id=1))
    dataset.derive_schema()
    with pytest.raises(TimeFValidationError, match="two different record objects get the final id 1"):
        _write(tmp_path / "second", dataset)
    assert not (tmp_path / "second" / dataset.metadata.dataset_id / "1.0.0").exists()


# ---- streamed tasks -------------------------------------------------------------------------------


def _stream_dataset(*, n_tasks: int, explicit: tuple[int, ...] = ()) -> TimeFDataset:
    """A stream whose second task derives from the last one, so the parent streams after its child."""
    dataset = TimeFDataset(metadata=_metadata())
    record = dataset.add_record(record=_record(_signal("a", 8)))
    context = record.annotate(Annotation(key="context", value="rest"))
    captured_signal_id = record.signals[0].id

    def tasks():
        ids = [*explicit, *(new_id() for _ in range(n_tasks - len(explicit)))]
        parent = ClassificationTask(id=ids[-1], inputs=(record,), targets=("parent",))
        for index, task_id in enumerate(ids):
            if index == 1:
                task = AnswerTask(id=task_id, inputs=(record,), prompt="q", targets=("a",), from_tasks=(parent,))
                task.annotate(Annotation(key="context", value="rest"))
            elif index == n_tasks - 1:
                task = parent
            else:
                task = ClassificationTask(
                    id=task_id,
                    inputs=(record,),
                    targets=(str(index),),
                    scope=TimeInterval.seconds(0.0, 1.0, time_series_ids=(captured_signal_id,)),
                    input_annotations=(context,),
                )
                task.annotate(Annotation(key="batch", value=index % 3))
            yield task

    dataset.set_task_stream([ClassificationTask, AnswerTask], tasks)
    return dataset


def test_streamed_tasks_are_numbered_in_stream_order(tmp_path, monkeypatch):
    monkeypatch.setattr(control_writer, "_TASK_BATCH_SIZE", 4)
    dataset = _stream_dataset(n_tasks=25)
    version_dir, ids = _write(tmp_path, dataset)

    assert _column(version_dir, "SELECT task_id FROM tasks ORDER BY rowid") == list(range(1, 26))
    assert sorted(ids.tasks.values()) == list(range(1, 26))
    assert _column(version_dir, "SELECT task_id, parent_task_id FROM task_dependencies") == [(2, 25)]
    signal_id = ids.signals[dataset.records[0].signals[0].id]
    assert set(_column(version_dir, "SELECT DISTINCT list_max(scope_signal_ids) FROM tasks")) == {signal_id, None}
    occurrences = _column(version_dir, "SELECT occurrence_id, object_type FROM annotation_occurrences ORDER BY rowid")
    assert [occurrence_id for occurrence_id, _ in occurrences] == list(range(1, len(occurrences) + 1))
    assert occurrences[0][1] == "Record"
    assert {object_type for _, object_type in occurrences[1:]} == {"Task"}
    assert _column(version_dir, "SELECT content_id, name FROM annotation_contents ORDER BY rowid") == [
        (1, "context"),
        (2, "batch"),
        (3, "batch"),
        (4, "batch"),
    ]
    assert _column(version_dir, "SELECT content_id FROM annotation_occurrences WHERE object_id = 2") == [1]
    referenced = _column(version_dir, "SELECT DISTINCT occurrence_id FROM task_annotation_refs")
    assert referenced == [ids.occurrences[_occurrence(dataset.records[0].annotations[0])]]

    restored = {task.id: task for task in _read(version_dir).tasks}
    assert [parent.id for parent in restored[2].from_tasks] == [25]
    assert restored[2].from_tasks[0] is restored[25]


def test_a_stream_that_builds_new_tasks_numbers_them_above_the_last_write(tmp_path):
    dataset = _stream_dataset(n_tasks=6)
    first, _ = _write(tmp_path / "first", dataset)
    second, _ = _write(tmp_path / "second", dataset)

    assert _column(first, "SELECT task_id FROM tasks ORDER BY rowid") == list(range(1, 7))
    assert _column(second, "SELECT task_id FROM tasks ORDER BY rowid") == list(range(7, 13))
    first_tables, second_tables = _tables(first), _tables(second)
    for table in ("records", "sources", "signals", "axes", "annotation_contents"):
        assert first_tables[table] == second_tables[table], table


def test_a_stream_of_the_same_task_objects_keeps_its_ids_across_writes(tmp_path):
    dataset = TimeFDataset(metadata=_metadata())
    record = dataset.add_record(record=_record(_signal("a", 8)))
    tasks = [ClassificationTask(inputs=(record,), targets=(str(index),)) for index in range(3)]
    for index, task in enumerate(tasks):
        task.annotate(Annotation(key="batch", value=index % 2))
    dataset.set_task_stream([ClassificationTask], lambda: iter(tasks))
    first, ids1 = _write(tmp_path / "first", dataset)
    second, ids2 = _write(tmp_path / "second", dataset)

    assert _tables(first) == _tables(second)
    assert ids1 == ids2

    later = ClassificationTask(inputs=(record,), targets=("3",))
    added = later.annotate(Annotation(key="batch", value=0))
    tasks.insert(0, later)
    third, ids3 = _write(tmp_path / "third", dataset)

    _assert_kept(ids1, ids3)
    assert ids3.tasks[later.id] == 4
    assert ids3.occurrences[_occurrence(added)] == 4
    assert ids3.contents[added.id] == ids1.contents[tasks[1].annotations[0].id]
    assert _column(third, "SELECT task_id FROM tasks ORDER BY rowid") == [4, 1, 2, 3]


def test_a_stream_with_explicit_ids_keeps_them(tmp_path):
    dataset = _stream_dataset(n_tasks=4, explicit=(40, 41, 42, 43))
    version_dir, ids = _write(tmp_path, dataset)
    assert dict(ids.tasks) == {40: 40, 41: 41, 42: 42, 43: 43}
    assert _column(version_dir, "SELECT task_id, parent_task_id FROM task_dependencies") == [(41, 43)]


def test_a_stream_that_mixes_explicit_and_provisional_ids_is_rejected(tmp_path):
    dataset = _stream_dataset(n_tasks=4, explicit=(40,))
    with pytest.raises(TimeFValidationError, match="explicit id below 2\\*\\*52, or none"):
        _write(tmp_path, dataset)


def test_a_stream_that_repeats_a_task_reports_the_duplicate(tmp_path):
    dataset = TimeFDataset(metadata=_metadata())
    record = dataset.add_record(record=_record(_signal("a")))
    task = ClassificationTask(inputs=(record,), targets=("x",))
    dataset.set_task_stream([ClassificationTask], lambda: iter((task, task)))
    with pytest.raises(TimeFValidationError, match=f"task id {task.id} is not unique"):
        _write(tmp_path, dataset)


def test_a_streamed_parent_that_never_streams_is_rejected(tmp_path):
    dataset = TimeFDataset(metadata=_metadata())
    record = dataset.add_record(record=_record(_signal("a")))
    missing = ClassificationTask(inputs=(record,), targets=("missing",))
    child = AnswerTask(inputs=(record,), prompt="q", targets=("a",), from_tasks=(missing,))
    dataset.set_task_stream([AnswerTask], lambda: iter((child,)))
    with pytest.raises(TimeFValidationError, match=f"depends on task {missing.id}, which is not in the dataset"):
        _write(tmp_path, dataset)


def test_an_explicit_occurrence_id_in_a_stream_cannot_take_an_assigned_id(tmp_path):
    dataset = TimeFDataset(metadata=_metadata())
    record = dataset.add_record(record=_record(_signal("a")))
    record.annotate(Annotation(key="k", value=1))
    task = ClassificationTask(inputs=(record,), targets=("x",))
    task.annotations = (Annotation(key="t", value=1, occurrence_id=1),)
    dataset.set_task_stream([ClassificationTask], lambda: iter((task,)))
    with pytest.raises(TimeFValidationError, match="explicit occurrence id 1, which the writer may already"):
        _write(tmp_path, dataset)


# ---- reads ----------------------------------------------------------------------------------------


def test_keyed_annotation_reads_stay_within_their_object_type(tmp_path):
    dataset = TimeFDataset(metadata=_metadata())
    signal = _signal("a")
    source = Source(name="s", signals=(signal,))
    record = dataset.add_record(record=Record(sources=(source,)))
    record.annotate(Annotation(key="owner", value="record"))
    source.annotate(Annotation(key="owner", value="source"))
    signal.annotate(Annotation(key="owner", value="signal"))
    task = ClassificationTask(inputs=(record,), targets=("x",))
    task.annotate(Annotation(key="owner", value="task"))
    dataset.add_task(task=task)
    version_dir, ids = _write(tmp_path, dataset)
    assert {ids.records[record.id], ids.sources[source.id], ids.signals[signal.id], ids.tasks[task.id]} == {1}

    with DuckDBControlReader(version_dir / "control.duckdb") as reader:
        keyed = reader._read_annotations({"Record": [1]})
        assert list(keyed) == [("Record", 1)]
        assert [item.value for item in keyed["Record", 1]] == ["record"]
        (back,) = reader.read_records((1,))
    assert [item.value for item in back.annotations] == ["record"]
    assert [item.value for item in back.sources[0].annotations] == ["source"]
    assert [item.value for item in back.signals[0].annotations] == ["signal"]


@pytest.mark.parametrize("workers", [2, 3, 4])
def test_records_shard_by_id_modulo_and_by_id_range(tmp_path, workers):
    dataset = _records_dataset([None] * 13)
    version_dir, _ = _write(tmp_path, dataset)

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        all_ids = reader.record_ids()
        assert all_ids == tuple(range(1, 14))
        by_modulo = [{record_id for record_id in all_ids if record_id % workers == worker} for worker in range(workers)]
        assert sum(len(shard) for shard in by_modulo) == len(all_ids)
        assert set().union(*by_modulo) == set(all_ids)
        for worker, shard in enumerate(by_modulo):
            names = [
                signal.name for record in reader.iter_records(record_ids=sorted(shard)) for signal in record.signals
            ]
            assert names == [f"s{record_id - 1}" for record_id in sorted(shard)], worker
        size = -(-len(all_ids) // workers)
        names: list[str] = []
        for worker in range(workers):
            first = worker * size + 1
            last = min(first + size - 1, len(all_ids))
            shard = list(reader.iter_records(record_ids=range(first, last + 1)))
            names.extend(signal.name for record in shard for signal in record.signals)
    assert names == [f"s{index}" for index in range(13)]


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_values_are_read_back_through_final_ids(tmp_path, backend):
    dataset = _records_dataset([None] * 3)
    version_dir, ids = _write(tmp_path, dataset, values_backend=backend)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        (record,) = reader.iter_records(record_ids=(ids.records[dataset.records[1].id],))
        assert record.signals[0].to_arrow().equals(pa.array([0.0, 1.0, 2.0, 3.0], pa.float32()))
    assert _column(version_dir, "SELECT signal_id FROM signal_chunks ORDER BY rowid") == [1, 2, 3]


# ---- explicit ids and limits ----------------------------------------------------------------------


def test_explicit_ids_at_or_above_the_limit_are_provisional(tmp_path):
    dataset = _records_dataset([FINAL_ID_LIMIT + 5, None])
    version_dir, ids = _write(tmp_path, dataset)
    assert ids.records[FINAL_ID_LIMIT + 5] == 1
    assert _column(version_dir, "SELECT record_id FROM records ORDER BY rowid") == [1, 2]


def test_explicit_duplicates_are_still_rejected(tmp_path):
    dataset = _records_dataset([3, None])
    dataset._records[1].id = 3
    with pytest.raises(TimeFValidationError, match="record id 3 is not unique"):
        _write(tmp_path, dataset)


def test_running_out_of_final_ids_fails_before_anything_is_written(tmp_path):
    dataset = _records_dataset([FINAL_ID_LIMIT - 2, None, None])
    before = _snapshot(dataset)
    with pytest.raises(TimeFValidationError, match="2 new record ids"):
        _write(tmp_path, dataset)
    assert _snapshot(dataset) == before
    assert not (tmp_path / dataset.metadata.dataset_id / "1.0.0").exists()


def test_the_control_writer_on_its_own_stores_only_final_ids(tmp_path):
    path = tmp_path / "control.duckdb"
    dataset = _tree_dataset()
    DuckDBControlWriter(path).write_hierarchy(dataset)
    with duckdb.connect(str(path), read_only=True) as connection:
        assert connection.execute("SELECT record_id FROM records ORDER BY rowid").fetchall() == [(1,), (2,)]
        largest = connection.execute("SELECT max(task_id), max(parent_task_id) FROM task_dependencies").fetchone()
    assert largest == (2, 1)
    assert all(is_provisional(record.id) for record in dataset.records)


def test_task_is_not_merged_with_its_parent_when_both_are_equal(tmp_path):
    dataset = _records_dataset([None])
    (record,) = dataset.records
    parent = ClassificationTask(inputs=(record,), targets=("x",))
    child = ClassificationTask(inputs=(record,), targets=("x",), from_tasks=(parent,))
    dataset.add_tasks(tasks=[parent, child])
    assert isinstance(child, Task)

    version_dir, ids = _write(tmp_path, dataset)

    assert (ids.tasks[parent.id], ids.tasks[child.id]) == (1, 2)
    assert _column(version_dir, "SELECT task_id, parent_task_id FROM task_dependencies") == [(2, 1)]


def test_the_control_writer_on_its_own_numbers_a_task_stream(tmp_path):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(_stream_dataset(n_tasks=3))
    with duckdb.connect(str(path), read_only=True) as connection:
        assert connection.execute("SELECT task_id FROM tasks ORDER BY rowid").fetchall() == [(1,), (2,), (3,)]
        assert connection.execute("SELECT task_id, parent_task_id FROM task_dependencies").fetchall() == [(2, 3)]


def test_the_dataset_exposes_the_ids_of_its_last_write(tmp_path):
    dataset = make_dataset()
    assert dataset.final_ids is None
    _, ids = _write(tmp_path, dataset)
    assert dataset.final_ids == ids
