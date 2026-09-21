from fractions import Fraction
from types import SimpleNamespace

import pyarrow as pa
import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.format.control_writer import DuckDBControlWriter
from timenet.format.duckdb import connect_control
from timenet.types import Annotation, AnswerTask, DatasetMetadata, License, TimeSeriesSpec, Version, ureg


SPEC = TimeSeriesSpec(
    spec_type="voltage",
    name="Voltage",
    unit_value=ureg.millivolt,
    dtype="float32",
)


def _dataset() -> TimeFDataset:
    axis = RegularAxis(axis_id="axis-1", period_us=Fraction(2_000))
    lead_i = Signal.from_loader(
        id="lead-i",
        name="I",
        spec=SPEC,
        time_axis=axis,
        n_values=2,
        loader=lambda: pa.array([1.0, 2.0], type=pa.float32()),
    )
    lead_ii = Signal.from_loader(
        id="lead-ii",
        name="II",
        spec=SPEC,
        time_axis=axis,
        n_values=2,
        loader=lambda: pa.array([3.0, 4.0], type=pa.float32()),
    )
    ecg = Source(id="ecg", name="ECG", signals=(lead_i, lead_ii))
    monitor = Source(id="monitor", name="Monitor", sources=(ecg,))
    record = Record(record_id="record-1", sources=(monitor,))
    record.add_annotation(Annotation(id="sex-male", key="patient_sex", value="male"))
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="org/control-writer",
            dataset_version=Version(1, 0, 0),
            name="Control writer",
            description="Test hierarchy",
            license=License.MIT,
        )
    )
    dataset.add_record(record=record)
    dataset.annotate(Annotation(id="site", key="site", value="lab"))
    task = AnswerTask(id="task-1", inputs=(record,), prompt="Alive?", target="Yes")
    task.annotate(Annotation(id="task-kind", key="task_kind", value="diagnosis"))
    dataset.add_task(task=task)
    return dataset


def test_control_writer_serializes_recursive_hierarchy_and_shared_axis(tmp_path):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(_dataset())

    with connect_control(path, read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM records").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM sources").fetchone() == (2,)
        assert connection.execute("SELECT count(*) FROM signals").fetchone() == (2,)
        assert connection.execute("SELECT count(*) FROM axes").fetchone() == (1,)
        assert connection.execute("SELECT task_type, prompt, payload FROM tasks").fetchone() == (
            "answer",
            "Alive?",
            '{"target":"Yes"}',
        )
        assert connection.execute("SELECT field, record_id FROM task_record_refs").fetchone() == (
            "inputs",
            "record-1",
        )
        assert connection.execute("SELECT parent_source_id FROM sources WHERE source_id = 'ecg'").fetchone() == (
            "monitor",
        )
        assert connection.execute(
            """SELECT content_id, object_type, object_id
               FROM annotation_occurrences WHERE content_id = 'sex-male'"""
        ).fetchone() == ("sex-male", "Record", "record-1")
        assert connection.execute(
            """SELECT content_id, object_type, object_id
               FROM annotation_occurrences WHERE content_id = 'site'"""
        ).fetchone() == ("site", "Dataset", "org/control-writer")


def test_control_writer_removes_database_after_validation_failure(tmp_path):
    dataset = _dataset()
    annotation = dataset.records[0].annotations[0]
    dataset.records[0].annotate(
        Annotation(
            id=annotation.content_id,
            key="patient_sex",
            value="female",
        )
    )
    path = tmp_path / "control.duckdb"

    with pytest.raises(TimeFValidationError, match="reused with different content"):
        DuckDBControlWriter(path).write_hierarchy(dataset)

    assert not path.exists()


def test_control_writer_rejects_ambiguous_annotation_target_id(tmp_path):
    dataset = _dataset()
    dataset.records[0].sources[0].id = dataset.records[0].record_id
    path = tmp_path / "control.duckdb"

    with pytest.raises(TimeFValidationError, match="ambiguous across object types"):
        DuckDBControlWriter(path).write_hierarchy(dataset)

    assert not path.exists()


def test_control_writer_stores_value_chunk_locations(tmp_path):
    path = tmp_path / "control.duckdb"
    placement = SimpleNamespace(
        chunk_file="values/part-00000000.parquet",
        data_index=SimpleNamespace(major_idx=2, minor_idx=3),
        spec_type="voltage",
        signal="I",
        n_values=2,
    )
    placements = {
        ("lead-i", 0): placement,
        ("lead-ii", 0): SimpleNamespace(
            chunk_file=placement.chunk_file,
            data_index=SimpleNamespace(major_idx=2, minor_idx=4),
            spec_type="voltage",
            signal="II",
            n_values=2,
        ),
    }

    DuckDBControlWriter(path).write_hierarchy(_dataset(), placements)  # ty: ignore[invalid-argument-type]

    with connect_control(path, read_only=True) as connection:
        assert connection.execute(
            """SELECT value_path, chunk_major_index, chunk_minor_index, n_values
               FROM signal_chunks WHERE signal_id = 'lead-i'"""
        ).fetchone() == ("values/part-00000000.parquet", 2, 3, 2)
