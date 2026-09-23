from fractions import Fraction

import pyarrow as pa
import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.format.control_audit import audit_control_database
from timenet.format.control_writer import DuckDBControlWriter
from timenet.format.duckdb import connect_control
from timenet.types import Annotation, AnswerTask, DatasetMetadata, License, TimeSeriesSpec, Version, ureg


SPEC = TimeSeriesSpec(spec_type="voltage", name="Voltage", unit_value=ureg.millivolt, dtype="float32")


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
    ecg = Source(id="ecg", name="ECG", signals=(lead_i,))
    monitor = Source(id="monitor", name="Monitor", sources=(ecg,))
    record = Record(record_id="record-1", sources=(monitor,))
    record.annotate(Annotation(id="sex-male", key="patient_sex", value="male"))
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="org/control-audit",
            dataset_version=Version(1, 0, 0),
            name="Control audit",
            description="Test hierarchy",
            license=License.MIT,
        )
    )
    dataset.add_record(record=record)
    dataset.add_task(task=AnswerTask(id="task-1", inputs=(record,), prompt="Alive?", targets=("Yes",)))
    return dataset


@pytest.fixture
def control_path(tmp_path):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(_dataset())
    return path


def test_audit_accepts_a_freshly_written_database(control_path):
    with connect_control(control_path, read_only=True) as connection:
        audit_control_database(connection, require_chunks=False)


def test_audit_requires_chunks_unless_told_otherwise(control_path):
    with (
        connect_control(control_path, read_only=True) as connection,
        pytest.raises(TimeFFormatError, match="value chunks whose lengths do not match"),
    ):
        audit_control_database(connection)


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        (
            "INSERT INTO records (record_key, record_id, metadata) SELECT record_key, record_id, metadata FROM records",
            "records contains duplicate values for record_key",
        ),
        (
            "UPDATE sources SET record_key = -1 WHERE source_id = 'monitor'",
            r"sources\.record_key refers to missing records\.record_key",
        ),
        (
            "UPDATE sources SET parent_source_key = source_key WHERE source_id = 'ecg'",
            r"invalid sources\.parent_source_key value",
        ),
        (
            "UPDATE signals SET n_values = 0",
            r"invalid signals\.n_values value",
        ),
        (
            "INSERT INTO records (record_key, record_id, metadata) VALUES (-1, 'ghost', '{}'); "
            "UPDATE sources SET record_key = -1 WHERE source_id = 'ecg'",
            "belong to different records",
        ),
        (
            "UPDATE sources SET parent_source_key = (SELECT source_key FROM sources WHERE source_id = 'ecg') "
            "WHERE source_id = 'monitor'",
            "contains a cycle",
        ),
        (
            "INSERT INTO axis_offsets SELECT axis_key, 0, 0 FROM axes",
            "axis length that does not match",
        ),
    ],
)
def test_audit_names_the_first_violated_invariant(control_path, statement, message):
    with connect_control(control_path) as connection:
        for part in statement.split("; "):
            connection.execute(part)
        with pytest.raises(TimeFFormatError, match=message):
            audit_control_database(connection, require_chunks=False)
