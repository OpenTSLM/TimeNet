from fractions import Fraction

import pyarrow as pa
import pytest

from timenet.dataset import IrregularAxis, Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.format.control_writer import DuckDBControlWriter
from timenet.format.duckdb import CONTROL_FILE
from timenet.format.storage import StorageSources
from timenet.format.stored_values import PackedValueWriter
from timenet.types import Annotation, AnswerTask, DatasetMetadata, License, TimeSeriesSpec, Version, ureg


SPEC = TimeSeriesSpec(spec_type="voltage", name="Voltage", unit_value=ureg.millivolt, dtype="float32")


@pytest.fixture
def control_dataset() -> TimeFDataset:
    axis = RegularAxis(axis_id="axis-1", period_us=Fraction(2_000))
    lead_i = Signal.from_loader(
        id="lead-i",
        name="I",
        spec=SPEC,
        time_axis=axis,
        n_values=2,
        loader=lambda: pa.array([1.0, 2.0], type=pa.float32()),
    )
    pulse = Signal.from_loader(
        id="pulse",
        name="Pulse",
        spec=SPEC,
        time_axis=IrregularAxis(axis_id="axis-2", first_us=0, last_us=5_000),
        n_values=2,
        loader=lambda: pa.array([1.0, 2.0], type=pa.float32()),
        time_offsets_loader=lambda: pa.array([0, 5_000], type=pa.int64()),
    )
    ecg = Source(id="ecg", name="ECG", signals=(lead_i, pulse))
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
def control_path(tmp_path, control_dataset):
    path = tmp_path / CONTROL_FILE
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent, StorageSources())).write_hierarchy(
        control_dataset
    )
    return path
