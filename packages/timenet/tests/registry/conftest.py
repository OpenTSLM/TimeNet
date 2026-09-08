from pathlib import Path
import sys

import pytest


# Make sibling test helpers (e.g. ``_fake_registry``) importable by name under
# pytest's importlib import mode, which does not add test dirs to ``sys.path``.
sys.path.insert(0, str(Path(__file__).parent))

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.testing import make_dataset, sine_loader
from timenet.types import (
    Annotation,
    ClassificationTask,
    DatasetMetadata,
    Domain,
    License,
    TimeSeriesSpec,
    Version,
    ureg,
)
from timenet.writer import TimeFWriter


def _ecg_dataset() -> TimeFDataset:
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="demo/ecg",
            dataset_version=Version(2, 0, 0),
            name="ECG Dataset",
            description="A clinical ECG dataset.",
            license=License.MIT,
            domains=(Domain.CARDIOLOGY,),
            tags=("clinical",),
        )
    )
    spec = TimeSeriesSpec(
        spec_type="ecg_lead",
        name="ECG Lead",
        unit_value=ureg.millivolt,
    )
    series = TimeSeries(
        spec=spec,
        signal="II",
        time_axis=RegularAxis.from_rate_hz(16),
        loader=sine_loader(n=16, sampling_rate_hz=16.0),
        time_series_id="ecg-ts-0",
        n_values=16,
    )
    record = dataset.add_record(time_series=(series,), record_id="ecg-record-0")
    record.add_annotation(Annotation(key="age", value=70, unit="years", id="ecg-age-0"))
    dataset.add_task(record, ClassificationTask(target="afib", id="ecg-task-0"))
    return dataset


@pytest.fixture
def registry_root(tmp_path) -> Path:
    """A local registry directory holding two datasets: timenet/hello-world and demo/ecg."""
    for dataset in (make_dataset(), _ecg_dataset()):
        dataset.derive_schema()
        with TimeFWriter(tmp_path, dataset) as writer:
            writer.write()
    return tmp_path
