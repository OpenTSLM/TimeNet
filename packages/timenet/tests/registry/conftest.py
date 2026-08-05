from pathlib import Path

import pytest

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.testing import make_dataset, sine_loader
from timenet.types import (
    Annotation,
    ClassificationTask,
    DatasetMetadata,
    Domain,
    License,
    TimeSeriesSpec,
    Version,
    View,
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
        unit_sampling_rate=ureg.hertz,
        unit_timestamp=ureg.second,
        unit_value=ureg.millivolt,
    )
    series = TimeSeries(
        spec=spec,
        channel="II",
        sampling_rate_hz=16.0,
        loader=sine_loader(n=16, sampling_rate_hz=16.0),
        time_series_id="ecg-ts-0",
        t_start_s=0.0,
        t_end_s=1.0,
    )
    sample = dataset.add_sample(time_series=(series,), view=View.FULL, sample_id="ecg-sample-0")
    sample.add_annotation(Annotation(key="age", value=70, unit="years", id="ecg-age-0"))
    dataset.add_task(sample, ClassificationTask(target="afib", id="ecg-task-0"))
    return dataset


@pytest.fixture
def registry_root(tmp_path) -> Path:
    """A local registry directory holding two datasets: timenet/hello-world and demo/ecg."""
    for dataset in (make_dataset(), _ecg_dataset()):
        dataset.derive_schema()
        with TimeFWriter(tmp_path, dataset) as writer:
            writer.write()
    return tmp_path
