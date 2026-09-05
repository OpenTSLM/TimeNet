import pytest


torch = pytest.importorskip("torch")

from torch.utils.data import Dataset  # noqa: E402

from timenet.dataset import TimeFDataset, TimeSeries  # noqa: E402
from timenet.dataset.axis import RegularAxis  # noqa: E402
from timenet.errors import TimeFValidationError  # noqa: E402
from timenet.testing import make_dataset  # noqa: E402
from timenet.torch import TimeFTorchDataset  # noqa: E402
from timenet.types import (  # noqa: E402
    DatasetMetadata,
    Domain,
    License,
    TimeSeriesSpec,
    Version,
    ureg,
)


def _ds():
    return TimeFTorchDataset(make_dataset())


def test_is_torch_dataset():
    assert isinstance(_ds(), Dataset)


def test_len_matches_samples():
    assert len(_ds()) == len(make_dataset().records)


def test_getitem_structure():
    item = _ds()[0]
    assert isinstance(item["record_id"], str)
    assert item["series"]
    assert all(isinstance(t, torch.Tensor) and t.dtype == torch.float32 for t in item["series"])


def test_getitem_values_match():
    dataset = make_dataset()
    item = TimeFTorchDataset(dataset)[0]
    expected = dataset.records[0].time_series[0].to_numpy()
    assert item["series"][0].numpy().tolist() == expected.tolist()


def test_tasks_resolved():
    ds = _ds()
    assert any(ds[i]["tasks"] for i in range(len(ds))), "expected at least one record with a task"


def test_transform_applied():
    ds = TimeFTorchDataset(make_dataset(), transform=lambda item: item["series"][0])
    assert isinstance(ds[0], torch.Tensor)


def test_dangling_task_id_raises():
    dataset = make_dataset()
    dataset.records[0].task_ids = ("no-such-task",)
    with pytest.raises(TimeFValidationError, match="unknown task id"):
        TimeFTorchDataset(dataset)[0]


def _typed_dataset(dtype, values):
    spec = TimeSeriesSpec(
        spec_type=f"chan_{dtype}",
        name=dtype,
        unit_value=ureg.dimensionless,
        dtype=dtype,
    )
    ts = TimeSeries.from_values(values, spec=spec, signal="c", time_axis=RegularAxis.from_rate_hz(1))
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/torch",
            dataset_version=Version(1, 0, 0),
            name="Torch typed fixture",
            description="A single series of one dtype.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    dataset.add_record(time_series=(ts,), record_id="record-0")
    return dataset


def test_bool_series_stays_a_bool_tensor():
    item = TimeFTorchDataset(_typed_dataset("bool", [True, False, True]))[0]
    assert item["series"][0].dtype == torch.bool
    assert item["series"][0].tolist() == [True, False, True]


def test_int16_series_stays_an_int16_tensor():
    item = TimeFTorchDataset(_typed_dataset("int16", [1, 2, 3]))[0]
    assert item["series"][0].dtype == torch.int16


def test_str_series_has_no_tensor_representation():
    dataset = _typed_dataset("str", ["awake", "deep"])
    with pytest.raises(TimeFValidationError, match="no tensor representation"):
        TimeFTorchDataset(dataset)[0]
