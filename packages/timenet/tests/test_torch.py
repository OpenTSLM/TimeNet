import pytest


torch = pytest.importorskip("torch")

from torch.utils.data import Dataset  # noqa: E402

from timenet.errors import TimeFValidationError  # noqa: E402
from timenet.testing import make_dataset  # noqa: E402
from timenet.torch import TimeFTorchDataset  # noqa: E402


def _ds():
    return TimeFTorchDataset(make_dataset())


def test_is_torch_dataset():
    assert isinstance(_ds(), Dataset)


def test_len_matches_samples():
    assert len(_ds()) == len(make_dataset().samples)


def test_getitem_structure():
    item = _ds()[0]
    assert isinstance(item["sample_id"], str)
    assert item["series"]
    assert all(isinstance(t, torch.Tensor) and t.dtype == torch.float32 for t in item["series"])


def test_getitem_values_match():
    dataset = make_dataset()
    item = TimeFTorchDataset(dataset)[0]
    expected = dataset.samples[0].time_series[0].to_numpy()
    assert item["series"][0].numpy().tolist() == expected.tolist()


def test_tasks_resolved():
    ds = _ds()
    assert any(ds[i]["tasks"] for i in range(len(ds))), "expected at least one sample with a task"


def test_transform_applied():
    ds = TimeFTorchDataset(make_dataset(), transform=lambda item: item["series"][0])
    assert isinstance(ds[0], torch.Tensor)


def test_dangling_task_id_raises():
    dataset = make_dataset()
    dataset.samples[0].task_ids = ("no-such-task",)
    with pytest.raises(TimeFValidationError, match="unknown task id"):
        TimeFTorchDataset(dataset)[0]
