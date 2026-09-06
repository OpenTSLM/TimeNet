from collections import Counter
from pathlib import Path

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.testing import assert_datasets_equal
from timenet.types import ClassificationTask
from timenet_connectors.datasets.timenet.test_mean import TestMeanConnector


def _convert() -> TimeFDataset:
    return TestMeanConnector().convert([])


def _labels_by_record(dataset: TimeFDataset) -> dict[str, str | None]:
    return {
        record_id: task.target
        for task in dataset.tasks
        if isinstance(task, ClassificationTask)
        for record_id in task.record_ids
    }


def test_is_a_connector():
    assert isinstance(TestMeanConnector(), BaseConnector)


def test_metadata():
    assert TestMeanConnector().metadata().dataset_id == "timenet/test-mean"


def test_download_defaults_to_no_op():
    # Synthetic connector: it generates everything in convert and inherits the no-op download.
    assert TestMeanConnector().download(Path("cache")) == []


def test_convert_is_deterministic():
    assert_datasets_equal(_convert(), _convert())


def test_has_balanced_binary_labels():
    labels = Counter(_labels_by_record(_convert()).values())
    assert labels == Counter({"above_zero": 500, "below_zero": 500})


def test_target_has_no_schema():
    assert all(task.target_schema is None for task in _convert().tasks_of(ClassificationTask))


def test_every_record_is_a_single_signal():
    dataset = _convert()
    assert len(dataset.records) == 1000
    for record in dataset.records:
        assert len(record.time_series) == 1


def test_label_agrees_with_signal_mean():
    # The task is solvable from the data: a mean-based classifier would recover every label.
    dataset = _convert()
    labels = _labels_by_record(dataset)
    for record in dataset.records:
        mean = float(record.time_series[0].to_numpy().mean())
        expected = "above_zero" if mean > 0 else "below_zero"
        assert labels[record.record_id] == expected
