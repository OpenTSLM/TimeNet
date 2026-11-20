from collections import Counter
from typing import cast

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.testing import assert_datasets_equal
from timenet.types import ClassificationTask, read_value
from timenet_connectors.datasets.timenet.test_mean import TestMeanConnector


def _convert() -> TimeFDataset:
    return TestMeanConnector().convert(None)


def _labels_by_record(dataset: TimeFDataset) -> dict[int | None, str | None]:
    return {
        record.id: cast(str, read_value(task.targets[0]))
        for task in dataset.tasks
        if isinstance(task, ClassificationTask) and task.targets is not None
        for record in task.inputs
    }


def test_is_a_connector():
    assert isinstance(TestMeanConnector(), BaseConnector)


def test_metadata():
    assert TestMeanConnector().metadata().dataset_id == "timenet/test-mean"


def test_convert_is_deterministic():
    assert_datasets_equal(_convert(), _convert())


def test_has_balanced_binary_labels():
    labels = Counter(_labels_by_record(_convert()).values())
    assert labels == Counter({"above_zero": 500, "below_zero": 500})


def test_every_task_shares_the_mean_sign_label_set():
    label_sets = [task.label_set for task in _convert().tasks_of(ClassificationTask)]
    names = {read_value(a.value) for label_set in label_sets for a in label_set.annotations if a.key == "name"}
    assert names == {"mean_sign"}
    assert len({id(label_set) for label_set in label_sets}) == 1


def test_every_record_is_a_single_signal():
    dataset = _convert()
    assert len(dataset.records) == 1000
    for record in dataset.records:
        assert len(record.signals) == 1


def test_label_agrees_with_signal_mean():
    # The task is solvable from the data: a mean-based classifier would recover every label.
    dataset = _convert()
    labels = _labels_by_record(dataset)
    for record in dataset.records:
        mean = float(record.signals[0].to_numpy().mean())
        expected = "above_zero" if mean > 0 else "below_zero"
        assert labels[record.id] == expected
