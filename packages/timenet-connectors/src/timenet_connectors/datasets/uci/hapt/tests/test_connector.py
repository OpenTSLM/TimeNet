from pathlib import Path

import numpy as np
import pytest

from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import assert_datasets_equal
from timenet.types import ClassificationTask, Split, TimeInterval, read_value
from timenet_connectors.datasets.uci.hapt.connector import (
    ACTIVITY,
    ACTIVITY_NAMES,
    US_PER_SAMPLE,
    HaptConnector,
    HaptSource,
)


def _write_imu(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, values, fmt="%.17g")


def _write_tree(root: Path) -> HaptSource:
    raw_dir = root / "RawData"
    train_dir = root / "Train"
    test_dir = root / "Test"
    for directory in (raw_dir, train_dir, test_dir):
        directory.mkdir(parents=True)

    accelerometer_1 = np.arange(24, dtype=np.float64).reshape(8, 3) * 0.1
    gyroscope_1 = accelerometer_1 + 1.0
    accelerometer_2 = np.arange(18, dtype=np.float64).reshape(6, 3) * 0.2
    gyroscope_2 = accelerometer_2 + 2.0
    _write_imu(raw_dir / "acc_exp01_user01.txt", accelerometer_1)
    _write_imu(raw_dir / "gyro_exp01_user01.txt", gyroscope_1)
    _write_imu(raw_dir / "acc_exp02_user02.txt", accelerometer_2)
    _write_imu(raw_dir / "gyro_exp02_user02.txt", gyroscope_2)

    (raw_dir / "labels.txt").write_text("1 1 1 1 4\n1 1 7 5 6\n2 2 5 1 6\n", encoding="utf-8")
    (root / "activity_labels.txt").write_text(
        "".join(f"{activity_id} {ACTIVITY_NAMES[activity_id].upper()}\n" for activity_id in range(1, 13)),
        encoding="utf-8",
    )
    (train_dir / "subject_id_train.txt").write_text("1\n3\n", encoding="utf-8")
    (test_dir / "subject_id_test.txt").write_text("2\n4\n", encoding="utf-8")
    return HaptSource(root=root, raw_dir=raw_dir)


def _convert(root: Path):
    return HaptConnector().convert(_write_tree(root))


def test_each_trial_becomes_a_record_and_each_segment_a_scoped_task(tmp_path):
    dataset = _convert(tmp_path)
    records = dataset.records
    assert len(records) == 2

    by_experiment = {}
    for task in dataset.tasks_of(ClassificationTask):
        (record,) = task.inputs
        by_experiment.setdefault(record.metadata["experiment_id"], []).append(task)
    assert set(by_experiment) == {1, 2}

    first_record = next(record for record in records if record.metadata["experiment_id"] == 1)
    assert [signal.name for signal in first_record.signals] == [
        "accelerometer_x",
        "accelerometer_y",
        "accelerometer_z",
        "gyroscope_x",
        "gyroscope_y",
        "gyroscope_z",
    ]
    expected_accelerometer = np.arange(24, dtype=np.float64).reshape(8, 3) * 0.1
    expected_gyroscope = expected_accelerometer + 1.0
    for signal, column in zip(first_record.signals[:3], expected_accelerometer.T, strict=True):
        np.testing.assert_allclose(signal.to_numpy(), column, rtol=1e-12, atol=1e-12)
    for signal, column in zip(first_record.signals[3:], expected_gyroscope.T, strict=True):
        np.testing.assert_allclose(signal.to_numpy(), column, rtol=1e-12, atol=1e-12)

    walking, transition = by_experiment[1]
    assert read_value(walking.targets[0]) == "walking"
    assert walking.label_set == ACTIVITY
    assert walking.split == Split.TRAIN
    assert isinstance(walking.scope, TimeInterval)
    assert walking.scope.start_us == 0
    assert walking.scope.exclusive_end == 4 * US_PER_SAMPLE

    assert read_value(transition.targets[0]) == "stand_to_sit"
    assert transition.split == Split.TRAIN
    assert isinstance(transition.scope, TimeInterval)
    assert transition.scope.start_us == 4 * US_PER_SAMPLE
    assert transition.scope.exclusive_end == 6 * US_PER_SAMPLE

    (standing,) = by_experiment[2]
    assert read_value(standing.targets[0]) == "standing"
    assert standing.split == Split.TEST
    assert isinstance(standing.scope, TimeInterval)
    assert standing.scope.start_us == 0
    assert standing.scope.exclusive_end == 6 * US_PER_SAMPLE


def test_the_dataset_round_trips_through_storage(tmp_path):
    dataset = _convert(tmp_path / "source")
    version_dir = dataset.write(path=tmp_path / "registry")
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        assert_datasets_equal(dataset, reader.read())


@pytest.mark.parametrize(
    ("labels", "match"),
    [
        ("1 1 13 1 4\n", "unknown activity id 13"),
        ("1 1 1 1 99\n", "leaves the 8-sample trial"),
        ("1 1 1 5 4\n", "leaves the 8-sample trial"),
    ],
    ids=["unknown activity", "end past the trial", "end before start"],
)
def test_a_malformed_label_raises(tmp_path, labels, match):
    source = _write_tree(tmp_path)
    (source.raw_dir / "labels.txt").write_text(labels, encoding="utf-8")
    with pytest.raises(TimeNetConnectorError, match=match):
        HaptConnector().convert(source)


def test_a_mismatched_activity_label_file_raises(tmp_path):
    source = _write_tree(tmp_path)
    (source.root / "activity_labels.txt").write_text("1 WRONG\n", encoding="utf-8")
    with pytest.raises(TimeNetConnectorError, match="does not match the connector's twelve activity labels"):
        HaptConnector().convert(source)
