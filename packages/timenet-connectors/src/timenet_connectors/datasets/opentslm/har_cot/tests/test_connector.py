import csv
import json
from pathlib import Path

import numpy as np
import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import assert_datasets_equal
from timenet.types import ClassificationTask, Split
from timenet.types.text import read_value
from timenet_connectors.datasets.opentslm.har_cot.connector import (
    ACTIVITY,
    COLUMNS,
    SAMPLES_PER_WINDOW,
    HarCotConnector,
    HarCotSplitFile,
)


_STEP_BETWEEN_VALUES = 1 / 3
"""A step with no exact binary form. Like in the release, the values get up to 17 significant digits,
so a float32 cast changes them."""
_Y_AXIS_SHIFT = 1.0
"""The y axis starts above the x axis, so a swap of the two axes changes the values."""
_Z_AXIS_SHIFT = 9.81
"""The z axis starts at gravity in m/s², as on an accelerometer at rest."""
_AXIS_ERROR = f"x_axis must hold {SAMPLES_PER_WINDOW} finite numbers"


def _window(start: float) -> list[float]:
    return [start + index * _STEP_BETWEEN_VALUES for index in range(SAMPLES_PER_WINDOW)]


def _row(label: str, distractor: str, *, start: float) -> dict[str, str]:
    # Synthetic rows in the shape of the release. Like the real prompt, this one spans several lines.
    return {
        "x_axis": json.dumps(_window(start)),
        "y_axis": json.dumps(_window(start + _Y_AXIS_SHIFT)),
        "z_axis": json.dumps(_window(start + _Z_AXIS_SHIFT)),
        "label": label,
        "prompt": f"\n    Synthetic prompt: {label} or {distractor}?\n    End with Answer: {label}\n    ",
        "rationale": f"Synthetic rationale for the window at {start}. Answer: {label}.",
    }


def _write_split_file(directory: Path, split: Split, rows: list[dict[str, str]]) -> HarCotSplitFile:
    # Like the validation and test files of the release, repeat the header before each row after the first.
    path = directory / f"{split}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for index, row in enumerate(rows):
            if index > 0:
                writer.writeheader()
            writer.writerow(row)
    return HarCotSplitFile(split=split, path=path)


_TRAIN_ROWS = [_row("sitting", "walking", start=0.0)]
_TEST_ROWS = [_row("walking_up", "lying", start=2.0), _row("biking", "sitting", start=4.0)]


def _convert(directory: Path) -> TimeFDataset:
    train = _write_split_file(directory, Split.TRAIN, _TRAIN_ROWS)
    test = _write_split_file(directory, Split.TEST, _TEST_ROWS)
    return HarCotConnector().convert([train, test])


def test_each_window_becomes_a_record_and_a_classification_task(tmp_path):
    expected = [(Split.TRAIN, row) for row in _TRAIN_ROWS] + [(Split.TEST, row) for row in _TEST_ROWS]
    tasks = _convert(tmp_path).tasks_of(ClassificationTask)
    assert len(tasks) == len(expected)
    for task, (split, row) in zip(tasks, expected, strict=True):
        (record,) = task.inputs
        assert [signal.name for signal in record.signals] == ["accelerometer_x", "accelerometer_y", "accelerometer_z"]
        for signal, column in zip(record.signals, ("x_axis", "y_axis", "z_axis"), strict=True):
            # Exact equality: the values keep every digit of the release.
            np.testing.assert_array_equal(signal.to_numpy(), np.array(json.loads(row[column]), dtype=np.float64))
        assert task.split == split
        assert task.label_set == ACTIVITY
        assert task.targets is not None
        assert read_value(task.targets[0]) == row["label"]
        assert read_value(task.rationale) == row["rationale"]
        # The generation prompt of the release names the answer, so the dataset does not keep it.
        assert task.prompt is None
        assert task.annotations == ()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"x_axis": json.dumps(_window(0.0)[:-1])}, _AXIS_ERROR),
        ({"x_axis": json.dumps([None, *_window(0.0)[1:]])}, _AXIS_ERROR),
        ({"label": "jogging"}, "label 'jogging' is not one of"),
    ],
    ids=["one value too few", "a null value", "an unknown label"],
)
def test_a_malformed_row_raises_with_its_location(tmp_path, change, message):
    rows = [*_TEST_ROWS, _row("lying", "walking", start=6.0) | change]
    split_file = _write_split_file(tmp_path, Split.TEST, rows)
    with pytest.raises(TimeNetConnectorError, match=rf"test\.csv data row 3: {message}"):
        HarCotConnector().convert([split_file])


def test_the_dataset_round_trips_through_storage(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    dataset = _convert(source)
    version_dir = dataset.write(path=tmp_path / "registry")
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        assert_datasets_equal(dataset, reader.read())
