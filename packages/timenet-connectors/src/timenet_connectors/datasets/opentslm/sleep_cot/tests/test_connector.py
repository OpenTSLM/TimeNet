import csv
import json
from pathlib import Path

import numpy as np
import pytest

from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import assert_datasets_equal
from timenet.types import ClassificationTask, Split
from timenet.types.text import read_value
from timenet_connectors.datasets.opentslm.sleep_cot.connector import (
    COLUMNS,
    SAMPLES_PER_WINDOW,
    SLEEP_COT_SHA256,
    SLEEP_STAGE,
    SleepCotConnector,
    SleepCotReader,
)


_STEP_BETWEEN_VALUES = 1e-6 / 3
"""A third of a microvolt has no exact binary form. Like in the release, the values get up to 17
significant digits, so a float32 cast changes them."""
_SUBJECTS = {"sc4102e0": "sleep-cassette-10", "st7022j0": "sleep-telemetry-02"}
"""One recording of each Sleep-EDF study, with the subject id that its name states."""
_WINDOW_ERROR = f"time_series must hold one list of {SAMPLES_PER_WINDOW} finite numbers"


def _window(start: float) -> list[list[float]]:
    # Like the release, the cell holds one list of values inside a list.
    return [[start + index * _STEP_BETWEEN_VALUES for index in range(SAMPLES_PER_WINDOW)]]


def _row(recording_id: str, label: str, *, start: float) -> dict[str, str]:
    # Synthetic rows in the shape of the release. Like the real prompt, this one spans several lines.
    return {
        "id": recording_id,
        "time_series": json.dumps(_window(start)),
        "label": label,
        "prompt": f"\n    Synthetic prompt: {label} or another stage?\n    Answer: {label}\n    ",
        "rationale": f"Synthetic rationale for the window at {start}. Answer: {label}",
    }


_ROWS = [
    _row("sc4102e0", "W", start=0.0),
    _row("st7022j0", "N2", start=1.0),
    _row("sc4102e0", "REM", start=2.0),
    _row("st7022j0", "N4", start=3.0),
]
_VALIDATION_POSITIONS = [1]
_TEST_POSITIONS = [3]


def _write_release(
    directory: Path,
    rows: list[dict[str, str]] = _ROWS,
    *,
    validation: list[int] = _VALIDATION_POSITIONS,
    test: list[int] = _TEST_POSITIONS,
    source_sha256: str = SLEEP_COT_SHA256,
) -> SleepCotReader:
    csv_path = directory / "sleep_cot.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    splits_path = directory / "splits.json"
    splits = {"source_sha256": source_sha256, "validation": validation, "test": test}
    splits_path.write_text(json.dumps(splits), encoding="utf-8")
    return SleepCotReader(csv_path=csv_path, splits_path=splits_path)


def test_each_window_becomes_a_record_and_a_classification_task(tmp_path):
    tasks = SleepCotConnector().convert([_write_release(tmp_path)]).tasks_of(ClassificationTask)
    assert [task.split for task in tasks] == [Split.TRAIN, Split.VALIDATION, Split.TRAIN, Split.TEST]
    for task, row in zip(tasks, _ROWS, strict=True):
        (record,) = task.inputs
        (signal,) = record.signals
        assert signal.name == "EEG Fpz-Cz"
        # Exact equality: the values keep every digit of the release.
        np.testing.assert_array_equal(signal.to_numpy(), np.array(json.loads(row["time_series"])[0]))
        assert record.metadata == {"recording_id": row["id"]}
        (subject,) = record.annotations
        assert (subject.key, read_value(subject.value)) == ("subject_ids", [_SUBJECTS[row["id"]]])
        assert task.label_set == SLEEP_STAGE
        assert task.targets is not None
        assert read_value(task.targets[0]) == row["label"]
        assert read_value(task.rationale) == row["rationale"]
        # The generation prompt of the release names the answer, so the dataset does not keep it.
        assert task.prompt is None
        assert task.annotations == ()


@pytest.mark.parametrize(
    ("splits", "message"),
    [
        ({"source_sha256": "0" * 64}, "the split was made from the CSV 0+, not from the pinned"),
        ({"test": [1]}, r"rows \[1\] are named more than once"),
        ({"test": [len(_ROWS)]}, rf"names rows \[{len(_ROWS)}\], past the {len(_ROWS)} rows"),
    ],
    ids=["another release CSV", "a row in two splits", "a row past the end"],
)
def test_a_split_file_that_does_not_fit_the_release_raises(tmp_path, splits, message):
    with pytest.raises(TimeNetConnectorError, match=message):
        SleepCotConnector().convert([_write_release(tmp_path, **splits)])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"time_series": json.dumps([_window(0.0)[0][:-1]])}, _WINDOW_ERROR),
        ({"time_series": json.dumps([[None, *_window(0.0)[0][1:]]])}, _WINDOW_ERROR),
        ({"id": "sc4102"}, "id 'sc4102' is not a Sleep-EDF recording name"),
        ({"label": "N5"}, "label 'N5' is not one of"),
    ],
    ids=["one value too few", "a null value", "a cut recording name", "an unknown label"],
)
def test_a_malformed_row_raises_with_its_location(tmp_path, change, message):
    rows = [*_ROWS[:2], _ROWS[2] | change]
    with pytest.raises(TimeNetConnectorError, match=rf"sleep_cot\.csv data row 3: {message}"):
        SleepCotConnector().convert([_write_release(tmp_path, rows)])


def test_the_dataset_round_trips_through_storage(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    dataset = SleepCotConnector().convert([_write_release(source)])
    version_dir = dataset.write(path=tmp_path / "registry")
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        assert_datasets_equal(dataset, reader.read())
