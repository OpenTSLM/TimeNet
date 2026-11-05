import csv
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import assert_datasets_equal
from timenet.types import ClassificationTask, Split
from timenet.types.text import read_value
from timenet_connectors.datasets.opentslm.sleep_cot.connector import (
    COLUMNS,
    SAMPLES_PER_WINDOW,
    SLEEP_STAGE,
    SleepCotConnector,
)


_STEP_BETWEEN_VALUES = 1e-6 / 3
"""A third of a microvolt has no exact binary form. Like in the release, the values get up to 17
significant digits, so a float32 cast changes them."""
_RECORDINGS = {"sc4102e0": "sleep-cassette-10", "st7022j0": "sleep-telemetry-02"}
"""One recording of each Sleep-EDF study, with the subject id that its name states."""
_STAGES = ("W", "N2", "REM")
_ROWS_PER_STAGE = 10
"""Enough rows for each stage that both stratified splits can take some of each."""
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


def _fixture_rows() -> list[dict[str, str]]:
    # Alternate the two recordings and the three stages, so that each recording has each stage.
    recording_ids = list(_RECORDINGS)
    rows = []
    for index in range(len(_STAGES) * _ROWS_PER_STAGE):
        recording_id = recording_ids[index % len(recording_ids)]
        stage = _STAGES[index % len(_STAGES)]
        rows.append(_row(recording_id, stage, start=float(index)))
    return rows


_ROWS = _fixture_rows()


def _write_csv(directory: Path, rows: list[dict[str, str]]) -> Path:
    path = directory / "sleep_cot.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _convert(directory: Path, rows: list[dict[str, str]] = _ROWS) -> TimeFDataset:
    return SleepCotConnector().convert([_write_csv(directory, rows)])


def test_each_window_becomes_a_record_and_a_classification_task(tmp_path):
    tasks = _convert(tmp_path).tasks_of(ClassificationTask)
    assert len(tasks) == len(_ROWS)
    for task, row in zip(tasks, _ROWS, strict=True):
        (record,) = task.inputs
        (signal,) = record.signals
        assert signal.name == "EEG Fpz-Cz"
        # Exact equality: the values keep every digit of the release.
        np.testing.assert_array_equal(signal.to_numpy(), np.array(json.loads(row["time_series"])[0]))
        assert record.metadata == {"recording_id": row["id"]}
        (subject,) = record.annotations
        assert (subject.key, read_value(subject.value)) == ("subject_ids", [_RECORDINGS[row["id"]]])
        assert task.label_set == SLEEP_STAGE
        assert task.targets is not None
        assert read_value(task.targets[0]) == row["label"]
        assert read_value(task.rationale) == row["rationale"]
        # The generation prompt names the answer. It is task provenance and never a model input.
        assert task.prompt is None
        assert task.input_annotations == ()
        (annotation,) = task.annotations
        assert annotation.key == "rationale_generation_prompt"
        assert read_value(annotation.value) == row["prompt"]


def test_each_window_gets_the_split_of_the_opentslm_loader(tmp_path):
    # OpenTSLM's load_sleepedf_cot_splits splits the rows themselves, as a DataFrame, with these settings.
    train_and_validation, test = train_test_split(
        _ROWS, test_size=0.1, random_state=42, stratify=[row["label"] for row in _ROWS]
    )
    train, validation = train_test_split(
        train_and_validation,
        test_size=0.1 / (1.0 - 0.1),
        random_state=43,
        stratify=[row["label"] for row in train_and_validation],
    )
    expected = {}
    for split, rows in ((Split.TRAIN, train), (Split.VALIDATION, validation), (Split.TEST, test)):
        for row in rows:
            expected[row["rationale"]] = split
    tasks = _convert(tmp_path).tasks
    assert {read_value(task.rationale): task.split for task in tasks} == expected


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"time_series": json.dumps([_window(0.0)[0][:-1]])}, _WINDOW_ERROR),
        ({"time_series": json.dumps([[None, *_window(0.0)[0][1:]]])}, _WINDOW_ERROR),
        ({"id": "sc4102"}, "id 'sc4102' is not a Sleep-EDF recording name"),
    ],
    ids=["one value too few", "a null value", "a cut recording name"],
)
def test_a_malformed_row_raises_with_its_location(tmp_path, change, message):
    rows = [*_ROWS[:2], _ROWS[2] | change]
    with pytest.raises(TimeNetConnectorError, match=rf"sleep_cot\.csv data row 3: {message}"):
        _convert(tmp_path, rows)


def test_the_dataset_round_trips_through_storage(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    dataset = _convert(source)
    version_dir = dataset.write(path=tmp_path / "registry")
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        assert_datasets_equal(dataset, reader.read())
