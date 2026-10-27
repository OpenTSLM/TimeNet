from pathlib import Path
from typing import ClassVar

import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import License, Split
from timenet.types.text import read_value
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.mikeam.time_series_reasoning import CONNECTOR
from timenet_connectors.datasets.mikeam.time_series_reasoning.connector import TimeSeriesReasoningConnector
from timenet_connectors.datasets.mikeam.time_series_reasoning.records import record_metadata
from timenet_connectors.datasets.mikeam.time_series_reasoning.release import SourceFile
from timenet_connectors.datasets.mikeam.time_series_reasoning.tests.release_fixtures import fixture_path


class _FixtureConnector(TimeSeriesReasoningConnector):
    """The connector without the release's pins, which the fixtures don't meet."""

    CARD = Path(__file__).parents[1] / "dataset.yaml"
    PINNED: ClassVar[bool] = False


def _refs() -> list[HubFileRef]:
    return [HubFileRef(fixture_path(file), file.part) for file in SourceFile]


def _output(dataset: TimeFDataset) -> dict:
    """Return what a dataset holds, comparable before and after storage."""
    records = {}
    for record in dataset.records:
        metadata = record_metadata(record)
        (signal,) = record.signals
        records[metadata.file, metadata.line, metadata.uuid] = (
            signal.spec.dtype,
            signal.to_numpy().tolist(),
            {annotation.key: read_value(annotation.value) for annotation in record.annotations},
        )
    tasks = [
        (
            read_value(task.prompt),
            tuple(read_value(target) for target in task.targets or ()),
            task.split,
            dict(task.metadata),
            tuple((record_metadata(record).file, record_metadata(record).line) for record in task.inputs),
            task.input_annotations,
        )
        for task in dataset.iter_tasks()
    ]
    annotations = {annotation.key: read_value(annotation.value) for annotation in dataset.annotations}
    return {"annotations": annotations, "records": records, "tasks": tasks}


def test_declares_the_pinned_source_files():
    # The base's download() turns these declarations into refs; pin the declarations themselves.
    assert CONNECTOR is TimeSeriesReasoningConnector
    # The round trip opens the stored dataset by the card's id and version; the license is pinned here.
    assert TimeSeriesReasoningConnector().metadata().license is License.MIT
    assert TimeSeriesReasoningConnector.HF_REPO == "mikeam/time-series-reasoning"
    assert TimeSeriesReasoningConnector.SOURCE_REVISION == "d6a1c7d48d8a454b9db0b2137383fc30847e6ec0"
    assert TimeSeriesReasoningConnector.SOURCE_FILES == {
        "series": "TS_Dataset.jsonl",
        "description_train": "TS_Dataset_MCQ/train.json",
        "description_val": "TS_Dataset_MCQ/val.json",
        "description_test": "TS_Dataset_MCQ/test.json",
        "questions": "MCQ_1_TS.jsonl",
        "counterfactuals": "MCQ_2_TS.jsonl",
    }
    assert TimeSeriesReasoningConnector.PINNED


def test_the_pinned_connector_rejects_files_that_are_not_the_release():
    with pytest.raises(TimeFValidationError, match="expected 8787 rows"):
        TimeSeriesReasoningConnector().convert(_refs())


def test_storage_round_trip_keeps_the_whole_conversion(tmp_path):
    original = _FixtureConnector().convert(_refs())
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("mikeam/time-series-reasoning", "1.0.0")) as reader:
        restored = reader.read()
    expected = _output(original)
    assert expected["annotations"] == {
        "source_repository": "mikeam/time-series-reasoning",
        "source_revision": "d6a1c7d48d8a454b9db0b2137383fc30847e6ec0",
    }
    assert (len(expected["records"]), len(expected["tasks"])) == (16, 12)
    # The description tasks keep the authors' split; the question tasks have none.
    splits = [split for _, _, split, _, _, _ in expected["tasks"]]
    assert splits == [Split.TRAIN] * 5 + [Split.VALIDATION, Split.TEST] + [None] * 5
    # A prompt holds all the authors show a model, so no annotation is a task's input.
    assert all(input_annotations == () for *_, input_annotations in expected["tasks"])
    # The writer read every lazy signal and the task stream already, so this pass reads them again.
    assert _output(restored) == expected
