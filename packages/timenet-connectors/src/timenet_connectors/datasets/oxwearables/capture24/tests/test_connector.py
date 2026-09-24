"""Offline contract tests for CAPTURE-24 participant records."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from timenet.engine import store_dataset
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import ClassificationTask, TimeInterval
from timenet_connectors.datasets.oxwearables.capture24 import connector as capture24
from timenet_connectors.datasets.oxwearables.capture24.connector import Capture24Connector, Capture24Participant


def _scope(task: ClassificationTask) -> TimeInterval:
    """Return a classification task's required activity interval."""
    assert isinstance(task.scope, TimeInterval)
    return task.scope


def test_fixture_preserves_source_text_mappings_tasks_and_roundtrip(tmp_path: Path) -> None:
    """A native mini-record preserves values, source text, mappings, and labels."""
    path = tmp_path / "P001.csv.gz"
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("time", "x", "y", "z", "annotation"))
        writer.writeheader()
        writer.writerows(
            (
                {
                    "time": "2016-01-24 00:37:00.000000",
                    "x": "-0.46501154",
                    "y": "0.418018",
                    "z": "0.80145556",
                    "annotation": "sleep",
                },
                {"time": "2016-01-24 00:37:00.010000", "x": "1", "y": "2", "z": "3", "annotation": "walk"},
            )
        )
    mapping = {f"label:{index}": "sleep" for index in range(6)}
    dataset = Capture24Connector().convert([Capture24Participant(path, "P001", "38-52", "F", {"sleep": mapping})])
    record = dataset.records[0]
    assert record.subject_ids == ("P001",)
    signals = {signal.name: signal for signal in record.signals}
    assert capture24._PARTICIPANT_CACHE.path != path
    assert signals["x"].to_arrow().to_pylist() == [-0.46501154, 1.0]
    assert signals["time"].to_arrow().to_pylist()[0] == "2016-01-24 00:37:00.000000"
    assert record.sources[0].metadata == {"data_source_type": "wearable", "provider": "University of Oxford"}
    annotation_values = {item.key: item.value for item in record.annotations}
    assert json.loads(annotation_values["annotation_label_dictionary"]) == {"sleep": mapping}
    tasks = tuple(task for task in dataset.tasks if isinstance(task, ClassificationTask))
    assert [task.targets for task in tasks] == [("sleep",), ("walk",)]
    assert [(_scope(task).start_us, _scope(task).end_us) for task in tasks] == [(0, 10_000), (10_000, 20_000)]
    dataset.derive_schema()
    version = store_dataset(dataset, tmp_path / "out")
    with TimeFReader(DatasetVersion.open_local(version)) as reader:
        restored = reader.read().records[0]
        restored_signals = {signal.name: signal for signal in restored.signals}
        assert restored_signals["z"].to_arrow().to_pylist() == [
            0.80145556,
            3.0,
        ]
        assert restored_signals["annotation"].to_arrow().to_pylist() == ["sleep", "walk"]
        restored_annotation_values = {item.key: item.value for item in restored.annotations}
        assert restored_annotation_values["age"] == "38-52"
        assert json.loads(restored_annotation_values["annotation_label_dictionary"]) == {"sleep": mapping}
        assert restored.sources[0].metadata == {"data_source_type": "wearable", "provider": "University of Oxford"}
