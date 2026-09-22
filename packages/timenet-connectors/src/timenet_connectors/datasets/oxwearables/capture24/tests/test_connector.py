"""Offline contract tests for CAPTURE-24 participant records."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

from timenet.engine import store_dataset
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import ClassificationTask
from timenet_connectors.datasets.oxwearables.capture24.connector import Capture24Connector, Capture24Participant


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
    assert record.time_series[0].to_arrow().to_pylist() == [-0.46501154, 1.0]
    assert record.time_series[3].to_arrow().to_pylist()[0] == "2016-01-24 00:37:00.000000"
    assert {item.key: item.value for item in record.annotations}["annotation_label_dictionary"] == {"sleep": mapping}
    assert [task.target for task in dataset.tasks if isinstance(task, ClassificationTask)] == ["sleep", "walk"]
    dataset.derive_schema()
    version = store_dataset(dataset, tmp_path / "out")
    with TimeFReader(DatasetVersion.open_local(version)) as reader:
        assert reader.read().records[0].time_series[2].to_arrow().to_pylist() == [0.80145556, 3.0]
