"""Check the ARFBench charts connector.

Each chart is a record and each question is a task over its charts. The dataset round-trips through Zarr.
A question without its charts raises an error."""

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.types import ClassificationTask, InputModality, Split, read_values
from timenet.writer import TimeFWriter
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.datadog.arfbench.tests.release import CORRELATION_ROW, ROWS, write_qa_csv
from timenet_connectors.datasets.datadog.arfbench_charts.connector import CHART_DIR, ARFBenchChartsConnector


def _write_chart_dir(root: Path) -> Path:
    chart_dir = root / CHART_DIR
    chart_dir.mkdir()
    for metric in ("35997_0", "35997_1"):
        Image.new("RGBA", (3, 2), (10, 20, 30, 42)).save(chart_dir / f"{metric}.png")
    Image.new("RGB", (6, 2), (255, 255, 255)).save(chart_dir / "35997_0-35997_1.png")
    return chart_dir


@pytest.fixture
def release(tmp_path: Path) -> list[HubFileRef]:
    """The question table and the three charts of the test release."""
    return [HubFileRef(write_qa_csv(tmp_path, ROWS), "questions"), HubFileRef(_write_chart_dir(tmp_path), "charts")]


def _ids(records) -> list[str]:
    return [record.metadata["source_record_id"] for record in records]


def test_each_question_is_a_task_over_its_charts(release):
    dataset = ARFBenchChartsConnector().convert(release)
    presence, correlation = tasks = dataset.get_all()

    assert _ids(dataset.records) == [
        "arfbench-chart-35997_0-35997_1",
        "arfbench-chart-35997_0",
        "arfbench-chart-35997_1",
    ]
    assert [task.metadata["question_index"] for task in tasks] == [0, 1]
    assert [read_values(task.targets) for task in tasks] == [("Yes",), ("No",)]
    assert all(task.input_modalities == frozenset({InputModality.TEXT, InputModality.IMAGE}) for task in tasks)
    assert all(task.split is Split.TEST for task in tasks)
    assert isinstance(correlation, ClassificationTask)
    assert [label.name for label in correlation.label_set.labels] == ["Yes", "No"]
    # For a pair, the combined chart comes first, then the chart of each metric.
    assert _ids(correlation.inputs) == [
        "arfbench-chart-35997_0-35997_1",
        "arfbench-chart-35997_0",
        "arfbench-chart-35997_1",
    ]
    assert _ids(presence.inputs) == ["arfbench-chart-35997_1"]


def test_the_dataset_round_trips_through_zarr(release, tmp_path):
    connector = ARFBenchChartsConnector()
    dataset = connector.convert(release)
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "built", dataset, values_backend=connector.values_backend) as output:
        output.write()
    with TimeFReader(open_local_version(tmp_path / "built/datadog/arfbench-charts/1.0.0")) as reader:
        reader.verify()
        assert reader.values_backend == "zarr"
        back = {record.metadata["source_record_id"]: record for record in reader.iter_records()}
        tasks = reader.read().get_all()
    chart = back["arfbench-chart-35997_0"].signals[0]
    assert chart.spec.spec_type == "rgba_image_2x3"
    np.testing.assert_array_equal(chart.to_numpy()[0, 0, 0], [10, 20, 30, 42])
    assert [task.metadata["question_index"] for task in tasks] == [0, 1]


def test_a_pair_without_a_combined_chart_raises(release, tmp_path):
    write_qa_csv(tmp_path, [*ROWS, {**CORRELATION_ROW, "Unnamed: 0": "2", "query_group": "35997_1, 35997_0"}])
    with pytest.raises(TimeNetConnectorError, match=r"question 2: no chart for 35997_1-35997_0"):
        ARFBenchChartsConnector().convert(release)
