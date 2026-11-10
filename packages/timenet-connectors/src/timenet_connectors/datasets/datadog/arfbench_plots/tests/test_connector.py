"""Check that the plots dataset adds a chart record per chart and a chart task per question to the series
dataset, that it round-trips through Zarr, and that a question without its charts raises."""

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.types import InputModality, Split, read_value, read_values
from timenet.writer import TimeFWriter
from timenet_connectors.datasets.datadog.arfbench.connector import ARFBenchConnector, ARFBenchSource
from timenet_connectors.datasets.datadog.arfbench.tests.release import (
    CORRELATION_ROW,
    PRESENCE_ROW,
    facts,
    write_qa_csv,
    write_series_dir,
)
from timenet_connectors.datasets.datadog.arfbench_plots.connector import CHART_DIR, ARFBenchPlotsConnector


def _write_chart_dir(root: Path) -> Path:
    chart_dir = root / CHART_DIR
    chart_dir.mkdir()
    for metric in ("35997_0", "35997_1"):
        Image.new("RGBA", (3, 2), (10, 20, 30, 42)).save(chart_dir / f"{metric}.png")
    Image.new("RGB", (6, 2), (255, 255, 255)).save(chart_dir / "35997_0-35997_1.png")
    return chart_dir


@pytest.fixture
def release(tmp_path: Path) -> ARFBenchSource:
    """A release with two metrics, their three charts, and one question about each metric count."""
    write_qa_csv(tmp_path, [PRESENCE_ROW, CORRELATION_ROW])
    write_series_dir(tmp_path)
    _write_chart_dir(tmp_path)
    return ARFBenchSource(root=tmp_path)


def test_the_connector_adds_the_charts_to_the_series_connector():
    connector = ARFBenchPlotsConnector()
    patterns = connector.HUB_PATTERNS

    assert connector.metadata().dataset_id == "datadog/arfbench-plots"
    assert connector.values_backend == "zarr"
    assert patterns == (*ARFBenchConnector.HUB_PATTERNS, f"{CHART_DIR}/*.png")


def test_each_question_gets_a_series_task_and_a_chart_task(release):
    dataset = ARFBenchPlotsConnector().convert([release])
    tasks = dataset.get_all()

    assert [record.metadata["source_record_id"] for record in dataset.records] == [
        "arfbench-35997_0",
        "arfbench-35997_1",
        "arfbench-chart-35997_0-35997_1",
        "arfbench-chart-35997_0",
        "arfbench-chart-35997_1",
    ]
    # The series tasks first, then the chart tasks, paired by the question's row index.
    assert dataset.get_test() == tasks and dataset.get_train() == ()
    assert [task.metadata["question_index"] for task in tasks] == [0, 1, 0, 1]
    assert [read_value(task.prompt) for task in tasks[1::2]] == [CORRELATION_ROW["question"]] * 2
    assert [read_values(task.targets) for task in tasks] == [("Yes",), ("No",), ("Yes",), ("No",)]
    # The chart task reads the combined chart first, then each metric's own.
    assert [record.metadata["source_record_id"] for record in tasks[3].inputs] == [
        "arfbench-chart-35997_0-35997_1",
        "arfbench-chart-35997_0",
        "arfbench-chart-35997_1",
    ]
    assert [record.metadata["source_record_id"] for record in tasks[2].inputs] == ["arfbench-chart-35997_1"]
    assert tasks[1].input_modalities == frozenset({InputModality.TEXT, InputModality.TIME_SERIES})
    assert tasks[3].input_modalities == frozenset({InputModality.TEXT, InputModality.IMAGE})
    assert all(task.split is Split.TEST for task in tasks)
    assert facts(tasks[3]) == facts(tasks[1])


def test_the_dataset_round_trips_through_zarr(release, tmp_path):
    connector = ARFBenchPlotsConnector()
    dataset = connector.convert([release])
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "built", dataset, values_backend=connector.values_backend) as output:
        output.write()
    with TimeFReader(open_local_version(tmp_path / "built/datadog/arfbench-plots/1.0.0")) as reader:
        reader.verify()
        back = {record.metadata["source_record_id"]: record for record in reader.iter_records()}
        tasks = reader.read().get_all()
    assert back["arfbench-35997_0"].sources[0].signals[1].to_arrow().to_pylist() == [1.5, None, 2.5]
    chart = back["arfbench-chart-35997_0"].signals[0]
    assert chart.spec.spec_type == "rgba_image_2x3"
    np.testing.assert_array_equal(chart.to_numpy()[0, 0, 0], [10, 20, 30, 42])
    assert [task.metadata["question_index"] for task in tasks] == [0, 1, 0, 1]


def test_a_pair_without_a_combined_chart_raises(release):
    rows = [PRESENCE_ROW, CORRELATION_ROW, {**CORRELATION_ROW, "Unnamed: 0": "2", "query_group": "35997_1, 35997_0"}]
    write_qa_csv(release.root, rows)
    with pytest.raises(TimeNetConnectorError, match=r"question 2: no chart for 35997_1-35997_0"):
        ARFBenchPlotsConnector().convert([release])
