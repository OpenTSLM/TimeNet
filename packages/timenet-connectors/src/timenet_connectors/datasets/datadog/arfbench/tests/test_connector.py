"""Check that a metric keeps every rollup per tag group, that a question gets a series task and a chart task,
and that a malformed source raises with its location."""

import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.types import InputModality, MissingReason, Split, read_value, read_values
from timenet.writer import TimeFWriter
from timenet_connectors.datasets.datadog.arfbench.connector import ARFBenchConnector, ARFBenchSource
from timenet_connectors.datasets.datadog.arfbench.questions import QUESTION_COLUMNS


START = datetime(2025, 3, 7, 9, 30)  # naive, as the release stores its epochs; the connector reads it as UTC
START_US = 1_741_339_800_000_000


def _series_file(path: Path, rows: list[tuple[int, str, float | None]]) -> None:
    """Write (seconds after START, tag label, value) rows the way the release does, in arbitrary order."""
    pq.write_table(
        pa.table(
            {
                "epoch": pa.array([START + timedelta(seconds=s) for s, _, _ in rows], type=pa.timestamp("ns")),
                "group": [label for _, label, _ in rows],
                "value": pa.array([value for _, _, value in rows], type=pa.float64()),
                "num_groups": [1] * len(rows),
                "query_name": ["35997_9"] * len(rows),
                "__index_level_0__": list(range(len(rows))),
            }
        ),
        path,
    )


def _write_series_dir(directory: Path) -> Path:
    series_dir = directory / "arfbench-ts-data"
    series_dir.mkdir()
    # Metric 35997_0: two tag groups at 10 s, where dc:2 skips a step. At 60 s a third group appears,
    # as it does in the wider windows of the release, but only from the second grid step: its first
    # step is grid padding. dc:1 has a null inside its window, and dc:4 is on the grid without a value.
    _series_file(
        series_dir / "35997_0_10.parquet", [(10, "dc:1", 2.0), (0, "dc:2", 5.0), (0, "dc:1", 1.0), (20, "dc:2", 6.0)]
    )
    _series_file(
        series_dir / "35997_0_60.parquet",
        [
            (0, "dc:3", None),
            (60, "dc:3", 9.0),
            (60, "dc:1", None),
            (0, "dc:1", 1.5),
            (120, "dc:1", 2.5),
            (0, "dc:4", None),
            (60, "dc:4", None),
            (120, "dc:4", None),
        ],
    )
    # Metric 35997_1 has no tag grouping, so its label is empty, and only a 10 s file.
    _series_file(series_dir / "35997_1_10.parquet", [(30, "", 0.5), (40, "", 0.6)])
    return series_dir


def _write_chart_dir(directory: Path) -> Path:
    chart_dir = directory / "arfbench-images"
    chart_dir.mkdir()
    for metric in ("35997_0", "35997_1"):
        Image.new("RGBA", (3, 2), (10, 20, 30, 42)).save(chart_dir / f"{metric}.png")
    Image.new("RGB", (6, 2), (255, 255, 255)).save(chart_dir / "35997_0-35997_1.png")
    return chart_dir


_PRESENCE_ROW = {
    "Unnamed: 0": "0",
    "question": "Is there an anomaly?\nTime-series: A gauge.",
    "task_category": "Anomaly Presence",
    "difficulty": "Tier 1",
    "options_str": '["Yes", "No"]',
    "correct_answer": "Yes",
    "query_group": "35997_1",
    "options": '[{"value": "Yes"}, {"value": "No"}]',
    "interpolate_1": "0",
    "interpolate_2": "0",
}
_CORRELATION_ROW = {
    "Unnamed: 0": "1",
    "question": "Are they correlated?\nTime-series 1: A gauge.\nTime-series 2: A count.",
    "task_category": "Anomaly Correlation",
    "difficulty": "Tier 3",
    "options_str": '["Yes", "No"]',
    "correct_answer": "No",
    "query_group": "35997_0, 35997_1",
    "options": '[{"value": "Yes"}, {"value": "No"}]',
    "interpolate_1": "1",
    "interpolate_2": "0",
}


def _write_qa_csv(directory: Path, rows: list[dict[str, str]], *, columns: tuple[str, ...] = QUESTION_COLUMNS) -> Path:
    qa_csv = directory / "arfbench-qa.csv"
    with qa_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return qa_csv


@pytest.fixture
def release(tmp_path: Path) -> ARFBenchSource:
    """A release with two metrics, their three charts, and one question about each metric count."""
    return ARFBenchSource(
        qa_csv=_write_qa_csv(tmp_path, [_PRESENCE_ROW, _CORRELATION_ROW]),
        series_dir=_write_series_dir(tmp_path),
        chart_dir=_write_chart_dir(tmp_path),
    )


def _facts(annotated):
    return {annotation.key: read_value(annotation.value) for annotation in annotated.annotations}


def test_a_metric_record_keeps_every_rollup_per_tag_group(release):
    dataset = ARFBenchConnector().convert([release])
    first, second = dataset.records[:2]

    assert [record.metadata["source_record_id"] for record in dataset.records] == [
        "arfbench-35997_0",
        "arfbench-35997_1",
        "arfbench-chart-35997_0-35997_1",
        "arfbench-chart-35997_0",
        "arfbench-chart-35997_1",
    ]
    # A source is a tag group, its signals that group's rollups, finest first.
    assert [source.name for source in first.sources] == ["dc:1", "dc:2", "dc:3"]
    assert [[signal.name for signal in source.signals] for source in first.sources] == [
        ["10s", "60s"],
        ["10s"],
        ["60s"],
    ]
    assert [source.name for source in second.sources] == ["untagged"]
    dc1_10s, dc1_60s = first.sources[0].signals
    assert dc1_10s.to_arrow().to_pylist() == [1.0, 2.0]
    # A null inside the observed window stays a missing timestep, and the spec says why.
    assert dc1_60s.to_arrow().to_pylist() == [1.5, None, 2.5]
    assert dc1_60s.spec.missing is not None and dc1_60s.spec.missing.reason is MissingReason.NOT_RECORDED
    assert dc1_10s.spec.unit_value is None
    assert isinstance(dc1_10s.time_axis, RegularAxis) and dc1_10s.time_axis.offset_us == 0
    # dc:2 skips the 10 s step, which is a missing timestep on the grid like a null row.
    (dc2_10s,) = first.sources[1].signals
    assert dc2_10s.to_arrow().to_pylist() == [5.0, None, 6.0]
    assert isinstance(dc2_10s.time_axis, RegularAxis)
    # dc:3 starts at its first value, not at the grid's first step, and dc:4 has no value, so no source.
    (dc3_60s,) = first.sources[2].signals
    assert dc3_60s.to_arrow().to_pylist() == [9.0]
    assert isinstance(dc3_60s.time_axis, RegularAxis) and dc3_60s.time_axis.offset_us == 60 * 1_000_000
    assert first.start_time.timestamp == START_US and second.start_time.timestamp == START_US + 30 * 1_000_000
    assert _facts(first) == {
        "metric_id": "35997_0",
        "incident_id": "35997",
        "query_name": "35997_9",
        "finest_interval_s": 10,
    }


def test_each_question_gets_a_series_task_and_a_chart_task(release):
    dataset = ARFBenchConnector().convert([release])
    tasks = dataset.get_all()

    # Two tasks per question, in the test split, paired by the question's row index.
    assert dataset.get_test() == tasks and dataset.get_train() == ()
    assert [task.metadata["question_index"] for task in tasks] == [0, 0, 1, 1]
    assert [read_value(task.prompt) for task in tasks[2:]] == [_CORRELATION_ROW["question"]] * 2
    assert [read_values(task.targets) for task in tasks] == [("Yes",), ("Yes",), ("No",), ("No",)]
    # The series task reads the cited metrics in citation order, the chart task the combined chart
    # first and then each metric's own.
    assert [record.metadata["source_record_id"] for record in tasks[2].inputs] == [
        "arfbench-35997_0",
        "arfbench-35997_1",
    ]
    assert [record.metadata["source_record_id"] for record in tasks[3].inputs] == [
        "arfbench-chart-35997_0-35997_1",
        "arfbench-chart-35997_0",
        "arfbench-chart-35997_1",
    ]
    assert [record.metadata["source_record_id"] for record in tasks[1].inputs] == ["arfbench-chart-35997_1"]
    assert tasks[2].input_modalities == frozenset({InputModality.TEXT, InputModality.TIME_SERIES})
    assert tasks[3].input_modalities == frozenset({InputModality.TEXT, InputModality.IMAGE})
    assert all(task.split is Split.TEST for task in tasks)
    assert _facts(tasks[3]) == {
        "answer_options": ["Yes", "No"],
        "task_category": "Anomaly Correlation",
        "difficulty": "Tier 3",
        "interpolate_1": True,
        "interpolate_2": False,
    }


def test_the_dataset_round_trips_through_storage(release, tmp_path):
    dataset = ARFBenchConnector().convert([release])
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "built", dataset, values_backend="zarr") as output:
        output.write()
    with TimeFReader(open_local_version(tmp_path / "built/datadog/arfbench/1.0.0")) as reader:
        reader.verify()
        back = {record.metadata["source_record_id"]: record for record in reader.iter_records()}
        tasks = reader.read().get_all()
    assert [source.name for source in back["arfbench-35997_0"].sources] == ["dc:1", "dc:2", "dc:3"]
    assert back["arfbench-35997_0"].sources[0].signals[1].to_arrow().to_pylist() == [1.5, None, 2.5]
    chart = back["arfbench-chart-35997_0"].signals[0]
    assert chart.spec.spec_type == "rgba_image_2x3"
    np.testing.assert_array_equal(chart.to_numpy()[0, 0, 0], [10, 20, 30, 42])
    assert [task.metadata["question_index"] for task in tasks] == [0, 0, 1, 1]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"query_group": "35997_2"}, r"question 2: no series file for cited metric 35997_2"),
        ({"query_group": "35997_1, 35997_0"}, r"question 2: no chart for 35997_1-35997_0"),
        ({"correct_answer": "Maybe"}, r"data row 3: correct_answer 'Maybe' is not one of"),
        ({"options_str": "Yes"}, r"data row 3: options_str is not JSON"),
        ({"interpolate_1": "yes"}, r"data row 3: interpolate_1 must be 0 or 1"),
        ({"task_category": "Anomaly Vibes"}, r"data row 3: task_category must be one of \["),
        ({"query_group": "35997_0, 35997_1, 35997_1"}, r"data row 3: query_group must name one or two metrics"),
    ],
    ids=[
        "a cited metric without series",
        "a pair without a combined chart",
        "an answer outside the options",
        "options that are not JSON",
        "a flag that is not 0 or 1",
        "an unknown task category",
        "three cited metrics",
    ],
)
def test_a_malformed_question_raises_with_its_location(release, change, message):
    rows = [_PRESENCE_ROW, _CORRELATION_ROW, {**_CORRELATION_ROW, "Unnamed: 0": "2"} | change]
    _write_qa_csv(release.qa_csv.parent, rows)
    with pytest.raises(TimeNetConnectorError, match=rf"(arfbench-qa\.csv )?{message}"):
        ARFBenchConnector().convert([release])


def test_an_unexpected_header_raises(release):
    _write_qa_csv(release.qa_csv.parent, [_PRESENCE_ROW], columns=QUESTION_COLUMNS[1:])
    with pytest.raises(TimeNetConnectorError, match=r"arfbench-qa\.csv: expected columns"):
        ARFBenchConnector().convert([release])


def test_a_row_off_the_rollup_grid_raises(release):
    _series_file(release.series_dir / "35997_1_10.parquet", [(30, "", 0.5), (35, "", 0.6)])
    with pytest.raises(TimeNetConnectorError, match=r"35997_1_10\.parquet group '': rows are off the 10 s grid"):
        ARFBenchConnector().convert([release])


def test_a_series_file_without_an_interval_raises(release):
    (release.series_dir / "35997_1_10.parquet").rename(release.series_dir / "35997_1_10s.parquet")
    with pytest.raises(TimeNetConnectorError, match=r"35997_1_10s\.parquet is not named"):
        ARFBenchConnector().convert([release])
