"""Check that a metric keeps every rollup per tag group, and that a question gets a series task and a chart task."""

import csv
from datetime import datetime, timedelta

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.dataset import IrregularAxis, RegularAxis
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import InputModality, Split
from timenet.writer import TimeFWriter
from timenet_connectors.datasets.datadog.arfbench.connector import ARFBenchConnector, ARFBenchSource


START = datetime(2025, 3, 7, 9, 30)  # naive UTC, as the release stores its epochs
START_US = 1_741_339_800_000_000


def _series_file(path, rows):
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


def _facts(annotated):
    return {annotation.key: annotation.value for annotation in annotated.annotations}


def test_metrics_keep_every_rollup_per_tag_group_and_questions_get_two_tasks(tmp_path):
    series_dir = tmp_path / "arfbench-ts-data"
    series_dir.mkdir()
    # Metric 35997_0: two tag groups at 10 s, where dc:2 skips a step; at 60 s a third group appears,
    # as it does in the wider windows of the release, and dc:1 has a null.
    _series_file(
        series_dir / "35997_0_10.parquet", [(10, "dc:1", 2.0), (0, "dc:2", 5.0), (0, "dc:1", 1.0), (20, "dc:2", 6.0)]
    )
    _series_file(series_dir / "35997_0_60.parquet", [(0, "dc:3", 9.0), (60, "dc:1", None), (0, "dc:1", 1.5)])
    # Metric 35997_1 has no tag grouping, so its label is empty, and only a 10 s file.
    _series_file(series_dir / "35997_1_10.parquet", [(30, "", 0.5), (40, "", 0.6)])
    chart_dir = tmp_path / "arfbench-images"
    chart_dir.mkdir()
    for metric in ("35997_0", "35997_1"):
        Image.new("RGBA", (3, 2), (10, 20, 30, 42)).save(chart_dir / f"{metric}.png")
    Image.new("RGB", (6, 2), (255, 255, 255)).save(chart_dir / "35997_0-35997_1.png")
    qa_csv = tmp_path / "arfbench-qa.csv"
    with qa_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "Unnamed: 0",
                "question",
                "task_category",
                "difficulty",
                "options_str",
                "correct_answer",
                "query_group",
                "options",
                "interpolate_1",
                "interpolate_2",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
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
        )
        writer.writerow(
            {
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
        )
    source = ARFBenchSource(qa_csv=qa_csv, series_dir=series_dir, chart_dir=chart_dir)

    dataset = ARFBenchConnector().convert([source])
    tasks = dataset.get_all()
    first, second = dataset.records[:2]

    assert [record.id for record in dataset.records] == [
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
    assert dc1_60s.to_arrow().to_pylist() == [1.5, None]  # the release's null stays a missing timestep
    assert isinstance(dc1_10s.time_axis, RegularAxis) and dc1_10s.time_axis.offset_us == 0
    assert isinstance(first.sources[1].signals[0].time_axis, IrregularAxis)  # dc:2 skips the 10 s step
    assert first.start_time.timestamp == START_US and second.start_time.timestamp == START_US + 30 * 1_000_000
    assert _facts(first) == {
        "metric_id": "35997_0",
        "incident_id": "35997",
        "query_name": "35997_9",
        "finest_interval_s": 10,
    }
    assert dc1_10s.spec.unit_value is None
    # Two tasks per question, in the test split: the series task reads the cited metrics in citation
    # order, the chart task the combined chart first and then each metric's own.
    assert [task.id for task in tasks] == [
        "arfbench-000-series",
        "arfbench-000-chart",
        "arfbench-001-series",
        "arfbench-001-chart",
    ]
    assert dataset.get_test() == tasks and dataset.get_train() == ()
    assert [record.id for record in tasks[2].inputs] == ["arfbench-35997_0", "arfbench-35997_1"]
    assert [record.id for record in tasks[3].inputs] == [
        "arfbench-chart-35997_0-35997_1",
        "arfbench-chart-35997_0",
        "arfbench-chart-35997_1",
    ]
    assert tasks[2].input_modalities == frozenset({InputModality.TEXT, InputModality.TIME_SERIES})
    assert tasks[3].input_modalities == frozenset({InputModality.TEXT, InputModality.IMAGE})
    assert all(task.split is Split.TEST for task in tasks)
    assert [task.targets for task in tasks] == [("Yes",), ("Yes",), ("No",), ("No",)]
    assert _facts(tasks[3]) == {
        "answer_options": ["Yes", "No"],
        "task_category": "Anomaly Correlation",
        "difficulty": "Tier 3",
        "interpolate_1": True,
        "interpolate_2": False,
    }

    dataset.derive_schema()
    with TimeFWriter(tmp_path / "built", dataset, values_backend="zarr") as output:
        output.write()
    with TimeFReader(DatasetVersion.open_local(tmp_path / "built/datadog/arfbench/1.0.0")) as reader:
        back = {record.id: record for record in reader.iter_records()}
    assert [source.name for source in back["arfbench-35997_0"].sources] == ["dc:1", "dc:2", "dc:3"]
    assert back["arfbench-35997_0"].sources[0].signals[1].to_arrow().to_pylist() == [1.5, None]
    chart = back["arfbench-chart-35997_0"].signals[0]
    assert chart.spec.spec_type == "rgba_image_2x3"
    np.testing.assert_array_equal(chart.to_numpy()[0, 0, 0], [10, 20, 30, 42])
