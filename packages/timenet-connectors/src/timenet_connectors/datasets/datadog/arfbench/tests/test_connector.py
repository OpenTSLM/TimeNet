"""Check that each ARFBench question retains its resolutions and visual inputs."""

import csv
from datetime import UTC, datetime, timedelta

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.types import InputModality
from timenet_connectors.datasets.datadog.arfbench.connector import ARFBenchConnector, ARFBenchSource


def _metric(path, step_seconds):
    start = datetime(2025, 3, 7, tzinfo=UTC)
    table = pa.table(
        {
            "epoch": pa.array([start, start + timedelta(seconds=step_seconds)], type=pa.timestamp("us", tz="UTC")),
            "group": ["", ""],
            "value": pa.array([1.0, None], type=pa.float64()),
        }
    )
    pq.write_table(table, path)


def test_one_question_reuses_records_at_two_resolutions_and_reads_charts(tmp_path):
    ts_dir = tmp_path / "arfbench-ts-data"
    ts_dir.mkdir()
    image_dir = tmp_path / "arfbench-images"
    image_dir.mkdir()
    for metric in ("35997_0", "35997_1"):
        for resolution in (10, 60):
            _metric(ts_dir / f"{metric}_{resolution}.parquet", resolution)
        Image.new("RGB", (3, 2), (255, 255, 255)).save(image_dir / f"{metric}.png")
    Image.new("RGB", (6, 2), (255, 255, 255)).save(image_dir / "35997_0-35997_1.png")
    qa_path = tmp_path / "arfbench-qa.csv"
    with qa_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "query_group",
                "question",
                "correct_answer",
                "options_str",
                "task_category",
                "difficulty",
                "interpolate_1",
                "interpolate_2",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "query_group": "35997_0,35997_1",
                "question": "Which metric changed?",
                "correct_answer": "A",
                "options_str": '["A", "B"]',
                "task_category": "Indicator",
                "difficulty": "easy",
                "interpolate_1": "0",
                "interpolate_2": "0",
            }
        )
    source = ARFBenchSource(
        qa_csv=qa_path,
        ts_dir=ts_dir,
        image_dir=image_dir,
        published={"35997_0": (10, 60), "35997_1": (10, 60)},
    )

    dataset = ARFBenchConnector().convert([source])
    tasks = list(dataset.iter_tasks())

    assert [task.id for task in tasks] == [
        "arfbench-000-series-10s",
        "arfbench-000-series-60s",
        "arfbench-000-image",
    ]
    assert len(dataset.records) == 7
    assert tasks[0].input_modalities == frozenset({InputModality.TEXT, InputModality.TIME_SERIES})
    assert tasks[2].input_modalities == frozenset({InputModality.TEXT, InputModality.IMAGE})
    assert [record.id for record in tasks[2].inputs] == [
        "arfbench-image-35997_0-35997_1",
        "arfbench-image-35997_0",
        "arfbench-image-35997_1",
    ]
    assert tasks[0].inputs[0].signals[0].to_arrow().to_pylist() == [1.0, None]
    assert tasks[0].targets == tasks[1].targets == tasks[2].targets == ("A",)
