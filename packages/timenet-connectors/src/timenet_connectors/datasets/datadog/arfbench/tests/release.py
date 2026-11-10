"""Write a small ARFBench release for the connector tests: two metrics and the question table.

The charts tests add the charts. This module does not import Pillow. Thus the environment of the
series connector can run it.
"""

import csv
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.types import SupportsAnnotate, read_value
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.datadog.arfbench.connector import QA_CSV, SERIES_DIR
from timenet_connectors.datasets.datadog.arfbench.questions import QUESTION_COLUMNS


START = datetime(2025, 3, 7, 9, 30)  # naive, as the release stores its epochs. The connector reads it as UTC.
START_US = 1_741_339_800_000_000


def write_series_file(path: Path, rows: list[tuple[int, str, float | None]]) -> None:
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


def write_series_dir(root: Path) -> Path:
    """Write the series files of metrics 35997_0 and 35997_1.

    Returns:
        The series directory.
    """
    series_dir = root / SERIES_DIR
    series_dir.mkdir()
    # Metric 35997_0 has two tag groups at 10 s, and dc:2 skips a step. At 60 s a third group appears,
    # as in the wider windows of the release. The values of this group start at the second grid step.
    # Its first step is grid padding. dc:1 has a null inside its window. dc:4 is on the grid without a value.
    write_series_file(
        series_dir / "35997_0_10.parquet", [(10, "dc:1", 2.0), (0, "dc:2", 5.0), (0, "dc:1", 1.0), (20, "dc:2", 6.0)]
    )
    write_series_file(
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
    # Metric 35997_1 has no tag grouping, so its label is empty. The metric has only a 10 s file.
    write_series_file(series_dir / "35997_1_10.parquet", [(30, "", 0.5), (40, "", 0.6)])
    return series_dir


PRESENCE_ROW = {
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
CORRELATION_ROW = {
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
ROWS = [PRESENCE_ROW, CORRELATION_ROW]


def write_qa_csv(root: Path, rows: list[dict[str, str]], *, columns: tuple[str, ...] = QUESTION_COLUMNS) -> Path:
    """Write the question table.

    Returns:
        The table path.
    """
    qa_csv = root / QA_CSV
    with qa_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return qa_csv


def facts(annotated: SupportsAnnotate) -> dict[str, object]:
    """Read the annotations of a record or task as a dict."""
    return {annotation.key: read_value(annotation.value) for annotation in annotated.annotations}


def write_release(root: Path) -> list[HubFileRef]:
    """Write the question table and the series files.

    Returns:
        The refs that the series connector downloads.
    """
    return [
        HubFileRef(write_qa_csv(root, ROWS), "questions"),
        HubFileRef(write_series_dir(root), "series"),
    ]
