"""Check four properties of the ARFBench series connector.

A metric keeps every rollup per tag group. A question becomes a classification task over its options. The dataset
round-trips through Parquet with its missing timesteps. A malformed source raises an error with its location."""

from pathlib import Path

import pytest

from timenet.dataset import RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.types import ClassificationTask, InputModality, MissingReason, Split, read_value, read_values
from timenet.writer import TimeFWriter
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.datadog.arfbench.connector import ARFBenchConnector
from timenet_connectors.datasets.datadog.arfbench.questions import QUESTION_COLUMNS
from timenet_connectors.datasets.datadog.arfbench.tests.release import (
    CORRELATION_ROW,
    PRESENCE_ROW,
    ROWS,
    START_US,
    facts,
    write_qa_csv,
    write_release,
    write_series_file,
)


@pytest.fixture
def release(tmp_path: Path) -> list[HubFileRef]:
    """A release with two metrics and one question about each metric count."""
    return write_release(tmp_path)


def _path(refs: list[HubFileRef], part: str) -> Path:
    return next(ref.path for ref in refs if ref.part == part)


def test_the_connector_downloads_the_table_and_the_series_by_pattern():
    connector = ARFBenchConnector()

    assert connector.metadata().dataset_id == "datadog/arfbench"
    assert connector.values_backend == "parquet"
    assert dict(connector.SOURCE_FILES) == {"questions": "arfbench-qa.csv", "series": "arfbench-ts-data/*.parquet"}


def test_a_metric_record_keeps_every_rollup_per_tag_group(release):
    dataset = ARFBenchConnector().convert(release)
    first, second = dataset.records

    assert [record.metadata["source_record_id"] for record in dataset.records] == [
        "arfbench-35997_0",
        "arfbench-35997_1",
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
    assert facts(first) == {
        "metric_id": "35997_0",
        "incident_id": "35997",
        "query_name": "35997_9",
        "finest_interval_s": 10,
    }


def test_each_question_is_a_classification_task_over_its_options(release):
    dataset = ARFBenchConnector().convert(release)
    presence, correlation = tasks = dataset.get_all()

    assert dataset.get_test() == tasks and dataset.get_train() == ()
    assert [task.metadata["question_index"] for task in tasks] == [0, 1]
    assert [read_value(task.prompt) for task in tasks] == [row["question"] for row in ROWS]
    assert [read_values(task.targets) for task in tasks] == [("Yes",), ("No",)]
    # The options are the task's label set, named after the question's row.
    assert isinstance(presence, ClassificationTask) and isinstance(correlation, ClassificationTask)
    assert [presence.label_set.name, correlation.label_set.name] == ["question_0", "question_1"]
    assert [label.name for label in correlation.label_set.labels] == ["Yes", "No"]
    assert not correlation.label_set.multi_label
    # The task reads the cited metrics in citation order.
    assert [record.metadata["source_record_id"] for record in correlation.inputs] == [
        "arfbench-35997_0",
        "arfbench-35997_1",
    ]
    assert [record.metadata["source_record_id"] for record in presence.inputs] == ["arfbench-35997_1"]
    assert all(task.input_modalities == frozenset({InputModality.TEXT, InputModality.TIME_SERIES}) for task in tasks)
    assert all(task.split is Split.TEST for task in tasks)
    assert facts(correlation) == {
        "task_category": "Anomaly Correlation",
        "difficulty": "Tier 3",
        "interpolate_1": True,
        "interpolate_2": False,
    }


def test_the_dataset_round_trips_through_parquet_with_its_missing_timesteps(release, tmp_path):
    connector = ARFBenchConnector()
    dataset = connector.convert(release)
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "built", dataset, values_backend=connector.values_backend) as output:
        output.write()
    with TimeFReader(open_local_version(tmp_path / "built/datadog/arfbench/1.0.0")) as reader:
        reader.verify()
        assert reader.values_backend == "parquet"
        back = {record.metadata["source_record_id"]: record for record in reader.iter_records()}
        tasks = reader.read().get_all()
    first = back["arfbench-35997_0"]
    assert [source.name for source in first.sources] == ["dc:1", "dc:2", "dc:3"]
    dc1_60s = first.sources[0].signals[1]
    assert dc1_60s.to_arrow().to_pylist() == [1.5, None, 2.5]
    assert dc1_60s.spec.missing is not None and dc1_60s.spec.missing.reason is MissingReason.NOT_RECORDED
    assert first.sources[1].signals[0].to_arrow().to_pylist() == [5.0, None, 6.0]
    assert [task.metadata["question_index"] for task in tasks] == [0, 1]
    assert isinstance(tasks[1], ClassificationTask)
    assert [label.name for label in tasks[1].label_set.labels] == ["Yes", "No"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"query_group": "35997_2"}, r"question 2: no series file for cited metric 35997_2"),
        ({"correct_answer": "Maybe"}, r"data row 3: correct_answer 'Maybe' is not one of"),
        ({"options_str": "Yes"}, r"data row 3: options_str is not JSON"),
        ({"interpolate_1": "yes"}, r"data row 3: interpolate_1 must be 0 or 1"),
        ({"task_category": "Anomaly Vibes"}, r"data row 3: task_category must be one of \["),
        ({"query_group": "35997_0, 35997_1, 35997_1"}, r"data row 3: query_group must name one or two metrics"),
    ],
    ids=[
        "a cited metric without series",
        "an answer outside the options",
        "options that are not JSON",
        "a flag that is not 0 or 1",
        "an unknown task category",
        "three cited metrics",
    ],
)
def test_a_malformed_question_raises_with_its_location(release, change, message):
    rows = [*ROWS, {**CORRELATION_ROW, "Unnamed: 0": "2"} | change]
    write_qa_csv(_path(release, "questions").parent, rows)
    with pytest.raises(TimeNetConnectorError, match=rf"(arfbench-qa\.csv )?{message}"):
        ARFBenchConnector().convert(release)


def test_an_unexpected_header_raises(release):
    write_qa_csv(_path(release, "questions").parent, [PRESENCE_ROW], columns=QUESTION_COLUMNS[1:])
    with pytest.raises(TimeNetConnectorError, match=r"arfbench-qa\.csv: expected columns"):
        ARFBenchConnector().convert(release)


def test_a_row_off_the_rollup_grid_raises(release):
    write_series_file(_path(release, "series") / "35997_1_10.parquet", [(30, "", 0.5), (35, "", 0.6)])
    with pytest.raises(TimeNetConnectorError, match=r"35997_1_10\.parquet group '': rows are off the 10 s grid"):
        ARFBenchConnector().convert(release)


def test_a_series_file_without_an_interval_raises(release):
    series_dir = _path(release, "series")
    (series_dir / "35997_1_10.parquet").rename(series_dir / "35997_1_10s.parquet")
    with pytest.raises(TimeNetConnectorError, match=r"35997_1_10s\.parquet is not named"):
        ARFBenchConnector().convert(release)
