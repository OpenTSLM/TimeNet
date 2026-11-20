from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.types import ClassificationTask, Split, read_value
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.autonlab.timeseriesexam1 import TimeSeriesExam1Connector


# Eight real rows of the published test split, in this order. The fixture keeps the published schema.
_FIXTURE = Path(__file__).parent / "fixtures" / "test.parquet"
_SERIES_COLUMNS = ("ts", "ts1", "ts2")


def _source_rows() -> list[dict]:
    return pq.read_table(_FIXTURE).to_pylist()


def _convert(tmp_path: Path, rows: list[dict] | None = None) -> TimeFDataset:
    path = _FIXTURE
    if rows is not None:
        path = tmp_path / "rows.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(_FIXTURE)), path)
    return TimeSeriesExam1Connector().convert([HubFileRef(path, "test")])


def _task_for(dataset: TimeFDataset, row_id: int) -> ClassificationTask:
    return next(task for task in dataset.tasks_of(ClassificationTask) if task.metadata["id"] == row_id)


def test_metadata():
    metadata = TimeSeriesExam1Connector().metadata()
    assert metadata.dataset_id == "autonlab/timeseriesexam1"
    assert str(metadata.license) == "MIT"
    assert str(metadata.dataset_version) == "1.0.0"
    assert metadata.license_url
    assert metadata.citation


def test_records_and_tasks_match_rows(tmp_path):
    dataset = _convert(tmp_path)
    rows = _source_rows()
    two_series = [row for row in rows if row["ts1"] is not None]
    assert len(dataset.tasks) == len(dataset.tasks_of(ClassificationTask)) == len(rows)
    assert len(dataset.records) == len(rows) + len(two_series)
    roles = [(record.metadata["row_id"], record.metadata["role"]) for record in dataset.records]
    expected = [(row["id"], role) for row in rows for role in (("ts1", "ts2") if row["ts1"] is not None else ("ts",))]
    assert roles == expected


def test_series_values_equal_source(tmp_path):
    dataset = _convert(tmp_path)
    by_key: dict[object, list[float]] = {
        (row["id"], column): row[column] for row in _source_rows() for column in _SERIES_COLUMNS
    }
    for record in dataset.records:
        (signal,) = record.signals
        expected = by_key[record.metadata["row_id"], record.metadata["role"]]
        assert signal.n_values == len(expected)
        assert signal.to_numpy().tolist() == expected


def test_inputs_follow_ts1_then_ts2(tmp_path):
    dataset = _convert(tmp_path)
    assert [record.metadata["role"] for record in _task_for(dataset, 1).inputs] == ["ts1", "ts2"]
    assert [record.metadata["role"] for record in _task_for(dataset, 18).inputs] == ["ts"]


def test_records_are_not_shared_between_rows(tmp_path):
    dataset = _convert(tmp_path)
    ids = [id(record) for task in dataset.tasks for record in task.inputs]
    assert len(set(ids)) == len(ids)


def test_task_fields(tmp_path):
    dataset = _convert(tmp_path)
    rows = {row["id"]: row for row in _source_rows()}
    for task in dataset.tasks_of(ClassificationTask):
        row = rows[task.metadata["id"]]
        assert task.split == Split.TEST
        assert task.targets == (row["answer"],)
        assert task.label_set.names == tuple(row["options"])
        assert not task.label_set.ordered
        assert task.label_set.multi_label is False
        assert read_value(task.prompt) == row["question"]


def test_metadata_is_as_published(tmp_path):
    dataset = _convert(tmp_path)
    keys = ["category", "subcategory", "question_type", "difficulty", "relevant_concepts", "question_hint"]
    keys += ["format_hint", "tid"]
    for row in _source_rows():
        metadata = _task_for(dataset, row["id"]).metadata
        assert metadata == {"id": row["id"], **{key: row[key] for key in keys}}


def test_options_and_answers_survive_a_write(tmp_path):
    with TimeFReader(open_local_version(_convert(tmp_path).write(path=tmp_path / "out"))) as reader:
        stored = reader.read()
    rows = {row["id"]: row for row in _source_rows()}
    for task in stored.tasks_of(ClassificationTask):
        row = rows[task.metadata["id"]]
        assert set(task.label_set.names) == set(row["options"])
        assert [read_value(target) for target in task.targets or ()] == [row["answer"]]


@pytest.mark.parametrize(
    ("row_id", "changes", "found"),
    [
        (18, {"ts1": [1.0, 2.0], "ts2": [1.0, 2.0]}, ["ts", "ts1", "ts2"]),
        (18, {"ts": None}, []),
        (1, {"ts2": None}, ["ts1"]),
    ],
)
def test_rejects_row_without_ts_or_ts1_and_ts2(tmp_path, row_id, changes, found):
    rows = [{**row, **changes} if row["id"] == row_id else row for row in _source_rows()]
    message = f"row id {row_id}: expected ts, or ts1 and ts2, found {found}"
    with pytest.raises(TimeNetConnectorError, match=re.escape(message)):
        _convert(tmp_path, rows)
