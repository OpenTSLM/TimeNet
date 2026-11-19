from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.types import ClassificationTask, Split
from timenet.types.text import read_value
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.autonlab.timeseriesexam1 import TimeSeriesExam1Connector


# Eight real rows of the published test split, in this order. The fixture keeps the published schema.
_FIXTURE = Path(__file__).parent / "fixtures" / "test.parquet"
_IDS = [1, 5, 18, 30, 2, 65, 85, 99]
_SERIES_COLUMNS = ("ts", "ts1", "ts2")


def _source_rows() -> list[dict]:
    return pq.read_table(_FIXTURE).to_pylist()


def _refs(tmp_path: Path, rows: list[dict] | None = None) -> list[HubFileRef]:
    if rows is None:
        return [HubFileRef(_FIXTURE, "test")]
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(_FIXTURE)), path)
    return [HubFileRef(path, "test")]


def _convert(tmp_path: Path, rows: list[dict] | None = None) -> TimeFDataset:
    return TimeSeriesExam1Connector().convert(_refs(tmp_path, rows))


def _classification_tasks(dataset: TimeFDataset) -> list[ClassificationTask]:
    tasks = list(dataset.tasks)
    assert all(isinstance(task, ClassificationTask) for task in tasks)
    return [task for task in tasks if isinstance(task, ClassificationTask)]


def _task_for(dataset: TimeFDataset, row_id: int) -> ClassificationTask:
    return next(task for task in _classification_tasks(dataset) if task.metadata["id"] == row_id)


def test_metadata():
    metadata = TimeSeriesExam1Connector().metadata()
    assert metadata.dataset_id == "autonlab/timeseriesexam1"
    assert str(metadata.license) == "MIT"
    assert str(metadata.dataset_version) == "1.0.0"
    assert metadata.license_url
    assert metadata.citation


def test_source_is_pinned():
    assert TimeSeriesExam1Connector.HF_REPO == "AutonLab/TimeSeriesExam1"
    assert TimeSeriesExam1Connector.SOURCE_REVISION == "628f686757bb484d9b608b6855fb41f6a302ecda"
    assert dict(TimeSeriesExam1Connector.SOURCE_FILES) == {"test": "data/test-00000-of-00001.parquet"}


def test_fixture_rows():
    assert [row["id"] for row in _source_rows()] == _IDS


def test_records_and_tasks_match_rows(tmp_path):
    dataset = _convert(tmp_path)
    rows = _source_rows()
    two_series = [row for row in rows if row["ts1"] is not None]
    assert len(dataset.tasks) == len(rows)
    assert len(dataset.records) == len(rows) + len(two_series)
    roles = [(record.metadata["row_id"], record.metadata["role"]) for record in dataset.records]
    expected = [(row["id"], role) for row in rows for role in (("ts1", "ts2") if row["ts1"] is not None else ("ts",))]
    assert roles == expected


def test_lazy_values_equal_source(tmp_path):
    dataset = _convert(tmp_path)
    by_key: dict[object, list[float]] = {
        (row["id"], column): row[column] for row in _source_rows() for column in _SERIES_COLUMNS
    }
    for record in dataset.records:
        (signal,) = record.signals
        expected = by_key[record.metadata["row_id"], record.metadata["role"]]
        assert signal.n_values == len(expected)
        assert signal.to_numpy().tolist() == expected


def test_series_lengths_are_as_published(tmp_path):
    dataset = _convert(tmp_path)
    task = _task_for(dataset, 85)
    first, second = task.inputs
    assert (first.signals[0].n_values, second.signals[0].n_values) == (1024, 1180)


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
    for task in _classification_tasks(dataset):
        row = rows[task.metadata["id"]]
        assert task.split == Split.TEST
        assert task.targets == (row["answer"],)
        assert task.label_set.names == tuple(row["options"])
        assert task.label_set.multi_label is False
        assert read_value(task.prompt) == row["question"]


def test_option_counts(tmp_path):
    dataset = _convert(tmp_path)
    counts = {task.metadata["id"]: len(task.label_set.labels) for task in _classification_tasks(dataset)}
    assert counts == {1: 2, 5: 2, 18: 2, 30: 2, 2: 3, 65: 4, 85: 3, 99: 3}


def test_metadata_is_as_published(tmp_path):
    dataset = _convert(tmp_path)
    keys = ["category", "subcategory", "question_type", "difficulty", "relevant_concepts", "question_hint"]
    keys += ["format_hint", "tid"]
    for row in _source_rows():
        metadata = _task_for(dataset, row["id"]).metadata
        assert metadata == {"id": row["id"], **{key: row[key] for key in keys}}


def test_the_dataset_has_no_annotations(tmp_path):
    assert _convert(tmp_path).annotations == ()


def test_reordered_options_are_different_label_sets(tmp_path):
    dataset = _convert(tmp_path)
    yes_no, no_yes = _task_for(dataset, 30), _task_for(dataset, 18)
    assert yes_no.label_set.names == ("Yes", "No")
    assert no_yes.label_set.names == ("No", "Yes")
    assert yes_no.label_set.ordered
    assert yes_no.label_set != no_yes.label_set


def test_each_option_order_is_stored_once_after_a_write(tmp_path):
    dataset = _convert(tmp_path)
    registry = LocalRegistry(tmp_path / "registry")
    registry.put_dataset(dataset)
    with TimeFReader(open_local_version(registry.root / "autonlab/timeseriesexam1/1.0.0")) as reader:
        stored = reader.read()
    tasks = {task.metadata["id"]: task for task in _classification_tasks(stored)}
    rows = {row["id"]: row for row in _source_rows()}
    assert tasks[1].label_set is not tasks[5].label_set
    assert tasks[18].label_set is not tasks[30].label_set
    assert len({id(task.label_set) for task in tasks.values()}) == len({tuple(row["options"]) for row in rows.values()})
    for row_id, task in tasks.items():
        assert task.label_set.names == tuple(rows[row_id]["options"])
    for row_id, task in tasks.items():
        answer = rows[row_id]["answer"]
        assert [read_value(target) for target in task.targets or ()] == [answer]
        assert answer in task.label_set.names
        assert set(task.label_set.names) == set(rows[row_id]["options"])


def _with(rows: list[dict], row_id: int, **changes) -> list[dict]:
    return [{**row, **changes} if row["id"] == row_id else row for row in rows]


def test_rejects_row_with_ts_and_ts1_and_ts2(tmp_path):
    rows = _source_rows()
    rows = _with(rows, 18, ts1=[1.0, 2.0], ts2=[1.0, 2.0])
    message = "row id 18: expected ts, or ts1 and ts2, found ['ts', 'ts1', 'ts2']"
    with pytest.raises(TimeNetConnectorError, match=re.escape(message)):
        _convert(tmp_path, rows)


def test_rejects_row_without_series(tmp_path):
    rows = _with(_source_rows(), 18, ts=None)
    message = "row id 18: expected ts, or ts1 and ts2, found []"
    with pytest.raises(TimeNetConnectorError, match=re.escape(message)):
        _convert(tmp_path, rows)


def test_rejects_row_with_one_of_ts1_and_ts2(tmp_path):
    rows = _with(_source_rows(), 1, ts2=None)
    message = "row id 1: expected ts, or ts1 and ts2, found ['ts1']"
    with pytest.raises(TimeNetConnectorError, match=re.escape(message)):
        _convert(tmp_path, rows)
