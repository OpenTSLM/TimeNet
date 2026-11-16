from datetime import UTC, datetime
from fractions import Fraction
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import Record, TimeFDataset
from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.types import ForecastingTask
from timenet.types.clock import unix_us
from timenet.types.splits import Split
from timenet.types.text import read_value
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.servicenow.context_is_key import ContextIsKeyConnector


_HOUR_US = 3_600_000_000
_DAY_US = 24 * _HOUR_US


def _row(**overrides):
    # One row shaped exactly like a Hub row. The values are hand-written for the tests and not
    # taken from the real benchmark, so the repo ships no dataset bytes.
    row = {
        "name": "SyntheticHourlyTask",
        "seed": 1,
        "weight": "1/2",
        "context_sources": ["c_cov", "c_f"],
        "skills": ["forecasting", "instruction following"],
        "background": "This is a synthetic hourly load in kW.",
        "scenario": "Suppose the load doubles from 2020-01-01 04:00:00 for 1 hour.",
        "constraints": "",
        "seasonal_period": 24,
        "past_time": json.dumps(
            {
                "0": {
                    "2020-01-01T00:00:00.000": 1.0,
                    "2020-01-01T01:00:00.000": 2,
                    "2020-01-01T02:00:00.000": 3.5,
                }
            }
        ),
        "future_time": json.dumps({"0": {"2020-01-01T03:00:00.000": 4.0, "2020-01-01T04:00:00.000": 8.0}}),
        "metric_scaling": 0.25,
        "region_of_interest": [1],
        "constraint_min": float("-inf"),
        "constraint_max": float("inf"),
        "constraint_variable_max_index": [],
        "constraint_variable_max_values": [],
    }
    row.update(overrides)
    return row


_HOURLY = _row()
# Monthly steps differ in length, and the source names them without a day.
_MONTHLY = _row(
    name="SyntheticMonthlyTask",
    seed=2,
    background="This is a synthetic monthly count.",
    scenario="",
    constraints="The values are at least 0 and at most 10.",
    past_time=json.dumps({"value": {"2021-01": 5.0, "2021-02": 6.0, "2021-03": 7.0}}),
    future_time=json.dumps({"value": {"2021-04": 8.0, "2021-05": 9.0}}),
    region_of_interest=[],
    constraint_min=0.0,
    constraint_max=10.0,
    constraint_variable_max_index=[0, 1],
    constraint_variable_max_values=[9.0, 9.5],
)
# The third timestamp format, with a gap of one day between the history and the future.
_GAPPED = _row(
    name="SyntheticGappedTask",
    seed=3,
    past_time=json.dumps({"GHI": {"2022-06-01 10:00": 1.0, "2022-06-01 11:00": 2.0, "2022-06-01 12:00": 3.0}}),
    future_time=json.dumps({"GHI": {"2022-06-02 10:00": 1.5, "2022-06-02 11:00": 2.5}}),
)
_FIXTURE_ROWS = [_HOURLY, _MONTHLY, _GAPPED]

_SCHEMA = pa.schema(
    [
        ("name", pa.string()),
        ("seed", pa.int64()),
        ("weight", pa.string()),
        ("context_sources", pa.list_(pa.string())),
        ("skills", pa.list_(pa.string())),
        ("background", pa.string()),
        ("scenario", pa.string()),
        ("constraints", pa.string()),
        ("seasonal_period", pa.int64()),
        ("past_time", pa.string()),
        ("future_time", pa.string()),
        ("metric_scaling", pa.float64()),
        ("region_of_interest", pa.list_(pa.int64())),
        ("constraint_min", pa.float64()),
        ("constraint_max", pa.float64()),
        ("constraint_variable_max_index", pa.list_(pa.int64())),
        ("constraint_variable_max_values", pa.list_(pa.float64())),
    ]
)


def _refs(tmp_path, rows):
    # Write the rows to a parquet file with the Hub file's schema; convert() reads it back.
    path = tmp_path / f"rows-{len(list(tmp_path.iterdir()))}.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=_SCHEMA), path)
    return [HubFileRef(path, "test")]


def _convert(tmp_path, rows=_FIXTURE_ROWS) -> TimeFDataset:
    return ContextIsKeyConnector().convert(_refs(tmp_path, rows))


def _task(dataset, name) -> ForecastingTask:
    return next(task for task in dataset.tasks if task.metadata["task_name"] == name)


def _records(task) -> tuple[Record, Record]:
    # The history record a task reads and the future record it asks for.
    (history,) = task.inputs
    assert task.targets is not None
    (future,) = task.targets
    assert isinstance(future, Record)
    return history, future


def test_metadata():
    metadata = ContextIsKeyConnector().metadata()
    assert metadata.dataset_id == "servicenow/context-is-key"
    assert str(metadata.license) == "Apache-2.0"
    assert str(metadata.dataset_version) == "1.0.0"


def test_source_is_pinned():
    # Literal pins, so moving to another repo, commit, or file is a deliberate test edit.
    assert ContextIsKeyConnector.HF_REPO == "ServiceNow/context-is-key"
    assert ContextIsKeyConnector.SOURCE_REVISION == "32da467ade142a16d099675e78b1129745c5ef7d"
    assert dict(ContextIsKeyConnector.SOURCE_FILES) == {"test": "data/test-00000-of-00001.parquet"}


def test_each_row_becomes_two_records_and_one_test_forecasting_task(tmp_path):
    dataset = _convert(tmp_path)
    assert len(dataset.records) == 2 * len(_FIXTURE_ROWS)
    assert len(dataset.tasks) == len(_FIXTURE_ROWS)
    for task in dataset.tasks:
        assert isinstance(task, ForecastingTask)
        assert task.split == Split.TEST
        history, future = _records(task)
        assert history.metadata == {
            "task_name": task.metadata["task_name"],
            "seed": task.metadata["seed"],
            "part": "history",
        }
        assert future.metadata == {**history.metadata, "part": "future"}


def test_values_and_signal_name_come_from_the_row(tmp_path):
    history_record, future_record = _records(_task(_convert(tmp_path), "SyntheticGappedTask"))
    history = history_record.signals[0]
    future = future_record.signals[0]
    assert history.name == future.name == "GHI"
    assert history.to_numpy().tolist() == [1.0, 2.0, 3.0]
    assert future.to_numpy().tolist() == [1.5, 2.5]


def test_integer_cells_become_floats(tmp_path):
    history, _ = _records(_task(_convert(tmp_path), "SyntheticHourlyTask"))
    assert history.signals[0].to_numpy().tolist() == [1.0, 2.0, 3.5]


def test_regular_series_share_an_origin_and_the_future_follows_the_history(tmp_path):
    history, future = _records(_task(_convert(tmp_path), "SyntheticHourlyTask"))
    assert history.start_time is future.start_time
    assert history.start_time.timestamp == unix_us(datetime(2020, 1, 1, tzinfo=UTC))
    assert history.signals[0].time_axis == RegularAxis(period_us=Fraction(_HOUR_US))
    assert future.signals[0].time_axis == RegularAxis(period_us=Fraction(_HOUR_US), offset_us=3 * _HOUR_US)


def test_gap_between_history_and_future_is_kept(tmp_path):
    history, future = _records(_task(_convert(tmp_path), "SyntheticGappedTask"))
    assert history.start_time.timestamp == unix_us(datetime(2022, 6, 1, 10, tzinfo=UTC))
    assert future.signals[0].time_axis == RegularAxis(period_us=Fraction(_HOUR_US), offset_us=_DAY_US)


def test_monthly_series_are_irregular_and_start_on_the_first_day(tmp_path):
    history, future = _records(_task(_convert(tmp_path), "SyntheticMonthlyTask"))
    assert history.start_time.timestamp == unix_us(datetime(2021, 1, 1, tzinfo=UTC))
    assert isinstance(history.signals[0].time_axis, IrregularAxis)
    assert history.signals[0].time_offsets_us().tolist() == [0, 31 * _DAY_US, 59 * _DAY_US]
    assert future.signals[0].time_offsets_us().tolist() == [90 * _DAY_US, 120 * _DAY_US]


def test_prompt_joins_the_nonempty_context_parts(tmp_path):
    dataset = _convert(tmp_path)
    assert read_value(_task(dataset, "SyntheticHourlyTask").prompt) == (
        "This is a synthetic hourly load in kW.\n\nSuppose the load doubles from 2020-01-01 04:00:00 for 1 hour."
    )
    assert read_value(_task(dataset, "SyntheticMonthlyTask").prompt) == (
        "This is a synthetic monthly count.\n\nThe values are at least 0 and at most 10."
    )


def test_task_metadata_keeps_the_metric_fields(tmp_path):
    dataset = _convert(tmp_path)
    assert _task(dataset, "SyntheticHourlyTask").metadata == {
        "task_name": "SyntheticHourlyTask",
        "seed": 1,
        "weight": "1/2",
        "context_sources": ["c_cov", "c_f"],
        "skills": ["forecasting", "instruction following"],
        "background": "This is a synthetic hourly load in kW.",
        "scenario": "Suppose the load doubles from 2020-01-01 04:00:00 for 1 hour.",
        "constraints": "",
        "seasonal_period": 24,
        "metric_scaling": 0.25,
        "region_of_interest": [1],
        # The source writes "no bound" as an infinity, which JSON cannot hold.
        "constraint_min": None,
        "constraint_max": None,
        "constraint_variable_max_index": [],
        "constraint_variable_max_values": [],
    }
    monthly = _task(dataset, "SyntheticMonthlyTask").metadata
    assert (monthly["constraint_min"], monthly["constraint_max"]) == (0.0, 10.0)
    assert monthly["constraint_variable_max_index"] == [0, 1]
    assert monthly["constraint_variable_max_values"] == [9.0, 9.5]


def test_unknown_timestamp_format_is_rejected(tmp_path):
    row = _row(past_time=json.dumps({"0": {"01/02/2020": 1.0}}))
    with pytest.raises(TimeNetConnectorError, match="01/02/2020"):
        _convert(tmp_path, [row])


def test_empty_history_is_rejected(tmp_path):
    row = _row(past_time=json.dumps({"0": {}}))
    with pytest.raises(TimeNetConnectorError, match="no history"):
        _convert(tmp_path, [row])


def test_tasks_survive_storage_round_trip(tmp_path):
    original = _convert(tmp_path)
    registry = LocalRegistry(tmp_path / "registry")
    registry.put_dataset(original)
    with TimeFReader(open_local_version(registry.root / "servicenow/context-is-key/1.0.0")) as reader:
        restored = reader.read()
    assert len(restored.records) == len(original.records)
    by_name = {task.metadata["task_name"]: task for task in restored.tasks}
    for task in original.tasks:
        stored = by_name[task.metadata["task_name"]]
        assert isinstance(stored, ForecastingTask)
        assert stored.split == Split.TEST
        assert stored.metadata == task.metadata
        assert read_value(stored.prompt) == read_value(task.prompt)
        history, future = _records(task)
        stored_history, stored_future = _records(stored)
        assert stored_history.signals[0].to_numpy().tolist() == history.signals[0].to_numpy().tolist()
        assert stored_future.signals[0].to_numpy().tolist() == future.signals[0].to_numpy().tolist()
        assert stored_future.signals[0].time_axis == future.signals[0].time_axis
