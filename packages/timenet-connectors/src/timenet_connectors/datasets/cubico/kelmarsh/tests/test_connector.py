from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.types import (
    ClassificationTask,
    ScalarPredictionTask,
    Split,
    Task,
    TemporalLocalizationTask,
    TimeInterval,
    TimePoint,
    read_value,
    unix_us,
)
from timenet_connectors.datasets.cubico.kelmarsh import connector as kelmarsh_connector, tasks
from timenet_connectors.datasets.cubico.kelmarsh.connector import KelmarshConnector
from timenet_connectors.datasets.cubico.kelmarsh.farm import SOURCE_COLUMNS, TurbineYearFiles
from timenet_connectors.datasets.cubico.kelmarsh.greenbyte import STEPS_PER_DAY


# A synthetic export of one invented turbine for two days. It has the shape of the release and
# holds no value of it.
_TURBINE = "Testfarm 1"
_START = datetime(2022, 3, 1, tzinfo=UTC)
_DAYS = 2
_COMMENTS = f"# Synthetic export.\n#\n# Turbine: {_TURBINE}\n# Time zone: UTC\n#\n#\n#\n#\n#\n"
_GAP_STEP = 5
"""The step of the first day at which every signal is empty."""
_STATIC = (
    "﻿Wind Farm,Title,Alternative Title,Identity,Manufacturer,Model,Rated power (kW),Hub Height (m),"
    "Rotor Diameter (m),Latitude,Longitude,Elevation (m),Country,Commercial Operations Date\n"
    f"Testfarm,{_TURBINE},TF1,X 1,Maker,M1,2000,80.5,90,1.5,-2.5,100,UK,01/01/2016\n"
)
_STATUS_HEADER = (
    "Timestamp start,Timestamp end,Duration,Status,Code,Message,Comment,Service contract category,IEC category\n"
)
# Day 1 has two stops that overlap from 01:00 to 03:00, and a stop that runs into day 2 from 23:00
# to 00:30. Day 2 also has a curtailment and a stop without a category.
_STATUS_ROWS = (
    "2022-03-01 00:10:00,-,-,Informational,0,System OK,,System OK (32),Full Performance\n"
    "2022-03-01 01:00:00,2022-03-01 02:30:00,01:30:00,Stop,1,Manual stop,,Manual stop (service)  (9),"
    "Scheduled Maintenance\n"
    "2022-03-01 02:00:00,2022-03-01 03:00:00,01:00:00,Stop,2,Grid loss,,External stop (grid) (4),"
    "Out of Electrical Specification\n"
    "2022-03-01 23:00:00,2022-03-02 00:30:00,01:30:00,Stop,3,Low hydraulic pressure,,Mechanical error (23),"
    "Forced outage\n"
    "2022-03-02 05:00:00,2022-03-02 05:10:00,00:10:00,Stop,4,Check vane!,,Sensor error (21),\n"
    "2022-03-02 10:00:00,2022-03-02 12:00:00,02:00:00,Curtailment,108,Technical curtailment,,,Partial Performance\n"
)
_HOUR_US = 3_600_000_000


def _value(column: int, step: int) -> float:
    return column * 1000 + step / 3


def _write_data(path: Path, *, skip_step: int | None = None) -> None:
    lines = [_COMMENTS, ",".join(["# Date and time", "Unused (x)", *SOURCE_COLUMNS]) + "\n"]
    for step in range(_DAYS * STEPS_PER_DAY):
        if step == skip_step:
            continue
        cells = ["" if step == _GAP_STEP else repr(_value(column, step)) for column in range(len(SOURCE_COLUMNS))]
        lines.append(",".join([f"{_START + timedelta(minutes=10 * step):%Y-%m-%d %H:%M:%S}", "7", *cells]) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def _write_release(
    directory: Path, *, status_rows: str = _STATUS_ROWS, skip_step: int | None = None
) -> TurbineYearFiles:
    ref = TurbineYearFiles(
        data_path=directory / "Turbine_Data_Testfarm_1.csv",
        status_path=directory / "Status_Testfarm_1.csv",
        static_path=directory / "static.csv",
    )
    _write_data(ref.data_path, skip_step=skip_step)
    ref.status_path.write_text(_COMMENTS + _STATUS_HEADER + status_rows, encoding="utf-8")
    ref.static_path.write_text(_STATIC, encoding="utf-8")
    return ref


@pytest.fixture
def dataset(tmp_path) -> TimeFDataset:
    return KelmarshConnector().convert([_write_release(tmp_path)])


def _targets(dataset: TimeFDataset, task_type: type[Task], prompt: str) -> list[object]:
    found = [task for task in dataset.tasks_of(task_type) if read_value(task.prompt) == prompt]
    return [tuple(read_value(target) for target in task.targets or ()) for task in found]


def test_each_day_becomes_a_record_with_the_source_values(dataset):
    assert len(dataset.records) == _DAYS
    for day, record in enumerate(dataset.records):
        assert record.start_time.timestamp == unix_us(_START + timedelta(days=day))
        assert record.metadata["turbine"] == _TURBINE
        assert record.metadata["hub_height_m"] == pytest.approx(80.5)
        by_name = {signal.name: signal for signal in record.signals}
        assert set(by_name) == {
            "power",
            "wind_speed",
            "pitch_angle",
            "rotor_speed",
            "nacelle_direction",
            "ambient_temperature",
        }
        steps = range(day * STEPS_PER_DAY, (day + 1) * STEPS_PER_DAY)
        expected = np.array([_value(0, step) for step in steps])
        values = by_name["power"].to_arrow()
        if day == 0:
            # The empty cell stays missing. The connector does not fill it.
            assert values.null_count == 1
            assert values[_GAP_STEP].as_py() is None
            expected[_GAP_STEP] = np.nan
        else:
            assert values.null_count == 0
        np.testing.assert_array_equal(values.to_numpy(zero_copy_only=False), expected)


def test_stops_and_curtailments_become_annotations_clipped_to_the_day(dataset):
    first, second = ([(a.key, read_value(a.value), a.span) for a in record.annotations] for record in dataset.records)
    assert (tasks.STOP_KEY, "maintenance", TimeInterval.micros(_HOUR_US, _HOUR_US * 5 // 2)) in first
    assert (tasks.STOP_MESSAGE_KEY, "Grid loss", TimeInterval.micros(2 * _HOUR_US, 3 * _HOUR_US)) in first
    assert (tasks.STOP_KEY, "fault", TimeInterval.micros(23 * _HOUR_US, 24 * _HOUR_US)) in first
    assert (tasks.STOP_KEY, "fault", TimeInterval.micros(0, _HOUR_US // 2)) in second
    assert (
        tasks.STOP_KEY,
        tasks.UNKNOWN_REASON,
        TimeInterval.micros(5 * _HOUR_US, 5 * _HOUR_US + _HOUR_US // 6),
    ) in second
    assert (tasks.CURTAILMENT_KEY, "Technical curtailment", TimeInterval.micros(10 * _HOUR_US, 12 * _HOUR_US)) in second
    assert len(first) == 6
    assert len(second) == 5


def test_tasks_state_the_stops_of_each_day(dataset):
    assert _targets(dataset, ClassificationTask, tasks.STOPPED_PROMPT) == [("stopped",), ("stopped",)]
    # Day 1: 01:00 to 03:00 counts once, plus 23:00 to 24:00. Day 2: 00:00 to 00:30, plus 10 minutes.
    assert _targets(dataset, ScalarPredictionTask, tasks.STOPPED_MINUTES_PROMPT) == [(180.0,), (40.0,)]
    assert _targets(dataset, TemporalLocalizationTask, tasks.STOP_INTERVALS_PROMPT) == [
        (TimeInterval.micros(_HOUR_US, 3 * _HOUR_US), TimeInterval.micros(23 * _HOUR_US, 24 * _HOUR_US)),
        (TimeInterval.micros(0, _HOUR_US // 2), TimeInterval.micros(5 * _HOUR_US, 5 * _HOUR_US + _HOUR_US // 6)),
    ]
    # Day 1: maintenance has 90 minutes, grid and fault have 60 each. Day 2 has a stop without a category.
    assert _targets(dataset, ClassificationTask, tasks.STOP_REASON_PROMPT) == [("maintenance",)]
    assert _targets(dataset, ClassificationTask, tasks.CURTAILED_PROMPT) == [("not_curtailed",), ("curtailed",)]
    assert {task.split for task in dataset.tasks} == {Split.TEST}


def test_a_day_without_events_is_running_and_has_no_curtailment_task(tmp_path):
    dataset = KelmarshConnector().convert([_write_release(tmp_path, status_rows="")])
    assert _targets(dataset, ClassificationTask, tasks.STOPPED_PROMPT) == [("running",), ("running",)]
    assert _targets(dataset, ScalarPredictionTask, tasks.STOPPED_MINUTES_PROMPT) == [(0.0,), (0.0,)]
    assert dataset.tasks_of(TemporalLocalizationTask) == ()
    # The log records no curtailment, so a day is not known to be free of it.
    assert _targets(dataset, ClassificationTask, tasks.CURTAILED_PROMPT) == []
    assert all(record.annotations == () for record in dataset.records)


class _CutoffAtDayTwo(KelmarshConnector):
    CARD = Path(kelmarsh_connector.__file__).with_name("dataset.yaml")
    SPLIT_CUTOFFS = tasks.SplitCutoffs(validation=_START, test=_START + timedelta(days=1))


def test_split_follows_the_dates_and_no_train_day_is_after_a_test_day():
    cutoffs = KelmarshConnector.SPLIT_CUTOFFS
    days = [unix_us(datetime(2019, 1, 1, tzinfo=UTC) + timedelta(days=day)) for day in range(4 * 365 + 1)]
    splits = [cutoffs.split_of_day(day) for day in days]
    assert splits == sorted(splits, key=[Split.TRAIN, Split.VALIDATION, Split.TEST].index)
    assert splits[0] == Split.TRAIN
    assert cutoffs.split_of_day(unix_us(datetime(2021, 6, 30, tzinfo=UTC))) == Split.TRAIN
    assert cutoffs.split_of_day(unix_us(datetime(2021, 7, 1, tzinfo=UTC))) == Split.VALIDATION
    assert cutoffs.split_of_day(unix_us(datetime(2021, 12, 31, tzinfo=UTC))) == Split.VALIDATION
    assert cutoffs.split_of_day(unix_us(datetime(2022, 1, 1, tzinfo=UTC))) == Split.TEST


def test_no_event_of_the_log_has_tasks_in_two_splits(tmp_path):
    # The cutoff to test is at the start of day 2. The stop from 23:00 to 00:30 runs across it.
    dataset = _CutoffAtDayTwo().convert([_write_release(tmp_path)])
    first, second = dataset.records
    # The Records keep their annotations. Only the tasks are left out.
    assert len(first.annotations) == 6
    assert len(second.annotations) == 5
    assert dataset.tasks == ()


def test_an_event_that_ends_at_a_cutoff_keeps_its_tasks(tmp_path):
    rows = (
        "2022-03-01 23:00:00,2022-03-02 00:00:00,01:00:00,Stop,3,Low hydraulic pressure,,Mechanical error (23),"
        "Forced outage\n"
        "2022-03-02 05:00:00,2022-03-02 05:10:00,00:10:00,Stop,4,Check vane!,,Sensor error (21),Forced outage\n"
    )
    dataset = _CutoffAtDayTwo().convert([_write_release(tmp_path, status_rows=rows)])
    _assert_no_event_in_two_splits(dataset, rows)
    assert [task.split for task in dataset.tasks_of(ScalarPredictionTask)] == [Split.VALIDATION, Split.TEST]


def test_stops_that_touch_across_a_cutoff_are_one_incident(tmp_path):
    # Two entries of the log: the first ends at the cutoff, and the second starts there.
    rows = (
        "2022-03-01 23:00:00,2022-03-02 00:00:00,01:00:00,Stop,3,Low hydraulic pressure,,Mechanical error (23),"
        "Forced outage\n"
        "2022-03-02 00:00:00,2022-03-02 00:30:00,00:30:00,Stop,3,Low hydraulic pressure,,Mechanical error (23),"
        "Forced outage\n"
    )
    dataset = _CutoffAtDayTwo().convert([_write_release(tmp_path, status_rows=rows)])
    assert dataset.tasks == ()


def _assert_no_event_in_two_splits(dataset: TimeFDataset, status_rows: str) -> None:
    """Check the intervals of the source log, not the clipped annotations, against the splits."""
    split_of_day = {task.inputs[0].start_time.timestamp: task.split for task in dataset.tasks}
    day_us = 86_400_000_000
    for row in status_rows.splitlines():
        start, end = (unix_us(datetime.fromisoformat(cell).replace(tzinfo=UTC)) for cell in row.split(",")[:2])
        days = range(start // day_us * day_us, end, day_us)
        assert len({split_of_day[day] for day in days if day in split_of_day}) <= 1


def test_a_table_with_a_missing_row_is_rejected(tmp_path):
    ref = _write_release(tmp_path, skip_step=STEPS_PER_DAY)
    with pytest.raises(TimeNetConnectorError, match="whole UTC days"):
        KelmarshConnector().convert([ref])


def test_a_stop_without_an_end_time_leaves_its_days_without_tasks(tmp_path):
    # The log does not state when the turbine ran again. The next System OK entry is on the same day.
    rows = (
        "2022-03-01 01:00:00,-,-,Stop,1,Manual stop,,Manual stop (service)  (9),Scheduled Maintenance\n"
        "2022-03-01 02:00:00,-,-,Informational,0,System OK,,System OK (32),Full Performance\n"
    )
    dataset = KelmarshConnector().convert([_write_release(tmp_path, status_rows=rows)])
    first, second = dataset.records
    assert [(a.key, read_value(a.value), a.span) for a in first.annotations] == [
        (tasks.OPEN_STOP_KEY, "Manual stop", TimePoint.micros(_HOUR_US))
    ]
    assert second.annotations == ()
    assert {task.inputs[0].start_time for task in dataset.tasks} == {second.start_time}
    assert _targets(dataset, ClassificationTask, tasks.STOPPED_PROMPT) == [("running",)]


def test_a_stop_without_an_end_time_and_no_later_system_ok_leaves_every_later_day_without_tasks(tmp_path):
    row = "2022-03-01 01:00:00,-,-,Stop,1,Manual stop,,Manual stop (service)  (9),Scheduled Maintenance\n"
    dataset = KelmarshConnector().convert([_write_release(tmp_path, status_rows=row)])
    assert dataset.tasks == ()


def test_a_curtailment_without_an_end_time_is_rejected(tmp_path):
    row = "2022-03-01 01:00:00,-,-,Curtailment,108,Technical curtailment,,,Partial Performance\n"
    ref = _write_release(tmp_path, status_rows=row)
    with pytest.raises(TimeNetConnectorError, match="no end time"):
        KelmarshConnector().convert([ref])


def _downtime_targets(tmp_path: Path, status_rows: str) -> list[object]:
    dataset = KelmarshConnector().convert([_write_release(tmp_path, status_rows=status_rows)])
    return _targets(dataset, ClassificationTask, tasks.DOWNTIME_60_PROMPT)


def test_the_downtime_task_asks_for_an_hour_of_stop(dataset):
    # Day 1 has 180 minutes of stop. Day 2 has 40 minutes of stop and a long curtailment: no task.
    assert _targets(dataset, ClassificationTask, tasks.DOWNTIME_60_PROMPT) == [("downtime",)]


def test_downtime_starts_at_exactly_sixty_minutes_of_stop(tmp_path):
    hour = "2022-03-01 01:00:00,2022-03-01 02:00:00,01:00:00,Stop,1,Manual stop,,Manual stop (service)  (9),Scheduled Maintenance\n"
    short = hour.replace("02:00:00,01:00:00", "01:59:00,00:59:00")
    assert _downtime_targets(tmp_path, hour) == [("downtime",), ("no_downtime",)]
    # A stop of 59 minutes is a short stop: no label for that day, and the next day has no stop.
    assert _downtime_targets(tmp_path, short) == [("no_downtime",)]


def test_a_power_limit_is_not_downtime_and_a_long_one_leaves_the_day_without_the_task(tmp_path):
    def curtailment(end: str) -> str:
        return f"2022-03-02 10:00:00,2022-03-02 {end},00:00:00,Curtailment,108,Technical curtailment,,,Partial Performance\n"

    assert _downtime_targets(tmp_path, curtailment("10:30:00")) == [("no_downtime",), ("no_downtime",)]
    assert _downtime_targets(tmp_path, curtailment("11:00:00")) == [("no_downtime",)]


def test_the_downtime_rule_by_minutes():
    minute_us = 60_000_000
    assert tasks._downtime_60(60 * minute_us, 0) == "downtime"
    assert tasks._downtime_60(60 * minute_us, 600 * minute_us) == "downtime"
    assert tasks._downtime_60(59 * minute_us, 0) is None
    assert tasks._downtime_60(0, 59 * minute_us) == "no_downtime"
    assert tasks._downtime_60(0, 60 * minute_us) is None


def test_a_record_states_how_many_power_values_are_missing(dataset):
    # The first day has one step without any value. The second day has none.
    assert [record.metadata["power_missing_steps"] for record in dataset.records] == [1, 0]


def test_the_card_has_the_new_version():
    assert str(KelmarshConnector().metadata().dataset_version) == "1.1.0"
