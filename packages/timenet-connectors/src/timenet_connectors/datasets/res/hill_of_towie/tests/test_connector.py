from datetime import UTC, datetime, timedelta
from pathlib import Path
import zipfile

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
    read_value,
    unix_us,
)
from timenet_connectors.datasets.cubico.kelmarsh.tasks import STOPPED_MINUTES_PROMPT, STOPPED_PROMPT, SplitCutoffs
from timenet_connectors.datasets.res.hill_of_towie import connector as hill_of_towie
from timenet_connectors.datasets.res.hill_of_towie.connector import TABLES, HillOfTowieConnector, HillOfTowieFiles
from timenet_connectors.datasets.res.hill_of_towie.scada import STEPS_PER_DAY


# A synthetic release of two invented turbines for three days in two months. It has the shape of
# the release and holds no value of it.
_START = datetime(2020, 1, 30, tzinfo=UTC)
_DAYS = 3
_STATIONS = {"T01": 2304510, "T02": 2304511}
_METADATA = (
    "﻿Wind Farm,Turbine Name,Latitude,Longitude,Station ID,Manufacturer,Model,Rated power (kW),"
    "Hub Height (m),Rotor Diameter (m),Country,Commercial Operations Date\n"
    "Testfarm,T02,1.6,-2.6,2304511,Maker,M1,2300,59,82,UK,2012-05-12\n"
    "Testfarm,T01,1.5,-2.5,2304510,Maker,M1,2300,59,82,UK,2012-05-12\n"
)
_ABSENT_STEP = 7
"""The step of the first day for which T01 has no row in any table."""
_EMPTY_DAY = 1
"""The day for which T02 has no row in any table."""
_TEN_MINUTES_US = 600_000_000
# Seconds of shutdown of T01 by step: a run on day 1 that ends partly stopped, and a run that
# crosses from day 1 into day 2.
_SHUTDOWN = {10: 600, 11: 600, 12: 120, 143: 30, 144: 600}


class _TestRelease(HillOfTowieConnector):
    CARD = Path(hill_of_towie.__file__).with_name("dataset.yaml")
    FIRST_DAY = _START
    END_DAY = _START + timedelta(days=_DAYS)


def _value(turbine: int, column: int, step: int) -> float:
    return turbine * 10_000 + column * 1000 + step / 3


def _has_row(name: str, step: int) -> bool:
    if name == "T01":
        return step != _ABSENT_STEP
    return step // STEPS_PER_DAY != _EMPTY_DAY


def _table_files(table: str, columns: list[str], *, conflict: bool) -> dict[str, str]:
    """Write a table as one file for each month, with the first period of a month in both files."""
    column_offset = [name for names in TABLES.values() for name in names].index(columns[0])
    header = ",".join(["TimeStamp", "StationId", "wtc_Other_mean", *columns]) + "\n"
    months: dict[str, list[str]] = {"2020_01": [], "2020_02": []}
    # The step before the first one ends at the start of the window. The reader skips it.
    for step in range(-1, _DAYS * STEPS_PER_DAY):
        end = _START + timedelta(minutes=10 * (step + 1))
        # The source writes midnight with or without the time of day.
        stamp = f"{end:%Y-%m-%d}" if (end.hour, end.minute, step % 2) == (0, 0, 0) else f"{end:%Y-%m-%d %H:%M:%S}"
        for turbine, (name, station) in enumerate(_STATIONS.items()):
            if not _has_row(name, step):
                continue
            cells = [repr(_value(turbine, column_offset + column, step)) for column in range(len(columns))]
            if step == 0 and table == "tblSCTurGrid":
                cells[0] = ""
            row = ",".join([stamp, str(station), "7", *cells]) + "\n"
            february = datetime(2020, 2, 1, tzinfo=UTC)
            if end <= february:
                months["2020_01"].append(row)
            if end >= february:
                other = conflict and end == february and table == "tblSCTurGrid"
                months["2020_02"].append(",".join([stamp, str(station), "7", "1"]) + "\n" if other else row)
    return {f"{table}_{month}.csv": header + "".join(rows) for month, rows in months.items()}


def _write_release(directory: Path, *, conflict: bool = False, shutdown_rows: int | None = None) -> HillOfTowieFiles:
    files = HillOfTowieFiles(
        archives=(directory / "2020.zip",),
        shutdown_path=directory / "shutdown.zip",
        metadata_path=directory / "metadata.csv",
    )
    with zipfile.ZipFile(files.archives[0], "w") as archive:
        for table, columns in TABLES.items():
            for name, text in _table_files(table, columns, conflict=conflict).items():
                archive.writestr(name, text)
        archive.writestr("tblAlarmLog_2020_01.csv", "TimeStamp,StationId\n")
    rows = [
        f"{_START + timedelta(minutes=10 * step):%Y-%m-%d %H:%M:%S}+00:00,{name},"
        f"{_SHUTDOWN.get(step, 0) if name == 'T01' else 0}\n"
        for step in range(_DAYS * STEPS_PER_DAY + 1)
        for name in _STATIONS
    ]
    with zipfile.ZipFile(files.shutdown_path, "w") as archive:
        archive.writestr(
            "ShutdownDuration.csv",
            "TimeStamp_StartFormat,TurbineName,ShutdownDuration\n" + "".join(rows[:shutdown_rows]),
        )
    files.metadata_path.write_text(_METADATA, encoding="utf-8")
    return files


@pytest.fixture
def dataset(tmp_path) -> TimeFDataset:
    return _TestRelease().convert([_write_release(tmp_path)])


def _targets(dataset: TimeFDataset, task_type: type[Task], prompt: str) -> list[object]:
    found = [task for task in dataset.tasks_of(task_type) if read_value(task.prompt) == prompt]
    return [tuple(read_value(target) for target in task.targets or ()) for task in found]


def test_each_day_with_a_row_becomes_a_record_at_the_start_of_each_period(dataset):
    days = [(record.metadata["turbine"], record.start_time.timestamp) for record in dataset.records]
    assert days == [
        *[("T01", unix_us(_START + timedelta(days=day))) for day in range(_DAYS)],
        *[("T02", unix_us(_START + timedelta(days=day))) for day in range(_DAYS) if day != _EMPTY_DAY],
    ]
    first = dataset.records[0]
    assert first.metadata["rated_power_kw"] == pytest.approx(2300)
    by_name = {signal.name: signal for signal in first.signals}
    assert set(by_name) == {
        "power",
        "wind_speed",
        "pitch_angle",
        "rotor_speed",
        "nacelle_direction",
        "ambient_temperature",
    }
    # The row that ends at 00:10 holds the first period of the day. Power is column 0, and
    # ambient temperature is the last column.
    for name, column in (("power", 0), ("ambient_temperature", 5)):
        expected = np.array([_value(0, column, step) for step in range(STEPS_PER_DAY)])
        expected[_ABSENT_STEP] = np.nan
        if name == "power":
            expected[0] = np.nan
        values = by_name[name].to_arrow()
        # A period without a row and an empty cell stay missing. The connector does not fill them.
        assert values[_ABSENT_STEP].as_py() is None
        np.testing.assert_array_equal(values.to_numpy(zero_copy_only=False), expected)


def test_the_signal_columns_of_a_table_keep_their_order(dataset):
    record = dataset.records[_DAYS]  # the first day of T02
    by_name = {signal.name: signal.to_arrow()[3].as_py() for signal in record.signals}
    columns = [name for names in TABLES.values() for name in names]
    assert by_name["wind_speed"] == pytest.approx(_value(1, columns.index("wtc_AcWindSp_mean"), 3))
    assert by_name["rotor_speed"] == pytest.approx(_value(1, columns.index("wtc_MainSRpm_mean"), 3))
    assert by_name["nacelle_direction"] == pytest.approx(_value(1, columns.index("wtc_NacelPos_mean"), 3))


def test_runs_of_shutdown_become_annotations_and_tasks(dataset):
    first, second, third = dataset.records[:_DAYS]
    spans = [[(a.key, a.value, a.span) for a in record.annotations] for record in (first, second, third)]
    assert spans == [
        [
            ("shutdown", 1320, TimeInterval.micros(10 * _TEN_MINUTES_US, 13 * _TEN_MINUTES_US)),
            ("shutdown", 30, TimeInterval.micros(143 * _TEN_MINUTES_US, 144 * _TEN_MINUTES_US)),
        ],
        [("shutdown", 600, TimeInterval.micros(0, _TEN_MINUTES_US))],
        [],
    ]
    assert _targets(dataset, ClassificationTask, STOPPED_PROMPT) == [
        ("stopped",),
        ("stopped",),
        ("running",),
        ("running",),
        ("running",),
    ]
    assert _targets(dataset, ScalarPredictionTask, STOPPED_MINUTES_PROMPT) == [(22.5,), (10.0,), (0.0,), (0.0,), (0.0,)]
    assert _targets(dataset, TemporalLocalizationTask, hill_of_towie.SHUTDOWN_PERIODS_PROMPT) == [
        (
            TimeInterval.micros(10 * _TEN_MINUTES_US, 13 * _TEN_MINUTES_US),
            TimeInterval.micros(143 * _TEN_MINUTES_US, 144 * _TEN_MINUTES_US),
        ),
        (TimeInterval.micros(0, _TEN_MINUTES_US),),
    ]
    assert {task.split for task in dataset.tasks} == {Split.TRAIN}


def test_a_run_of_shutdown_across_a_cutoff_leaves_its_days_without_tasks(tmp_path):
    # The run of T01 from step 143 to step 144 runs across the start of day 2.
    class CutoffAtDayTwo(_TestRelease):
        SPLIT_CUTOFFS = SplitCutoffs(validation=_START + timedelta(days=1), test=_START + timedelta(days=2))

    dataset = CutoffAtDayTwo().convert([_write_release(tmp_path)])
    first, second = dataset.records[:2]
    assert len(first.annotations) == 2
    assert len(second.annotations) == 1
    splits = [(task.inputs[0].metadata["turbine"], task.split) for task in dataset.tasks_of(ScalarPredictionTask)]
    assert splits == [("T01", Split.TEST), ("T02", Split.TRAIN), ("T02", Split.TEST)]


def test_split_follows_the_dates():
    cutoffs = HillOfTowieConnector.SPLIT_CUTOFFS
    assert cutoffs.split_of_day(unix_us(datetime(2021, 12, 31, tzinfo=UTC))) == Split.TRAIN
    assert cutoffs.split_of_day(unix_us(datetime(2022, 1, 1, tzinfo=UTC))) == Split.VALIDATION
    assert cutoffs.split_of_day(unix_us(datetime(2022, 12, 31, tzinfo=UTC))) == Split.VALIDATION
    assert cutoffs.split_of_day(unix_us(datetime(2023, 1, 1, tzinfo=UTC))) == Split.TEST


def test_a_repeated_row_with_other_values_is_rejected(tmp_path):
    files = _write_release(tmp_path, conflict=True)
    with pytest.raises(TimeNetConnectorError, match="repeats a row"):
        _TestRelease().convert([files])


def test_a_period_without_a_shutdown_duration_is_rejected(tmp_path):
    files = _write_release(tmp_path, shutdown_rows=-4)
    with pytest.raises(TimeNetConnectorError, match="no row or more than one row"):
        _TestRelease().convert([files])


def test_an_archive_without_a_table_is_rejected(tmp_path):
    files = _write_release(tmp_path)
    with zipfile.ZipFile(files.archives[0], "w") as archive:
        archive.writestr("tblSCTurGrid_2020_01.csv", "TimeStamp,StationId,wtc_ActPower_mean\n")
    with pytest.raises(TimeNetConnectorError, match="no file for table"):
        _TestRelease().convert([files])


def test_the_downtime_task_counts_only_shutdown_at_wind_of_4_ms_or_more(dataset):
    # No day of the release has an hour of shutdown, and the wind values are high. A day with a shutdown
    # of 22.5 or 10 minutes is in between and has no task; a day with none is no downtime.
    assert _targets(dataset, ClassificationTask, hill_of_towie.DOWNTIME_60_PROMPT) == [
        ("no_downtime",),
        ("no_downtime",),
        ("no_downtime",),
    ]


def test_an_hour_of_shutdown_in_wind_is_downtime(tmp_path, monkeypatch):
    monkeypatch.setitem(globals(), "_SHUTDOWN", dict.fromkeys(range(20, 26), 600))
    dataset = _TestRelease().convert([_write_release(tmp_path)])
    labels = []
    for task in dataset.tasks_of(ClassificationTask):
        if read_value(task.prompt) == hill_of_towie.DOWNTIME_60_PROMPT:
            (record,) = task.inputs
            assert task.targets is not None
            labels.append((record.metadata["turbine"], read_value(task.targets[0])))
    assert labels == [
        ("T01", "downtime"),
        ("T01", "no_downtime"),
        ("T01", "no_downtime"),
        ("T02", "no_downtime"),
        ("T02", "no_downtime"),
    ]


def test_the_downtime_rule_by_wind():
    seconds = np.zeros(144)
    wind = np.full(144, 8.0)
    assert hill_of_towie._downtime_60(seconds, wind) == "no_downtime"
    seconds[:6] = 600
    assert hill_of_towie._downtime_60(seconds, wind) == "downtime"
    seconds[5] = 599
    assert hill_of_towie._downtime_60(seconds, wind) is None
    # Shutdown in light wind does not count and does not block the label.
    assert hill_of_towie._downtime_60(seconds, np.full(144, 3.9)) == "no_downtime"
    # Shutdown in periods without a wind value: under ten minutes is fine, ten minutes or more is not known.
    unread = np.full(144, 8.0)
    unread[:1] = np.nan
    seconds = np.zeros(144)
    seconds[0] = 599
    assert hill_of_towie._downtime_60(seconds, unread) == "no_downtime"
    seconds[0] = 600
    assert hill_of_towie._downtime_60(seconds, unread) is None


def test_a_record_states_how_many_power_values_are_missing(dataset):
    for record in dataset.records:
        power = next(signal for signal in record.signals if signal.name == "power")
        assert record.metadata["power_missing_steps"] == power.null_count
    assert dataset.records[0].metadata["power_missing_steps"] >= 1


def test_the_card_has_the_new_version():
    assert str(HillOfTowieConnector().metadata().dataset_version) == "1.1.0"
