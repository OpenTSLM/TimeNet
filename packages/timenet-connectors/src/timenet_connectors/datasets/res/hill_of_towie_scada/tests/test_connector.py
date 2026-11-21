"""Round-trip tests for Hill of Towie's SCADA forecasts."""

from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from timenet import TimeNet
from timenet.dataset.axis import RegularAxis
from timenet.engine import store_dataset
from timenet.errors import TimeNetConnectorError
from timenet.types import ForecastingTask, TimeInterval, unix_us, ureg
from timenet_connectors.datasets.res.hill_of_towie_scada.connector import HillOfTowieFiles, HillOfTowieSCADAConnector


_CAMPAIGN_BOUNDARIES = {
    (2025, 10): datetime(2025, 10, 28, 13),
    (2026, 1): datetime(2026, 1, 7, 13),
    (2026, 2): datetime(2026, 2, 3, 18),
}


def _month_contents(first: datetime, january_gap: bool) -> tuple[list[str], list[str]]:
    power = ["TimeStamp,StationId,wtc_ActPower_mean"]
    telemetry = ["TimeStamp,StationId,wtc_AcWindSp_mean,wtc_GenRpm_mean"]
    starts = [first]
    if boundary := _CAMPAIGN_BOUNDARIES.get((first.year, first.month)):
        starts.extend((boundary - timedelta(hours=7), boundary))
    for start in starts:
        for step in range(0 if start == first else 1, 43):
            stamp = (start + timedelta(minutes=step * 10)).isoformat(sep=" ")
            for station, offset in ((2304510, 0), (2304511, 100)):
                power.append(f"{stamp},{station},{first.month * 100 + step + offset}")
                if step <= 36 or start != first:
                    gap = january_gap and station == 2304510 and step == 24
                    wind = "" if gap else 8 + offset
                    telemetry.append(f"{stamp},{station},{wind},{900 + offset}")
    return power, telemetry


def _source(tmp_path: Path, *, january_gap: bool = False, duplicate: bool = False) -> HillOfTowieFiles:
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "\ufeffStation ID,Turbine Name,Rated power (kW)\n2304510,T01,2300\n2304511,T02,2300\n",
        encoding="utf-8",
    )
    paths = []
    for year in range(2016, 2027):
        path = tmp_path / f"{year}.zip"
        paths.append(path)
        with ZipFile(path, "w") as archive:
            for month in range(1, 5 if year == 2026 else 13):
                power, telemetry = _month_contents(datetime(year, month, 1), january_gap and (year, month) == (2016, 1))
                if duplicate and (year, month) == (2016, 1):
                    power.append(power[-1])
                archive.writestr(f"tblSCTurGrid_{year}_{month:02}.csv", "\n".join(power))
                archive.writestr(
                    f"tblSCTurbine_{year}_{month:02}.csv", "\n".join([telemetry[0], *reversed(telemetry[1:])])
                )
    return HillOfTowieFiles(tuple(paths), metadata)


def test_forecasts_roundtrip_preserves_observations_and_units(tmp_path):
    converted = HillOfTowieSCADAConnector().convert(_source(tmp_path))
    assert converted.has_task_stream
    registry = tmp_path / "registry"
    store_dataset(converted, registry)
    with TimeNet(sources=[registry]).open("res/hill-of-towie-scada@2.0.0") as reader:
        task = next(reader.iter_tasks(split="train", batch_size=1))
        assert isinstance(task, ForecastingTask)
        assert isinstance(task.scope, TimeInterval)
        (record,) = task.inputs
        assert task.targets is not None
        (target,) = task.targets
        assert isinstance(target, TimeInterval)
        assert target.time_series is not None
        (target_signal,) = target.time_series
        assert record.metadata["turbine_id"] == "T01"
        assert task.metadata["window_start_utc"] == "2016-01-01T00:00:00+00:00"
        assert task.metadata["forecast_cutoff_utc"] == "2016-01-01T06:00:00+00:00"
        assert record.start_time.timestamp == unix_us(datetime(2016, 1, 1, 0, 10, tzinfo=UTC))
        channels = {signal.name: signal for signal in record.signals}
        assert channels.keys() == {"active_power", "wind_speed", "generator_speed"}
        power, wind, rpm = (channels[name] for name in ("active_power", "wind_speed", "generator_speed"))
        assert target_signal is power
        assert power.spec.unit_value == ureg.kilowatt
        assert wind.spec.unit_value == ureg.meter / ureg.second
        assert rpm.spec.unit_value == ureg.rpm
        for signal in (power, wind, rpm):
            assert isinstance(signal.time_axis, RegularAxis)
            assert signal.time_axis.period_us == 600_000_000
            assert signal.step_range(task.scope) == (0, 36)
        assert power.step_range(target) == (36, 42)
        np.testing.assert_array_equal(power.slice(task.scope).to_numpy(), 100 + np.arange(1, 37))
        np.testing.assert_array_equal(wind.slice(task.scope).to_numpy(), np.full(36, 8))
        np.testing.assert_array_equal(rpm.slice(task.scope).to_numpy(), np.full(36, 900))
        np.testing.assert_array_equal(power.slice(target).to_numpy(), 100 + np.arange(37, 43))
        assert wind.read_steps(36, 42).null_count == 6


def test_chronological_split_and_campaign_boundaries(tmp_path):
    dataset = HillOfTowieSCADAConnector().convert(_source(tmp_path))
    tasks = list(dataset.iter_tasks())
    assert Counter(task.split for task in tasks) == {"train": 210, "validation": 6, "test": 44}
    by_cohort = Counter(task.metadata["evaluation_cohort"] for task in tasks)
    assert by_cohort == {
        "historical_train": 210,
        "post_tuneup_validation": 6,
        "pre_commissioning": 22,
        "dynamic_yaw_commissioning": 10,
        "dynamic_yaw_west_trial": 6,
        "dynamic_yaw_west_and_east_trial": 6,
    }
    all_cutoffs = set()
    for task in tasks:
        cutoff = task.metadata["forecast_cutoff_utc"]
        assert isinstance(cutoff, str)
        all_cutoffs.add(datetime.fromisoformat(cutoff))
    for boundary in _CAMPAIGN_BOUNDARIES.values():
        limit = boundary.replace(tzinfo=UTC)
        assert limit - timedelta(hours=1) in all_cutoffs
        assert limit + timedelta(hours=6) in all_cutoffs
        assert not any(limit + timedelta(hours=hours) in all_cutoffs for hours in range(6))


def test_missing_history_excludes_only_affected_windows(tmp_path):
    dataset = HillOfTowieSCADAConnector().convert(_source(tmp_path, january_gap=True))
    tasks = list(dataset.iter_tasks(split="train"))
    assert len(tasks) == 209
    assert tasks[0].inputs[0].metadata["turbine_id"] == "T02"


def test_duplicate_measurement_fails(tmp_path):
    with pytest.raises(TimeNetConnectorError, match="duplicate measurement"):
        HillOfTowieSCADAConnector().convert(_source(tmp_path, duplicate=True))
