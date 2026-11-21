"""Test the Hill of Towie connector against a small SCADA archive."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from timenet import TimeNet
from timenet.dataset import Record
from timenet.dataset.axis import RegularAxis
from timenet.engine import store_dataset
from timenet.errors import TimeNetConnectorError
from timenet.types import ForecastingTask, unix_us, ureg
from timenet_connectors.datasets.res.hill_of_towie_scada.connector import (
    HillOfTowieFiles,
    HillOfTowieSCADAConnector,
)


def _source(tmp_path: Path, *, january_gap: bool = False, duplicate: bool = False) -> HillOfTowieFiles:
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "\ufeffStation ID,Turbine Name,Rated power (kW)\n2304510,T01,2300\n2304511,T02,2300\n",
        encoding="utf-8",
    )
    path = tmp_path / "2026.zip"
    with ZipFile(path, "w") as archive:
        for month in range(1, 5):
            origin = datetime(2026, month, 1)
            power = ["TimeStamp,StationId,wtc_ActPower_mean"]
            telemetry = ["TimeStamp,StationId,wtc_AcWindSp_mean,wtc_GenRpm_mean"]
            for step in range(43):
                stamp = (origin + timedelta(minutes=step * 10)).isoformat(sep=" ")
                for station, offset in ((2304510, 0), (2304511, 100)):
                    value = month * 100 + step + offset
                    power.append(f"{stamp},{station},{value}")
                    if step <= 36:  # Future telemetry is intentionally absent.
                        wind = "" if january_gap and month == 1 and station == 2304510 and step == 24 else 8 + offset
                        telemetry.append(f"{stamp},{station},{wind},{900 + offset}")
            if duplicate and month == 1:
                power.append(power[-1])
            archive.writestr(f"tblSCTurGrid_2026_{month:02}.csv", "\n".join(power))
            archive.writestr(f"tblSCTurbine_2026_{month:02}.csv", "\n".join([telemetry[0], *reversed(telemetry[1:])]))
    return HillOfTowieFiles(path, metadata)


def test_forecasts_roundtrip_preserves_station_time_units_and_splits(tmp_path):
    converted = HillOfTowieSCADAConnector().convert(_source(tmp_path))
    registry = tmp_path / "registry"
    store_dataset(converted, registry)
    dataset = TimeNet(sources=[registry]).load("res/hill-of-towie-scada@1.0.0")
    assert [len(dataset.get_train()), len(dataset.get_validation()), len(dataset.get_test())] == [4, 2, 2]
    for tasks, months in (
        (dataset.get_train(), [1, 1, 2, 2]),
        (dataset.get_validation(), [3, 3]),
        (dataset.get_test(), [4, 4]),
    ):
        for index, (task, month) in enumerate(zip(tasks, months, strict=True)):
            assert isinstance(task, ForecastingTask)
            context = task.inputs[0]
            assert task.targets is not None
            target = task.targets[0]
            assert isinstance(target, Record)
            turbine = index % 2 + 1
            offset = 100 * (turbine - 1)
            assert context.metadata["turbine_id"] == target.metadata["turbine_id"] == f"T{turbine:02}"
            assert context.metadata["window_start_utc"] == f"2026-{month:02}-01T00:00:00+00:00"
            assert context.start_time.timestamp == unix_us(datetime(2026, month, 1, 0, 10, tzinfo=UTC))
            assert target.start_time.timestamp == unix_us(datetime(2026, month, 1, 6, 10, tzinfo=UTC))

            channels = {signal.name: signal for signal in context.signals}
            assert channels.keys() == {"active_power", "wind_speed", "generator_speed"}
            power = channels["active_power"]
            wind = channels["wind_speed"]
            rpm = channels["generator_speed"]
            (future_power,) = target.signals
            assert power.spec.unit_value == future_power.spec.unit_value == ureg.kilowatt
            assert wind.spec.unit_value == ureg.meter / ureg.second
            assert rpm.spec.unit_value == ureg.rpm
            for signal in (power, wind, rpm, future_power):
                assert isinstance(signal.time_axis, RegularAxis)
                assert signal.time_axis.period_us == 600_000_000
            np.testing.assert_array_equal(power.to_numpy(), month * 100 + offset + np.arange(1, 37))
            np.testing.assert_array_equal(wind.to_numpy(), np.full(36, 8 + offset))
            np.testing.assert_array_equal(rpm.to_numpy(), np.full(36, 900 + offset))
            np.testing.assert_array_equal(future_power.to_numpy(), month * 100 + offset + np.arange(37, 43))


def test_missing_history_excludes_only_affected_windows(tmp_path):
    dataset = HillOfTowieSCADAConnector().convert(_source(tmp_path, january_gap=True))
    assert len(dataset.get_train()) == 3
    assert dataset.get_train()[0].inputs[0].metadata["turbine_id"] == "T02"


def test_duplicate_measurement_fails(tmp_path):
    with pytest.raises(TimeNetConnectorError, match="duplicate measurement"):
        HillOfTowieSCADAConnector().convert(_source(tmp_path, duplicate=True))
