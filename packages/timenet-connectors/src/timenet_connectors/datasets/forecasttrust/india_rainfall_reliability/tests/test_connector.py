from datetime import date, timedelta
from pathlib import Path

import pytest

from timenet.dataset import Record, TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.types import ClassificationTask, ScalarPredictionTask
from timenet.types.splits import Split
from timenet.types.text import read_value
from timenet_connectors.datasets.forecasttrust.india_rainfall_reliability import (
    IndiaRainfallReliabilityConnector,
    connector as module,
)
from timenet_connectors.datasets.forecasttrust.india_rainfall_reliability.connector import CityFiles


_FIRST_DAY = date(2025, 5, 20)
_N_DAYS = 30
_LOCATION = "latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation\n19.0,73.0,9.0,19800,Asia/Kolkata,GMT+5:30\n\n"


def _forecast(day_index: int) -> tuple[float, float, float]:
    # Hand-written, deterministic values; the total of a day is 24 times its hourly value.
    return (day_index * 0.5, day_index * 0.5 + 1.0, day_index * 0.5 + 2.0)


def _observed(day_index: int) -> float:
    return 100.0 + day_index


def _write(tmp_path: Path, *, drop_hour_of_day: int | None = None, drop_obs_day: int | None = None) -> CityFiles:
    forecast_lines = [
        "time,precipitation_previous_day1 (mm),precipitation_previous_day2 (mm),precipitation_previous_day3 (mm)"
    ]
    for index in range(_N_DAYS):
        day = _FIRST_DAY + timedelta(days=index)
        for hour in range(24):
            if index == drop_hour_of_day and hour == 5:
                forecast_lines.append(f"{day.isoformat()}T{hour:02d}:00,NaN,NaN,NaN")
                continue
            values = ",".join(f"{total / 24:.6f}" for total in _forecast(index))
            forecast_lines.append(f"{day.isoformat()}T{hour:02d}:00,{values}")
    observed_lines = ["time,precipitation_sum (mm)"]
    for index in range(_N_DAYS):
        value = "NaN" if index == drop_obs_day else f"{_observed(index):.2f}"
        observed_lines.append(f"{(_FIRST_DAY + timedelta(days=index)).isoformat()},{value}")
    forecast = tmp_path / "forecast.csv"
    observed = tmp_path / "observed.csv"
    forecast.write_text(_LOCATION + "\n".join(forecast_lines) + "\n")
    observed.write_text(_LOCATION + "\n".join(observed_lines) + "\n")
    return CityFiles("Testville", 19.0, 73.0, forecast, observed)


def _convert(tmp_path: Path, **kwargs: int | None) -> TimeFDataset:
    return IndiaRainfallReliabilityConnector().convert([_write(tmp_path, **kwargs)])


def _valid_day(record: Record) -> date:
    value = record.metadata["valid_date"]
    assert isinstance(value, str)
    return date.fromisoformat(value)


def _signals(record: Record) -> dict[str, list[float]]:
    return {signal.name: signal.to_numpy().tolist() for source in record.sources for signal in source.signals}


def test_metadata():
    metadata = IndiaRainfallReliabilityConnector().metadata()
    assert metadata.dataset_id == "forecasttrust/india-rainfall-reliability"
    assert str(metadata.license) == "CC-BY-4.0"


def test_every_city_has_pinned_checksums():
    for city in module.CITIES:
        for name in module._file_names(city):
            assert len(module._SHA256[name]) == 64


def test_one_record_per_complete_window(tmp_path):
    dataset = _convert(tmp_path)
    assert len(dataset.records) == _N_DAYS - module.WINDOW_DAYS + 1


def test_observations_stop_two_days_before_the_valid_day(tmp_path):
    dataset = _convert(tmp_path)
    for record in dataset.records:
        valid = _valid_day(record)
        index = (valid - _FIRST_DAY).days
        signals = _signals(record)
        assert len(signals["observed_verified"]) == module.WINDOW_DAYS - module.OBS_LAG_DAYS
        # The last verified observation is day D-2; the target day and D-1 never appear.
        assert signals["observed_verified"][-1] == pytest.approx(_observed(index - 2))
        assert _observed(index) not in signals["observed_verified"]
        assert _observed(index - 1) not in signals["observed_verified"]
        assert signals["forecast_day1"][-1] == pytest.approx(_forecast(index)[0], rel=1e-5)
        assert len(signals["forecast_day1"]) == module.WINDOW_DAYS


def test_labels_and_scalar_target(tmp_path):
    dataset = _convert(tmp_path)
    classification = [t for t in dataset.iter_tasks() if isinstance(t, ClassificationTask)]
    scalar = {t.inputs[0].id: t for t in dataset.iter_tasks() if isinstance(t, ScalarPredictionTask)}
    assert len(classification) == len(scalar) == len(dataset.records)
    for task in classification:
        record = task.inputs[0]
        index = (_valid_day(record) - _FIRST_DAY).days
        issued = _forecast(index)[0]
        error = abs(_observed(index) - issued)
        expected = "failure" if error > max(5.0, 0.5 * issued) else "reliable"
        assert task.targets is not None
        assert read_value(task.targets[0]) == expected
        targets = scalar[record.id].targets
        assert targets is not None
        target = targets[0]
        assert isinstance(target, float)
        assert target == pytest.approx(error)


def test_splits_are_chronological(tmp_path):
    dataset = _convert(tmp_path)
    for task in dataset.iter_tasks():
        valid = _valid_day(task.inputs[0])
        expected = Split.VALIDATION if valid <= module.VALIDATION_END else Split.TEST
        assert task.split == expected


def test_a_missing_forecast_hour_removes_only_the_windows_that_need_that_day(tmp_path):
    complete_dir = tmp_path / "complete"
    complete_dir.mkdir()
    full = len(_convert(complete_dir).records)
    gapped_dir = tmp_path / "gapped"
    gapped_dir.mkdir()
    gapped = _convert(gapped_dir, drop_hour_of_day=20)
    assert len(gapped.records) < full
    for record in gapped.records:
        valid = _valid_day(record)
        assert not (valid - timedelta(days=module.WINDOW_DAYS - 1) <= _FIRST_DAY + timedelta(days=20) <= valid)


def test_a_missing_observation_is_never_filled(tmp_path):
    dataset = _convert(tmp_path, drop_obs_day=15)
    for record in dataset.records:
        assert all(value == value for value in _signals(record)["observed_verified"])


def test_missing_column_raises(tmp_path):
    files = _write(tmp_path)
    files.observed.write_text(_LOCATION + "time,something_else (mm)\n2025-05-20,1.0\n")
    with pytest.raises(TimeNetConnectorError):
        IndiaRainfallReliabilityConnector().convert([files])
