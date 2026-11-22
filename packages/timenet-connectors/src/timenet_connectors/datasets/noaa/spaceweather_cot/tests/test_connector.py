from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.types import AnswerTask, ScalarPredictionTask, read_value
from timenet_connectors.datasets.noaa.spaceweather_cot.connector import (
    SpaceWeatherCotConnector,
    SpaceWeatherRaw,
    _Issuance,
    _parse_discussion,
    _parse_three_day_forecast,
    _read_goes_xray,
    _read_omni2,
)


# A real (trimmed) Forecast Discussion, shaped exactly like NOAA SWPC's archived reports.
_DISCUSSION_TEXT = """:Product: 06170030forecast_discussion.txt
:Issued: 2024 Jun 17 0030 UTC
# Prepared by the U.S. Dept. of Commerce, NOAA, Space Weather Prediction Center
#
Solar Activity

.24 hr Summary...
Only C-class events during the past 24 hours. Region 3712 was responsible
for most, including the largest of the period, a C6.7 at 1546 UTC.

.Forecast...
Solar activity is likely to reach moderate levels (R1-R2/Minor-Moderate)
during 17-19 Jun.

Energetic Particle

.24 hr Summary...
The greater than 2 MeV electron flux was at normal to moderate levels.

.Forecast...
The greater than 2 MeV electron flux is expected to remain at normal to
moderate levels through 18 Jun.

Solar Wind

.24 hr Summary...
The solar wind environment was disturbed after 16/1115 UTC.

.Forecast...
Positive polarity CH, fast wind stream influences are expected on 17-18 Jun.

Geospace

.24 hr Summary...
The geomagnetic field ranged from quiet to active levels.

.Forecast...
The geomagnetic field is expected to be at quiet to unsettled levels on 17 Jun.
"""

# A real (trimmed) 3-Day Forecast, shaped exactly like NOAA SWPC's archived reports.
_FORECAST_TEXT = """:Product: 06170030three_day_forecast.txt
:Issued: 2024 Jun 17 0030 UTC
A. NOAA Geomagnetic Activity Observation and Forecast

The greatest observed 3 hr Kp over the past 24 hours was 4 (below NOAA
Scale levels).
The greatest expected 3 hr Kp for Jun 17-Jun 19 2024 is 4.00 (below NOAA
Scale levels).

B. NOAA Solar Radiation Activity Observation and Forecast

Solar Radiation Storm Forecast for Jun 17-Jun 19 2024

              Jun 17  Jun 18  Jun 19
S1 or greater   10%     10%     10%

C. NOAA Radio Blackout Activity and Forecast

Radio Blackout Forecast for Jun 17-Jun 19 2024

              Jun 17        Jun 18        Jun 19
R1-R2           55%           55%           55%
R3 or greater   10%           10%           10%
"""


def test_parse_discussion_splits_summary_and_forecast():
    summary, forecast = _parse_discussion(_DISCUSSION_TEXT)
    assert "Solar Activity: Only C-class events" in summary
    assert "Geospace: The geomagnetic field ranged from quiet to active" in summary
    assert "Solar Activity: Solar activity is likely to reach moderate levels" in forecast
    assert "Geospace: The geomagnetic field is expected to be at quiet to unsettled" in forecast


def test_parse_discussion_rejects_wrong_sections():
    with pytest.raises(TimeNetConnectorError, match="section headers"):
        _parse_discussion("Solar Activity\n.24 hr Summary...\nx\n.Forecast...\ny\n")


def test_parse_three_day_forecast_extracts_numeric_fields():
    numeric = _parse_three_day_forecast(_FORECAST_TEXT)
    assert numeric == {"kp_max_3day": 4.0, "r1_r2_pct_day1": 55.0, "s1_pct_day1": 10.0}


def test_parse_three_day_forecast_rejects_missing_fields():
    with pytest.raises(TimeNetConnectorError, match="numeric fields"):
        _parse_three_day_forecast("no numbers here")


def _write_goes_fixture(path: Path, *, start: datetime, minutes: list[int], flags: list[int]) -> None:
    """Write a tiny ``xrsf-l2-avg1m_science``-shaped NetCDF file with one sample per listed minute."""
    netcdf4 = pytest.importorskip("netCDF4")
    epoch = datetime(2000, 1, 1, 12, tzinfo=UTC)
    seconds = [(start.replace(tzinfo=UTC) + timedelta(minutes=m) - epoch).total_seconds() for m in minutes]
    with netcdf4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", len(seconds))
        time_var = dataset.createVariable("time", "f8", ("time",))
        time_var.units = "seconds since 2000-01-01 12:00:00 UTC"
        time_var[:] = seconds
        flux_var = dataset.createVariable("xrsb_flux", "f4", ("time",))
        flux_var[:] = [1e-7 * (i + 1) for i in range(len(seconds))]
        flag_var = dataset.createVariable("xrsb_flag", "u1", ("time",))
        flag_var[:] = flags


def test_read_goes_xray_nulls_flagged_samples(tmp_path):
    start = datetime(2024, 6, 17, 0, 0)
    path = tmp_path / "goes.nc"
    _write_goes_fixture(path, start=start, minutes=[0, 1, 2], flags=[0, 1, 0])
    xray = _read_goes_xray(path)
    times, values = xray.window(start.replace(tzinfo=UTC), start.replace(tzinfo=UTC) + timedelta(minutes=3))
    assert len(times) == 3
    assert values[0] == pytest.approx(1e-7)
    assert values[1] is None  # flagged sample is nulled, not passed through
    assert values[2] == pytest.approx(3e-7)


def _omni_line(*, when: tuple[int, int, int], bz: float, speed: float, density: float, kp_code: int) -> str:
    """Build one syntactically valid OMNI2 line, with fill values everywhere except the fields tested.

    Args:
        when: ``(year, day_of_year, hour)``.
        bz: IMF Bz (GSM), nT.
        speed: Plasma flow speed, km/s.
        density: Proton density, cm^-3.
        kp_code: The OMNI2 Kp code (tens digit the level, ones digit the third).
    """
    year, day_of_year, hour = when
    fields = ["9"] * 55
    fields[0], fields[1], fields[2] = str(year), str(day_of_year), str(hour)
    fields[16] = str(bz)  # Bz (GSM)
    fields[23] = str(density)  # proton density
    fields[24] = str(speed)  # plasma speed
    fields[38] = str(kp_code)  # Kp code
    return " ".join(fields)


def test_read_omni2_decodes_kp_and_nulls_fill_values(tmp_path):
    path = tmp_path / "omni2_2024.dat"
    lines = [
        _omni_line(when=(2024, 169, 12), bz=-5.0, speed=400.0, density=6.0, kp_code=33),
        _omni_line(when=(2024, 169, 13), bz=999.9, speed=9999.0, density=999.9, kp_code=57),
    ]
    path.write_text("\n".join(lines) + "\n")
    omni = _read_omni2(path)
    first = omni.rows[datetime(2024, 6, 17, 12)]
    assert first == {"bz_gsm": -5.0, "speed": 400.0, "density": 6.0, "kp": pytest.approx(3 + 1 / 3)}
    second = omni.rows[datetime(2024, 6, 17, 13)]
    assert second["bz_gsm"] is None
    assert second["speed"] is None
    assert second["density"] is None
    assert second["kp"] == pytest.approx(5 - 1 / 3)  # code 57 -> tens=5, ones=7 ("-")


def _build_raw(tmp_path: Path, issued_at: datetime) -> SpaceWeatherRaw:
    """Build a minimal but shape-correct :class:`SpaceWeatherRaw` for one issuance."""
    discussion_path = tmp_path / "discussion.txt"
    discussion_path.write_text(_DISCUSSION_TEXT)
    forecast_path = tmp_path / "forecast.txt"
    forecast_path.write_text(_FORECAST_TEXT)

    xray_path = tmp_path / "goes.nc"
    window_start = issued_at - timedelta(hours=24)
    _write_goes_fixture(xray_path, start=window_start, minutes=[0, 60, 120], flags=[0, 0, 0])

    omni_path = tmp_path / "omni2_2024.dat"
    lines = [
        _omni_line(
            when=(window_start.year, window_start.timetuple().tm_yday, window_start.hour),
            bz=-5.0,
            speed=400.0,
            density=6.0,
            kp_code=33,
        ),
        _omni_line(
            when=(issued_at.year, issued_at.timetuple().tm_yday, issued_at.hour),
            bz=-3.0,
            speed=420.0,
            density=5.0,
            kp_code=40,
        ),
    ]
    omni_path.write_text("\n".join(lines) + "\n")

    issuance = _Issuance(issued_at=issued_at, discussion_path=discussion_path, forecast_path=forecast_path)
    return SpaceWeatherRaw(issuances=(issuance,), xray_path=xray_path, omni_path=omni_path)


def test_metadata():
    assert SpaceWeatherCotConnector().metadata().dataset_id == "noaa/spaceweather-cot"
    assert str(SpaceWeatherCotConnector().metadata().license) == "CC0-1.0"


def test_convert_builds_one_record_with_five_signals(tmp_path):
    issued_at = datetime(2024, 6, 17, 12, 30, tzinfo=UTC)
    dataset = SpaceWeatherCotConnector().convert(_build_raw(tmp_path, issued_at))
    assert isinstance(dataset, TimeFDataset)
    assert len(dataset.records) == 1
    (record,) = dataset.records
    assert {signal.name for signal in record.signals} == {"xrsb_flux", "bz_gsm", "speed", "density", "kp"}
    assert {annotation.key for annotation in record.annotations} == {"issued_at", "satellite"}


def test_convert_builds_answer_task_from_forecaster_text(tmp_path):
    issued_at = datetime(2024, 6, 17, 12, 30, tzinfo=UTC)
    dataset = SpaceWeatherCotConnector().convert(_build_raw(tmp_path, issued_at))
    (answer,) = [task for task in dataset.tasks if isinstance(task, AnswerTask)]
    assert answer.rationale is not None
    assert answer.targets is not None
    target_text = read_value(answer.targets[0])
    assert isinstance(target_text, str)
    assert "Only C-class events" in read_value(answer.rationale)
    assert "Solar activity is likely to reach moderate levels" in target_text


def test_convert_builds_scalar_prediction_tasks_from_noaa_numbers(tmp_path):
    issued_at = datetime(2024, 6, 17, 12, 30, tzinfo=UTC)
    dataset = SpaceWeatherCotConnector().convert(_build_raw(tmp_path, issued_at))
    scalars = {
        task.target_name: task.targets[0]
        for task in dataset.tasks
        if isinstance(task, ScalarPredictionTask) and task.targets is not None
    }
    assert scalars == {
        "kp_max_3day": pytest.approx(4.0),
        "r1_r2_radio_blackout_pct": pytest.approx(55.0),
        "s1_radiation_storm_pct": pytest.approx(10.0),
    }


def test_convert_raises_when_window_has_no_samples(tmp_path):
    issued_at = datetime(2024, 6, 17, 12, 30, tzinfo=UTC)
    raw = _build_raw(tmp_path, issued_at)
    # Rebuild the GOES file with samples far outside the issuance's lookback window.
    _write_goes_fixture(raw.xray_path, start=datetime(2020, 1, 1), minutes=[0], flags=[0])
    with pytest.raises(TimeNetConnectorError, match="no GOES or OMNI2 samples"):
        SpaceWeatherCotConnector().convert(raw)
