from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.types import AnswerTask, ScalarPredictionTask, Split, read_value
from timenet_connectors.datasets.noaa.spaceweather_cot.connector import (
    SpaceWeatherCotConnector,
    SpaceWeatherRaw,
    _decode_omni_kp,
    _Issuance,
    _Omni2,
    _parse_discussion,
    _parse_issued_at,
    _parse_three_day_forecast,
    _read_goes_xray,
    _read_omni2,
    _split_for,
)


def _discussion_text(issued_at: datetime) -> str:
    """The fixture Forecast Discussion, with its ``:Issued:`` header set to ``issued_at``."""
    stamp = issued_at.strftime("%Y %b %d %H%M")
    return _DISCUSSION_TEMPLATE.replace(":Issued: 2024 Jun 17 0030 UTC", f":Issued: {stamp} UTC")


# A real (trimmed) Forecast Discussion, shaped exactly like NOAA SWPC's archived reports.
_DISCUSSION_TEMPLATE = """:Product: 06170030forecast_discussion.txt
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
    summary, forecast = _parse_discussion(_DISCUSSION_TEMPLATE)
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


def test_decode_omni_kp_matches_nasa_documented_examples():
    # NASA's omni2 format doc: "Kp Planetary Geomagnetic Activity Index (e.g. 3+ = 33, 6- = 57,
    # 4 = 40, etc.)". "6-" means 6 minus a third, i.e. 5.667 — the ones digit 7 rounds the level
    # UP to the next integer minus one third, not down.
    assert _decode_omni_kp(33) == pytest.approx(3 + 1 / 3)
    assert _decode_omni_kp(57) == pytest.approx(5 + 2 / 3)
    assert _decode_omni_kp(40) == pytest.approx(4.0)
    # Code 07 is "1-" = 0.667. Kp can never be negative, which the tens-minus-a-third reading
    # of this code would produce.
    assert _decode_omni_kp(7) == pytest.approx(2 / 3)
    assert _decode_omni_kp(0) == pytest.approx(0.0)


# The first row of the pinned omni2_2024.dat (2024-01-01 00:00 UTC), verbatim. Expected values
# come from NASA's format description, not from this connector's own column positions.
_REAL_OMNI_LINE = (
    "2024   1  0 2596 51 52  60  36   5.3   5.1  -7.3 322.9   4.0  -3.0  -0.6  -2.8  -1.3   0.1"
    "   1.5   0.3   0.4   1.5   31114.   7.4  306.  -2.3   2.8 0.042  1.35    4691.   0.4    2."
    "   0.3   1.3 0.007   0.40   1.75   7.9  7  55     0   20 999999.99 99999.99 99999.99"
    " 99999.99 99999.99 99999.99  0   3 131.2   0.3    -9    11  5.0"
)


def test_read_omni2_parses_real_row_against_nasa_format_doc(tmp_path):
    path = tmp_path / "omni2_2024.dat"
    path.write_text(_REAL_OMNI_LINE + "\n")
    omni = _read_omni2(path)
    row = omni.rows[datetime(2024, 1, 1, 0)]
    assert row["bz_gsm"] == pytest.approx(-1.3)
    assert row["density"] == pytest.approx(7.4)
    assert row["speed"] == pytest.approx(306.0)
    assert row["kp"] == pytest.approx(2 / 3)  # code 7 is Kp "1-" = 0.667 per NASA's doc


def test_read_omni2_nulls_fill_values(tmp_path):
    path = tmp_path / "omni2_2024.dat"
    path.write_text(_omni_line(when=(2024, 169, 13), bz=999.9, speed=9999.0, density=999.9, kp_code=33) + "\n")
    omni = _read_omni2(path)
    row = omni.rows[datetime(2024, 6, 17, 13)]
    assert row["bz_gsm"] is None
    assert row["speed"] is None
    assert row["density"] is None
    assert row["kp"] == pytest.approx(3 + 1 / 3)


def _omni2_with_rows(rows: dict[datetime, dict[str, float | None]]) -> _Omni2:
    return _Omni2(rows=rows)


def _full_row(kp: float) -> dict[str, float | None]:
    return {"bz_gsm": -1.0, "speed": 400.0, "density": 5.0, "kp": kp}


def test_omni_window_excludes_hours_incomplete_at_window_end():
    # OMNI2's row for hour H averages [H, H+1). At a 00:30 issuance the row labelled 00:00 still
    # contains 30 minutes that had not happened yet, so only rows through 23:00 may be used.
    rows = {
        datetime(2024, 6, 16, 23): _full_row(2.0),
        datetime(2024, 6, 17, 0): _full_row(3.0),
    }
    omni = _omni2_with_rows(rows)
    start = datetime(2024, 6, 16, 0, 30, tzinfo=UTC)
    end = datetime(2024, 6, 17, 0, 30, tzinfo=UTC)
    times, fields = omni.window(start, end)
    assert times == [datetime(2024, 6, 16, 23, tzinfo=UTC)]
    assert fields["kp"] == [2.0]


def test_omni_window_includes_hour_closing_exactly_at_window_end():
    rows = {datetime(2024, 6, 16, 23): _full_row(2.0)}
    omni = _omni2_with_rows(rows)
    start = datetime(2024, 6, 16, 0, 30, tzinfo=UTC)
    end = datetime(2024, 6, 17, 0, 0, tzinfo=UTC)  # the 23:00 row closes exactly here
    times, _ = omni.window(start, end)
    assert times == [datetime(2024, 6, 16, 23, tzinfo=UTC)]


def test_omni_window_nulls_kp_whose_three_hour_interval_is_unfinished():
    # Kp is a 3-hour index repeated in each hourly row. For an issuance at 20:20 the rows of
    # hours 18 and 19 are complete hours, but their Kp interval (18:00-21:00) had not ended, so
    # carrying it would leak up to 40 minutes of a quantity tied to the kp_max_3day target.
    rows = {
        datetime(2024, 6, 17, 17): _full_row(4.0),  # interval 15-18 ended at 18:00, fully past
        datetime(2024, 6, 17, 18): _full_row(4.0),  # interval 18-21 ends at 21:00, after 20:20
        datetime(2024, 6, 17, 19): _full_row(4.0),
    }
    omni = _omni2_with_rows(rows)
    start = datetime(2024, 6, 16, 20, 20, tzinfo=UTC)
    end = datetime(2024, 6, 17, 20, 20, tzinfo=UTC)
    times, fields = omni.window(start, end)
    assert times == [
        datetime(2024, 6, 17, 17, tzinfo=UTC),
        datetime(2024, 6, 17, 18, tzinfo=UTC),
        datetime(2024, 6, 17, 19, tzinfo=UTC),
    ]
    assert fields["kp"] == [4.0, None, None]
    assert fields["bz_gsm"] == [-1.0, -1.0, -1.0]  # hourly fields are unaffected


def test_omni_window_keeps_kp_for_interval_closing_exactly_at_window_end():
    rows = {datetime(2024, 6, 17, 18): _full_row(4.0)}  # interval 18-21 ends at 21:00
    omni = _omni2_with_rows(rows)
    start = datetime(2024, 6, 17, 0, 0, tzinfo=UTC)
    end = datetime(2024, 6, 17, 21, 0, tzinfo=UTC)
    _, fields = omni.window(start, end)
    assert fields["kp"] == [4.0]


def test_parse_issued_at_reads_the_report_header():
    text = ":Product: 06170030forecast_discussion.txt\n:Issued: 2024 Jun 17 0030 UTC\n# comment\n"
    assert _parse_issued_at(text) == datetime(2024, 6, 17, 0, 30, tzinfo=UTC)


def test_parse_issued_at_reads_late_issuance_times():
    # Real examples from the archive: the 202404021230 report was issued at 20:20 and the
    # 202405010030 report at 10:55. The file name does not carry the true issuance time.
    text = ":Product: 04021230forecast_discussion.txt\n:Issued: 2024 Apr 02 2020 UTC\n"
    assert _parse_issued_at(text) == datetime(2024, 4, 2, 20, 20, tzinfo=UTC)


def test_parse_issued_at_rejects_a_missing_header():
    with pytest.raises(TimeNetConnectorError, match="Issued"):
        _parse_issued_at(":Product: 06170030forecast_discussion.txt\nno header here\n")


def _build_raw(tmp_path: Path, issued_at: datetime) -> SpaceWeatherRaw:
    """Build a minimal but shape-correct :class:`SpaceWeatherRaw` for one issuance."""
    discussion_path = tmp_path / "discussion.txt"
    discussion_path.write_text(_discussion_text(issued_at))
    forecast_path = tmp_path / "forecast.txt"
    forecast_path.write_text(_FORECAST_TEXT)

    xray_path = tmp_path / "goes.nc"
    window_start = issued_at - timedelta(hours=24)
    _write_goes_fixture(xray_path, start=window_start, minutes=[0, 60, 120], flags=[0, 0, 0])

    omni_path = tmp_path / "omni2_2024.dat"
    # Rows at hours comfortably inside the lookback window: an hour is only usable once it has
    # closed completely before the issuance, so rows near the window edges are excluded.
    lines = [
        _omni_line(
            when=(window_start.year, window_start.timetuple().tm_yday, (window_start.hour + 3) % 24),
            bz=-5.0,
            speed=400.0,
            density=6.0,
            kp_code=33,
        ),
        _omni_line(
            when=(window_start.year, window_start.timetuple().tm_yday, (window_start.hour + 4) % 24),
            bz=-3.0,
            speed=420.0,
            density=5.0,
            kp_code=40,
        ),
    ]
    omni_path.write_text("\n".join(lines) + "\n")

    issuance = _Issuance(issued_at=issued_at, discussion_path=discussion_path, forecast_path=forecast_path)
    return SpaceWeatherRaw(issuances=(issuance,), xray_paths=(xray_path,), omni_paths=(omni_path,))


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
    _write_goes_fixture(raw.xray_paths[0], start=datetime(2020, 1, 1), minutes=[0], flags=[0])
    with pytest.raises(TimeNetConnectorError, match="no GOES or OMNI2 samples"):
        SpaceWeatherCotConnector().convert(raw)


def test_convert_anchors_window_to_issued_header_not_file_name(tmp_path):
    # The file is archived under the nominal 0030 slot, but the report header says it was really
    # issued at 10:55 (a real archive case: 202405010030). The lookback window must anchor to the
    # true issuance time, because that is what the forecaster actually saw.
    nominal = datetime(2024, 6, 17, 0, 30, tzinfo=UTC)
    real = datetime(2024, 6, 17, 10, 55, tzinfo=UTC)
    raw = _build_raw(tmp_path, nominal)
    # Rewrite the discussion with the late :Issued: header, and place one OMNI row at hour 9:
    # inside the true window ([Jun 16 10:55, Jun 17 10:55]) but after the nominal one ends.
    raw.issuances[0].discussion_path.write_text(_discussion_text(real))
    omni_lines = [
        _omni_line(when=(2024, 169, 9), bz=-7.0, speed=500.0, density=8.0, kp_code=33),
    ]
    raw.omni_paths[0].write_text("\n".join(omni_lines) + "\n")
    # GOES fixture must cover the true window too; rebuild it around the real issuance.
    _write_goes_fixture(raw.xray_paths[0], start=real - timedelta(hours=24), minutes=[0, 60, 120], flags=[0, 0, 0])

    dataset = SpaceWeatherCotConnector().convert(raw)
    (record,) = dataset.records
    annotations = {annotation.key: annotation.value for annotation in record.annotations}
    assert annotations["issued_at"] == real.isoformat()
    bz = next(signal for signal in record.signals if signal.name == "bz_gsm")
    values = bz.to_arrow().to_pylist()
    assert -7.0 in values  # the hour-9 row is present only if the true issuance time anchored the window


# --- Time-stratified split ------------------------------------------------------------------


def test_split_boundaries():
    # Train ends before 2024-12-01; the 3-day gap keeps windows and forecast periods apart.
    assert _split_for(datetime(2024, 11, 30, 23, 59, tzinfo=UTC)) is Split.TRAIN
    assert _split_for(datetime(2024, 12, 1, 0, 30, tzinfo=UTC)) is None  # inside the train/validation gap
    assert _split_for(datetime(2024, 12, 3, 23, 59, tzinfo=UTC)) is None
    assert _split_for(datetime(2024, 12, 4, 0, 30, tzinfo=UTC)) is Split.VALIDATION
    assert _split_for(datetime(2024, 12, 31, 23, 59, tzinfo=UTC)) is Split.VALIDATION
    assert _split_for(datetime(2025, 1, 1, 0, 30, tzinfo=UTC)) is None  # inside the validation/test gap
    assert _split_for(datetime(2025, 1, 4, 0, 30, tzinfo=UTC)) is Split.TEST
    assert _split_for(datetime(2025, 4, 6, 12, 30, tzinfo=UTC)) is Split.TEST


def test_convert_excludes_issuances_in_split_gaps(tmp_path):
    # 2024-12-02 is inside the train/validation gap, so no record is built for it at all.
    issued_at = datetime(2024, 12, 2, 0, 30, tzinfo=UTC)
    dataset = SpaceWeatherCotConnector().convert(_build_raw(tmp_path, issued_at))
    assert len(dataset.records) == 0
    assert len(dataset.tasks) == 0


def test_convert_assigns_the_time_stratified_split_to_every_task(tmp_path):
    issued_at = datetime(2025, 2, 17, 12, 30, tzinfo=UTC)  # test split
    dataset = SpaceWeatherCotConnector().convert(_build_raw(tmp_path, issued_at))
    assert len(dataset.tasks) > 0
    assert {task.split for task in dataset.tasks} == {Split.TEST}


# --- Observed outcomes -----------------------------------------------------------------------


def test_greatest_kp_between_returns_max_over_fully_covered_window():
    rows = {datetime(2024, 6, 17, h): _full_row(2.0) for h in range(20, 24)}
    rows[datetime(2024, 6, 18, 0)] = _full_row(2.0)  # the window rounds up to hour 21 and ends after hour 0
    rows[datetime(2024, 6, 17, 22)] = _full_row(6.5)
    omni = _omni2_with_rows(rows)
    start = datetime(2024, 6, 17, 20, 30, tzinfo=UTC)
    end = datetime(2024, 6, 18, 0, 30, tzinfo=UTC)
    assert omni.greatest_kp_between(start, end) == pytest.approx(6.5)


def test_greatest_kp_between_returns_none_when_any_hour_is_missing():
    rows = {datetime(2024, 6, 17, h): _full_row(2.0) for h in (21, 23)}  # hour 22 absent
    omni = _omni2_with_rows(rows)
    start = datetime(2024, 6, 17, 20, 30, tzinfo=UTC)
    end = datetime(2024, 6, 18, 0, 30, tzinfo=UTC)
    assert omni.greatest_kp_between(start, end) is None  # a truncated maximum is never reported


def test_greatest_flux_between_returns_none_when_coverage_is_partial(tmp_path):
    start = datetime(2024, 6, 17, 12, 0)
    path = tmp_path / "goes.nc"
    _write_goes_fixture(path, start=start, minutes=[0, 1, 2], flags=[0, 0, 0])
    xray = _read_goes_xray(path)
    # The series covers 3 minutes; a 3-day outcome window is mostly uncovered.
    assert xray.greatest_flux_between(start.replace(tzinfo=UTC), start.replace(tzinfo=UTC) + timedelta(days=3)) is None
    # A fully covered 2-minute window reports the greatest unflagged flux.
    assert xray.greatest_flux_between(
        start.replace(tzinfo=UTC), start.replace(tzinfo=UTC) + timedelta(minutes=2)
    ) == pytest.approx(2e-7)


def test_convert_adds_observed_outcome_tasks_when_coverage_is_complete(tmp_path):
    issued_at = datetime(2024, 6, 17, 12, 30, tzinfo=UTC)
    raw = _build_raw(tmp_path, issued_at)
    # Cover the lookback and the 72-hour outcome window fully: one OMNI row per hour from the
    # lookback through the outcome end, and one GOES sample per minute over the same span.
    first_hour = (issued_at - timedelta(hours=24)).replace(tzinfo=None)
    hours = 24 + 72 + 2
    omni_lines = []
    for h in range(hours):
        moment = first_hour + timedelta(hours=h)
        kp_code = 70 if h == 60 else 20  # one hour of Kp 7.0 among Kp 2.0, inside the outcome window
        omni_lines.append(
            _omni_line(
                when=(moment.year, moment.timetuple().tm_yday, moment.hour),
                bz=-5.0,
                speed=400.0,
                density=6.0,
                kp_code=kp_code,
            )
        )
    raw.omni_paths[0].write_text("\n".join(omni_lines) + "\n")
    minutes = 4 * 24 * 60  # the 24-hour lookback plus the 72-hour outcome window
    _write_goes_fixture(
        raw.xray_paths[0],
        start=issued_at - timedelta(hours=24),
        minutes=list(range(minutes)),
        flags=[0] * minutes,
    )
    dataset = SpaceWeatherCotConnector().convert(raw)
    scalars = {
        task.target_name: task.targets[0]
        for task in dataset.tasks
        if isinstance(task, ScalarPredictionTask) and task.targets is not None
    }
    assert scalars["greatest_observed_kp_next_3_days"] == pytest.approx(7.0)
    assert "greatest_observed_xray_flux_next_3_days" in scalars
    # The fixture assigns flux (index + 1) * 1e-7; the outcome window ends at the last minute.
    assert scalars["greatest_observed_xray_flux_next_3_days"] == pytest.approx(5760e-7, rel=1e-3)


def test_convert_omits_xray_outcome_when_goes_coverage_is_truncated(tmp_path):
    # Default fixture: the GOES file holds only 3 minutes of the lookback, so the 72-hour outcome
    # window is mostly uncovered and no X-ray outcome task may be built.
    issued_at = datetime(2024, 6, 17, 12, 30, tzinfo=UTC)
    dataset = SpaceWeatherCotConnector().convert(_build_raw(tmp_path, issued_at))
    names = {task.target_name for task in dataset.tasks if isinstance(task, ScalarPredictionTask)}
    assert "greatest_observed_xray_flux_next_3_days" not in names
    assert "greatest_observed_kp_next_3_days" not in names  # only 2 of 72 rows exist in the fixture
