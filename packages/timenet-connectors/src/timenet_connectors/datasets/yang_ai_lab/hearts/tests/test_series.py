from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.yang_ai_lab.hearts.series import (
    _audio_rate_hz,
    axis_for,
    series_for,
    time_offsets_us,
)


# This module is given payloads as values. The paths below never exist: a spec and an axis are read
# off the payload, and only the lazy loader would open the file.

_PATH = Path("/nowhere/0.pkl")
_RECORD = "hearts-cgmacros-iauc_calculation-00"
_MINUTES = [0.0, 1.0, 2.0, 9.0]
_GLUCOSE = np.array([88.0, 91.5, 96.25, 101.0])


def test_a_wall_clock_string_column_counts_from_its_own_first_row():
    column = pd.Series(["2022-01-03 10:44:00", "2022-01-03 10:45:00", "2022-01-03 10:52:00"])
    assert list(time_offsets_us(column)) == [0, 60_000_000, 480_000_000]


def test_a_datetime_and_a_timedelta_column_both_count_in_microseconds():
    stamps = pd.Series(pd.to_datetime(["2022-01-03 10:44:00.000", "2022-01-03 10:44:00.010"]))
    assert list(time_offsets_us(stamps)) == [0, 10_000]
    elapsed = pd.Series(pd.to_timedelta([0, 10_000, 20_000], unit="us"))
    assert list(time_offsets_us(elapsed)) == [0, 10_000, 20_000]


def test_a_float_minute_column_becomes_whole_microseconds():
    assert list(time_offsets_us(pd.Series(_MINUTES))) == [0, 60_000_000, 120_000_000, 540_000_000]


def test_a_time_column_this_connector_cannot_read_fails_by_name():
    # An integer column could be minutes, seconds or a row number, and the release says which
    # nowhere, so it is refused rather than guessed.
    with pytest.raises(TimeFFormatError, match="does not read as time"):
        time_offsets_us(pd.Series([0, 1, 2], dtype="int64"))


def test_a_constant_step_is_a_cadence_and_anything_else_keeps_its_offsets():
    assert axis_for(np.array([0, 10_000, 20_000])) == RegularAxis(period_us=Fraction(10_000))
    wavering = np.array([0, 60_000_000, 120_000_000, 540_000_000])
    assert axis_for(wavering) == IrregularAxis(first_us=0, last_us=540_000_000)


def test_a_single_row_frame_keeps_its_offsets_because_no_step_places_it():
    assert axis_for(np.array([0])) == IrregularAxis(first_us=0, last_us=0)


def test_an_audio_rate_comes_from_the_payload_and_vctk_from_the_reference_implementation():
    assert _audio_rate_hz("vctk", {}, ("waveform",)) == 16_000
    assert _audio_rate_hz("coughvid", {"sr": 48_000}, ("audio",)) == 48_000
    assert _audio_rate_hz("coswara", {"data": {"sr": 44_100}}, ("data", "signal")) == 44_100


@pytest.mark.parametrize("rate", [0, -1, 48_000.0, True, None])
def test_an_audio_rate_that_is_not_a_positive_integer_fails_by_name(rate):
    with pytest.raises(TimeFFormatError, match="not a positive integer"):
        _audio_rate_hz("coughvid", {"sr": rate}, ("audio",))


def test_a_frame_becomes_one_series_per_value_column():
    frame = pd.DataFrame({"Time (min)": _MINUTES, "CGM (mg/dL)": _GLUCOSE})
    series = series_for("cgmacros", _PATH, {"cgm_df": frame, "GT": 1.0}, _RECORD)
    assert [item.signal for item in series] == ["cgm_df.CGM (mg/dL)"]
    only = series[0]
    assert only.spec.spec_type == "cgm"
    assert only.n_values == 4
    assert only.time_series_id == f"{_RECORD}-cgm_df.CGM (mg/dL)"
    assert only.time_axis == IrregularAxis(first_us=0, last_us=540_000_000)


def test_the_series_of_one_case_are_sorted_by_signal_name():
    values = np.linspace(0.2, 0.6, 3)
    payload = {
        "hr_dfs": {"hr_1": pd.DataFrame({"timestamp": pd.to_timedelta([0, 1, 2], unit="s"), "hr": values})},
        "respiration_dfs": {
            "respiration_B": pd.DataFrame({"timestamp": pd.to_timedelta([0, 10, 20], unit="ms"), "rsp": values}),
            "respiration_A": pd.DataFrame({"timestamp": pd.to_timedelta([0, 10, 20], unit="ms"), "rsp": values}),
        },
        "GT": {"A": "1", "B": "2"},
    }
    series = series_for("harespod", _PATH, payload, _RECORD)
    assert [item.signal for item in series] == [
        "hr_dfs.hr_1.hr",
        "respiration_dfs.respiration_A.rsp",
        "respiration_dfs.respiration_B.rsp",
    ]


def test_the_index_column_is_read_as_the_axis_and_not_as_a_series():
    frame = pd.DataFrame(
        {
            "Timestamp": ["2022-01-03 10:44:00", "2022-01-03 10:45:00"],
            "Libre GL": [87.0, 88.0],
            "timestamp_min": np.array([0, 1], dtype="int64"),
        }
    )
    series = series_for("cgmacros", _PATH, {"window_df": frame, "GT": 1}, _RECORD)
    assert [item.signal for item in series] == ["window_df.Libre GL"]


def test_an_index_column_that_counts_from_somewhere_else_fails_by_name():
    # timestamp_min is read as part of the axis, so it is stored nowhere. If it counted absolute
    # minutes instead of minutes into the window, dropping it would lose what it says.
    frame = pd.DataFrame(
        {
            "Timestamp": ["2022-01-03 10:44:00", "2022-01-03 10:45:00"],
            "Libre GL": [87.0, 88.0],
            "timestamp_min": np.array([28_319_779, 28_319_780], dtype="int64"),
        }
    )
    with pytest.raises(TimeFFormatError, match="in whole minutes"):
        series_for("cgmacros", _PATH, {"window_df": frame, "GT": 1}, _RECORD)


def test_the_walk_never_descends_into_the_answer():
    frame = pd.DataFrame({"Time (min)": _MINUTES, "CGM (mg/dL)": _GLUCOSE})
    payload = {"cgm_df": frame, "GT": {"held_out": frame}}
    assert [item.signal for item in series_for("cgmacros", _PATH, payload, _RECORD)] == ["cgm_df.CGM (mg/dL)"]


def test_an_unknown_value_column_fails_by_name():
    frame = pd.DataFrame({"Time (min)": _MINUTES, "Insulin": _GLUCOSE})
    with pytest.raises(TimeFFormatError, match="Insulin"):
        series_for("cgmacros", _PATH, {"cgm_df": frame, "GT": 1.0}, _RECORD)


def test_a_frame_with_no_time_column_fails_by_name():
    frame = pd.DataFrame({"Libre GL": _GLUCOSE})
    with pytest.raises(TimeFFormatError, match="has no time column"):
        series_for("cgmacros", _PATH, {"window_df": frame, "GT": 1.0}, _RECORD)


def test_a_frame_with_no_rows_fails_by_name():
    # A zero-row frame has no first offset to count from. No released frame is empty, so this path
    # guards a repin, and it has to fail by name rather than through an IndexError.
    frame = pd.DataFrame({"Time (min)": pd.Series(dtype="float64"), "CGM (mg/dL)": pd.Series(dtype="float64")})
    with pytest.raises(TimeFFormatError, match="holds no rows"):
        series_for("cgmacros", _PATH, {"cgm_df": frame, "GT": 1.0}, _RECORD)


@pytest.mark.parametrize(
    ("array", "shape_text"),
    [
        # An integer PCM buffer and a two-signal one. Neither is corrupt, and neither is a shape
        # this connector knows how to place, so both have to stop the build instead of vanishing.
        (np.linspace(-3000, 3000, 320).astype(np.int16), "dtype int16 and ndim 1"),
        (np.zeros((2, 160), dtype=np.float32), "dtype float32 and ndim 2"),
    ],
)
def test_an_array_the_connector_cannot_read_fails_by_name(array, shape_text):
    with pytest.raises(TimeFFormatError) as refusal:
        series_for("vctk", _PATH, {"waveform": array, "GT": 0}, _RECORD)
    message = str(refusal.value)
    assert "waveform" in message
    assert _PATH.name in message
    assert shape_text in message
    # The refusal is a modelling decision, not a corrupt file, so the message has to say so.
    assert "PCM" in message


def test_a_bare_array_in_a_corpus_with_no_audio_spec_fails_by_name():
    with pytest.raises(TimeFFormatError, match="no audio spec"):
        series_for("cgmacros", _PATH, {"waveform": np.zeros(8), "GT": 0}, _RECORD)
