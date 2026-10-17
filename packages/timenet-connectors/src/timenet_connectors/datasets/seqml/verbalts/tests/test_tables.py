import pytest

from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.seqml.verbalts import tables


# Every case below passes literal values. This module needs no file and no fixture, because
# tables.py does no I/O.

_ETTM1_COLUMNS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")


def test_the_var_id_code_indexes_the_upstream_column_order():
    assert tables.signal_names("ETTm1", (3, 0), 1) == ("MULL",)
    assert tables.signal_names("istanbul_traffic", (2, 0), 1) == ("TI_Av",)


def test_every_ettm1_code_names_a_distinct_upstream_column():
    named = [tables.signal_names("ETTm1", (code,), 1)[0] for code in range(len(_ETTM1_COLUMNS))]
    assert tuple(named) == _ETTM1_COLUMNS


def test_a_var_id_outside_the_column_list_is_rejected():
    with pytest.raises(TimeFFormatError, match=r"var_id 7; expected one signal and a var_id in \[0, 7\)"):
        tables.signal_names("ETTm1", (7, 0), 1)


def test_a_var_id_component_with_more_than_one_signal_is_rejected():
    with pytest.raises(TimeFFormatError, match="istanbul_traffic window has 3 signal"):
        tables.signal_names("istanbul_traffic", (0, 0), 3)


def test_blindways_names_every_joint_and_axis():
    names = tables.signal_names("BlindWays", (0,), 72)
    assert names[0] == "j00_x"
    assert names[-1] == "j23_z"
    assert names == tuple(f"j{joint:02d}_{axis}" for joint in range(24) for axis in ("x", "y", "z"))


def test_a_blindways_window_with_the_wrong_signal_count_is_rejected():
    with pytest.raises(TimeFFormatError, match="71 signals, expected 72"):
        tables.signal_names("BlindWays", (0,), 71)


def test_weather_takes_the_upstream_column_names_in_file_order():
    names = tables.signal_names("Weather", (0, 0), 21)
    assert names == tables.WEATHER_SIGNALS
    assert names[0] == "p (mbar)"
    assert names[-1] == "CO2 (ppm)"


def test_the_transliterated_weather_names_carry_no_replacement_character():
    # The paper prints these two units with the degree-square and the micro sign. Both are spelled
    # in ASCII here, so no name carries the replacement character a lost encoding leaves behind.
    assert "SWDR (W/m2)" in tables.WEATHER_SIGNALS
    assert "PAR (umol/m2/s)" in tables.WEATHER_SIGNALS
    assert all("\ufffd" not in name for name in tables.WEATHER_SIGNALS)


def test_a_weather_window_with_the_wrong_signal_count_is_rejected():
    with pytest.raises(TimeFFormatError, match="3 signals, expected 21"):
        tables.signal_names("Weather", (0, 0), 3)


def test_a_synthetic_window_is_named_by_position():
    assert tables.signal_names("synthetic_u", (0,), 1) == ("x0",)
    assert tables.signal_names("synthetic_m", (0,), 2) == ("x0", "x1")
