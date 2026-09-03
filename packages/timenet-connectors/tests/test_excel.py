from datetime import time
from pathlib import Path
import struct

import pytest

from timenet.errors import TimeFFormatError
from timenet_connectors.bases import excel


# The workbook name and the row number are error-message context only, so every test here passes
# the same invented pair and each raise test asserts that both reach the message.
_WORKBOOK = "subjects.xls"
_ROW = 7


def test_a_file_that_is_not_a_workbook_raises_and_names_it(tmp_path: Path):
    path = tmp_path / "subjects.xls"
    path.write_text("this is not a workbook", encoding="utf-8")
    with pytest.raises(TimeFFormatError, match=r"subjects\.xls"):
        excel.read_table_rows(path)


def test_a_workbook_is_read_as_one_tuple_for_each_row(tmp_path: Path):
    # A minimal legacy BIFF worksheet, written byte by byte: the stream header, one number cell
    # holding 44.0, then the end record. xlrd reads this flat form of the format, so the test
    # needs no writer and no checked-in binary.
    path = tmp_path / "subjects.xls"
    path.write_bytes(
        struct.pack("<HHHH", 0x0009, 4, 0x0002, 0x0010)
        + struct.pack("<HH", 0x0003, 15)
        + struct.pack("<HH3sd", 0, 0, b"\x00\x00\x00", 44.0)
        + struct.pack("<HH", 0x000A, 0)
    )
    assert excel.read_table_rows(path) == [(44.0,)]


def test_a_whole_number_decodes():
    # Excel writes every number as a float, so a whole number reaches the decoder as 44.0.
    assert excel.decode_whole_number(44.0, "age", _WORKBOOK, _ROW) == 44


def test_a_fraction_is_no_whole_number_and_raises():
    with pytest.raises(TimeFFormatError, match=r"subjects\.xls row 7: age holds 44\.5"):
        excel.decode_whole_number(44.5, "age", _WORKBOOK, _ROW)


def test_a_bool_is_no_whole_number_and_raises():
    # A bool is an int in Python, so the decoder has to turn it down by name.
    with pytest.raises(TimeFFormatError, match=r"subjects\.xls row 7: age holds True"):
        excel.decode_whole_number(True, "age", _WORKBOOK, _ROW)


def test_text_is_no_whole_number_and_raises():
    with pytest.raises(TimeFFormatError, match=r"subjects\.xls row 7: age holds '44'"):
        excel.decode_whole_number("44", "age", _WORKBOOK, _ROW)


def test_a_day_fraction_of_zero_is_midnight():
    assert excel.decode_day_fraction_as_time(0.0, _WORKBOOK, _ROW) == time(0, 0, 0)


def test_half_a_day_is_noon():
    assert excel.decode_day_fraction_as_time(0.5, _WORKBOOK, _ROW) == time(12, 0, 0)


def test_a_day_fraction_reads_to_the_second():
    assert excel.decode_day_fraction_as_time(0.04513888888888889, _WORKBOOK, _ROW) == time(1, 5, 0)


def test_a_whole_day_is_no_time_of_day_and_raises():
    with pytest.raises(TimeFFormatError, match=r"subjects\.xls row 7: the cell holds 1\.0"):
        excel.decode_day_fraction_as_time(1.0, _WORKBOOK, _ROW)


def test_more_than_a_day_raises():
    with pytest.raises(TimeFFormatError, match=r"subjects\.xls row 7: the cell holds 1\.5"):
        excel.decode_day_fraction_as_time(1.5, _WORKBOOK, _ROW)


def test_a_negative_day_fraction_raises():
    with pytest.raises(TimeFFormatError, match=r"subjects\.xls row 7: the cell holds -0\.25"):
        excel.decode_day_fraction_as_time(-0.25, _WORKBOOK, _ROW)


def test_a_cell_holding_text_is_no_day_fraction_and_raises():
    with pytest.raises(TimeFFormatError, match=r"subjects\.xls row 7: the cell holds '01:05:00'"):
        excel.decode_day_fraction_as_time("01:05:00", _WORKBOOK, _ROW)
