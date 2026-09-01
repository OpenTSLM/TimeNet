from datetime import time
from pathlib import Path

import pytest

from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.physionet.sleep_edfx import reader, tables
from timenet_connectors.datasets.physionet.sleep_edfx.keys import Condition


# Every recording id in this file is invented. Each one keeps the shape the release uses, three
# study characters then two subject digits, a night and two more characters, because the parser
# needs that shape. Each one names no recording of the release. The subject numbers are all above
# 89, and the release numbers no subject that high.

# Hand-written rows in the shape the cassette sheet gives back: a header row, then one row for
# each recording, with every cell a float. The numbers are invented and no row of the real sheet
# holds them. The lights-off cells are the Excel day fractions for 01:05:00 and 23:12:00.
_CASSETTE_ROWS = [
    ("subject", "night", "age", "sex (F=1)", "LightsOff"),
    (90.0, 1.0, 44.0, 1.0, 0.04513888888888889),
    (90.0, 2.0, 44.0, 1.0, 0.9666666666666667),
    (91.0, 1.0, 57.0, 2.0, 0.9791666666666666),
]

# Hand-written rows in the shape the telemetry sheet gives back: two header rows whose merged
# group labels read back empty, then one row for each subject. The numbers are invented. The
# subject numbers skip 93, so a reader that used the position of a row would name the wrong
# subject.
_TELEMETRY_ROWS = [
    ("Subject - age - sex", "", "", "Placebo night", "", "Temazepam night", ""),
    ("Nr", "Age", "M1/F2", "night nr", "lights off", "night nr", "lights off"),
    (91.0, 62.0, 1.0, 1.0, 0.9270833333333334, 2.0, 0.9861111111111112),
    (92.0, 29.0, 2.0, 2.0, 0.9513888888888888, 1.0, 0.0),
    (94.0, 41.0, 2.0, 1.0, 0.9618055555555556, 2.0, 0.9305555555555556),
]


def test_cassette_rows_key_by_subject_and_night():
    table = tables.parse_subject_table(tables.CASSETTE_SHEET, _CASSETTE_ROWS)
    assert set(table) == {(90, 1), (90, 2), (91, 1)}


def test_cassette_row_states_age_sex_and_lights_off():
    row = tables.parse_subject_table(tables.CASSETTE_SHEET, _CASSETTE_ROWS)[90, 1]
    assert (row.age, row.sex, row.nights[0].lights_off) == (44, "F", time(1, 5, 0))


def test_a_lights_off_time_before_midnight_reads_the_same_way():
    table = tables.parse_subject_table(tables.CASSETTE_SHEET, _CASSETTE_ROWS)
    assert table[90, 2].nights[0].lights_off == time(23, 12, 0)


def test_the_code_1_means_the_opposite_thing_in_each_sheet():
    cassette = tables.parse_subject_table(tables.CASSETTE_SHEET, _CASSETTE_ROWS)[90, 1]
    telemetry = tables.parse_subject_table(tables.TELEMETRY_SHEET, _TELEMETRY_ROWS)[91,]
    assert (cassette.sex, telemetry.sex) == ("F", "M")


def test_the_cassette_sheet_decodes_code_2_as_male():
    assert tables.parse_subject_table(tables.CASSETTE_SHEET, _CASSETTE_ROWS)[91, 1].sex == "M"


def test_a_sex_code_the_sheet_does_not_define_raises():
    rows = [_CASSETTE_ROWS[0], (90.0, 1.0, 44.0, 3.0, 0.5)]
    with pytest.raises(TimeFFormatError, match=r"row 2.*code 3"):
        tables.parse_subject_table(tables.CASSETTE_SHEET, rows)


def test_a_lights_off_cell_holding_text_raises():
    rows = [_CASSETTE_ROWS[0], (90.0, 1.0, 44.0, 1.0, "01:05:00")]
    with pytest.raises(TimeFFormatError, match=r"SC-subjects\.xls.*'01:05:00'"):
        tables.parse_subject_table(tables.CASSETTE_SHEET, rows)


def test_a_lights_off_cell_outside_one_day_raises():
    rows = [_CASSETTE_ROWS[0], (90.0, 1.0, 44.0, 1.0, 1.5)]
    with pytest.raises(TimeFFormatError, match=r"SC-subjects\.xls.*1\.5"):
        tables.parse_subject_table(tables.CASSETTE_SHEET, rows)


def test_one_pair_of_subject_and_night_twice_raises():
    rows = [*_CASSETTE_ROWS, (90.0, 1.0, 44.0, 1.0, 0.5)]
    with pytest.raises(TimeFFormatError, match="appears twice"):
        tables.parse_subject_table(tables.CASSETTE_SHEET, rows)


def test_a_short_cassette_row_raises():
    with pytest.raises(TimeFFormatError, match="columns"):
        tables.parse_subject_table(tables.CASSETTE_SHEET, [_CASSETTE_ROWS[0], (90.0, 1.0, 44.0)])


def test_telemetry_rows_key_by_the_subject_number_and_not_by_the_position():
    table = tables.parse_subject_table(tables.TELEMETRY_SHEET, _TELEMETRY_ROWS)
    assert sorted(table) == [(91,), (92,), (94,)]


def test_a_telemetry_row_becomes_two_nights():
    subject = tables.parse_subject_table(tables.TELEMETRY_SHEET, _TELEMETRY_ROWS)[91,]
    assert (subject.age, subject.sex) == (62, "M")
    assert [(night.night, night.condition, night.lights_off) for night in subject.nights] == [
        (1, Condition.PLACEBO, time(22, 15, 0)),
        (2, Condition.TEMAZEPAM, time(23, 40, 0)),
    ]


def test_a_lights_off_time_of_midnight_reads_as_zero():
    subject = tables.parse_subject_table(tables.TELEMETRY_SHEET, _TELEMETRY_ROWS)[92,]
    assert tables.find_night(subject, 1, "ST7921J0").lights_off == time(0, 0, 0)


def test_the_condition_comes_from_the_column_the_night_number_sits_in():
    table = tables.parse_subject_table(tables.TELEMETRY_SHEET, _TELEMETRY_ROWS)
    first = tables.find_night(table[91,], 1, "ST7911J0")
    second = tables.find_night(table[92,], 1, "ST7921J0")
    assert (first.condition, second.condition) == (Condition.PLACEBO, Condition.TEMAZEPAM)


def test_a_night_number_that_matches_neither_column_raises():
    subject = tables.parse_subject_table(tables.TELEMETRY_SHEET, _TELEMETRY_ROWS)[91,]
    with pytest.raises(TimeFFormatError, match="ST7911J0"):
        tables.find_night(subject, 3, "ST7911J0")


def test_a_night_number_that_matches_both_columns_raises():
    subject = tables.SubjectRow(
        subject=95,
        age=47,
        sex="F",
        nights=(
            tables.SubjectNight(night=1, condition=Condition.PLACEBO, lights_off=time(23, 0, 0)),
            tables.SubjectNight(night=1, condition=Condition.TEMAZEPAM, lights_off=time(23, 30, 0)),
        ),
    )
    with pytest.raises(TimeFFormatError, match="ST7951J0"):
        tables.find_night(subject, 1, "ST7951J0")


def test_a_recording_with_no_row_raises_and_names_the_recording_and_the_table():
    table = tables.parse_subject_table(tables.CASSETTE_SHEET, _CASSETTE_ROWS)
    with pytest.raises(TimeFFormatError, match=r"SC4991E0.*SC-subjects\.xls"):
        tables.find_subject_row(table, tables.CASSETTE_SHEET, 99, 1, "SC4991E0")


def test_a_telemetry_subject_with_no_row_raises():
    table = tables.parse_subject_table(tables.TELEMETRY_SHEET, _TELEMETRY_ROWS)
    with pytest.raises(TimeFFormatError, match=r"ST7991J0.*ST-subjects\.xls"):
        tables.find_subject_row(table, tables.TELEMETRY_SHEET, 99, 1, "ST7991J0")


def test_a_file_that_is_not_a_workbook_raises_and_names_it(tmp_path: Path):
    path = tmp_path / "SC-subjects.xls"
    path.write_text("this is not a workbook", encoding="utf-8")
    with pytest.raises(TimeFFormatError, match=r"SC-subjects\.xls"):
        reader.read_table_rows(path)
