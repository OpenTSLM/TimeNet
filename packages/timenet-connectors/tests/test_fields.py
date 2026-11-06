from enum import StrEnum
import math

import numpy as np
import pytest

from timenet.errors import TimeNetConnectorError, TimeNetDownloadError
from timenet_connectors.sources.fields import (
    as_list,
    as_object,
    error_context,
    field,
    numbers,
    parse_int,
    parse_whole_number,
    text,
    text_field,
)


class _Color(StrEnum):
    RED = "red"
    BLUE = "blue"


def test_error_context_prefixes_the_message_and_keeps_the_type():
    with pytest.raises(TimeNetConnectorError) as info, error_context("train.jsonl line 4"):
        text_field({"answer": " "}, "answer")
    assert str(info.value) == "train.jsonl line 4: answer: expected text, got ' '"
    assert type(info.value) is TimeNetConnectorError


def test_error_contexts_nest_from_outside_in():
    with (
        pytest.raises(TimeNetConnectorError, match=r"^exams\.csv: row 17: age: expected an integer"),
        error_context("exams.csv"),
        error_context("row 17"),
    ):
        parse_int("x", "age")


def test_error_context_keeps_a_subclass():
    with pytest.raises(TimeNetDownloadError, match=r"^part 2: missing file$"), error_context("part 2"):
        raise TimeNetDownloadError("missing file")


def test_error_context_leaves_other_errors_alone():
    with pytest.raises(KeyError) as info, error_context("row 1"):
        raise KeyError("x")
    assert info.value.args == ("x",)


def test_long_values_are_shortened_in_the_message():
    with pytest.raises(TimeNetConnectorError) as info:
        as_list("y" * 500, "note")
    assert len(str(info.value)) < 120
    assert "..." in str(info.value)


def test_objects_lists_and_fields():
    assert as_object({"a": 1}, "row") == {"a": 1}
    assert as_list([], "items") == []
    assert field({"a": None}, "a") is None
    with pytest.raises(TimeNetConnectorError, match=r"^row: expected an object, got \[1\]$"):
        as_object([1], "row")
    with pytest.raises(TimeNetConnectorError, match="expected a nonempty list"):
        as_list([], "items", nonempty=True)
    with pytest.raises(TimeNetConnectorError, match=r"^missing field 'b'$"):
        field({"a": 1}, "b")
    with pytest.raises(TimeNetConnectorError, match=r"^missing field 'red'$"):
        field({}, _Color.RED)


@pytest.mark.parametrize("value", [None, 3, "", "  \n"])
def test_text_rejects_non_strings_and_blank_text(value):
    with pytest.raises(TimeNetConnectorError, match=r"^note: expected text"):
        text(value, "note")


def test_text_can_allow_blank_but_not_other_types():
    assert text("", "note", allow_blank=True) == ""
    assert text(" a ", "note") == " a "
    with pytest.raises(TimeNetConnectorError, match="expected a string"):
        text(None, "note", allow_blank=True)


def test_numbers_build_typed_arrays():
    floats = numbers([1, 2.5], "series")
    assert floats.dtype == np.float64
    assert floats.tolist() == [1.0, 2.5]
    ints = numbers([1, -2], "series", dtype=np.int64)
    assert ints.dtype == np.int64
    assert numbers([], "series", nonempty=False).shape == (0,)


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([], np.float64),
        ([1, True], np.float64),
        ([1, "2"], np.float64),
        ([[1, 2]], np.float64),
        ([1, math.nan], np.float64),
        ([1, 2.0], np.int64),
        ([2**70], np.int64),
        ("1,2", np.float64),
    ],
)
def test_numbers_reject_anything_else(values, dtype):
    with pytest.raises(TimeNetConnectorError, match=r"^series(\[\d+\])?: expected a"):
        numbers(values, "series", dtype=dtype)


@pytest.mark.parametrize("cell", ["42", "-3", "+7", "007"])
def test_parse_int_accepts_plain_digits(cell):
    assert parse_int(cell, "n") == int(cell)


@pytest.mark.parametrize("cell", ["", " 4", "4 ", "1_000", "4.0", "4e2", "four", "٣", "9" * 4301])
def test_parse_int_rejects_anything_else(cell):
    with pytest.raises(TimeNetConnectorError, match="expected an integer"):
        parse_int(cell, "n")


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("56.0", 56),
        ("56", 56),
        ("5.6e1", 56),
        ("-0.0", 0),
        ("1e20", 10**20),
        ("9007199254740993.0", 9007199254740993),
        ("0e-999", 0),
    ],
)
def test_parse_whole_number_accepts_float_notation_without_rounding(cell, expected):
    assert parse_whole_number(cell, "age") == expected


@pytest.mark.parametrize(
    "cell", ["56.5", "nan", "inf", "", "1_0", " 56", "2.0000000000000001", "1e-999", "1e999999", "9" * 4301 + ".0"]
)
def test_parse_whole_number_rejects_fractions_and_inexact_values(cell):
    with pytest.raises(TimeNetConnectorError, match="age: expected a whole number"):
        parse_whole_number(cell, "age")


def test_parse_whole_number_checks_the_range():
    with pytest.raises(TimeNetConnectorError, match="at most 120"):
        parse_whole_number("300.0", "age", maximum=120)


@pytest.mark.parametrize("cell", ["1e-99999999999999999999999", "0e-99999999999999999999999", "1e99999999999999999999"])
def test_parse_whole_number_rejects_exponents_that_decimal_cannot_hold(cell):
    with pytest.raises(TimeNetConnectorError, match="age: expected a whole number"):
        parse_whole_number(cell, "age")


@pytest.mark.parametrize("parse", [parse_int, parse_whole_number])
@pytest.mark.parametrize("cell", [None, 1.5, 7])
def test_parsers_reject_a_cell_that_is_not_text(parse, cell):
    with pytest.raises(TimeNetConnectorError, match="n: expected"):
        parse(cell, "n")


def test_integers_pass_as_floats_only_when_exact():
    assert numbers([2**53], "x").tolist() == [float(2**53)]
    with pytest.raises(TimeNetConnectorError, match="holds exactly"):
        numbers([2**53 + 1], "x")


def test_error_messages_survive_an_integer_too_long_to_print():
    with pytest.raises(TimeNetConnectorError, match="an int of about 5001 digits"):
        as_list(10**5000, "x")


def test_numbers_names_the_first_bad_item():
    with pytest.raises(TimeNetConnectorError, match=r"s\[3\]: expected a finite number, got nan"):
        numbers([0.5, 0.5, 0.5, math.nan, 0.5], "s")
    with pytest.raises(TimeNetConnectorError, match=r"s\[2\]: expected a finite number, got 'x'"):
        numbers([1, 2, "x"], "s")


@pytest.mark.parametrize(
    ("values", "dtype", "index"),
    [
        ([1, 2**63], np.int64, 1),
        ([1, 2, 2**1100], np.float64, 2),
        ([2**1024], np.float64, 0),
        ([-(2**1024)], np.float64, 0),
        ([-(2**64)], np.int64, 0),
    ],
)
def test_numbers_names_an_item_out_of_range(values, dtype, index):
    with pytest.raises(TimeNetConnectorError, match=rf"^s\[{index}\]: expected .* in range"):
        numbers(values, "s", dtype=dtype)


def test_numbers_checks_exactness_only_for_large_ints():
    assert numbers([2**53, -(2**53), 1], "s").tolist() == [float(2**53), -float(2**53), 1.0]
    assert numbers([2**60], "s").tolist() == [float(2**60)]


def test_numbers_takes_only_the_two_documented_dtypes():
    with pytest.raises(TypeError, match=r"np\.int64 or np\.float64"):
        numbers([1.5], "s", dtype=np.int32)  # ty: ignore[invalid-argument-type]


def test_a_large_list_is_shortened_before_it_is_written_out():
    with pytest.raises(TimeNetConnectorError) as info:
        text(list(range(100_000)), "ids")
    assert len(str(info.value)) < 120


def test_a_list_holding_an_integer_too_long_to_print_is_shown():
    with pytest.raises(TimeNetConnectorError, match=r"\[1, an int of about 5001 digits\]"):
        text([1, 10**5000], "x")
