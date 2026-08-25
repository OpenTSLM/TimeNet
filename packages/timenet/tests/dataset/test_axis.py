from datetime import UTC, datetime
from fractions import Fraction

import numpy as np
import pytest

from timenet.dataset.axis import (
    AxisType,
    IrregularAxis,
    OrdinalAxis,
    RegularAxis,
    time_offsets_from_datetimes,
    to_time_offsets_us,
)
from timenet.errors import TimeFValidationError


@pytest.mark.parametrize(
    ("rate_hz", "expected_period_us"),
    [
        (500, Fraction(2000)),
        (16, Fraction(62500)),
        (1_000_000, Fraction(1)),
        (44100, Fraction(10000, 441)),  # not a whole microsecond
        (360, Fraction(25000, 9)),  # nor this: MIT-BIH ECG
        (256, Fraction(15625, 4)),  # nor this: standard EEG
        (Fraction(1, 3600), Fraction(3_600_000_000)),  # hourly bars
    ],
)
def test_from_rate_hz_is_exact(rate_hz, expected_period_us):
    assert RegularAxis.from_rate_hz(rate_hz).period_us == expected_period_us


def test_a_non_whole_rate_is_stated_as_a_fraction():
    # A float is refused by the type, not at runtime: every real rate is whole, so 500.0 should be
    # 500. One that genuinely is not whole has no single reading, so the builder states which.
    assert RegularAxis.from_rate_hz(Fraction(30000, 1001)).period_us == Fraction(100_100, 3)
    assert RegularAxis.from_rate_hz(Fraction("29.97")).period_us == Fraction(1_000_000 * 100, 2997)


def test_a_sub_microsecond_period_is_refused():
    # Above 1 MHz the period is finer than the format can address, so it is rejected rather than
    # silently rounded to something the axis cannot invert.
    with pytest.raises(TimeFValidationError, match="finer than the one microsecond"):
        RegularAxis.from_rate_hz(2_000_000)


@pytest.mark.parametrize("rate_hz", [44100, 32000, 360, 256, 1024, 500, 16])
def test_placing_a_value_and_locating_it_are_exactly_inverse(rate_hz):
    # This is what deletes the float grid snap: a floor paired with an integer ceiling is exact at
    # every period of one microsecond or coarser, including the ones that are not whole microseconds.
    axis = RegularAxis.from_rate_hz(rate_hz)
    for index in range(0, 200_000, 997):
        assert axis.index_at_or_after(axis.time_offset_us(index)) == index


def test_a_window_keeps_its_place_exactly():
    # At 44.1 kHz only 3 of 1000 window starts land on a whole microsecond, so the origin is an index
    # rather than a time. Every window start is then exact instead of almost all of them being wrong.
    base = RegularAxis.from_rate_hz(44100)
    window = base.at_index(1009)
    assert window.start_index == 1009
    assert window.time_offset_us(0) == base.time_offset_us(1009)
    assert window.index_at_or_after(window.time_offset_us(0)) == 0


def test_windows_compose():
    base = RegularAxis.from_rate_hz(500)
    assert base.at_index(10).at_index(5).start_index == 15


def test_the_period_reduces_itself():
    assert RegularAxis(period_us=Fraction(2000, 2)) == RegularAxis(period_us=Fraction(1000))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"period_us": Fraction(0)},
        {"period_us": Fraction(-1)},
        {"period_us": Fraction(1, 2)},  # sub-microsecond, above 1 MHz
        {"period_us": Fraction(1000), "start_index": -1},
    ],
)
def test_validation_rejects(kwargs):
    with pytest.raises(TimeFValidationError):
        RegularAxis(**kwargs)


def test_an_ordinal_axis_offers_no_route_to_an_time_offset():
    # Not a runtime guard: the attribute does not exist, so a caller narrowed to this class cannot
    # ask a time-valued question and ty rejects the attempt.
    axis = OrdinalAxis()
    assert not hasattr(axis, "time_offset_us")
    assert not hasattr(axis, "index_at_or_after")


def test_axis_types_are_the_three_shapes():
    assert {t.value for t in AxisType} == {"regular", "irregular", "ordinal"}


def test_every_shape_declares_its_own_tag():
    # The discriminator lives on each shape rather than in a function that derives it by branching,
    # so a shape cannot be added without giving itself a tag: writer.py reads axis.axis_type directly.
    assert {RegularAxis.axis_type, IrregularAxis.axis_type, OrdinalAxis.axis_type} == set(AxisType)


def test_irregular_axis_takes_its_endpoints_from_the_stream():
    axis = IrregularAxis.spanning([0, 812_000, 1_601_000, 2_444_000])
    assert (axis.first_us, axis.last_us) == (0, 2_444_000)
    assert axis.axis_type is AxisType.IRREGULAR


def test_irregular_axis_offers_no_constant_time_answer():
    # Same precedent as OrdinalAxis: both need a read, so neither method exists to be called.
    axis = IrregularAxis.spanning([0, 5])
    assert not hasattr(axis, "time_offset_us")
    assert not hasattr(axis, "index_at_or_after")


def test_irregular_axis_compares_and_hashes():
    # _series_identity puts the axis in a tuple and compares it. An axis holding the array instead
    # would raise "truth value of an array with more than one element is ambiguous" right here.
    a, b = IrregularAxis.spanning([0, 7]), IrregularAxis.spanning([0, 7])
    assert a == b
    assert len({a, b}) == 1
    assert ("ibi", "rr", a) == ("ibi", "rr", b)


def test_endpoints_are_not_an_identity_for_the_stream():
    # Documented, deliberate: equal endpoints say nothing about the middle. Nothing may use axis
    # equality to conclude two streams agree, which is why assert_datasets_equal compares time offsets.
    assert IrregularAxis.spanning([0, 5, 10]) == IrregularAxis.spanning([0, 9, 10])


def test_irregular_axis_rejects_backwards_endpoints():
    with pytest.raises(TimeFValidationError, match="runs backwards"):
        IrregularAxis(first_us=10, last_us=5)


def test_time_offsets_must_be_non_decreasing():
    with pytest.raises(TimeFValidationError, match="non-decreasing"):
        to_time_offsets_us([0, 500, 200])


def test_time_offsets_reject_a_datetime64_array():
    # pandas DatetimeIndex.values is datetime64[ns]; read as int64 every time offset lands 1000x out.
    moments = np.array(["2026-08-05T09:00:00", "2026-08-05T09:00:01"], dtype="datetime64[ns]")
    with pytest.raises(TimeFValidationError, match="datetime64"):
        to_time_offsets_us(moments)


def test_time_offsets_reject_floats():
    with pytest.raises(TimeFValidationError, match="float array"):
        to_time_offsets_us([0.0, 1.5])  # ty: ignore[invalid-argument-type]


def test_time_offsets_reject_an_empty_stream():
    with pytest.raises(TimeFValidationError, match="at least one"):
        to_time_offsets_us([])


def test_time_offsets_from_datetimes_measures_against_the_anchor():
    anchor = datetime(2026, 8, 5, 9, 0, 0, tzinfo=UTC)
    time_offsets = time_offsets_from_datetimes([anchor, datetime(2026, 8, 5, 9, 0, 5, tzinfo=UTC)], start_time=anchor)
    assert time_offsets.tolist() == [0, 5_000_000]
    assert time_offsets.dtype == np.int64


def test_time_offsets_from_datetimes_needs_an_anchored_sample():
    with pytest.raises(TimeFValidationError, match="start_time"):
        time_offsets_from_datetimes([datetime(2026, 8, 5, tzinfo=UTC)], start_time=None)


@pytest.mark.parametrize("rate", [29.97, True, "500", 1.5])
def test_from_rate_hz_rejects_a_float_or_bool(rate):
    with pytest.raises(TimeFValidationError, match="integer or Fraction rate"):
        RegularAxis.from_rate_hz(rate)


def test_regular_axis_rejects_a_float_period():
    with pytest.raises(TimeFValidationError, match="must be a Fraction"):
        RegularAxis(period_us=1.5)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("start_index", [1.0, True])
def test_regular_axis_rejects_a_non_integer_start_index(start_index):
    with pytest.raises(TimeFValidationError, match="must be an integer"):
        RegularAxis(period_us=Fraction(2000), start_index=start_index)


def test_irregular_axis_rejects_endpoints_past_int64():
    with pytest.raises(TimeFValidationError, match="fit int64"):
        IrregularAxis(first_us=0, last_us=2**63)


def test_to_time_offsets_us_rejects_a_uint64_past_int64():
    # np.uint64(2**63) would wrap to a negative int64 on the cast; the range check catches it first.
    with pytest.raises(TimeFValidationError, match="fit int64"):
        to_time_offsets_us(np.array([0, 2**63], dtype=np.uint64))


def test_regular_axis_rejects_a_period_past_int64():
    with pytest.raises(TimeFValidationError, match="fit int64"):
        RegularAxis(period_us=Fraction(2**63))


def test_regular_axis_rejects_a_start_index_past_int64():
    with pytest.raises(TimeFValidationError, match="fit int64"):
        RegularAxis(period_us=Fraction(2000), start_index=2**63)
