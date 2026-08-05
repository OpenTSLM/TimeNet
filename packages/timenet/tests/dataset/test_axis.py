from fractions import Fraction

import pytest

from timenet.dataset.axis import AxisType, OrdinalAxis, RegularAxis
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
    # 500. One that genuinely is not whole has no single reading, so the curator states which.
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


def test_axis_types_are_the_two_shapes():
    assert {t.value for t in AxisType} == {"regular", "ordinal"}


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
