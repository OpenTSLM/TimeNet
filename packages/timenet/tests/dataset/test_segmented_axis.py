from fractions import Fraction

import numpy as np
import pytest

from timenet.dataset import RegularAxis, RegularRun, SegmentedRegularAxis, Signal
from timenet.errors import TimeFValidationError
from timenet.types import TimeInterval, TimeSeriesSpec


def axis_example():
    return SegmentedRegularAxis(
        runs=(
            RegularRun(axis=RegularAxis.from_rate_hz(100), n_values=3),
            RegularRun(axis=RegularAxis.from_rate_hz(100, offset_us=50_000), n_values=2),
        )
    )


def signal_example(axis=None):
    axis = axis or axis_example()
    return Signal(
        name="acceleration",
        spec=TimeSeriesSpec(spec_type="acceleration", name="Acceleration", unit_value="g", value_shape=(3,)),
        time_axis=axis,
        data=np.arange(axis.n_values * 3).reshape(-1, 3),
    )


def test_lower_bounds_match_explicit_timestamps():
    axis = axis_example()
    times = [0, 10_000, 20_000, 50_000, 60_000]
    assert [axis.time_offset_us(i) for i in range(5)] == times
    assert axis.time_offset_us(5) == 70_000
    for offset in range(-1, 80_000, 137):
        assert axis.index_at_or_after(offset) == np.searchsorted(times, offset)
    assert hash(axis) == hash(axis_example())


def test_slicing_preserves_gaps_and_values():
    signal = signal_example()
    part = signal.slice(TimeInterval.micros(10_000, 60_000))
    assert isinstance(part.time_axis, SegmentedRegularAxis)
    assert [part.time_axis.time_offset_us(i) for i in range(3)] == [10_000, 20_000, 50_000]
    np.testing.assert_array_equal(part.to_numpy(), signal.to_numpy()[1:4])
    inner = part.slice(TimeInterval.micros(50_000, 60_000))
    assert isinstance(inner.time_axis, RegularAxis)
    assert inner.span_us == (50_000, 60_000)
    assert signal.span_us == (0, 70_000)
    assert not signal.overlaps_with(TimeInterval.micros(30_000, 50_000))
    with pytest.raises(TimeFValidationError, match="no steps"):
        signal.slice(TimeInterval.micros(30_000, 50_000))
    with pytest.raises(TimeFValidationError, match="past"):
        signal.step_range(TimeInterval.micros(0, 80_000))
    assert signal.slice(TimeInterval.micros(0, 80_000), clip=True).n_values == 5


@pytest.mark.parametrize("rate", [360, 44_100])
def test_fractional_phase_survives_repeated_slices(rate):
    axis = SegmentedRegularAxis(
        runs=(
            RegularRun(axis=RegularAxis.from_rate_hz(rate).at_index(7), n_values=20),
            RegularRun(axis=RegularAxis.from_rate_hz(rate, offset_us=1_000_000).at_index(3), n_values=20),
        )
    )
    times = [axis.time_offset_us(i) for i in range(axis.n_values)]
    for offset in times:
        assert axis.index_at_or_after(offset) == np.searchsorted(times, offset)
        assert axis.index_at_or_after(offset + 1) == np.searchsorted(times, offset + 1)
    part = axis.slice(3, 35)
    assert isinstance(part, SegmentedRegularAxis)
    again = part.slice(4, 30)
    assert [again.time_offset_us(i) for i in range(26)] == times[7:33]


@pytest.mark.parametrize(
    "runs",
    [
        (),
        (
            RegularRun(axis=RegularAxis.from_rate_hz(100), n_values=3),
            RegularRun(axis=RegularAxis.from_rate_hz(100, offset_us=20_000), n_values=2),
        ),
        (
            RegularRun(axis=RegularAxis.from_rate_hz(100), n_values=3),
            RegularRun(axis=RegularAxis.from_rate_hz(50, offset_us=50_000), n_values=2),
        ),
    ],
)
def test_invalid_runs_are_rejected(runs):
    with pytest.raises(TimeFValidationError):
        SegmentedRegularAxis(runs=runs)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_counts_are_rejected(count):
    with pytest.raises(TimeFValidationError):
        RegularRun(axis=RegularAxis.from_rate_hz(100), n_values=count)


def test_count_mismatch_and_overflow_are_rejected():
    with pytest.raises(TimeFValidationError, match="length"):
        Signal(
            name="x",
            spec=TimeSeriesSpec(spec_type="x", name="x", unit_value=None),
            time_axis=axis_example(),
            data=[1, 2],
        )
    with pytest.raises(TimeFValidationError):
        RegularRun(axis=RegularAxis(period_us=Fraction(2)), n_values=2**63 - 1)


def test_contiguity_uses_coverage_not_outer_span():
    signal = signal_example()
    assert signal.is_contiguous(TimeInterval.micros(0, 30_000))
    assert signal.is_contiguous(TimeInterval.micros(50_000, 70_000))
    assert not signal.is_contiguous(TimeInterval.micros(0, 70_000))
    assert not signal.is_contiguous(TimeInterval.micros(30_000, 40_000))
    assert not signal.is_contiguous(TimeInterval.micros(50_000, 70_001))
    adjacent = SegmentedRegularAxis(
        runs=(
            RegularRun(axis=RegularAxis.from_rate_hz(100), n_values=3),
            RegularRun(axis=RegularAxis.from_rate_hz(100, offset_us=30_000), n_values=2),
        )
    )
    assert adjacent.is_contiguous(0, 50_000)
