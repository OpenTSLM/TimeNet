from fractions import Fraction

import numpy as np
import pytest

from timenet.dataset import IrregularAxis, RegularAxis, SegmentedRegularAxis
from timenet.errors import TimeFFormatError
from timenet_connectors.time_axes import axis_for_offsets, offsets_from_origin


def test_shifted_grid_and_published_cadence_gaps():
    timestamps = np.array([1_700_000_000_000_003, 1_700_000_000_000_013, 1_700_000_000_000_023])
    offsets = offsets_from_origin(timestamps, 1_700_000_000_000_000)
    axis = axis_for_offsets(offsets, period_us=Fraction(10))
    assert isinstance(axis, RegularAxis)
    assert [axis.time_offset_us(i) for i in range(3)] == [3, 13, 23]
    assert isinstance(axis_for_offsets(offsets[[0, 2]], period_us=Fraction(10)), IrregularAxis)


def test_timestamp_subtraction_rejects_overflow_before_numpy_can_wrap():
    with pytest.raises(TimeFFormatError, match="origin"):
        offsets_from_origin(np.array([2**63 - 1], dtype=np.int64), -1)


def test_segmented_axis_is_opt_in_and_preserves_exact_offsets():
    offsets = np.concatenate((np.arange(20) * 10_000, 400_007 + np.arange(20) * 10_000))
    assert isinstance(axis_for_offsets(offsets, period_us=Fraction(10_000)), IrregularAxis)
    axis = axis_for_offsets(offsets, period_us=Fraction(10_000), allow_segmented=True)
    assert isinstance(axis, SegmentedRegularAxis)
    assert [axis.time_offset_us(i) for i in range(len(offsets))] == offsets.tolist()


def test_segmented_selection_keeps_dense_jitter_and_short_inputs_irregular():
    for offsets in (np.array([0, 20, 40]), np.arange(100) * 9):
        assert isinstance(axis_for_offsets(offsets, period_us=Fraction(10), allow_segmented=True), IrregularAxis)
