from fractions import Fraction

import numpy as np
import pytest

from timenet.dataset import IrregularAxis, RegularAxis
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
