"""Map source timestamps to a declared Record origin without resampling values."""

from fractions import Fraction

import numpy as np

from timenet.dataset.axis import IrregularAxis, RegularAxis, RegularRun, SegmentedRegularAxis, to_time_offsets_us
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.types.clock import check_int64


def offsets_from_origin(timestamps_us: np.ndarray, origin_us: int) -> np.ndarray:
    """Subtract an explicit origin from whole-microsecond source timestamps.

    The caller establishes alignment and chooses the origin. This function does not
    infer either fact. Input order is preserved, including interleaved tag groups.

    Returns:
        Integer offsets in the same shape and order as the source timestamps.

    Raises:
        TimeFFormatError: If timestamps are not a non-empty integer vector or offsets overflow.
    """
    if timestamps_us.ndim != 1 or not timestamps_us.size or timestamps_us.dtype.kind not in {"i", "u"}:
        raise TimeFFormatError("source timestamps must be a non-empty vector of whole microseconds")
    low, high = int(timestamps_us.min()) - origin_us, int(timestamps_us.max()) - origin_us
    try:
        check_int64("first relative timestamp", low)
        check_int64("last relative timestamp", high)
    except TimeFValidationError as exc:
        raise TimeFFormatError(f"source timestamps cannot use origin {origin_us}: {exc}") from exc
    # Python integers avoid intermediate unsigned wraparound before the checked int64 cast.
    return np.fromiter((int(value) - origin_us for value in timestamps_us), dtype=np.int64)


def axis_for_offsets(
    offsets: np.ndarray, *, period_us: Fraction | None = None, allow_segmented: bool = False
) -> RegularAxis | SegmentedRegularAxis | IrregularAxis:
    """Represent exact offsets compactly without filling gaps.

    An optional published period prevents a gapped sequence from becoming a slower
    regular signal. Without one, only a constant positive observed step implies regularity.
    Opting into segmented axes retains forward gaps at a declared period. Run metadata
    must use fewer integers than explicit offsets (three per run plus the shared period).

    Returns:
        A shifted regular axis or an axis that retains every explicit offset.

    Raises:
        TimeFFormatError: If the source offsets are invalid.
    """
    try:
        offsets = to_time_offsets_us(offsets)
        if period_us is None and len(offsets) > 1:
            step = int(offsets[1]) - int(offsets[0])
            if step > 0 and np.all(np.diff(offsets) == step):
                period_us = Fraction(step)
        if period_us is not None:
            axis = RegularAxis(period_us=period_us, offset_us=int(offsets[0]))
            if all(axis.time_offset_us(index) == int(value) for index, value in enumerate(offsets)):
                return axis
            if allow_segmented:
                segmented = _segmented_for_offsets(offsets, axis)
                if segmented is not None:
                    return segmented
        return IrregularAxis.spanning(offsets)
    except TimeFValidationError as exc:
        raise TimeFFormatError(f"invalid source time offsets: {exc}") from exc


def _segmented_for_offsets(offsets: np.ndarray, axis: RegularAxis) -> SegmentedRegularAxis | None:
    """Return an exact run encoding only when it is smaller than explicit offsets."""
    runs = []
    first = 0
    for index in range(1, len(offsets)):
        expected = axis.time_offset_us(index - first)
        actual = int(offsets[index])
        if actual == expected:
            continue
        if actual < expected:
            return None
        runs.append(RegularRun(axis=axis, n_values=index - first))
        first = index
        axis = RegularAxis(period_us=axis.period_us, offset_us=actual)
    runs.append(RegularRun(axis=axis, n_values=len(offsets) - first))
    return SegmentedRegularAxis(runs=tuple(runs)) if 3 * len(runs) + 2 < len(offsets) else None
