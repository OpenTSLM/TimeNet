"""Shared step and temporal window resolution for values and annotation overlays."""

from typing import Protocol

from timenet.dataset import IrregularAxis, OrdinalAxis, RegularAxis, Signal
from timenet.dataset.axis import TimeAxis
from timenet.errors import TimeFValidationError
from timenet.reader.inspection.queries import ReaderQueries


class WindowBoundsQuery(Protocol):
    """Selection bounds independent of plotting and HTTP."""

    record_id: str
    signal_id: str
    start: int
    stop: int | None
    start_us: int | None
    end_us: int | None
    full: bool


def window_bounds(inspection: ReaderQueries, signal: Signal, query: WindowBoundsQuery) -> tuple[int, int]:
    """Locate a common time window independently on each selected signal.

    Returns:
        The selected half-open step bounds.

    Raises:
        TimeFValidationError: If the window is reversed or time is used with an ordinal axis.
    """
    if query.start < 0 or (query.start_us is not None and query.end_us is not None and query.start_us > query.end_us):
        raise TimeFValidationError("window bounds must be ordered and steps must be nonnegative")
    start = query.start
    stop = query.stop
    axis = signal.time_axis

    def lower_bound(target: int) -> int:
        low, high = 0, signal.n_values
        while low < high:
            middle = (low + high) // 2
            value = x_coordinates(inspection, query.record_id, query.signal_id, axis, middle, middle + 1)[0]
            if value < target:
                low = middle + 1
            else:
                high = middle
        return low

    if query.start_us is not None or query.end_us is not None:
        if isinstance(axis, OrdinalAxis):
            raise TimeFValidationError("ordinal signals require a step window")
        start = lower_bound(query.start_us) if query.start_us is not None else 0
        stop = lower_bound(query.end_us) if query.end_us is not None else signal.n_values
    elif stop is None:
        if query.full:
            stop = signal.n_values
        elif isinstance(axis, OrdinalAxis):
            stop = min(signal.n_values, start + 2000)
        elif start < signal.n_values:
            first = x_coordinates(inspection, query.record_id, query.signal_id, axis, start, start + 1)[0]
            stop = lower_bound(first + 10_000_000)
        else:
            stop = start
    if start > stop or stop > signal.n_values or start > signal.n_values:
        raise TimeFValidationError("window must satisfy 0 <= start <= stop <= signal length")
    return start, stop


def x_coordinates(  # noqa: PLR0913, PLR0917
    inspection: ReaderQueries, record_id: str, signal_id: str, axis: TimeAxis, start: int, stop: int
) -> tuple[int, ...]:
    """Return exact step or time coordinates for a bounded window.

    Raises:
        TimeFValidationError: If the axis shape is unsupported.
    """
    if isinstance(axis, RegularAxis):
        return tuple(axis.time_offset_us(index) for index in range(start, stop))
    if isinstance(axis, IrregularAxis):
        return tuple(inspection.offsets(record_id, signal_id, start, stop).to_pylist())
    if isinstance(axis, OrdinalAxis):
        return tuple(range(start, stop))
    raise TimeFValidationError("unsupported signal axis")
