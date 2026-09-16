"""Deliver a record as an array-cell pandas frame.

A frame holds one row per record: ``record_id``, ``time_axis``, then one column per signal. Each
signal cell holds that signal's whole value array, in the dtype it was stored in. The signals of one
record can differ in length and in dtype, and nothing here pads, resamples, reindexes or casts.

Time is one :data:`TIME_AXIS_COLUMN` cell keyed by column name, not one offset per value. An axis is
two numbers whatever the length of the series. For a single series, :func:`series_frame` returns the
long shape instead: one row per value, with a time offset on each row.

pandas is not an SDK dependency. It is imported inside the functions that build a frame, so a caller
without pandas can still import this module and the rest of ``timenet``.
"""

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from timenet.dataset import Record, TimeSeries
from timenet.dataset.axis import IrregularAxis, OrdinalAxis, RegularAxis
from timenet.errors import TimeFValidationError


if TYPE_CHECKING:  # pragma: no cover - imported for the annotations only
    import pandas as pd

    from timenet.reader import TimeFReader


RECORD_ID_COLUMN = "record_id"
"""Which record a row came from."""

TIME_AXIS_COLUMN = "time_axis"
"""One dict cell, keyed by signal column name, that holds the axis of each signal."""

ARRAY_META_COLUMNS: tuple[str, ...] = (RECORD_ID_COLUMN, TIME_AXIS_COLUMN)
"""The columns of an array-cell frame that are not signals. Every column after them is a signal."""

TIME_COLUMN = "time_us"
"""Microseconds from the record's relative zero. An ordinal series has none, so the column is null."""

VALUE_COLUMN = "value"
"""The values of a scalar series. An N-D series spreads over ``value_0``, ``value_1`` and so on."""


def record_array_frame(record: Record) -> "pd.DataFrame":
    """Return one record as a single row, one column per signal, each cell a whole value array.

    A column takes the name of its signal. If the record repeats a signal name, the column name also
    carries the series id: ``I (record-000-lead-i)``.

    Args:
        record: The record to lay out. Its signals keep their stored order, and their values load
            here.

    Returns:
        A one-row frame: ``record_id``, ``time_axis``, then one object-typed column per signal
        holding that signal's values. A nullable signal holds zero where a value is missing. Read it
        with :meth:`~timenet.dataset.TimeSeries.to_arrow` to tell those apart from observations.

    Raises:
        TimeFValidationError: If the record holds no series, or the signal names cannot give a column
            each.
    """
    pd = _pandas()
    series = record.time_series
    if not series:
        raise TimeFValidationError(f"record {record.record_id!r} holds no series, so it has no frame")
    names = _column_names(record, series)
    columns: dict[str, list[Any]] = {
        RECORD_ID_COLUMN: [record.record_id],
        TIME_AXIS_COLUMN: [{name: ts.time_axis for name, ts in zip(names, series, strict=True)}],
    }
    for name, ts in zip(names, series, strict=True):
        columns[name] = [_dense_values(ts)]
    return pd.DataFrame(columns, copy=False)


def iter_record_frames(
    reader: "TimeFReader",
    record_ids: Iterable[str] | None = None,
) -> Iterator[tuple[str, "pd.DataFrame"]]:
    """Stream a dataset's records as array-cell frames, without materializing the dataset.

    The reader loads a record's values as it builds that record's frame, then drops them when the
    caller moves on. One record is in memory at a time.

    Args:
        reader: An open reader.
        record_ids: The records to read, or ``None`` for every record. Records come back in stored
            order, not in the order they were asked for.

    Yields:
        Each record's id and its one-row frame.
    """
    for record in reader.iter_records(record_ids):
        yield record.record_id, record_array_frame(record)


def series_frame(series: TimeSeries) -> "pd.DataFrame":
    """Return one series as a long frame of its time offsets and its values.

    Use this shape for one signal aligned to a timeline. It holds one time offset per value.

    A regular axis computes its offsets from the cadence. An irregular axis reads its stored offsets.
    An ordinal axis has no timeline, so its time column is null.

    Args:
        series: The series to read. Its values load here.

    Returns:
        A frame with ``time_us`` and either ``value`` or ``value_0 .. value_n``.
    """
    pd = _pandas()
    values = _dense_values(series)
    columns: dict[str, Any] = {TIME_COLUMN: _time_offsets(series)}
    if values.ndim == 1:
        columns[VALUE_COLUMN] = values
    else:
        flat = values.reshape(len(values), -1)
        for component in range(flat.shape[1]):
            columns[f"{VALUE_COLUMN}_{component}"] = flat[:, component]
    return pd.DataFrame(columns)


def _column_names(record: Record, series: Sequence[TimeSeries]) -> tuple[str, ...]:
    """Return one column name per series, unique inside the record.

    Args:
        record: The owning record, for the error message.
        series: The record's series, in stored order.

    Returns:
        The signal name, or ``signal (series id)`` if the record repeats a signal name.

    Raises:
        TimeFValidationError: If the names still collide, or one of them is a meta column.
    """
    seen = Counter(ts.signal for ts in series)
    names = tuple(ts.signal if seen[ts.signal] == 1 else f"{ts.signal} ({ts.time_series_id})" for ts in series)
    if len(set(names)) != len(names) or set(names) & set(ARRAY_META_COLUMNS):
        raise TimeFValidationError(
            f"the signals of record {record.record_id!r} ({list(names)}) cannot give a column each; "
            f"{ARRAY_META_COLUMNS} are the frame's own"
        )
    return names


def _dense_values(series: TimeSeries) -> np.ndarray:
    """Return a series' values as one dense array.

    Args:
        series: The series to read.

    Returns:
        The values. A nullable series holds zero where a value is missing. A string series comes
        back as an object array, because it has no numeric dtype.
    """
    if series.spec.dtype == "str":
        return series.to_arrow().to_numpy(zero_copy_only=False)
    return series.to_numpy_and_mask()[0]


def _time_offsets(series: TimeSeries) -> np.ndarray:
    """Return one time offset per value of a series.

    Args:
        series: The series whose timeline to compute or read.

    Returns:
        Int64 microseconds, or an all-null float array for an ordinal series, which has no timeline.

    Raises:
        TimeFValidationError: If the axis is of a type this module does not know.
    """
    axis = series.time_axis
    if isinstance(axis, RegularAxis):
        indices = np.arange(series.n_values, dtype=np.int64) + axis.start_index
        return (indices * axis.period_us.numerator) // axis.period_us.denominator
    if isinstance(axis, IrregularAxis):
        return series.time_offsets_us()
    if isinstance(axis, OrdinalAxis):
        return np.full(series.n_values, np.nan)
    raise TimeFValidationError(f"series {series.time_series_id!r} has an unknown axis {type(axis).__name__}")


def _pandas() -> Any:
    """Import pandas, or raise an error that names the extra to install.

    Returns:
        The ``pandas`` module.

    Raises:
        ModuleNotFoundError: If the ``pandas`` extra is not installed.
    """
    try:
        import pandas  # noqa: PLC0415 - imported lazily so the core SDK does not need pandas

        return pandas
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
        raise ModuleNotFoundError("the timenet pandas view needs pandas: pip install 'timenet[pandas]'") from exc


__all__: Sequence[str] = (
    "ARRAY_META_COLUMNS",
    "RECORD_ID_COLUMN",
    "TIME_AXIS_COLUMN",
    "TIME_COLUMN",
    "VALUE_COLUMN",
    "iter_record_frames",
    "record_array_frame",
    "series_frame",
)
