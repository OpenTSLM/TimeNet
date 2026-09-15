"""Deliver a record as an array-cell pandas frame.

The shape is one row per record: ``record_id``, ``time_axis``, then one column per signal whose cell
holds that signal's whole value array with the dtype it was stored in. The signals of one record
need not share a length or a dtype, and nothing here pads, resamples, reindexes or casts. A 500 Hz
ECG lead of 1,000 float32 values sits beside a 1 Hz temperature of 10 float64 values in the same row,
because the cells are separate arrays rather than columns of one table.

Time rides as one :data:`TIME_AXIS_COLUMN` cell keyed by column name, not as an offset per value.
Ragged signals cannot share one offset column, and one offset column per signal would cost more bytes
than the values do. An axis is two numbers whatever the length of the series.

A long frame, one row per value, was the earlier shape here. It is not the default any more, and the
reason is measured: restating the record id, the series id and the signal name on every row costs
2.35 GB on one Sleep-EDF record against 96 MB of values, and building it was 99.7 per cent of what a
pandas read measured. It also refused a record whose series disagree on dtype. :func:`series_frame`
still returns that shape for one series, where none of those costs apply.

pandas is not an SDK dependency. It is imported inside the functions that build a frame, so a caller
without it can still import this module and everything else in ``timenet``.
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
"""Per signal, the axis its values sit on, as one dict cell keyed by the signal's column name."""

ARRAY_META_COLUMNS: tuple[str, ...] = (RECORD_ID_COLUMN, TIME_AXIS_COLUMN)
"""The columns of an array-cell frame that are not a signal. Every column after them is one."""

TIME_COLUMN = "time_us"
"""Microseconds from the record's relative zero. An ordinal series has none, so the column is null."""

VALUE_COLUMN = "value"
"""The values of a scalar series. An N-D series spreads over ``value_0``, ``value_1`` and so on."""


def record_array_frame(record: Record) -> "pd.DataFrame":
    """Return one record as a single row, one column per signal, each cell a whole value array.

    A column is named after its signal. A record can hold the same signal name twice, at two rates or
    over two windows, and those two need two columns, so a name the record repeats carries the
    series id as well: ``I (record-000-lead-i)``.

    Args:
        record: The record to lay out. Its signals are taken in stored order, and their values load
            here rather than before.

    Returns:
        A one-row frame: ``record_id``, ``time_axis``, then one object-typed column per signal
        holding that signal's values. A nullable signal holds zero where a value is missing; read it
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

    The reader loads each record's values as the frame is built and drops them when the caller moves
    on, so the resident set holds one record rather than the corpus.

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

    This is the shape to reach for when a caller wants one signal aligned to a timeline. It costs one
    time offset per value, which is why :func:`record_array_frame` does not use it for a whole record.

    A regular axis computes its offsets from the cadence, an irregular one reads its stored stream,
    and an ordinal one has no timeline at all, so its time column is null.

    Args:
        series: The series to read. Its values load here, not before.

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
        The signal name, or ``signal (series id)`` where the record repeats a signal name.

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
        The values. A nullable series holds zero where a value is missing. A free-form string series
        comes back as an object array, since it has no numeric dtype.
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
        TimeFValidationError: If the axis is a shape this module does not know.
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
    """Import pandas, with the message that names the extra.

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
