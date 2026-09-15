"""Deliver a record as an array-cell pandas frame, straight off the control-plane reader.

The shape is one row per record: ``record_id``, ``time_axis``, then one column per signal whose cell
holds that signal's whole value array with the dtype it was stored in. The signals of one record
need not share a length or a dtype, and nothing here pads, resamples, reindexes or casts. A 500 Hz
ECG lead of 1,000 float32 values sits beside a 1 Hz temperature of 10 float64 values in the same
row, because the cells are separate arrays rather than columns of one table.

Time rides as one :data:`TIME_AXIS_COLUMN` cell keyed by column name, not as an offset per value.
Ragged signals cannot share one offset column, and one offset column per signal would cost more
bytes than the values do. An axis is two numbers whatever the length of the series.

Values come in from the caller rather than being read here. A batch of records is one
:meth:`~timenet.control_plane.reader.TimeFReader.values_for` call, which decodes each row group
once; reading a signal at a time re-opens the shard and re-decodes the row group for every signal.
:func:`iter_record_frames` and :class:`TimeFPandasDataset` do that batching for a whole walk.

pandas is not an SDK dependency. It is imported inside the function that builds a frame, so a caller
without it can still import this module and everything else in ``timenet``.
"""

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from timenet.control_plane.reader import RecordView, SignalView, TimeFReader
from timenet.dataset.axis import TimeAxis
from timenet.errors import TimeFValidationError


if TYPE_CHECKING:  # pragma: no cover - imported for the annotations only
    import pandas as pd


RECORD_ID_COLUMN = "record_id"
"""Which record a row came from: the id it was built under, or its surrogate id when it has none."""

TIME_AXIS_COLUMN = "time_axis"
"""Per signal, the axis its values sit on, as one dict cell keyed by the signal's column name.

The cell holds a :data:`~timenet.dataset.axis.TimeAxis`: the caller's when one is passed as
``axes=``, and otherwise the one the reader resolved, because a :class:`SignalView` carries the axis
itself and not only its kind. It falls back to the axis type alone (``regular``, ``irregular``,
``ordinal``) only for a version whose ``axes`` row the reader could not interpret.
"""

ARRAY_META_COLUMNS: tuple[str, ...] = (RECORD_ID_COLUMN, TIME_AXIS_COLUMN)
"""The columns of an array-cell frame that are not a signal. Every column after them is one."""

DEFAULT_BATCH_SIZE = 512
"""How many records a walk hydrates and reads values for at once. It is the reader's own default."""


def record_array_frame(
    record: RecordView,
    values: Mapping[int, np.ndarray],
    *,
    axes: Mapping[int, TimeAxis] | None = None,
) -> "pd.DataFrame":
    """Return one record as a single row, one column per signal, each cell a whole value array.

    A column is named after its signal. A record can hold the same signal name twice, at two rates
    or over two windows, and those two need two columns, so a name the record repeats carries the
    signal's id as well: ``I (record-000-lead-i)``.

    Args:
        record: The record to lay out. Its signals are taken in the order
            :meth:`~timenet.control_plane.reader.RecordView.signals` walks them.
        values: The values of this record's signals, keyed by ``signal_id``, as
            :meth:`~timenet.control_plane.reader.TimeFReader.values_for` returns them for a whole
            batch of records. A cell is the array as it comes back, not a copy or a cast of it.
        axes: The axis of each signal, keyed by ``signal_id``. Pass ``None`` to fill the
            :data:`TIME_AXIS_COLUMN` cell with each signal's axis type instead.

    Returns:
        A one-row frame: ``record_id``, ``time_axis``, then one object-typed column per signal
        holding that signal's values.

    Raises:
        ModuleNotFoundError: If pandas is not installed.
        TimeFValidationError: If the record holds no signal, if the signal names are not one column
            each, if ``values`` is missing a signal, or if ``axes`` is given and is missing one.
    """
    # A lane dependency: a caller that only reads values never needs pandas installed.
    try:
        import pandas as pd  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("the timenet pandas view needs pandas: pip install pandas") from exc

    signals = record.signals()
    if not signals:
        raise TimeFValidationError(f"record {_record_label(record)!r} holds no signal, so it has no frame")
    names = _column_names(record, signals)
    columns: dict[str, list[Any]] = {
        RECORD_ID_COLUMN: [_record_label(record)],
        TIME_AXIS_COLUMN: [{name: _axis_of(record, signal, axes) for name, signal in zip(names, signals, strict=True)}],
    }
    for name, signal in zip(names, signals, strict=True):
        found = values.get(signal.signal_id)
        if found is None:
            raise TimeFValidationError(
                f"record {_record_label(record)!r} signal {signal.name!r} (signal_id {signal.signal_id}) has no "
                f"values in the batch; read the whole record with reader.values_for()"
            )
        columns[name] = [found]
    return pd.DataFrame(columns, copy=False)


def iter_record_frames(
    reader: TimeFReader,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    worker_index: int = 0,
    num_workers: int = 1,
) -> Iterator[tuple[str, "pd.DataFrame"]]:
    """Walk a version and yield each record as an array-cell frame.

    Each round hydrates ``batch_size`` records and reads every value they need in one pass, then
    hands the frames out one at a time. The resident set holds a batch rather than the corpus.

    Args:
        reader: An open reader.
        batch_size: How many records to hydrate and read values for at once.
        worker_index: Which slice of the corpus this caller wants, from 0.
        num_workers: How many disjoint slices the corpus is split into.

    Yields:
        Each record's id and its one-row frame, in stored order.
    """
    for batch in _record_batches(reader, batch_size, worker_index, num_workers):
        wanted = sorted({signal.signal_id for record in batch for signal in record.signals()})
        values = reader.values_for(wanted)
        for record in batch:
            yield _record_label(record), record_array_frame(record, values)


class TimeFPandasDataset:
    """Shows a version's records as array-cell frames, one row per record, streaming by batch.

    Iterate it for the frames themselves and :meth:`items` for each frame with the id of the record
    it came from. Both read the corpus once, a batch of records at a time, so the first frame
    arrives without a read of the whole thing.
    """

    def __init__(
        self,
        reader: TimeFReader,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        worker_index: int = 0,
        num_workers: int = 1,
    ) -> None:
        """Wrap an open reader.

        Args:
            reader: The reader to walk. It stays open; this view does not close it.
            batch_size: How many records to hydrate and read values for at once.
            worker_index: Which slice of the corpus this view delivers, from 0.
            num_workers: How many disjoint slices the corpus is split into, so a DataLoader's
                workers together see every record exactly once.
        """
        self._reader = reader
        self._batch_size = batch_size
        self._worker_index = worker_index
        self._num_workers = num_workers

    def __iter__(self) -> Iterator["pd.DataFrame"]:
        """Yield every record in this view's slice as a one-row frame.

        Yields:
            One frame per record, in stored order.
        """
        for _record_id, frame in self.items():
            yield frame

    def items(self) -> Iterator[tuple[str, "pd.DataFrame"]]:
        """Walk every record in this view's slice, keeping the id it was built under.

        Returns:
            An iterator over each record's id and its one-row frame, in stored order.
        """
        return iter_record_frames(
            self._reader,
            batch_size=self._batch_size,
            worker_index=self._worker_index,
            num_workers=self._num_workers,
        )


def _record_batches(
    reader: TimeFReader, batch_size: int, worker_index: int, num_workers: int
) -> Iterator[list[RecordView]]:
    """Group a reader's records back into batches, so one values read serves many records.

    Args:
        reader: An open reader.
        batch_size: How many records per batch.
        worker_index: Which slice of the corpus this caller wants.
        num_workers: How many slices the corpus is split into.

    Yields:
        One batch of hydrated records at a time. The last batch is short.
    """
    batch: list[RecordView] = []
    for record in reader.iter_records(batch_size=batch_size, worker_index=worker_index, num_workers=num_workers):
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _column_names(record: RecordView, signals: Sequence[SignalView]) -> tuple[str, ...]:
    """Return one column name per signal of a record.

    Args:
        record: The record, named in the error message.
        signals: Its signals, in column order.

    Returns:
        The signal names, with a name the record repeats carrying that signal's id as well.

    Raises:
        TimeFValidationError: If the names are not one column each, because two of them collide or
            one is a column the frame needs for itself.
    """
    counts = Counter(signal.name for signal in signals)
    names = tuple(
        signal.name if counts[signal.name] == 1 else f"{signal.name} ({_signal_label(signal)})" for signal in signals
    )
    if len(set(names)) != len(names) or set(names) & set(ARRAY_META_COLUMNS):
        raise TimeFValidationError(
            f"record {_record_label(record)!r} names the signals {sorted(set(names))}, which an array-cell frame "
            f"cannot give a column each; {ARRAY_META_COLUMNS} are the frame's own"
        )
    return names


def _axis_of(record: RecordView, signal: SignalView, axes: Mapping[int, TimeAxis] | None) -> TimeAxis | str:
    """Return what the ``time_axis`` cell says about one signal.

    Args:
        record: The record, named in the error message.
        signal: The signal whose axis is wanted.
        axes: The caller's axes, keyed by ``signal_id``, or ``None``.

    Returns:
        The signal's axis: the caller's when one was supplied, the reader's otherwise, and the bare
        axis type only for a version whose ``axes`` row this reader could not interpret.

    Raises:
        TimeFValidationError: If ``axes`` was given and does not cover this signal.
    """
    if axes is None:
        # The reader resolves the axis from the row it already joined, so the cell carries the real
        # period or endpoints rather than only the word "regular".
        return signal.time_axis if signal.time_axis is not None else signal.axis_type
    found = axes.get(signal.signal_id)
    if found is None:
        raise TimeFValidationError(
            f"record {_record_label(record)!r} signal {signal.name!r} (signal_id {signal.signal_id}) has no axis in "
            f"the mapping passed as axes="
        )
    return found


def _record_label(record: RecordView) -> str:
    """Return the id to put in the ``record_id`` cell.

    Args:
        record: The record being laid out.

    Returns:
        The id the record was built under, or its surrogate id when it was built without one.
    """
    return record.external_id if record.external_id is not None else str(record.record_id)


def _signal_label(signal: SignalView) -> str:
    """Return the id that tells two signals of one name apart.

    Args:
        signal: The signal being named.

    Returns:
        The id the signal was built under, or its surrogate id when it was built without one.
    """
    return signal.external_id if signal.external_id is not None else str(signal.signal_id)


__all__: Sequence[str] = (
    "ARRAY_META_COLUMNS",
    "DEFAULT_BATCH_SIZE",
    "RECORD_ID_COLUMN",
    "TIME_AXIS_COLUMN",
    "TimeFPandasDataset",
    "iter_record_frames",
    "record_array_frame",
)
