"""A case's telemetry files: what each holds, checked once, and its columns reread on demand.

There are two shapes of file:

- A grid file has a ``time`` column of Unix seconds and one numeric column per series, one row per
  period: the metrics (1 s) and the authors' derived series ``logts``, ``tracets_err`` and
  ``tracets_lat`` (15 s). A RE1-OB ``data.csv`` can hold a second, separate table to the right of the
  first, with its own clock ``time.1``. Each column belongs to the clock to its left.
- An event file has one row per event at irregular times: the logs, timed in Unix nanoseconds, and the
  traces, one span per row, timed in Unix microseconds.

``read_grid`` and ``read_events`` read a whole file, check it, and keep only its shape: columns, clocks,
row counts and first and last times. ``TableReader`` rereads a file when a Signal is read, and returns one
column as TimeNet stores it.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum, unique
from functools import lru_cache
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv

from timenet.errors import TimeFFormatError
from timenet.types import US_PER_S
from timenet_connectors.datasets.phamquiluan.rcaeval.cases import ZipMember


_NS_PER_US = 1000
_MAX_EXACT_INTEGER = 2**53
"""The largest integer magnitude a float64 holds exactly."""


@unique
class EventKind(StrEnum):
    """The two event files."""

    LOGS = "logs"
    TRACES = "traces"


LOG_TIME = "timestamp"
"""The logs' time column: Unix nanoseconds."""
LOG_SUB_MICROSECOND = "timestamp_sub_us_ns"
"""The Signal that keeps the nanoseconds past each log line's microsecond, which a time axis can't hold."""
TRACE_TIME = "startTime"
"""The traces' time column: the span's start, in Unix microseconds."""
TRACE_SOURCE_ROW = "source_row"
"""The Signal that keeps each span's row in ``traces.csv``, which is not in start-time order."""
TRACE_COLUMNS = (
    "time",
    "traceID",
    "spanID",
    "serviceName",
    "methodName",
    "operationName",
    "startTimeMillis",
    "startTime",
    "duration",
    "statusCode",
    "parentSpanID",
)
"""The columns of every ``traces.csv``, in file order."""
TRACE_NUMBERS = frozenset({"startTimeMillis", "duration", "statusCode"})
"""The trace columns that hold numbers. The others are text, including the minute label ``time``."""


@dataclass(frozen=True)
class Clock:
    """One time column of a grid file, after the rows without a time are set aside."""

    column: str
    first_s: int
    n_rows: int
    """Rows with a time: consecutive, one period apart."""
    untimed_rows: int
    """Rows without a time, which the Signals leave out."""
    untimed_values: int
    """Values in those rows, in this clock's columns: the values the Signals leave out."""


@dataclass(frozen=True)
class GridTable:
    """A grid file's shape: its series columns, in file order, each timed by one of its clocks."""

    member: ZipMember
    period_s: int
    columns: Mapping[str, str]
    """Each series column, in file order, and the clock column that times it."""
    clocks: Mapping[str, Clock]

    def first_us(self) -> int:
        """Return the earliest time of any clock, in Unix microseconds."""
        return min(clock.first_s for clock in self.clocks.values()) * US_PER_S


@dataclass(frozen=True)
class EventTable:
    """An event file's shape: its columns, in file order, how many events, and when the first and last are."""

    member: ZipMember
    kind: EventKind
    columns: tuple[str, ...]
    n_rows: int
    first_us: int
    """The earliest event's time, in Unix microseconds; a log line's is rounded down."""
    last_us: int


def read_grid(member: ZipMember, period_s: int) -> GridTable:
    """Read a grid file and check it: whole-second clocks, ``period_s`` apart, with no gap, and exact numbers.

    Rows without a time are counted, not dropped silently: the caller compares them with the pinned
    release.

    Args:
        member: The file.
        period_s: The period every clock must have.

    Returns:
        The file's shape.

    Raises:
        TimeFFormatError: If a clock skips or repeats a period, or a series holds a value TimeNet can't store
            exactly.
    """
    table = _read_grid_csv(member)
    columns, clocks = {}, {}
    for clock, series in _clock_blocks(member, table.column_names).items():
        times = table.column(clock)
        values = table.select(series).filter(times.is_null())
        seconds = times.drop_null().to_numpy()
        if len(seconds) == 0:
            raise TimeFFormatError(f"{member.name}: clock {clock!r} has no time")
        if not np.all(np.diff(seconds) == period_s):
            raise TimeFFormatError(f"{member.name}: clock {clock!r} isn't one row every {period_s} s with no gap")
        clocks[clock] = Clock(
            column=clock,
            first_s=int(seconds[0]),
            n_rows=len(seconds),
            untimed_rows=values.num_rows,
            untimed_values=sum(len(column) - column.null_count for column in values.columns),
        )
        for name in series:
            _as_float64(table.column(name), member, name)
            columns[name] = clock
    return GridTable(member=member, period_s=period_s, columns=columns, clocks=clocks)


def read_events(member: ZipMember, kind: EventKind) -> EventTable | None:
    """Read an event file and check its times.

    Log lines must be in time order. Spans may come in any order: ``TableReader`` sorts them.

    Args:
        member: The file.
        kind: Whether it holds logs or traces.

    Returns:
        The file's shape, or ``None`` if it holds only its header.

    Raises:
        TimeFFormatError: If the logs are out of time order, a trace's ``startTimeMillis`` isn't its
            ``startTime`` in milliseconds, or a value can't be stored exactly.
    """
    table = _read_events_csv(member, kind)
    if table.num_rows == 0:
        return None
    time = LOG_TIME if kind is EventKind.LOGS else TRACE_TIME
    if table.column(time).null_count:
        raise TimeFFormatError(f"{member.name}: {table.column(time).null_count} rows have no {time}")
    if kind is EventKind.LOGS:
        nanoseconds = table.column(LOG_TIME).to_numpy()
        if np.any(np.diff(nanoseconds) < 0):
            raise TimeFFormatError(f"{member.name}: log lines are not in timestamp order")
        first_us, last_us = int(nanoseconds[0]) // _NS_PER_US, int(nanoseconds[-1]) // _NS_PER_US
    else:
        start = table.column(TRACE_TIME).to_numpy()
        if not np.array_equal(table.column("startTimeMillis").to_numpy(), start // 1000):
            raise TimeFFormatError(f"{member.name}: startTimeMillis is not startTime in milliseconds")
        for name in TRACE_NUMBERS:
            _as_float64(table.column(name), member, name)
        first_us, last_us = int(start.min()), int(start.max())
    return EventTable(
        member=member,
        kind=kind,
        columns=tuple(table.column_names),
        n_rows=table.num_rows,
        first_us=first_us,
        last_us=last_us,
    )


class TableReader:
    """Reread a case's files when their Signals are read, and return one column as TimeNet stores it.

    A Record's Signals load one after another, so the reader keeps the last few files it parsed, which
    covers every file of one case. It assumes the zips don't change between ``convert`` and writing:
    it checks each column's length, not its values, again.
    """

    def __init__(self) -> None:
        self._grid = lru_cache(maxsize=2)(_read_grid_csv)
        self._events = lru_cache(maxsize=2)(_sorted_events)

    def grid_values(self, table: GridTable, column: str) -> pa.Array:
        """Return one series of a grid file: its values at the rows its clock times, as float64.

        Returns:
            The values, one per row of the clock.
        """
        parsed = self._grid(table.member)
        clock = table.clocks[table.columns[column]]
        timed = parsed.column(clock.column).is_valid()
        values = _as_float64(parsed.column(column).filter(timed), table.member, column)
        return _checked(values, clock.n_rows, table.member, column)

    def event_offsets(self, table: EventTable, origin_us: int) -> pa.Array:
        """Return each event's time, in microseconds from the Record's origin, in the order of its values.

        Returns:
            The int64 offsets: log lines rounded down to the microsecond, spans by their exact start.
        """
        parsed, _ = self._events(table.member, table.kind)
        times = parsed.column(LOG_TIME if table.kind is EventKind.LOGS else TRACE_TIME).to_numpy()
        if table.kind is EventKind.LOGS:
            # The column's buffer is read-only, so this makes a new array.
            times = np.floor_divide(times, _NS_PER_US)
        return _checked(pa.array(times - origin_us, type=pa.int64()), table.n_rows, table.member, "offsets")

    def event_values(self, table: EventTable, column: str) -> pa.Array:
        """Return one column of an event file, in time order: text as strings, numbers as float64.

        ``timestamp_sub_us_ns`` is each log line's nanoseconds past its microsecond, and ``source_row`` each
        span's row in the file.

        Returns:
            The column's values, one per event.
        """
        parsed, rows = self._events(table.member, table.kind)
        if column == LOG_SUB_MICROSECOND:
            values = pa.array(parsed.column(LOG_TIME).to_numpy() % _NS_PER_US, type=pa.uint16())
        elif column == TRACE_SOURCE_ROW:
            values = pa.array(rows, type=pa.uint32())
        elif table.kind is EventKind.TRACES and column in TRACE_NUMBERS:
            values = _as_float64(parsed.column(column), table.member, column)
        else:
            values = parsed.column(column).combine_chunks()
        return _checked(values, table.n_rows, table.member, column)


def event_signal_columns(table: EventTable) -> tuple[str, ...]:
    """Return the Signals of an event file, in file order: every column but its time, and what the time can't keep.

    Returns:
        For logs, every column but ``timestamp``, then ``timestamp_sub_us_ns``. For traces, every column but
        ``startTime``, then ``source_row``.
    """
    if table.kind is EventKind.LOGS:
        return (*(column for column in table.columns if column != LOG_TIME), LOG_SUB_MICROSECOND)
    return (*(column for column in table.columns if column != TRACE_TIME), TRACE_SOURCE_ROW)


def _read_grid_csv(member: ZipMember) -> pa.Table:
    """Parse a grid file: its clocks as int64, its series as published.

    A repeated header ``time`` is renamed ``time.1``, as pandas reads it.

    Returns:
        The table.

    Raises:
        TimeFFormatError: If a clock holds a value that isn't a whole number of seconds.
    """
    with zipfile.ZipFile(member.archive) as archive:
        with archive.open(member.name) as handle:
            header = handle.readline().decode().rstrip("\r\n").split(",")
        seen: Counter[str] = Counter()
        names = []
        for name in header:
            names.append(f"{name}.{seen[name]}" if seen[name] else name)
            seen[name] += 1
        clocks = [name for name in names if name.split(".")[0] == "time"]
        with archive.open(member.name) as handle:
            # A clock with empty rows is written as floats, such as 1685221356.0, so it is read as float64.
            table = pcsv.read_csv(
                handle,
                read_options=pcsv.ReadOptions(column_names=names, skip_rows=1),
                convert_options=pcsv.ConvertOptions(column_types=dict.fromkeys(clocks, pa.float64())),
            )
    for name in clocks:
        try:
            # A safe cast refuses a fraction, and float64 holds every Unix second exactly.
            whole = table.column(name).cast(pa.int64(), safe=True)
        except pa.ArrowInvalid as error:
            raise TimeFFormatError(f"{member.name}: clock {name!r} isn't whole Unix seconds") from error
        table = table.set_column(table.column_names.index(name), name, whole)
    return table


def _clock_blocks(member: ZipMember, names: list[str]) -> dict[str, list[str]]:
    """Group a grid file's columns under the clock to their left.

    Returns:
        Each clock column and its series columns, in file order.

    Raises:
        TimeFFormatError: If the first column isn't ``time``, or a clock has no series.
    """
    if not names or names[0] != "time":
        raise TimeFFormatError(f"{member.name}: the first column is not 'time'")
    blocks: dict[str, list[str]] = {}
    for name in names:
        if name == "time" or (name.startswith("time.") and name.removeprefix("time.").isdigit()):
            blocks[name] = []
        else:
            blocks[next(reversed(blocks))].append(name)
    empty = [clock for clock, series in blocks.items() if not series]
    if empty:
        raise TimeFFormatError(f"{member.name}: clock columns {empty} time no series")
    return blocks


def _read_events_csv(member: ZipMember, kind: EventKind) -> pa.Table:
    """Parse an event file: text as text, with only an empty field as missing, and times as int64.

    A log message can span lines inside quotes, so quoted newlines are part of the value.

    Returns:
        The table.

    Raises:
        TimeFFormatError: If a ``traces.csv`` doesn't have the trace columns.
    """
    with zipfile.ZipFile(member.archive) as archive:
        with archive.open(member.name) as handle:
            header = tuple(handle.readline().decode().rstrip("\r\n").split(","))
        if kind is EventKind.TRACES and header != TRACE_COLUMNS:
            raise TimeFFormatError(f"{member.name}: columns {header} are not the trace columns {TRACE_COLUMNS}")
        time = LOG_TIME if kind is EventKind.LOGS else TRACE_TIME
        types = {name: pa.int64() if name in {time, "startTimeMillis"} else pa.string() for name in header}
        if kind is EventKind.TRACES:
            types |= {"duration": pa.int64(), "statusCode": pa.float64()}
        with archive.open(member.name) as handle:
            return pcsv.read_csv(
                handle,
                parse_options=pcsv.ParseOptions(newlines_in_values=True),
                convert_options=pcsv.ConvertOptions(
                    column_types=types, null_values=[""], strings_can_be_null=True, quoted_strings_can_be_null=True
                ),
            )


def _sorted_events(member: ZipMember, kind: EventKind) -> tuple[pa.Table, np.ndarray]:
    """Parse an event file and put its rows in time order, keeping each row's place in the file.

    Log lines are already in order. Spans are sorted by start time with a stable sort, so spans that
    start together keep their file order.

    Returns:
        The rows in time order, and each one's row in the file.
    """
    table = _read_events_csv(member, kind)
    if kind is EventKind.LOGS:
        return table, np.arange(table.num_rows)
    rows = np.argsort(table.column(TRACE_TIME).to_numpy(), kind="stable")
    return table.take(pa.array(rows)), rows


def _as_float64(values: pa.ChunkedArray | pa.Array, member: ZipMember, column: str) -> pa.Array:
    """Return a numeric column as float64, refusing a value float64 can't hold exactly or a non-finite one.

    Returns:
        The values, with missing values kept as nulls.

    Raises:
        TimeFFormatError: If the column isn't numeric, holds an integer past 2**53, or holds an infinity or NaN.
    """
    array = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    if pa.types.is_null(array.type):
        return pa.nulls(len(array), type=pa.float64())
    if pa.types.is_integer(array.type):
        present = array.drop_null().to_numpy()
        if len(present) and np.abs(present).max() > _MAX_EXACT_INTEGER:
            raise TimeFFormatError(f"{member.name}: {column!r} holds an integer that float64 can't hold exactly")
        return array.cast(pa.float64())
    if not pa.types.is_floating(array.type):
        raise TimeFFormatError(f"{member.name}: {column!r} is {array.type}, not numbers")
    array = array.cast(pa.float64())
    if not np.isfinite(array.drop_null().to_numpy()).all():
        raise TimeFFormatError(f"{member.name}: {column!r} holds an infinite or NaN value")
    return array


def _checked(values: pa.Array, n_rows: int, member: ZipMember, column: str) -> pa.Array:
    if len(values) != n_rows:
        raise TimeFFormatError(f"{member.name}: {column!r} has {len(values)} values, expected {n_rows}")
    return values
