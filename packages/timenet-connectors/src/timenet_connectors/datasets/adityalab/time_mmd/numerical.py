"""Read the numerical file of one domain into the signals of its record.

A numerical file is one CSV: a ``date`` column, the ``start_date`` and ``end_date`` that bound
the period each row covers, the ``OT`` column the release names as the target, and the other
variables of the domain. :mod:`~timenet_connectors.datasets.adityalab.time_mmd.domains` states
which columns become signals and what each one measures. Nothing here decides that.

The file states a cadence, and this module checks it. A regular axis states one period and
computes every time offset from it, so a row that broke the cadence would sit at the wrong
date with no warning. A build stops instead. It stops on a missing value for the same reason.
No spec of this connector is nullable, because neither domain holds a gap, and a gap that
appeared in a later snapshot must not read as a zero.
"""

from collections.abc import Sequence
import csv
from dataclasses import dataclass
from datetime import date, timedelta
from fractions import Fraction
import math
from pathlib import Path

import numpy as np

from timenet.dataset import TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.types import Annotation, TimeSeriesSpec
from timenet_connectors.datasets.adityalab.time_mmd.domains import DomainShape
from timenet_connectors.datasets.adityalab.time_mmd.timeline import ONE_DAY, US_PER_DAY, Timeline


# The three date columns every numerical file of the release carries.
DATE_COLUMN = "date"
START_COLUMN = "start_date"
END_COLUMN = "end_date"

# The dtypes whose values are text. Every other dtype of a spec is a NumPy number.
_TEXT_DTYPES = frozenset({"str", "enum"})


@dataclass(frozen=True)
class NumericalTable:
    """One numerical file read into memory: its dates, and the text of each column by header."""

    path: Path
    """The file the table was read from."""
    dates: tuple[date, ...]
    """The date of each row, in file order."""
    columns: dict[str, tuple[str, ...]]
    """The text of each column the shape names, by header, one entry for each row."""

    @property
    def first(self) -> date:
        """The date of the first row."""
        return self.dates[0]

    @property
    def last(self) -> date:
        """The date of the last row."""
        return self.dates[-1]


def read_table(path: Path, shape: DomainShape) -> NumericalTable:
    """Read one numerical file, and check that it holds what the shape states on the cadence it states.

    The release states each row's period twice: as its ``date``, and as the ``start_date`` and
    ``end_date`` beside it. The two must agree, and consecutive dates must sit one period apart.

    Args:
        path: The file.
        shape: What the domain's file holds.

    Returns:
        The table, with every column the shape names.

    Raises:
        TimeFFormatError: If the file holds no row, lacks a column the shape names, holds a row
            with a field missing, breaks the cadence, or bounds a row by dates other than its
            period.
    """
    needed = [
        DATE_COLUMN,
        START_COLUMN,
        END_COLUMN,
        *(one.column for one in shape.signals),
        *(one.column for one in shape.constants),
    ]
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or ()
        missing = [column for column in needed if column not in header]
        if missing:
            raise TimeFFormatError(f"{path.name}: lacks the columns {missing}")

        rows = list(reader)
    if not rows:
        raise TimeFFormatError(f"{path.name}: holds no row")

    # Line numbers count from 1, and the header is line 1.
    lines = range(2, 2 + len(rows))
    columns = {
        column: tuple(_field(row, column, path, line) for row, line in zip(rows, lines, strict=True))
        for column in needed
    }
    dates = tuple(
        _parse_date(text, path, DATE_COLUMN, line) for text, line in zip(columns[DATE_COLUMN], lines, strict=True)
    )

    period = timedelta(days=shape.period_days)
    for line, previous, current in zip(lines[1:], dates, dates[1:], strict=False):
        if current - previous != period:
            raise TimeFFormatError(
                f"{path.name} line {line}: {current.isoformat()} follows {previous.isoformat()}, and the cadence "
                f"of this domain is {shape.period_days} day(s)"
            )

    for line, day, start_text, end_text in zip(lines, dates, columns[START_COLUMN], columns[END_COLUMN], strict=True):
        start = _parse_date(start_text, path, START_COLUMN, line)
        end = _parse_date(end_text, path, END_COLUMN, line)
        if start != day or end != day + period - ONE_DAY:
            raise TimeFFormatError(
                f"{path.name} line {line}: the row of {day.isoformat()} states the period {start.isoformat()} to "
                f"{end.isoformat()}, and a period here is {shape.period_days} day(s) from its date"
            )

    return NumericalTable(path=path, dates=dates, columns=columns)


def build_series(
    table: NumericalTable, shape: DomainShape, timeline: Timeline, record_id: str
) -> tuple[TimeSeries, ...]:
    """Turn the columns of a table into the signals of a record, on one regular axis.

    Every signal shares the axis. Its period is the cadence of the file, and its origin is the
    number of periods between the record's zero and the first row, which is whole because the
    timeline was built on that cadence.

    Args:
        table: The file, as :func:`read_table` gives it.
        shape: What the domain's file holds.
        timeline: Where the record's zero sits.
        record_id: The record the signals belong to, which their ids start with.

    Returns:
        One series for each signal the shape names, in the shape's order.

    Raises:
        TimeFValidationError: If the first row does not sit a whole number of periods after the
            record's zero, so no regular axis can place it.
    """
    lead_days = (table.first - timeline.start).days
    if lead_days < 0 or lead_days % shape.period_days:
        raise TimeFValidationError(
            f"{table.path.name}: the first row, {table.first.isoformat()}, sits {lead_days} day(s) after the record's "
            f"zero, {timeline.start.isoformat()}, which is not a whole number of {shape.period_days}-day periods"
        )

    axis = RegularAxis(period_us=Fraction(shape.period_days * US_PER_DAY), start_index=lead_days // shape.period_days)
    source_id = shape.numerical_path(Path()).as_posix()
    return tuple(
        TimeSeries.from_values(
            _parse_values(one.spec, table.columns[one.column], table, one.column),
            spec=one.spec,
            signal=one.signal,
            time_axis=axis,
            source_id=source_id,
            time_series_id=f"{record_id}-{one.signal}",
        )
        for one in shape.signals
    )


def build_constants(table: NumericalTable, shape: DomainShape, record_id: str) -> list[Annotation]:
    """Turn the columns that hold one value on every row into annotations with no span.

    Args:
        table: The file, as :func:`read_table` gives it.
        shape: What the domain's file holds, and which columns are constant.
        record_id: The record the annotations belong to, which their ids start with.

    Returns:
        One annotation for each constant column, in the shape's order.

    Raises:
        TimeFFormatError: If a constant column holds more than one distinct value.
    """
    built: list[Annotation] = []
    for constant in shape.constants:
        distinct = sorted(set(table.columns[constant.column]))
        if len(distinct) != 1:
            raise TimeFFormatError(
                f"{table.path.name}: {constant.column!r} holds {len(distinct)} distinct values, of which "
                f"{distinct[:3]}, and a column that describes the whole domain holds one"
            )

        value: str | int = distinct[0]
        if constant.numeric:
            value = _whole_number(distinct[0], table.path, constant.column, line=2)
        built.append(
            Annotation(
                key=constant.key, value=value, description=constant.description, id=f"{record_id}-{constant.key}"
            )
        )
    return built


def _field(row: dict[str, str | None], column: str, path: Path, line: int) -> str:
    """Give one field of a row, which must be present.

    A short line gives ``None`` for the columns it lacks, and that is a broken file and not an
    empty value.

    Args:
        row: The row, as the CSV reader gives it.
        column: The header of the field.
        path: The file, for the error message.
        line: The line of the row in the file, counted from 1, for the error message.

    Returns:
        The text of the field.

    Raises:
        TimeFFormatError: If the row holds no such field.
    """
    text = row.get(column)
    if text is None:
        raise TimeFFormatError(f"{path.name} line {line}: holds no {column!r} field")

    return text


def _parse_date(text: str, path: Path, column: str, line: int) -> date:
    """Parse one ISO date.

    Args:
        text: The field.
        path: The file, for the error message.
        column: The header of the field, for the error message.
        line: The line of the row in the file, counted from 1, for the error message.

    Returns:
        The date.

    Raises:
        TimeFFormatError: If the field is not a date of the form ``YYYY-MM-DD``.
    """
    try:
        return date.fromisoformat(text.strip())
    except ValueError as exc:
        raise TimeFFormatError(
            f"{path.name} line {line}: {column!r} holds {text!r}, which is not a YYYY-MM-DD date"
        ) from exc


def _parse_values(
    spec: TimeSeriesSpec, texts: Sequence[str], table: NumericalTable, column: str
) -> np.ndarray | list[str]:
    """Parse the text of one column into the values its spec states.

    Args:
        spec: The spec of the signal the column becomes, which states the dtype.
        texts: The text of the column, one entry for each row.
        table: The file, for the error messages.
        column: The header of the column, for the error messages.

    Returns:
        The labels of a text spec as a list, or the numbers of a numeric spec as an array of
        the spec's dtype.

    Raises:
        TimeFFormatError: If a field is empty, holds a label the codebook of an enum spec does
            not name, holds a number that does not parse or is not finite, or holds a number
            that an integer dtype cannot carry.
    """
    for text, day in zip(texts, table.dates, strict=True):
        if not text.strip():
            raise TimeFFormatError(f"{table.path.name}: {column!r} holds no value on {day.isoformat()}")

    if spec.dtype in _TEXT_DTYPES:
        if spec.dtype == "enum":
            unknown = sorted({text for text in texts if text not in spec.categories})
            if unknown:
                raise TimeFFormatError(
                    f"{table.path.name}: {column!r} holds {unknown}, which the codebook {list(spec.categories)} does not name"
                )

        return list(texts)

    dtype = np.dtype(spec.dtype)
    numbers: list[float] = []
    for text, day in zip(texts, table.dates, strict=True):
        try:
            number = float(text)
        except ValueError as exc:
            raise TimeFFormatError(
                f"{table.path.name}: {column!r} holds {text!r} on {day.isoformat()}, which is not a number"
            ) from exc
        if not math.isfinite(number):
            raise TimeFFormatError(
                f"{table.path.name}: {column!r} holds {text!r} on {day.isoformat()}, and a value here is finite"
            )
        if np.issubdtype(dtype, np.integer):
            bounds = np.iinfo(spec.dtype)
            if not number.is_integer() or not bounds.min <= number <= bounds.max:
                raise TimeFFormatError(
                    f"{table.path.name}: {column!r} holds {text!r} on {day.isoformat()}, which is not a whole number "
                    f"between {bounds.min} and {bounds.max}"
                )
        numbers.append(number)
    return np.array(numbers, dtype=dtype)


def _whole_number(text: str, path: Path, column: str, line: int) -> int:
    """Parse one field that must hold a whole number.

    Args:
        text: The field.
        path: The file, for the error message.
        column: The header of the field, for the error message.
        line: The line of the row in the file, counted from 1, for the error message.

    Returns:
        The number.

    Raises:
        TimeFFormatError: If the field is not a whole number.
    """
    try:
        number = float(text)
    except ValueError as exc:
        raise TimeFFormatError(f"{path.name} line {line}: {column!r} holds {text!r}, which is not a number") from exc
    if not number.is_integer():
        raise TimeFFormatError(f"{path.name} line {line}: {column!r} holds {text!r}, which is not a whole number")

    return int(number)
