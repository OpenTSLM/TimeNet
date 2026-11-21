"""Readers for the Greenbyte exports of the release: the ten-minute turbine table and the status log.

Both files start with nine comment lines. The third one names the turbine. All times are UTC.
"""

from collections.abc import Iterator, Mapping
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pv

from timenet.errors import TimeNetConnectorError
from timenet.types import US_PER_S, unix_us


COMMENT_LINES = 9
TURBINE_PREFIX = "# Turbine: "
TIMESTAMP_COLUMN = "# Date and time"
STEP_US = 600 * US_PER_S
STEPS_PER_DAY = 144
DAY_US = STEPS_PER_DAY * STEP_US
MISSING_END = "-"
SYSTEM_OK = "System OK"


class EventKind(StrEnum):
    """The two values of the ``Status`` column that this connector keeps."""

    STOP = "Stop"
    CURTAILMENT = "Curtailment"


@dataclass(frozen=True)
class StatusEvent:
    """One stop or curtailment entry of the status log, with Unix times in microseconds."""

    kind: EventKind
    start_us: int
    end_us: int
    message: str
    iec_category: str
    """The IEC category that the operator gave the entry, or an empty string."""


@dataclass(frozen=True)
class OpenStop:
    """A stop entry without an end time. The log does not state when the turbine ran again."""

    start_us: int
    message: str


@dataclass(frozen=True)
class StatusLog:
    """What this connector keeps of one status log, with Unix times in microseconds."""

    events: tuple[StatusEvent, ...]
    open_stops: tuple[OpenStop, ...]
    ok_times_us: tuple[int, ...]
    """The start of each ``System OK`` entry."""


@dataclass(frozen=True)
class TurbineDay:
    """The values of one turbine for one UTC day: 144 steps for each column."""

    start_us: int
    values: Mapping[str, pa.Array]


def read_turbine_name(path: Path) -> str:
    """Read the turbine name from the comment lines of an export.

    Args:
        path: A turbine table or a status log.

    Returns:
        The name, such as ``Kelmarsh 1``.

    Raises:
        TimeNetConnectorError: If no comment line names the turbine.
    """
    with path.open(encoding="utf-8") as stream:
        for _, line in zip(range(COMMENT_LINES), stream, strict=False):
            if line.startswith(TURBINE_PREFIX):
                return line.removeprefix(TURBINE_PREFIX).strip()
    raise TimeNetConnectorError(f"{path.name}: no comment line starts with {TURBINE_PREFIX!r}")


def read_turbine_days(path: Path, columns: tuple[str, ...]) -> Iterator[TurbineDay]:
    """Read the columns of a turbine table and cut them into UTC days.

    Args:
        path: The turbine table.
        columns: The source names of the value columns to read.

    Yields:
        Each day of the table, in time order. An empty cell of the source is a null.

    Raises:
        TimeNetConnectorError: If a column is absent, the table does not start at midnight, the
            timestamps are not ten minutes apart, or the last day is not complete.
    """
    try:
        table = pv.read_csv(
            path,
            read_options=pv.ReadOptions(skip_rows=COMMENT_LINES),
            convert_options=pv.ConvertOptions(
                include_columns=[TIMESTAMP_COLUMN, *columns],
                column_types={TIMESTAMP_COLUMN: pa.timestamp("us"), **dict.fromkeys(columns, pa.float64())},
            ),
        ).combine_chunks()
    except (pa.ArrowInvalid, KeyError) as exc:
        raise TimeNetConnectorError(f"{path.name}: {exc}") from exc
    times = table[TIMESTAMP_COLUMN].to_numpy().astype("datetime64[us]").astype(np.int64)
    if times.size == 0 or times.size % STEPS_PER_DAY != 0 or times[0] % DAY_US != 0:
        raise TimeNetConnectorError(f"{path.name}: the table must hold whole UTC days, got {times.size} rows")
    if np.any(np.diff(times) != STEP_US):
        raise TimeNetConnectorError(f"{path.name}: the timestamps are not ten minutes apart")
    for first in range(0, times.size, STEPS_PER_DAY):
        yield TurbineDay(
            start_us=int(times[first]),
            values={column: table[column].chunk(0).slice(first, STEPS_PER_DAY) for column in columns},
        )


def read_status_log(path: Path) -> StatusLog:
    """Read the stop and curtailment entries of a status log.

    The log also holds informational entries, warnings, and communication losses. This function
    skips them, except that it keeps the time of each ``System OK`` entry.

    Args:
        path: The status log.

    Returns:
        The entries with an end time, the stops without one, and the ``System OK`` times.

    Raises:
        TimeNetConnectorError: If an entry ends before it starts, or a curtailment has no end time.
    """
    kinds = {kind.value for kind in EventKind}
    events: list[StatusEvent] = []
    open_stops: list[OpenStop] = []
    ok_times_us: list[int] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for _ in range(COMMENT_LINES):
            next(stream, None)
        for number, row in enumerate(csv.DictReader(stream), start=1):
            if row["Message"] == SYSTEM_OK:
                ok_times_us.append(_parse_utc(row["Timestamp start"]))
            if row["Status"] not in kinds:
                continue
            kind, start_us = EventKind(row["Status"]), _parse_utc(row["Timestamp start"])
            if row["Timestamp end"] == MISSING_END:
                if kind is not EventKind.STOP:
                    raise TimeNetConnectorError(f"{path.name} data row {number}: a {kind} entry has no end time")
                open_stops.append(OpenStop(start_us=start_us, message=row["Message"]))
                continue
            end_us = _parse_utc(row["Timestamp end"])
            if end_us < start_us:
                raise TimeNetConnectorError(f"{path.name} data row {number}: the entry ends before it starts")
            events.append(
                StatusEvent(
                    kind=kind,
                    start_us=start_us,
                    end_us=end_us,
                    message=row["Message"],
                    iec_category=row["IEC category"],
                )
            )
    return StatusLog(events=tuple(events), open_stops=tuple(open_stops), ok_times_us=tuple(ok_times_us))


def _parse_utc(text: str) -> int:
    return unix_us(datetime.fromisoformat(text).replace(tzinfo=UTC))
