"""Readers for the ten-minute SCADA tables and the shutdown durations of the Hill of Towie release.

A year archive holds one CSV file for each table and month. A row is one turbine for one
ten-minute period, and its ``TimeStamp`` is the end of the period in UTC. The shutdown file gives
the seconds of shutdown of each turbine and period, with the start of the period in UTC. The
readers put both on one grid of period starts.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import zipfile

import numpy as np
import pyarrow as pa
from pyarrow import csv as pa_csv

from timenet.errors import TimeNetConnectorError


STEP_S = 600
STEPS_PER_DAY = 144
DAY_S = STEP_S * STEPS_PER_DAY

TIMESTAMP_COLUMN = "TimeStamp"
STATION_COLUMN = "StationId"
SHUTDOWN_MEMBER = "ShutdownDuration.csv"
SHUTDOWN_START_COLUMN = "TimeStamp_StartFormat"
SHUTDOWN_TURBINE_COLUMN = "TurbineName"
SHUTDOWN_SECONDS_COLUMN = "ShutdownDuration"


@dataclass(frozen=True)
class Grid:
    """The turbines and the ten-minute periods that the connector reads."""

    station_ids: tuple[int, ...]
    turbine_names: tuple[str, ...]
    start_s: int
    """The start of the first period, in Unix seconds."""
    steps: int

    def __post_init__(self) -> None:
        if len(self.station_ids) != len(self.turbine_names):
            raise TimeNetConnectorError("The grid needs one name for each station.")


def read_signals(
    archives: Sequence[Path], tables: Mapping[str, Sequence[str]], grid: Grid
) -> tuple[np.ndarray, np.ndarray]:
    """Read the columns of the tables onto the grid.

    A period without a row in the source stays ``NaN``. The first period of a month is in the file
    of that month and in the file of the month before. Such a repeated row must state the same
    values.

    Args:
        archives: The year archives.
        tables: The columns to read from each table, in the order of the result.
        grid: The turbines and periods to keep. A row of another period is skipped.

    Returns:
        The values with shape ``(turbines, steps, columns)``, and whether each table has a row for
        each turbine and period, with shape ``(tables, turbines, steps)``.

    Raises:
        TimeNetConnectorError: If an archive has no file for a table.
    """
    widths = [len(columns) for columns in tables.values()]
    values = np.full((len(grid.station_ids), grid.steps, sum(widths)), np.nan)
    present = np.zeros((len(tables), len(grid.station_ids), grid.steps), dtype=bool)
    offsets = np.cumsum([0, *widths])
    for archive in archives:
        with zipfile.ZipFile(archive) as opened:
            for index, (table, columns) in enumerate(tables.items()):
                pattern = re.compile(rf"(?:.*/)?{re.escape(table)}_\d{{4}}_\d{{2}}\.csv")
                members = sorted(name for name in opened.namelist() if pattern.fullmatch(name))
                if not members:
                    raise TimeNetConnectorError(f"{archive}: no file for table {table}")
                for member in members:
                    _read_member(
                        opened,
                        member,
                        columns,
                        grid,
                        values=values[:, :, offsets[index] : offsets[index + 1]],
                        present=present[index],
                    )
    return values, present


def _read_member(  # ruff: ignore[too-many-arguments]
    opened: zipfile.ZipFile, member: str, columns: Sequence[str], grid: Grid, *, values: np.ndarray, present: np.ndarray
) -> None:
    """Put the rows of one table file onto the grid.

    Raises:
        TimeNetConnectorError: If a row is off the ten-minute grid, has an unknown station, or
            repeats another row with other values.
    """
    types = {TIMESTAMP_COLUMN: pa.string(), STATION_COLUMN: pa.int64(), **dict.fromkeys(columns, pa.float64())}
    try:
        with opened.open(member) as stream:
            table = pa_csv.read_csv(
                stream, convert_options=pa_csv.ConvertOptions(include_columns=list(types), column_types=types)
            )
    except (pa.ArrowInvalid, KeyError) as error:
        raise TimeNetConnectorError(f"{member}: {error}") from error
    end_s = _parse_utc_seconds(table[TIMESTAMP_COLUMN].to_numpy(zero_copy_only=False), member)
    offset_s = end_s - STEP_S - grid.start_s
    if (offset_s % STEP_S).any():
        raise TimeNetConnectorError(f"{member}: a timestamp is not on the ten-minute grid")
    steps = offset_s // STEP_S
    stations = table[STATION_COLUMN].to_numpy(zero_copy_only=False)
    turbines = np.searchsorted(grid.station_ids, stations)
    known = turbines < len(grid.station_ids)
    known[known] = np.asarray(grid.station_ids)[turbines[known]] == stations[known]
    if not known.all():
        raise TimeNetConnectorError(f"{member}: unknown station {stations[~known][0]}")
    keep = (steps >= 0) & (steps < grid.steps)
    turbines, steps = turbines[keep], steps[keep]
    rows = np.column_stack([table[column].to_numpy(zero_copy_only=False) for column in columns])[keep]
    cells = turbines * grid.steps + steps
    if len(np.unique(cells)) != len(cells):
        raise TimeNetConnectorError(f"{member}: two rows for the same station and timestamp")
    repeated = present[turbines, steps]
    if not np.array_equal(values[turbines[repeated], steps[repeated]], rows[repeated], equal_nan=True):
        raise TimeNetConnectorError(f"{member}: a row repeats a row of an earlier file with other values")
    values[turbines, steps] = rows
    present[turbines, steps] = True


def read_shutdown_seconds(archive: Path, grid: Grid) -> np.ndarray:
    """Read the seconds of shutdown of each turbine and period of the grid.

    Args:
        archive: The archive with the shutdown durations.
        grid: The turbines and periods to keep. A row of another period is skipped.

    Returns:
        The seconds with shape ``(turbines, steps)``.

    Raises:
        TimeNetConnectorError: If the file cannot be read, a row is off the grid, names an unknown
            turbine, or has a duration outside a period, or a turbine and period of the grid has
            no row or two rows.
    """
    types = {
        SHUTDOWN_START_COLUMN: pa.timestamp("s", tz="UTC"),
        SHUTDOWN_TURBINE_COLUMN: pa.string(),
        SHUTDOWN_SECONDS_COLUMN: pa.int64(),
    }
    try:
        with zipfile.ZipFile(archive) as opened, opened.open(SHUTDOWN_MEMBER) as stream:
            table = pa_csv.read_csv(
                stream, convert_options=pa_csv.ConvertOptions(include_columns=list(types), column_types=types)
            )
    except (pa.ArrowInvalid, KeyError) as error:
        raise TimeNetConnectorError(f"{SHUTDOWN_MEMBER}: {error}") from error
    offset_s = table[SHUTDOWN_START_COLUMN].cast(pa.int64()).to_numpy() - grid.start_s
    if (offset_s % STEP_S).any():
        raise TimeNetConnectorError(f"{SHUTDOWN_MEMBER}: a timestamp is not on the ten-minute grid")
    names = table[SHUTDOWN_TURBINE_COLUMN].dictionary_encode().combine_chunks()
    index_of_name = {name: index for index, name in enumerate(grid.turbine_names)}
    unknown = [name for name in names.dictionary.to_pylist() if name not in index_of_name]
    if unknown:
        raise TimeNetConnectorError(f"{SHUTDOWN_MEMBER}: unknown turbine {unknown[0]!r}")
    turbines = np.array([index_of_name[name] for name in names.dictionary.to_pylist()])[names.indices.to_numpy()]
    seconds = table[SHUTDOWN_SECONDS_COLUMN].to_numpy()
    if ((seconds < 0) | (seconds > STEP_S)).any():
        raise TimeNetConnectorError(f"{SHUTDOWN_MEMBER}: a duration is outside 0 to {STEP_S} seconds")
    steps = offset_s // STEP_S
    keep = (steps >= 0) & (steps < grid.steps)
    counts = np.zeros((len(grid.station_ids), grid.steps), dtype=np.int64)
    np.add.at(counts, (turbines[keep], steps[keep]), 1)
    if (counts != 1).any():
        raise TimeNetConnectorError(f"{SHUTDOWN_MEMBER}: a turbine and period has no row or more than one row")
    result = np.zeros((len(grid.station_ids), grid.steps), dtype=np.int64)
    result[turbines[keep], steps[keep]] = seconds[keep]
    return result


def _parse_utc_seconds(texts: np.ndarray, member: str) -> np.ndarray:
    """Parse UTC timestamps. The source writes midnight with or without the time of day.

    Returns:
        The timestamps in Unix seconds.

    Raises:
        TimeNetConnectorError: If a text is not a timestamp.
    """
    try:
        return texts.astype("datetime64[s]").astype(np.int64)
    except ValueError as error:
        raise TimeNetConnectorError(f"{member}: {error}") from error
