"""What the Time-MMD release states about each domain, and how to read the files of one.

A domain of Time-MMD ships three CSV files. The numerical file holds one row per period, with the
forecast target in a column named ``OT`` and a few covariates beside it. The two textual files,
``<Domain>_report.csv`` and ``<Domain>_search.csv``, hold one row per text excerpt, each dated by a
``start_date`` and an ``end_date`` so it aligns with the numerical rows.

:data:`DOMAINS` declares the domains this connector converts. Each declaration states the cadence
the file must keep, the columns that become signals, the columns whose single value describes the
whole record, and the forecasting protocol the Time-MMD paper evaluates with. The readers in this
module give the rows back as the file states them and check the shape the declaration promises.
They fill nothing in: a gap in the dates or a cell that does not parse raises.
"""

from collections.abc import Sequence
import csv
from dataclasses import dataclass
from datetime import date, timedelta
import math
from pathlib import Path
from typing import Literal

from timenet.errors import TimeFFormatError
from timenet.types import Domain, TimeSeriesSpec, ureg


# The git commit of https://github.com/AdityaLab/Time-MMD this connector reads, so that every
# rebuild converts the same files. The repository has no tagged release.
SOURCE_REPOSITORY = "AdityaLab/Time-MMD"
SOURCE_REVISION = "00281e2d86058286d5548b15a7670e8eda57ef62"

# The name of the forecast target column in every numerical file of the release.
TARGET_COLUMN = "OT"

# The two columns that date every row of a numerical or a textual file.
_START_DATE = "start_date"
_END_DATE = "end_date"

# The header of the two textual files. The first column is an unnamed row index that pandas wrote.
_TEXT_HEADER = ("", _START_DATE, _END_DATE, "fact", "preds")

ONE_DAY = timedelta(days=1)

TextKind = Literal["report", "search"]


@dataclass(frozen=True)
class Column:
    """One column of a numerical file that becomes a signal of the record."""

    header: str
    """The column name as the file writes it."""
    signal_name: str
    """The name the signal takes in TimeF."""
    spec: TimeSeriesSpec
    """What the column measures, which also states how to parse a cell."""


@dataclass(frozen=True)
class ConstantColumn:
    """A column of a numerical file whose one value describes the whole record.

    The reader checks that every row states the same value and gives it back once.
    """

    header: str
    """The column name as the file writes it."""
    key: str
    """The key of the record annotation that carries the value."""
    description: str
    """What the value states, for the annotation's descriptor."""


@dataclass(frozen=True)
class DomainSource:
    """What the release states about one domain, and how this connector reads it."""

    name: str
    """The folder and file stem the release uses, such as ``"Energy"``."""
    domain: Domain
    """The TimeNet domain the record belongs to."""
    frequency: str
    """The cadence in words, as the paper states it: ``"weekly"`` or ``"daily"``."""
    period: timedelta
    """The exact cadence of the numerical rows. Consecutive rows must be this far apart."""
    source_name: str
    """The name of the :class:`~timenet.dataset.Source` that groups the signals."""
    columns: tuple[Column, ...]
    """The columns that become signals, in the order the signals are declared."""
    constant_columns: tuple[ConstantColumn, ...]
    """The columns whose single value becomes a record annotation."""
    lookback: int
    """The context length, in steps, of the forecasting protocol the paper evaluates with."""
    horizons: tuple[int, ...]
    """The forecast horizons, in steps, of that protocol."""

    @property
    def numerical_file(self) -> str:
        """The repository-relative path of the numerical file."""
        return f"numerical/{self.name}/{self.name}.csv"

    def text_file(self, kind: TextKind) -> str:
        """Give the repository-relative path of one textual file.

        Args:
            kind: Which of the two textual files.

        Returns:
            The path, relative to the repository root.
        """
        return f"textual/{self.name}/{self.name}_{kind}.csv"


def _gasoline_column(region: str, signal_name: str) -> Column:
    """Declare one regional column of the Energy file.

    The Energy file names each covariate after the EIA series it comes from, with two spaces
    before the unit. Only the region differs between the names.

    Args:
        region: The region as the header spells it, such as ``"New England (PADD 1A)"``.
        signal_name: The name the signal takes.

    Returns:
        The column declaration.
    """
    return Column(
        header=f"Weekly {region} All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)",
        signal_name=signal_name,
        spec=GASOLINE_PRICE,
    )


# The unit registry carries no currency, so the spec states no unit and its name says dollars.
GASOLINE_PRICE = TimeSeriesSpec(
    spec_type="gasoline_price",
    name="Retail gasoline price, all grades and formulations (US dollars per gallon)",
    unit_value=None,
)

AQI = TimeSeriesSpec(
    spec_type="aqi",
    name="Air Quality Index",
    unit_value=ureg.dimensionless,
    dtype="int16",
)

# The six categories the EPA defines for the index, in order of severity. The release states
# five of them for New York; the codebook holds the sixth so that a later update still parses.
AQI_CATEGORY = TimeSeriesSpec(
    spec_type="aqi_category",
    name="AQI category",
    unit_value=None,
    dtype="enum",
    categories=("Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"),
)

# The six criteria pollutants the EPA computes the index from, as the release spells them.
AQI_DEFINING_PARAMETER = TimeSeriesSpec(
    spec_type="aqi_defining_parameter",
    name="Pollutant that defined the day's AQI",
    unit_value=None,
    dtype="enum",
    categories=("CO", "NO2", "Ozone", "PM2.5", "PM10", "SO2"),
)

AQI_DEFINING_SITE = TimeSeriesSpec(
    spec_type="aqi_defining_site",
    name="AQS id of the monitoring site that defined the day's AQI",
    unit_value=None,
    dtype="str",
)

AQI_SITES_REPORTING = TimeSeriesSpec(
    spec_type="aqi_sites_reporting",
    name="Number of monitoring sites reporting",
    unit_value=ureg.dimensionless,
    dtype="int16",
)

ENERGY = DomainSource(
    name="Energy",
    domain=Domain.ENERGY,
    frequency="weekly",
    period=timedelta(days=7),
    source_name="EIA weekly retail gasoline prices",
    columns=(
        Column(header=TARGET_COLUMN, signal_name=TARGET_COLUMN, spec=GASOLINE_PRICE),
        _gasoline_column("East Coast", "east_coast"),
        _gasoline_column("New England (PADD 1A)", "new_england"),
        _gasoline_column("Central Atlantic (PADD 1B)", "central_atlantic"),
        _gasoline_column("Lower Atlantic (PADD 1C)", "lower_atlantic"),
        _gasoline_column("Midwest", "midwest"),
        _gasoline_column("Gulf Coast", "gulf_coast"),
        _gasoline_column("Rocky Mountain", "rocky_mountain"),
        _gasoline_column("West Coast", "west_coast"),
    ),
    constant_columns=(),
    lookback=36,
    horizons=(12, 24, 36, 48),
)

ENVIRONMENT = DomainSource(
    name="Environment",
    domain=Domain.ENVIRONMENT,
    frequency="daily",
    period=ONE_DAY,
    source_name="EPA daily AQI",
    columns=(
        Column(header=TARGET_COLUMN, signal_name=TARGET_COLUMN, spec=AQI),
        Column(header="Category", signal_name="category", spec=AQI_CATEGORY),
        Column(header="Defining Parameter", signal_name="defining_parameter", spec=AQI_DEFINING_PARAMETER),
        Column(header="Defining Site", signal_name="defining_site", spec=AQI_DEFINING_SITE),
        Column(header="Number of Sites Reporting", signal_name="sites_reporting", spec=AQI_SITES_REPORTING),
    ),
    constant_columns=(
        ConstantColumn(
            header="CBSA",
            key="cbsa",
            description="The core-based statistical area the EPA reports the AQI for.",
        ),
        ConstantColumn(
            header="CBSA Code",
            key="cbsa_code",
            description="The Census code of that core-based statistical area.",
        ),
    ),
    lookback=96,
    horizons=(48, 96, 192, 336),
)

DOMAINS: tuple[DomainSource, ...] = (ENERGY, ENVIRONMENT)
"""The domains this connector converts, in record order."""


@dataclass(frozen=True)
class NumericalTable:
    """The rows of one numerical file, checked against its domain declaration."""

    first_date: date
    """The date of the first row. Every later row follows at the domain's period."""
    n_rows: int
    """How many rows the file holds."""
    values: dict[str, list[float | int | str]]
    """The parsed cells of each signal column, keyed by the column header."""
    constants: dict[str, str]
    """The one value of each constant column, keyed by the annotation key."""
    period: timedelta
    """The cadence of the rows, taken from the domain declaration."""

    @property
    def last_date(self) -> date:
        """The date of the last row."""
        return self.first_date + (self.n_rows - 1) * self.period


@dataclass(frozen=True)
class TextRow:
    """One row of a textual file: a dated excerpt with its facts and its predictions."""

    row: int
    """The row index the release wrote in the first column."""
    start: date
    """The first day the excerpt covers."""
    end: date
    """The last day the excerpt covers, inclusive."""
    fact: str
    """The objective facts the release extracted, verbatim."""
    prediction: str
    """The predictions the release extracted, verbatim: long-term, ``;``, short-term. ``"NA"`` in a
    half means it found none."""


def _parse_date(cell: str, field: str, path: Path, row_number: int) -> date:
    """Decode one ISO date cell.

    Args:
        cell: The cell as the file writes it.
        field: The column name, for the error message.
        path: The file, for the error message.
        row_number: The row of the file, counted from 1 with the header, for the error message.

    Returns:
        The date the cell states.

    Raises:
        TimeFFormatError: If the cell is not an ISO date.
    """
    try:
        return date.fromisoformat(cell)
    except ValueError as exc:
        raise TimeFFormatError(f"{path} row {row_number}: {field} holds {cell!r}, which is not an ISO date") from exc


def _parse_cell(cell: str, column: Column, path: Path, row_number: int) -> float | int | str:
    """Decode one signal cell according to the dtype its spec states.

    Args:
        cell: The cell as the file writes it.
        column: The column the cell belongs to.
        path: The file, for the error message.
        row_number: The row of the file, counted from 1 with the header, for the error message.

    Returns:
        The number or the text the cell states.

    Raises:
        TimeFFormatError: If the cell is empty, or does not parse as the dtype the spec states.
    """
    where = f"{path} row {row_number}: {column.header}"
    if not cell.strip():
        raise TimeFFormatError(f"{where} is empty, and this connector fills nothing in")
    dtype = column.spec.dtype
    try:
        if dtype in {"float32", "float64"}:
            value = float(cell)
            if not math.isfinite(value):
                raise TimeFFormatError(f"{where} holds {cell!r}, which is not a finite number")
            return value
        if dtype in {"int8", "int16", "int32", "uint8", "uint16", "uint32"}:
            return int(cell)
    except ValueError as exc:
        raise TimeFFormatError(f"{where} holds {cell!r}, which is not a {dtype}") from exc
    return cell


def _check_header(path: Path, found: Sequence[str] | None, required: tuple[str, ...]) -> None:
    """Check that a file names every required column.

    Args:
        path: The file, for the error message.
        found: The header the file states, or ``None`` for an empty file.
        required: The columns the declaration needs.

    Raises:
        TimeFFormatError: If the file is empty or lacks a required column.
    """
    if found is None:
        raise TimeFFormatError(f"{path} is empty")
    missing = [column for column in required if column not in found]
    if missing:
        raise TimeFFormatError(f"{path} lacks the columns {missing}; its header is {found}")


def read_numerical_table(path: Path, source: DomainSource) -> NumericalTable:
    """Read a numerical file and check it keeps the cadence its domain declares.

    Every row must date itself consistently: ``date`` equals ``start_date``, and ``end_date`` is
    the last day of the period. Consecutive rows must be exactly one period apart. A gap raises,
    because filling it would invent values.

    Args:
        path: The numerical CSV.
        source: The domain declaration the file must match.

    Returns:
        The parsed table.

    Raises:
        TimeFFormatError: If the file lacks a column, holds no row, dates a row inconsistently,
            skips or repeats a period, or holds a cell that does not parse. Also if a constant
            column changes value.
    """
    required = (
        "date",
        _START_DATE,
        _END_DATE,
        *(column.header for column in source.columns),
        *(constant.header for constant in source.constant_columns),
    )
    values: dict[str, list[float | int | str]] = {column.header: [] for column in source.columns}
    constants: dict[str, str] = {}
    first_date: date | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _check_header(path, reader.fieldnames, required)
        for index, row in enumerate(reader):
            # The physical line the row ends on, for messages a reader can look up in the file.
            row_number = reader.line_num
            start = _parse_date(row[_START_DATE], _START_DATE, path, row_number)
            end = _parse_date(row[_END_DATE], _END_DATE, path, row_number)
            expected = start if first_date is None else first_date + index * source.period
            # check that the date is the same as the start date and the end date is the last day of the period
            if row["date"] != row[_START_DATE] or end != start + source.period - ONE_DAY:
                raise TimeFFormatError(
                    f"{path} row {row_number}: dates ({row['date']}, {start}, {end}) do not describe one "
                    f"{source.frequency} period starting on the row's date"
                )
            # check that the start date is the same as the expected date to make sure the file is not skipping any dates
            if start != expected:
                raise TimeFFormatError(
                    f"{path} row {row_number}: starts on {start}, but the {source.frequency} cadence puts it on "
                    f"{expected}; this connector fills no gap"
                )
            first_date = first_date or start
            for column in source.columns:
                values[column.header].append(_parse_cell(row[column.header], column, path, row_number))
            for constant in source.constant_columns:
                _take_constant(constants, constant, row[constant.header], path, row_number)
    if first_date is None:
        raise TimeFFormatError(f"{path} holds no rows")
    return NumericalTable(
        first_date=first_date,
        n_rows=len(values[TARGET_COLUMN]),
        values=values,
        constants=constants,
        period=source.period,
    )


def _take_constant(constants: dict[str, str], column: ConstantColumn, cell: str, path: Path, row_number: int) -> None:
    """Record a constant column's value, or reject a row that states a different one.

    Args:
        constants: The values seen so far, keyed by annotation key.
        column: The constant column.
        cell: The cell as the file writes it.
        path: The file, for the error message.
        row_number: The row of the file, for the error message.

    Raises:
        TimeFFormatError: If an earlier row stated a different value.
    """
    seen = constants.setdefault(column.key, cell)
    if seen != cell:
        raise TimeFFormatError(
            f"{path} row {row_number}: {column.header} holds {cell!r}, but earlier rows hold {seen!r}; "
            f"one file describes one series"
        )


def read_text_rows(path: Path) -> list[TextRow]:
    """Read a textual file and give its rows as the release states them.

    Nothing here interprets the text. ``fact`` and ``preds`` come back verbatim, including the
    ``"NA"`` the release writes when it extracted nothing, and the ``;`` that joins a long-term
    prediction to a short-term one.

    Args:
        path: The report or search CSV.

    Returns:
        The rows, in file order.

    Raises:
        TimeFFormatError: If the header is not the one the release writes, a date does not parse,
            a row ends before it starts, or the row index is not a whole number.
    """
    rows: list[TextRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        # check that the header is of the expected format
        if header is None or tuple(header) != _TEXT_HEADER:
            raise TimeFFormatError(f"{path}: expected the header {list(_TEXT_HEADER)}, found {header}")
        for row in reader:
            # The physical line the row ends on. A quoted cell can span several lines, so this is
            # not the count of rows read.
            row_number = reader.line_num
            # check that the row matches the expected format
            if len(row) != len(_TEXT_HEADER):
                raise TimeFFormatError(f"{path} row {row_number}: holds {len(row)} cells, not {len(_TEXT_HEADER)}")
            try:
                source_row = int(row[0])
            except ValueError as exc:
                raise TimeFFormatError(f"{path} row {row_number}: the row index {row[0]!r} is not a number") from exc
            start = _parse_date(row[1], _START_DATE, path, row_number)
            end = _parse_date(row[2], _END_DATE, path, row_number)
            if end < start:
                raise TimeFFormatError(f"{path} row {row_number}: ends on {end}, before it starts on {start}")
            rows.append(TextRow(row=source_row, start=start, end=end, fact=row[3], prediction=row[4]))
    return rows
