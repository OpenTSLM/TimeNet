"""Read the files of the CARE to Compare release into plain NumPy arrays.

The release has one directory per wind farm. Each directory holds ``feature_description.csv``,
``event_info.csv``, and one ``datasets/<event_id>.csv`` per event. All three are semicolon-separated.
An event file holds one turbine's SCADA rows at a 10-minute resolution: about a year of training rows
followed by the prediction rows that contain the event. The rows are sorted and lie on the 10-minute
grid, with a few skipped steps.

This module only parses and validates. It imports no TimeNet type, so the connector can turn its
output into Records and the tests can check the parsing on its own.
"""

from __future__ import annotations

from collections.abc import Iterator
import csv
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from timenet.errors import TimeNetConnectorError


PERIOD_US = 600_000_000
"""One step of the SCADA grid: 10 minutes, in microseconds."""

DESCRIPTIVE_COLUMNS = ("time_stamp", "asset_id", "id", "train_test", "status_type_id")
"""The leading columns of every event file. The sensor columns follow them."""

TRAIN = "train"
PREDICTION = "prediction"

STATISTIC_SUFFIXES = {"average": "avg", "minimum": "min", "maximum": "max", "std_dev": "std"}
"""The statistic names of ``feature_description.csv`` and their column suffixes in the event files."""
SUFFIX_STATISTICS = {suffix: statistic for statistic, suffix in STATISTIC_SUFFIXES.items()}

STATUS_MISSING = -1
"""The status value of a grid step without a row. The release uses the IDs 0 to 5."""

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# The unit strings of the three feature descriptions, as pint expressions. Wind farm A's file stores
# the degree sign as U+FFFD, so both spellings map to the same unit. The release states reactive
# power and reactive energy (kVAr, kvar, VArh, kvarh), a vibration unit (mG), and "DL" without a
# definition that pint can express without guessing, so the connector keeps their source string
# as an annotation instead.
UNIT_EXPRESSIONS: dict[str, str] = {
    "°C": "degree_Celsius",
    "�C": "degree_Celsius",
    "Celsius": "degree_Celsius",
    "°": "degree",
    "�": "degree",
    "deg": "degree",
    "m/s": "meter / second",
    "m/s^2": "meter / second ** 2",
    "rpm": "revolution / minute",
    "1/min": "revolution / minute",
    "rpm/s": "revolution / minute / second",
    "rad/s": "radian / second",
    "A": "ampere",
    "V": "volt",
    "Hz": "hertz",
    "mHz": "millihertz",
    "kW": "kilowatt",
    "Wh": "watt * hour",
    "kWh": "kilowatt * hour",
    "kVA": "kilovolt * ampere",
    "Nm": "newton * meter",
    "kNm": "kilonewton * meter",
    "%": "percent",
    "bar": "bar",
    "hPa": "hectopascal",
    "Pa": "pascal",
    "L/min": "liter / minute",
    "l/h": "liter / hour",
    "L": "liter",
}

# The fault component of each anomaly event whose free-text description names one unambiguously.
# The keys are the lowercased descriptions of wind farms A and B. Wind farm C's descriptions are
# operator notes that often name several components, so the connector derives no component for them.
FAULT_COMPONENTS: dict[tuple[str, str], str] = {
    ("A", "transformer failure"): "transformer",
    ("A", "hydraulic group"): "hydraulic",
    ("A", "gearbox failure"): "gearbox",
    ("A", "gearbox bearings damaged"): "gearbox",
    ("A", "generator bearing failure"): "generator_bearing",
    ("B", "high temperature in transformer cell"): "transformer",
    ("B", "rotor bearing 2 - damage"): "rotor_bearing",
    ("B", "turbine is stopped due to a main bearing damage"): "rotor_bearing",
    ("B", "turbine is in standstill since 01.08 due to rotorbearing damage."): "rotor_bearing",
}


def parse_timestamp_us(text: str) -> int:
    """Parse a release timestamp such as ``2023-08-12 09:50:00`` into microseconds since the epoch.

    The release states no time zone and shifts every year for anonymization, so the value is read
    as UTC and stands only for the position in time, not a true calendar date.

    Returns:
        The timestamp in microseconds since 1970-01-01 UTC.

    Raises:
        TimeNetConnectorError: If the text is not in the release format.
    """
    try:
        moment = datetime.strptime(text.strip(), TIMESTAMP_FORMAT).replace(tzinfo=UTC)
    except ValueError as exc:
        raise TimeNetConnectorError(f"timestamp {text!r} is not in the format {TIMESTAMP_FORMAT}") from exc
    return int(moment.timestamp()) * 1_000_000


def _parse_bool(text: str, *, field: str) -> bool:
    if text == "True":
        return True
    if text == "False":
        return False
    raise TimeNetConnectorError(f"{field} must be True or False, not {text!r}")


def _read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read a semicolon-separated file into its header and rows.

    Returns:
        The header and the data rows.

    Raises:
        TimeNetConnectorError: If the file is empty.
    """
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream, delimiter=";"))
    if not rows:
        raise TimeNetConnectorError(f"{path.name} is empty")
    return rows[0], [row for row in rows[1:] if row]


@dataclass(frozen=True)
class SensorDescription:
    """One row of ``feature_description.csv``."""

    sensor_name: str
    """The anonymized sensor name, such as ``sensor_11`` or ``wind_speed_3``."""
    statistics: tuple[str, ...]
    """The 10-minute statistics that the event files hold for the sensor, such as ``average``."""
    description: str
    """The release's description of the sensor."""
    unit: str
    """The unit string of the release. It can be empty."""
    is_angle: bool
    is_counter: bool

    @property
    def unit_expression(self) -> str | None:
        """The unit as a pint expression, or ``None`` when the release unit has no safe mapping."""
        return UNIT_EXPRESSIONS.get(self.unit)


FEATURE_DESCRIPTION_COLUMNS = ("sensor_name", "statistics_type", "description", "unit", "is_angle", "is_counter")


def read_sensor_descriptions(path: Path) -> dict[str, SensorDescription]:
    """Read a wind farm's ``feature_description.csv``.

    Returns:
        Each sensor name mapped to its description, in file order.

    Raises:
        TimeNetConnectorError: If the header is not the expected one, a row is malformed, a statistic
            is unknown, or a sensor repeats.
    """
    header, rows = _read_rows(path)
    if tuple(header) != FEATURE_DESCRIPTION_COLUMNS:
        raise TimeNetConnectorError(f"{path.name} header {header} is not {list(FEATURE_DESCRIPTION_COLUMNS)}")
    sensors: dict[str, SensorDescription] = {}
    for number, row in enumerate(rows, start=1):
        if len(row) != len(FEATURE_DESCRIPTION_COLUMNS):
            raise TimeNetConnectorError(f"{path.name} row {number} has {len(row)} fields")
        name, statistics_text, description, unit, is_angle, is_counter = (cell.strip() for cell in row)
        statistics = tuple(statistic.strip() for statistic in statistics_text.split(","))
        unknown = [statistic for statistic in statistics if statistic not in STATISTIC_SUFFIXES]
        if unknown:
            raise TimeNetConnectorError(f"{path.name} row {number}: unknown statistics {unknown}")
        if name in sensors:
            raise TimeNetConnectorError(f"{path.name} row {number}: sensor {name} repeats")
        sensors[name] = SensorDescription(
            sensor_name=name,
            statistics=statistics,
            description=description,
            unit=unit,
            is_angle=_parse_bool(is_angle, field=f"{path.name} row {number} is_angle"),
            is_counter=_parse_bool(is_counter, field=f"{path.name} row {number} is_counter"),
        )
    return sensors


def split_feature_column(column: str) -> tuple[str, str]:
    """Split an event-file column such as ``wind_speed_3_avg`` into the sensor and the statistic.

    Returns:
        The sensor name and the statistic name, such as ``("wind_speed_3", "average")``.

    Raises:
        TimeNetConnectorError: If the column does not end in a known statistic suffix.
    """
    sensor, _, suffix = column.rpartition("_")
    if not sensor or suffix not in SUFFIX_STATISTICS:
        raise TimeNetConnectorError(f"column {column!r} does not end in one of {sorted(SUFFIX_STATISTICS)}")
    return sensor, SUFFIX_STATISTICS[suffix]


@dataclass(frozen=True)
class EventInfo:
    """One row of ``event_info.csv``."""

    event_id: int
    asset_id: int
    label: str
    """``anomaly`` or ``normal``."""
    start_us: int
    """The first timestamp of the event window."""
    end_us: int
    """The last timestamp of the event window. For an anomaly event, the turbine fault starts here."""
    description: str | None
    """The operator's note on the root cause of an anomaly event, if the release gives one."""

    def fault_component(self, wind_farm: str) -> str | None:
        """The component that the description names, for wind farms A and B.

        Returns:
            A label of the fault component label set, or ``None`` when the event is normal or the
            description names no single component.
        """
        if self.label != "anomaly" or self.description is None:
            return None
        return FAULT_COMPONENTS.get((wind_farm, self.description.strip().lower()))


EVENT_LABELS = ("anomaly", "normal")


def read_event_info(path: Path) -> dict[int, EventInfo]:
    """Read a wind farm's ``event_info.csv``.

    Wind farm A names the asset column ``asset``. Wind farms B and C name it ``asset_id``.

    Returns:
        Each event ID mapped to its row, in file order.

    Raises:
        TimeNetConnectorError: If a column is missing, a row is malformed, or an event ID repeats.
    """
    header, rows = _read_rows(path)
    asset_column = "asset_id" if "asset_id" in header else "asset"
    required = (asset_column, "event_id", "event_label", "event_start", "event_end", "event_description")
    missing = [column for column in required if column not in header]
    if missing:
        raise TimeNetConnectorError(f"{path.name} lacks the columns {missing}")
    index = {column: header.index(column) for column in required}
    events: dict[int, EventInfo] = {}
    for number, row in enumerate(rows, start=1):
        if len(row) != len(header):
            raise TimeNetConnectorError(f"{path.name} row {number} has {len(row)} fields, not {len(header)}")
        try:
            event_id = int(row[index["event_id"]])
            asset_id = int(row[index[asset_column]])
        except ValueError as exc:
            raise TimeNetConnectorError(f"{path.name} row {number}: {exc}") from exc
        label = row[index["event_label"]].strip()
        if label not in EVENT_LABELS:
            raise TimeNetConnectorError(f"{path.name} row {number}: event_label {label!r} is not one of {EVENT_LABELS}")
        start_us = parse_timestamp_us(row[index["event_start"]])
        end_us = parse_timestamp_us(row[index["event_end"]])
        if end_us < start_us:
            raise TimeNetConnectorError(f"{path.name} row {number}: the event ends before it starts")
        if event_id in events:
            raise TimeNetConnectorError(f"{path.name} row {number}: event {event_id} repeats")
        description = row[index["event_description"]].strip() or None
        events[event_id] = EventInfo(
            event_id=event_id,
            asset_id=asset_id,
            label=label,
            start_us=start_us,
            end_us=end_us,
            description=description,
        )
    return events


@dataclass(frozen=True)
class EventTable:
    """One event file placed on its 10-minute grid.

    Grid step ``i`` is the timestamp ``start_us + i * PERIOD_US``. A step that the file skips holds
    NaN in ``values`` and :data:`STATUS_MISSING` in ``status``.
    """

    asset_id: int
    start_us: int
    """The first timestamp of the file."""
    feature_columns: tuple[str, ...]
    """The sensor columns, in file order."""
    values: np.ndarray
    """float32, shape ``(n_features, n_steps)``. NaN marks an empty cell or a skipped step."""
    status: np.ndarray
    """int8, shape ``(n_steps,)``: the status ID of each step, or :data:`STATUS_MISSING`."""
    train_steps: tuple[int, int]
    """The grid steps ``[first, last]`` of the training rows."""
    prediction_steps: tuple[int, int]
    """The grid steps ``[first, last]`` of the prediction rows."""

    @property
    def n_steps(self) -> int:
        """The number of 10-minute steps from the first to the last row."""
        return int(self.status.shape[0])

    def step_of(self, timestamp_us: int) -> int:
        """The grid step of a timestamp.

        Returns:
            The step index.

        Raises:
            TimeNetConnectorError: If the timestamp is off the grid or outside the file.
        """
        offset = timestamp_us - self.start_us
        if offset % PERIOD_US or not 0 <= offset // PERIOD_US < self.n_steps:
            raise TimeNetConnectorError(f"timestamp {timestamp_us} us is off the grid or outside the file")
        return offset // PERIOD_US


def _to_float32(cells: list[list[str]], path: Path, first_row: int) -> np.ndarray:
    """Convert a block of sensor cells to float32, with NaN for empty cells.

    Returns:
        The block, shape ``(n_rows, n_features)``.

    Raises:
        TimeNetConnectorError: If a cell is not a number.
    """
    try:
        return np.array([[cell if cell else "nan" for cell in row] for row in cells], dtype=np.float32)
    except ValueError as exc:
        raise TimeNetConnectorError(f"{path.name} rows {first_row}+: {exc}") from exc


def _read_header(reader: Iterator[list[str]], path: Path) -> tuple[str, ...]:
    """Read and check the header of an event file.

    Returns:
        The sensor columns that follow the descriptive columns.

    Raises:
        TimeNetConnectorError: If the file is empty, the descriptive columns do not lead, or a column
            repeats.
    """
    header = next(reader, None)
    if header is None:
        raise TimeNetConnectorError(f"{path.name} is empty")
    if tuple(header[: len(DESCRIPTIVE_COLUMNS)]) != DESCRIPTIVE_COLUMNS:
        raise TimeNetConnectorError(f"{path.name} does not start with the columns {list(DESCRIPTIVE_COLUMNS)}")
    feature_columns = tuple(header[len(DESCRIPTIVE_COLUMNS) :])
    if len(set(feature_columns)) != len(feature_columns):
        raise TimeNetConnectorError(f"{path.name} repeats a column")
    return feature_columns


@dataclass
class _EventRows:
    """The rows of an event file, before they are placed on the grid."""

    times: list[int] = field(default_factory=list)
    statuses: list[int] = field(default_factory=list)
    is_train: list[bool] = field(default_factory=list)
    assets: set[str] = field(default_factory=set)
    blocks: list[np.ndarray] = field(default_factory=list)

    def add_descriptive(self, row: list[str], number: int, path: Path) -> None:
        """Check and keep the descriptive cells of one row.

        Raises:
            TimeNetConnectorError: If the row ID is not the row index, or a cell is malformed.
        """
        if row[2] != str(number):
            raise TimeNetConnectorError(f"{path.name} row {number}: id {row[2]!r} is not the row index")
        if row[3] not in {TRAIN, PREDICTION}:
            raise TimeNetConnectorError(f"{path.name} row {number}: train_test {row[3]!r} is unknown")
        try:
            status = int(row[4])
        except ValueError as exc:
            raise TimeNetConnectorError(f"{path.name} row {number}: status_type_id {row[4]!r}") from exc
        self.times.append(parse_timestamp_us(row[0]))
        self.assets.add(row[1])
        self.is_train.append(row[3] == TRAIN)
        self.statuses.append(status)


def _read_event_rows(path: Path, block_rows: int) -> tuple[tuple[str, ...], _EventRows]:
    """Read the header and the rows of an event file.

    Returns:
        The sensor columns and the rows.

    Raises:
        TimeNetConnectorError: If the header or a row is malformed.
    """
    rows = _EventRows()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream, delimiter=";")
        feature_columns = _read_header(reader, path)
        width = len(DESCRIPTIVE_COLUMNS) + len(feature_columns)
        cells: list[list[str]] = []
        for number, row in enumerate(reader):
            if not row:
                continue
            if len(row) != width:
                raise TimeNetConnectorError(f"{path.name} row {number}: {len(row)} fields, not {width}")
            rows.add_descriptive(row, number, path)
            cells.append(row[len(DESCRIPTIVE_COLUMNS) :])
            if len(cells) == block_rows:
                rows.blocks.append(_to_float32(cells, path, number - block_rows + 1))
                cells = []
        if cells:
            rows.blocks.append(_to_float32(cells, path, len(rows.times) - len(cells)))
    if not rows.times:
        raise TimeNetConnectorError(f"{path.name} has no rows")
    if len(rows.assets) != 1:
        raise TimeNetConnectorError(f"{path.name} holds the assets {sorted(rows.assets)}, not one")
    return feature_columns, rows


def _grid_steps(rows: _EventRows, path: Path) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Map each row to its step on the 10-minute grid.

    Returns:
        The step of each row, and the first and last steps of the training and prediction rows.

    Raises:
        TimeNetConnectorError: If the timestamps are off the grid or not rising, or the training rows
            do not all precede the prediction rows.
    """
    epochs = np.array(rows.times, dtype=np.int64)
    offsets = epochs - epochs[0]
    if np.any(offsets % PERIOD_US) or np.any(np.diff(offsets) <= 0):
        raise TimeNetConnectorError(f"{path.name}: timestamps are off the 10-minute grid or not rising")
    steps = offsets // PERIOD_US
    train = np.array(rows.is_train, dtype=bool)
    train_rows = np.flatnonzero(train)
    prediction_rows = np.flatnonzero(~train)
    if not len(train_rows) or not len(prediction_rows):
        raise TimeNetConnectorError(f"{path.name} lacks training or prediction rows")
    if train_rows[-1] > prediction_rows[0]:
        raise TimeNetConnectorError(f"{path.name}: a training row follows a prediction row")
    return (
        steps,
        (int(steps[train_rows[0]]), int(steps[train_rows[-1]])),
        (int(steps[prediction_rows[0]]), int(steps[prediction_rows[-1]])),
    )


def read_event_table(path: Path, *, block_rows: int = 4096) -> EventTable:
    """Read one event file and place its rows on the 10-minute grid.

    The parser checks the properties the connector relies on: the descriptive columns lead, the row
    IDs count up from zero, the timestamps rise on the 10-minute grid, every row has one asset, and the
    training rows precede the prediction rows.

    Args:
        path: The event file.
        block_rows: How many rows to convert to float at a time, to bound memory.

    Returns:
        The file's values on the grid.
    """
    feature_columns, rows = _read_event_rows(path, block_rows)
    steps, train_steps, prediction_steps = _grid_steps(rows, path)
    n_steps = int(steps[-1]) + 1
    values = np.full((len(feature_columns), n_steps), np.nan, dtype=np.float32)
    values[:, steps] = np.concatenate(rows.blocks, axis=0).T
    status = np.full(n_steps, STATUS_MISSING, dtype=np.int8)
    status[steps] = np.array(rows.statuses, dtype=np.int8)
    return EventTable(
        asset_id=int(next(iter(rows.assets))),
        start_us=int(rows.times[0]),
        feature_columns=feature_columns,
        values=values,
        status=status,
        train_steps=train_steps,
        prediction_steps=prediction_steps,
    )
