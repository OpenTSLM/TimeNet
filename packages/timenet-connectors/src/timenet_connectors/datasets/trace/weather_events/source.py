"""The TRACE archive as published: its pinned files, the authors' conventions, and the models of its rows."""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum, StrEnum, unique
import hashlib
import json
from pathlib import Path
from typing import Annotated, NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, Strict, StrictInt, StrictStr, StringConstraints

from timenet.errors import TimeFValidationError
from timenet.types import Split, TimeSeriesSpec, ureg


# The README's Google Drive link, as a direct download that skips the HTML confirmation page. The
# archive is 510,759,681 bytes (487 MiB) and unpacks to 1.75 GB.
SOURCE_URL = (
    "https://drive.usercontent.google.com/download?id=1hX4D91QbXa0UQlgf6Jnf-1ii96gfp1aY&export=download&confirm=t"
)
# Drive has no immutable revision, so pin the archive's bytes instead.
SOURCE_SHA256 = "a5d24d75a0b9f0f846a185caa46c4f77e0ef98272c12808df76fac25a181be58"


class Inventory(NamedTuple):
    """One source file's pinned counts."""

    rows: int
    tasks: int
    """Classification tasks (``pretrain``), forecasting tasks (``forecasting``), or rows with an event
    report (``retrieval``)."""


# Pinned inventory of the archive's split files, keyed by (task, split). The JSON row counts match
# Table 8 of the paper, and the pretrain task counts match classification/*_data/labels.npy.
SOURCE_FILES = {
    ("pretrain", "train"): Inventory(45339, 10110),
    ("pretrain", "val"): Inventory(12953, 2884),
    ("pretrain", "test"): Inventory(6484, 1452),
    ("forecasting", "train"): Inventory(6690, 6627),
    ("forecasting", "val"): Inventory(1914, 1900),
    ("forecasting", "test"): Inventory(957, 947),
    ("retrieval", "train"): Inventory(42180, 13787),
    ("retrieval", "test"): Inventory(2006, 659),
}
# Distinct measurement windows among the 74,337 windows of the full archive, one Record each.
DISTINCT_WINDOWS = 63366
# Fewest timestamps a window needs to get an event-type or forecasting task. The authors'
# classification arrays (classification/*_data/), Stage 2 retrieval files, and forecasting loader
# (load_forecasting_from_json) keep only windows at least this long.
MIN_TASK_VALUES = 100
# The values at the end of each forecasting window that form its target. It counts values, not time:
# 24 hours in a 7day_hourly window, 24 days in a 180day_daily one. configs/finetune.yaml says 7, but
# forecast_finetune.py replaces it with its --forecast_horizon argument, which defaults to 24, and
# the authors' README runs it without that argument.
FORECAST_HORIZON = 24


@unique
class EventType(IntEnum):
    """The nine NOAA event types, numbered as EVENT_MAP in the authors' ``src/common.py``.

    EVENT_MAP also maps 'None' to ``NO_EVENT``, which marks non-event windows; the authors'
    loss and metrics ignore it, so it is not a member.
    """

    LIGHTNING = 0
    DEBRIS_FLOW = 1
    FLASH_FLOOD = 2
    HEAVY_RAIN = 3
    TORNADO = 4
    FUNNEL_CLOUD = 5
    HAIL = 6
    FLOOD = 7
    THUNDERSTORM_WIND = 8

    @property
    def label(self) -> str:
        """Return the type as EVENT_MAP and NOAA spell it, such as ``Thunderstorm Wind``."""
        return self.name.replace("_", " ").title()


# EVENT_MAP's label for a window without a following event.
NO_EVENT = -100
EVENT_SCHEMA = "trace-noaa-event-types"
_SPLITS = {"train": Split.TRAIN, "val": Split.VALIDATION, "test": Split.TEST}


@unique
class Mode(StrEnum):
    """A window's span and sampling cadence, as its ``mode`` field names it.

    Window ids come in anchor triplets 3k, 3k+1, 3k+2, one per mode in member order.
    """

    HOURLY_7_DAYS = "7day_hourly"
    FOUR_HOURLY_28_DAYS = "28day_4hourly"
    DAILY_180_DAYS = "180day_daily"


MODES = tuple(Mode)
CADENCE_US = {
    Mode.HOURLY_7_DAYS: 3_600_000_000,
    Mode.FOUR_HOURLY_28_DAYS: 14_400_000_000,
    Mode.DAILY_180_DAYS: 86_400_000_000,
}
# GHCN-h sky cover codes 00-10, after NCEI's hourly coverage code table: 00-08 count oktas, and 09 and
# 10 are not amounts. The paper's prompt (Appendix B.4) labels the same codes with mixed fractions.
SKY_COVER = (
    "0 oktas",
    "1 okta",
    "2 oktas",
    "3 oktas",
    "4 oktas",
    "5 oktas",
    "6 oktas",
    "7 oktas",
    "8 oktas",
    "sky obscured",
    "partial obscuration",
)
SKY_COVER_LABELS = pa.array(SKY_COVER, type=pa.string())
# Units follow the Hugging Face release's field table; the archive itself states none. In 4-hourly
# and daily windows each value is the mean of the hourly values, so precipitation there is a mean
# hourly amount, not the period's total.
CHANNELS = (
    # Values carry at most two decimals, so float32 storage keeps them.
    ("temperature", TimeSeriesSpec(spec_type="air_temperature", name="Air temperature", unit_value=ureg.degC)),
    ("precipitation", TimeSeriesSpec(spec_type="precipitation", name="Precipitation", unit_value=ureg.mm)),
    (
        "relative_humidity",
        TimeSeriesSpec(spec_type="relative_humidity", name="Relative humidity", unit_value=ureg.percent),
    ),
    ("visibility", TimeSeriesSpec(spec_type="visibility", name="Visibility", unit_value=ureg.km)),
    ("wind_u", TimeSeriesSpec(spec_type="eastward_wind", name="Eastward wind", unit_value=ureg.m / ureg.s)),
    ("wind_v", TimeSeriesSpec(spec_type="northward_wind", name="Northward wind", unit_value=ureg.m / ureg.s)),
    (
        "sky_code",
        TimeSeriesSpec(
            spec_type="sky_cover", name="Sky cover", unit_value=ureg.dimensionless, dtype="enum", categories=SKY_COVER
        ),
    ),
)
CHANNEL_NAMES = tuple(name for name, _ in CHANNELS)

# Strict types: a JSON true or "1" must not pass as a number, and a blank string not as text.
_Text = Annotated[StrictStr, StringConstraints(pattern=r"\S")]
_Number = Annotated[FiniteFloat, Strict()]
_Numbers = Annotated[list[_Number], Field(min_length=1)]
_SkyCodes = Annotated[list[Annotated[StrictInt, Field(ge=0, lt=len(SKY_COVER))]], Field(min_length=1)]
# Matches YYYY-MM-DDTHH:MM:SS, such as 2023-01-03T00:00:00, with no fractional seconds or time zone.
# This rejects inputs NumPy would otherwise parse, such as "now", "NaT", or a "Z" suffix.
_Timestamp = Annotated[StrictStr, StringConstraints(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}$")]
# Validated as a strict int: an IntEnum field would also take True, 1.0, or "1".
_EventIndex = Annotated[StrictInt, Field(ge=0, lt=len(EventType))]
_NoEvent = Annotated[StrictInt, Field(ge=NO_EVENT, le=NO_EVENT)]


class WindowRow(BaseModel):
    """One window as the JSON files publish it, without the cross-field checks."""

    # The title names the model in validation errors, which reach users with the file and key.
    model_config = ConfigDict(extra="forbid", frozen=True, title="TRACE window")

    station_id: _Text
    latitude: _Number
    longitude: _Number
    # Pretrain rows publish "COUNTY,STATE"; most forecasting rows publish ["County", "State"].
    location: _Text | Annotated[list[_Text], Field(min_length=1)]
    # Unlike an IntEnum, a StrEnum field already rejects JSON numbers and booleans.
    mode: Mode
    DATE: Annotated[list[_Timestamp], Field(min_length=1)]
    temperature: _Numbers
    precipitation: _Numbers
    relative_humidity: _Numbers
    visibility: _Numbers
    wind_u: _Numbers
    wind_v: _Numbers
    sky_code: _SkyCodes

    @property
    def place(self) -> str:
        """Return the location as one string: the published string, or the published list joined."""
        return self.location if isinstance(self.location, str) else ", ".join(self.location)

    def measurement_key(self) -> str:
        """Return a digest of what makes two windows the same measurements.

        Two windows match when their station, mode, timestamps, and every value of every channel are
        equal. Floats keep their shortest round-trip form, so equal values give equal text.

        Returns:
            The SHA-256 hex digest.
        """
        fields = [self.station_id, self.mode.value, self.DATE, *(getattr(self, name) for name in CHANNEL_NAMES)]
        return hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest()


class PretrainRow(WindowRow):
    """A pretrain window, which adds its event label."""

    model_config = ConfigDict(title="TRACE pretrain window")

    event_type: _NoEvent | _EventIndex


class Description(BaseModel):
    """One window's generated description, as the retrieval files publish it."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="TRACE description")

    DATE: _Text
    labels: _Text
    location: _Text
    temperature: _Text
    precipitation: _Text
    relative_humidity: _Text
    visibility: _Text
    wind_u: _Text
    wind_v: _Text
    sky_code: _Text

    @property
    def channels(self) -> tuple[str, ...]:
        """Return the seven channel sentences in channel order, as ``generate_channel_description`` lists them."""
        return tuple(getattr(self, name) for name in CHANNEL_NAMES)

    @property
    def text(self) -> str:
        """Return the description as the authors' ``generate_dsp`` composes it for their text encoder."""
        # Spacing as published: generate_dsp joins the channel sentences with " \n ".
        channels = " \n ".join(self.channels)
        return (
            f"Weather time series location: {self.location} Time range: {self.DATE} "
            f"The weather is {self.labels}. {channels}"
        )


class EventReport(BaseModel):
    """One event window's report, as the retrieval files publish it."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="TRACE event report")

    event_id: StrictInt
    event_type: _EventIndex
    narrative: _Text

    @property
    def text(self) -> str:
        """Return the report as the authors' ``generate_er`` writes it for their text encoder."""
        return f"The weather event is {EventType(self.event_type).label}. {self.narrative}"


@dataclass(frozen=True)
class TraceFileRef:
    """One extracted split file of the archive."""

    path: Path
    task: str
    """The authors' task folder: ``pretrain``, ``forecasting``, or ``retrieval``."""
    split: str
    """The authors' split name: ``train``, ``val``, or ``test``."""
    expected: Inventory | None = None
    """Pinned counts to check the file against; ``None`` skips the checks, as local fixtures do."""

    @property
    def timenet_split(self) -> Split:
        """Return the TimeNet split of the file's rows."""
        return _SPLITS[self.split]

    @property
    def archive_path(self) -> str:
        """Return the file's path inside the archive."""
        return split_file_path(self.task, self.split)


def split_file_path(task: str, split: str) -> str:
    """Return a split file's path inside the archive: JSON windows, or Stage 2 parquet.

    Returns:
        The path below the archive root.
    """
    return f"dataset/{task}/{split}.{'parquet' if task == 'retrieval' else 'json'}"


def check_inventory(file: TraceFileRef, rows: int, tasks: int) -> None:
    """Compare one file's counts with its pinned inventory, when the reference carries one.

    A reference without one, such as a test fixture, skips the check.

    Raises:
        TimeFValidationError: If the counts differ from the pinned archive's.
    """
    if file.expected is not None and (rows, tasks) != file.expected:
        raise TimeFValidationError(
            f"{file.path}: expected {file.expected.rows} rows and {file.expected.tasks} tasks or events, "
            f"found {rows} and {tasks}"
        )


# Rows per parquet read, so neither the checks nor the task stream hold a whole file's descriptions.
_RETRIEVAL_BATCH_ROWS = 1024


def retrieval_rows(file: TraceFileRef) -> Iterator[tuple[str, Description, EventReport | None]]:
    """Read and validate a Stage 2 parquet file one batch at a time.

    Yields:
        Each row's window key, its description, and its event report (``None`` for a non-event window).

    Raises:
        TimeFValidationError: If a window id, a description, or a report is invalid.
    """
    # Skip the series column: the Records already carry the values from the JSON files.
    columns = ["id", "description", "events"]
    with pq.ParquetFile(file.path) as parquet:
        for batch in parquet.iter_batches(batch_size=_RETRIEVAL_BATCH_ROWS, columns=columns):
            for window_id, description, event in zip(
                *(batch.column(name).to_pylist() for name in columns), strict=True
            ):
                if type(window_id) is not int:
                    raise TimeFValidationError(f"{file.path}: expected an integer window id, got {window_id!r}")
                try:
                    validated = Description.model_validate(description)
                    # A non-event window has a null report.
                    report = None if event is None else EventReport.model_validate(event)
                # A pydantic ValidationError is a ValueError.
                except ValueError as exc:
                    raise TimeFValidationError(f"{file.path} (id {window_id}): {exc}") from exc
                yield str(window_id), validated, report
