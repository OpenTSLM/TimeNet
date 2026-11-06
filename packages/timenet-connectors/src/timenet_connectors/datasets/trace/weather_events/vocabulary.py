"""The names TRACE gives its values: event labels, samplings, channels and their units.

Nothing here reads a file. These types say what a published value means, so the readers and the task
builders can name it instead of comparing strings.
"""

from enum import IntEnum, StrEnum, unique
from typing import Annotated

import numpy as np
import pyarrow as pa
from pydantic import BeforeValidator, Field, StrictInt, TypeAdapter

from timenet.errors import TimeNetConnectorError
from timenet.types import US_PER_S, Label, LabelSet, TimeSeriesSpec, ureg
from timenet_connectors.sources.signals import numeric_signal


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
class EventLabel(IntEnum):
    """A window's published event label, numbered as EVENT_MAP in the authors' ``src/common.py``.

    The nine NOAA event types are 0-8. EVENT_MAP maps 'None' to ``NO_EVENT``, which marks a window
    without a following event; the authors' loss and metrics ignore it, so it is not in ``EVENT_TYPES``.
    """

    NO_EVENT = -100
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
    def is_event(self) -> bool:
        """Whether the label names an event type, not no event."""
        return self is not EventLabel.NO_EVENT

    @property
    def text(self) -> str:
        """The type as EVENT_MAP and NOAA spell it, such as ``Thunderstorm Wind``."""
        return self.name.replace("_", " ").title()


PublishedEventLabel = Annotated[
    EventLabel,
    Field(strict=False),
    BeforeValidator(TypeAdapter(StrictInt).validate_python),
]
"""A published integer event code, converted to its enum without accepting strings, floats or booleans."""


EVENT_TYPES = LabelSet(
    name="event_type",
    labels=tuple(Label(name=label.text) for label in EventLabel if label.is_event),
)
"""The nine event types, in label order: the labels of the event-type task. The files do not list them."""


@unique
class SourceField(StrEnum):
    """A field name in the authors' files: a window's JSON keys, and a retrieval row's columns and keys.

    A channel's field is its ``Channel`` value.
    """

    STATION_ID = "station_id"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    LOCATION = "location"
    MODE = "mode"
    DATE = "DATE"
    EVENT_TYPE = "event_type"
    ID = "id"
    DESCRIPTION = "description"
    LABELS = "labels"
    EVENTS = "events"
    EVENT_ID = "event_id"
    NARRATIVE = "narrative"


@unique
class AnnotationKey(StrEnum):
    """The key of an annotation the connector builds."""

    CHANNEL_DESCRIPTION = "channel_description"


@unique
class MetadataKey(StrEnum):
    """A key of the metadata the connector writes on a Record, a Source or a task. Metadata is never model input."""

    # A Record: one window.
    STATION = "station"
    COORDINATES = "coordinates"
    SAMPLING = "sampling"
    WINDOW_START = "window_start"
    """The first timestamp, as published: station local clock time, with no time zone."""
    WINDOW_END = "window_end"
    """The last timestamp, as published."""
    # A window, in its Record and in its tasks.
    WINDOW_ID = "window_id"
    ANCHOR = "anchor"
    SOURCE_FILE = "source_file"
    COUNTY = "county"
    EVENT_TYPE = "event_type"
    COPY_OF = "copy_of"
    """The lowest id of the other windows with the same measurements, or ``None``. Only on a Record."""
    # A task.
    EVENT_TYPE_INDEX = "event_type_index"
    EVENT_ID = "event_id"
    """A NOAA event's id, which joins a retrieval task to its event-type task."""
    # The station Source: its channels in source order, which TimeF does not keep on read.
    CHANNEL_ORDER = "channel_order"
    # A channel_description annotation: the channel it describes.
    CHANNEL = "channel"


@unique
class Sampling(StrEnum):
    """A window's span and sampling cadence, as its ``mode`` field names it.

    Window ids come in anchor triplets 3k, 3k+1, 3k+2, one per sampling in member order.
    """

    HOURLY_7_DAYS = "7day_hourly"
    FOUR_HOURLY_28_DAYS = "28day_4hourly"
    DAILY_180_DAYS = "180day_daily"

    @property
    def cadence_us(self) -> int:
        """The time between two consecutive timestamps, in microseconds."""
        return _CADENCE_US[self]

    @classmethod
    def for_window_id(cls, window_id: int) -> "Sampling":
        """Return the sampling that a window's position in its anchor triplet gives it."""
        return tuple(cls)[window_id % len(cls)]

    @classmethod
    def anchor_of(cls, window_id: int) -> int:
        """Return the anchor triplet that a window's id puts it in."""
        return window_id // len(cls)


_HOUR_US = 3600 * US_PER_S
_CADENCE_US = {
    Sampling.HOURLY_7_DAYS: _HOUR_US,
    Sampling.FOUR_HOURLY_28_DAYS: 4 * _HOUR_US,
    Sampling.DAILY_180_DAYS: 24 * _HOUR_US,
}

# GHCN-h sky cover codes 00-10, after NCEI's hourly coverage code table. Codes 00-08 count oktas, the
# eighths of the sky that cloud covers. Codes 09 and 10 are not amounts. The paper's prompt (Appendix
# B.4) labels the same codes with mixed fractions.
SKY_COVER_LABELS = (
    "0 oktas",
    "1 okta",
    *(f"{oktas} oktas" for oktas in range(2, 9)),
    "sky obscured",
    "partial obscuration",
)
SKY_COVER_CODEBOOK = pa.array(SKY_COVER_LABELS, type=pa.string())


@unique
class Channel(StrEnum):
    """A quantity a station measures at every timestamp of a window, one Signal each.

    The value is the key the authors' JSON uses for the channel, which also names its Signal. Member
    order is the authors' channel order, which the descriptions and the digests follow.
    """

    TEMPERATURE = "temperature"
    PRECIPITATION = "precipitation"
    RELATIVE_HUMIDITY = "relative_humidity"
    VISIBILITY = "visibility"
    # The wind: the eastward and northward components of one vector.
    EASTWARD_WIND = "wind_u"
    NORTHWARD_WIND = "wind_v"
    SKY_COVER = "sky_code"

    @property
    def spec(self) -> TimeSeriesSpec:
        """The channel's TimeF spec: what it measures, and in which unit."""
        return _SPECS[self]

    def parse_values(self, values: object, count: int) -> np.ndarray:
        """Convert schema-checked channel values, checking their count and numeric fidelity.

        Returns:
            Float64 values, or int64 sky cover codes, one per timestamp.

        Raises:
            TimeNetConnectorError: If the values are not one per timestamp, are not finite, or hold a code
                outside their numeric representation.
        """
        if not isinstance(values, list) or len(values) != count:
            raise TimeNetConnectorError(f"{self} must be a list of {count} values, one per timestamp")
        if self is Channel.SKY_COVER:
            return np.array(values, dtype=np.int64)
        # Preserve published precision for digest comparisons, then allow explicit
        # float-to-float rounding when storing the declared float32 channels.
        return numeric_signal(values, self, spec=self.spec.model_copy(update={"dtype": "float64"})).to_numpy()

    def to_arrow(self, values: np.ndarray) -> pa.Array:
        """Return checked values as the channel stores them: float32, or sky cover codes as codebook indices."""
        if self is Channel.SKY_COVER:
            # The published codes are already indices into the codebook.
            return pa.DictionaryArray.from_arrays(pa.array(values.astype(np.int32)), SKY_COVER_CODEBOOK)
        return numeric_signal(values, self, spec=self.spec, casting="float_rounding")


# Units follow the Hugging Face release's field table; the archive itself states none. In 4-hourly
# and daily windows each value is the mean of the hourly values, so precipitation there is a mean
# hourly amount, not the period's total. Values carry at most two decimals, so float32 storage keeps
# them.
_SPECS = {
    Channel.TEMPERATURE: TimeSeriesSpec(spec_type="air_temperature", name="Air temperature", unit_value=ureg.degC),
    Channel.PRECIPITATION: TimeSeriesSpec(spec_type="precipitation", name="Precipitation", unit_value=ureg.mm),
    Channel.RELATIVE_HUMIDITY: TimeSeriesSpec(
        spec_type="relative_humidity", name="Relative humidity", unit_value=ureg.percent
    ),
    Channel.VISIBILITY: TimeSeriesSpec(spec_type="visibility", name="Visibility", unit_value=ureg.km),
    Channel.EASTWARD_WIND: TimeSeriesSpec(spec_type="eastward_wind", name="Eastward wind", unit_value=ureg.m / ureg.s),
    Channel.NORTHWARD_WIND: TimeSeriesSpec(
        spec_type="northward_wind", name="Northward wind", unit_value=ureg.m / ureg.s
    ),
    Channel.SKY_COVER: TimeSeriesSpec(
        spec_type="sky_cover",
        name="Sky cover",
        unit_value=ureg.dimensionless,
        dtype="enum",
        categories=SKY_COVER_LABELS,
    ),
}
