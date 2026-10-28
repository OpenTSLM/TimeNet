"""The authors' conventions: event labels, samplings, sky cover codes, channels, and task thresholds."""

from enum import IntEnum, StrEnum, unique

from timenet.types import TimeSeriesSpec, ureg


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
    without a following event; the authors' loss and metrics ignore it, so it is not in the vocabulary.
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
        """Return whether the label names an event type, rather than no event."""
        return self is not EventLabel.NO_EVENT

    @property
    def label(self) -> str:
        """Return the type as EVENT_MAP and NOAA spell it, such as ``Thunderstorm Wind``."""
        return self.name.replace("_", " ").title()

    @classmethod
    def vocabulary(cls) -> list[str]:
        """Return the nine event types' names in label order: the targets of the event-type task.

        Returns:
            The name of every member except ``NO_EVENT``.
        """
        return [label.label for label in cls if label.is_event]


@unique
class WindowSampling(StrEnum):
    """A window's span and sampling cadence, as its ``mode`` field names it.

    Window ids come in anchor triplets 3k, 3k+1, 3k+2, one per sampling in member order.
    """

    HOURLY_7_DAYS = "7day_hourly"
    FOUR_HOURLY_28_DAYS = "28day_4hourly"
    DAILY_180_DAYS = "180day_daily"

    @property
    def cadence_us(self) -> int:
        """Return the time between two consecutive timestamps, in microseconds."""
        return _CADENCE_US[self]

    @classmethod
    def for_window_id(cls, window_id: int) -> "WindowSampling":
        """Return the sampling that a window's position in its anchor triplet gives it.

        Returns:
            The member at the id's position: 3k is hourly, 3k+1 4-hourly, 3k+2 daily.
        """
        samplings = tuple(cls)
        return samplings[window_id % len(samplings)]


_CADENCE_US: dict[WindowSampling, int] = {
    WindowSampling.HOURLY_7_DAYS: 3_600_000_000,
    WindowSampling.FOUR_HOURLY_28_DAYS: 14_400_000_000,
    WindowSampling.DAILY_180_DAYS: 86_400_000_000,
}


@unique
class SkyCover(IntEnum):
    """GHCN-h sky cover codes 00-10, after NCEI's hourly coverage code table.

    Codes 00-08 count oktas, the eighths of the sky that cloud covers. Codes 09 and 10 are not
    amounts. The paper's prompt (Appendix B.4) labels the same codes with mixed fractions.
    """

    OKTAS_0 = 0
    OKTAS_1 = 1
    OKTAS_2 = 2
    OKTAS_3 = 3
    OKTAS_4 = 4
    OKTAS_5 = 5
    OKTAS_6 = 6
    OKTAS_7 = 7
    OKTAS_8 = 8
    SKY_OBSCURED = 9
    PARTIAL_OBSCURATION = 10

    @property
    def label(self) -> str:
        """Return the code's category in the channel's codebook, such as ``1 okta`` or ``sky obscured``."""
        if self <= SkyCover.OKTAS_8:
            return f"{self.value} okta" if self is SkyCover.OKTAS_1 else f"{self.value} oktas"
        return self.name.replace("_", " ").lower()

    @classmethod
    def labels(cls) -> tuple[str, ...]:
        """Return every code's category, indexed by code: the ``sky_code`` channel's codebook.

        Returns:
            The labels, in code order.
        """
        return tuple(code.label for code in cls)


@unique
class Channel(StrEnum):
    """A quantity a station measures at every timestamp of a window, one Signal each.

    The value is the key the authors' JSON uses for the channel, which also names its Signal. Member
    order is the authors' channel order, which the descriptions and the measurement key follow.
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
        """Return the channel's TimeNet spec: its spec type, name, and unit."""
        return _SPECS[self]


# Units follow the Hugging Face release's field table; the archive itself states none. In 4-hourly
# and daily windows each value is the mean of the hourly values, so precipitation there is a mean
# hourly amount, not the period's total. Values carry at most two decimals, so float32 storage keeps
# them.
_SPECS: dict[Channel, TimeSeriesSpec] = {
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
        categories=SkyCover.labels(),
    ),
}
