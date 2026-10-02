"""Codebooks, channel names, and window strides for the pinned VerbalTS release.

``meta.json`` gives attribute names and code counts. Labels come from the paper's
dataset appendix and match the release captions. The connector rejects unknown
labeled codes. See https://proceedings.mlr.press/v267/gu25a.html.

ETTm1 and Istanbul captions name their single channel. Weather uses Jena station
column order. BlindWays gives no joint order, so its channels use their positions.
The synthetic captions name their channels ``variable 1`` and ``variable 2``.

ETTm1 and Istanbul windows share a split timeline. The release gives no absolute
time for that timeline.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Attribute:
    """An attribute's annotation key, description, and code values."""

    key: str
    """Annotation key shared by components with the same meaning."""
    description: str
    """Description stored in the dataset schema."""
    labels: tuple[str, ...] | tuple[int, ...] | None = None
    """Labels in code order, or ``None`` to keep integer codes."""
    codes: tuple[int, ...] | None = None
    """Valid integer codes in release order."""

    @property
    def vocabulary(self) -> tuple[str, ...] | tuple[int, ...] | None:
        """Return labels or integer codes in release order, if known.

        Returns:
            The known labels or codes, or ``None``.
        """
        return self.labels if self.labels is not None else self.codes


_TREND_TYPE = Attribute(
    key="trend_type",
    description="Shape of the generated trend. The captions abbreviate the labels to linear, quad, exp and log.",
    labels=("linear", "quadratic", "exponential", "logistic"),
)
_TREND_DIRECTION = Attribute(
    key="trend_direction",
    description="Direction of the generated trend.",
    labels=("up", "down"),
)
_SEASON_CYCLES = Attribute(
    key="season_cycles",
    description="Number of sinusoidal cycles the generator added over the window.",
    labels=(0, 1, 2, 4),
)
_VARIABLE_2_RULE = Attribute(
    key="variable_2_rule",
    description="How the generator derived variable 2 from variable 1. The shift distance is stated in the caption only.",
    labels=("x-axis flip", "y-axis flip", "shift forward", "shift backward"),
)

_SEASON = Attribute(
    key="season",
    description="Calendar season of the window's day, from the month the caption names.",
    labels=("spring", "summer", "fall", "winter"),
)
_TIME_OF_DAY = Attribute(
    key="time_of_day",
    description="Which six-hour block of its day the window covers.",
    labels=("early morning", "morning", "afternoon", "evening"),
)
_WEATHER_CONDITION = Attribute(
    key="weather_condition",
    description="Weather condition ChatGPT 3.5 extracted from the forecast text, or unknown when the text states none.",
    labels=("sunny", "cloudy", "rain", "foggy", "snowy", "unknown"),
)
_TEMPERATURE_TREND = Attribute(
    key="temperature_trend",
    description="Temperature trend ChatGPT 3.5 extracted from the forecast text, or unknown when the text states none.",
    labels=("increase", "decrease", "steady", "unknown"),
)
_WIND_DIRECTION = Attribute(
    key="wind_direction",
    description="Wind direction ChatGPT 3.5 extracted from the forecast text, or unknown when the text states none.",
    labels=("S", "N", "W", "E", "SW", "SE", "NW", "NE", "unknown"),
)
_PRESSURE_LEVEL = Attribute(
    key="pressure_level",
    description="Atmospheric pressure level ChatGPT 3.5 extracted from the forecast text, or unknown when the text states none.",
    labels=("low", "average", "high", "unknown"),
)
_HUMIDITY_LEVEL = Attribute(
    key="humidity_level",
    description="Humidity level ChatGPT 3.5 extracted from the forecast text, or unknown when the text states none.",
    labels=("low", "average", "high", "unknown"),
)

_GUIDE_METHOD = Attribute(
    key="guide_method",
    description="Mobility aid the pedestrian used, extracted from the BlindWays description by ChatGPT 3.5.",
    labels=("cane", "guide dog"),
)
_HAND = Attribute(
    key="hand",
    description="Hand holding the aid, extracted from the BlindWays description by ChatGPT 3.5, or unknown when it states none.",
    labels=("left", "right", "unknown"),
)

_VARIABLE_DESCRIPTION = "Source variable the window was cut from, as the release's var_id code and the caption name it."
_TREND = Attribute(
    key="trend",
    description="Sign of the linear-trend slope tsfresh fitted over the window.",
    labels=("upward", "downward"),
)
_SEASON_CODE = Attribute(
    key="season_code",
    description=(
        "Dominant-frequency index tsfresh found over the window, minus one, as the release codes it. The "
        "caption states it as 'around (code + 1) pi'; a constant window carries -1."
    ),
    codes=tuple(range(-1, 9)),
)
_SKEWNESS = Attribute(
    key="skewness",
    description="Sign of the value distribution's skewness over the window, from tsfresh.",
    labels=("negative", "positive", "symmetrical"),
)
_KURTOSIS = Attribute(
    key="kurtosis",
    description="Level of the value distribution's kurtosis over the window, from tsfresh.",
    labels=("low", "normal", "high"),
)

CODEBOOKS: dict[str, dict[str, Attribute]] = {
    "synthetic_u": {
        "trend_types": _TREND_TYPE,
        "trend_directions": _TREND_DIRECTION,
        "season_cycles": _SEASON_CYCLES,
    },
    "synthetic_m": {
        "trend_types_0": _TREND_TYPE,
        "trend_directions_0": _TREND_DIRECTION,
        "season_cycles_0": _SEASON_CYCLES,
        "ops_0": _VARIABLE_2_RULE,
    },
    "Weather": {
        "season": _SEASON,
        "time": _TIME_OF_DAY,
        "weather": _WEATHER_CONDITION,
        "temperature": _TEMPERATURE_TREND,
        "wind": _WIND_DIRECTION,
        "atmospher": _PRESSURE_LEVEL,
        "humidity": _HUMIDITY_LEVEL,
    },
    "BlindWays": {
        "guide": _GUIDE_METHOD,
        "hand": _HAND,
    },
    "ETTm1": {
        "var_id": Attribute(
            key="variable",
            description=_VARIABLE_DESCRIPTION,
            labels=("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"),
        ),
        "trend": _TREND,
        "season": _SEASON_CODE,
        "skewness": _SKEWNESS,
        "kurtosis": _KURTOSIS,
    },
    "istanbul_traffic": {
        "var_id": Attribute(key="variable", description=_VARIABLE_DESCRIPTION, labels=("TI", "TI_An", "TI_Av")),
        "trend": _TREND,
        "season": _SEASON_CODE,
        "skewness": _SKEWNESS,
        "kurtosis": _KURTOSIS,
    },
}
"""Attributes by component and ``meta.json`` column order."""

VARIABLE = "variable"
"""Key for a single-channel window's channel name."""

CHANNELS: dict[str, tuple[str, ...]] = {
    "synthetic_u": ("variable 1",),
    "synthetic_m": ("variable 1", "variable 2"),
    "Weather": (
        "p",
        "T",
        "Tpot",
        "Tdew",
        "rh",
        "VPmax",
        "VPact",
        "VPdef",
        "sh",
        "H2OC",
        "rho",
        "wv",
        "max. wv",
        "wd",
        "rain",
        "raining",
        "SWDR",
        "PAR",
        "max. PAR",
        "Tlog",
        "CO2",
    ),
    "BlindWays": tuple(f"joint{joint:02d}_{coordinate}" for joint in range(24) for coordinate in range(3)),
}
"""Fixed channel names in array order.

Other components use their ``variable`` attribute as the channel name.
"""

STRIDE: dict[str, int] = {"ETTm1": 30, "istanbul_traffic": 24}
"""Sliding-window strides in steps."""
