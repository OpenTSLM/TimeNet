"""Rule-based labels that describe one day of hourly candles against the six days before it.

Every label is computed from a seven-day window and nothing else: six days of context and the day
being described. A model that sees the window therefore has all the information that the label
needs. No rule looks past the end of the window, so the labels describe what happened; they never
predict what comes next.

The rules compare the day with the window's own context, not with fixed percentages, so one rule set
fits a calm market and a volatile one. Every hourly return is the log return of one candle,
``ln(close / open)``, so a return never reaches across a missing hour.
"""

from dataclasses import dataclass
import math

import numpy as np

from timenet.types import Annotation, Label, LabelSet


HOURS_PER_DAY = 24
CONTEXT_DAYS = 6
"""Days of context before the described day."""
WINDOW_HOURS = (CONTEXT_DAYS + 1) * HOURS_PER_DAY
"""Hours in one window: the context days plus the described day."""

SHARP_MOVE_Z = 2.0
"""A day's move at or beyond this many typical daily moves is sharp."""
MOVE_Z = 1.0
"""A day's move at or beyond this many typical daily moves is a rally or a sell-off."""
HIGH_RATIO = 2.0
"""A day at or above this multiple of the context's volatility or volume is a spike or a surge."""
LOW_RATIO = 0.5
"""A day at or below this multiple of the context's volatility or volume is calm or a drought."""

MOVE = LabelSet(
    [
        Label(name="sharp_selloff", description="The price fell by two or more typical daily moves."),
        Label(name="selloff", description="The price fell by one to two typical daily moves."),
        Label(name="flat", description="The price moved by less than one typical daily move."),
        Label(name="rally", description="The price rose by one to two typical daily moves."),
        Label(name="sharp_rally", description="The price rose by two or more typical daily moves."),
    ],
    ordered=True,
    annotations=[Annotation("name", value="price_move")],
)
"""How far the price moved over the day, measured in typical daily moves of the context."""

VOLATILITY = LabelSet(
    [
        Label(name="calm", description="Hourly swings were at most half those of the context days."),
        Label(name="normal", description="Hourly swings were close to those of the context days."),
        Label(name="spike", description="Hourly swings were at least twice those of the context days."),
    ],
    ordered=True,
    annotations=[Annotation("name", value="volatility")],
)
"""How large the day's hourly swings were against the context days."""

VOLUME = LabelSet(
    [
        Label(name="drought", description="Volume was at most half the context's daily average."),
        Label(name="normal", description="Volume was close to the context's daily average."),
        Label(name="surge", description="Volume was at least twice the context's daily average."),
    ],
    ordered=True,
    annotations=[Annotation("name", value="volume")],
)
"""How much traded over the day against the context's daily average."""


@dataclass(frozen=True)
class DayEvent:
    """The measurements and labels of one described day."""

    day_return: float
    """The day's simple return, from the first hour's open to the last hour's close."""
    move_z: float
    """The day's log return in typical daily moves of the context."""
    volatility_ratio: float
    """The day's hourly volatility over the context's hourly volatility."""
    volume_ratio: float
    """The day's volume over the context's average daily volume."""
    context_hourly_volatility: float
    """The standard deviation of the context's hourly log returns."""
    day_hourly_volatility: float
    """The standard deviation of the day's hourly log returns."""
    context_daily_volume: float
    """The context's average volume per day, in the base asset."""
    day_volume: float
    """The day's volume, in the base asset."""

    @property
    def move(self) -> str:
        """The :data:`MOVE` label of the day."""
        if self.move_z >= SHARP_MOVE_Z:
            return "sharp_rally"
        if self.move_z >= MOVE_Z:
            return "rally"
        if self.move_z <= -SHARP_MOVE_Z:
            return "sharp_selloff"
        if self.move_z <= -MOVE_Z:
            return "selloff"
        return "flat"

    @property
    def volatility(self) -> str:
        """The :data:`VOLATILITY` label of the day."""
        return _level(self.volatility_ratio, low="calm", high="spike")

    @property
    def volume(self) -> str:
        """The :data:`VOLUME` label of the day."""
        return _level(self.volume_ratio, low="drought", high="surge")


def _level(ratio: float, *, low: str, high: str) -> str:
    """Name a ratio against the context: ``low``, ``normal`` or ``high``.

    Args:
        ratio: The day's value over the context's value.
        low: The label at or below :data:`LOW_RATIO`.
        high: The label at or above :data:`HIGH_RATIO`.

    Returns:
        The label of the ratio.
    """
    if ratio >= HIGH_RATIO:
        return high
    if ratio <= LOW_RATIO:
        return low
    return "normal"


def measure_day(open_: np.ndarray, close: np.ndarray, volume: np.ndarray, window_start: int) -> DayEvent | None:
    """Measure the last day of the window that starts at ``window_start``.

    Args:
        open_: Hourly open prices of the whole series, with NaN for a missing hour.
        close: Hourly close prices of the whole series, with NaN for a missing hour.
        volume: Hourly base-asset volumes of the whole series, with NaN for a missing hour.
        window_start: The index of the window's first hour.

    Returns:
        The day's measurements, or ``None`` when the window cannot be measured: an hour is missing,
        or the context has no price movement or no volume, so no ratio against it exists.
    """
    window = slice(window_start, window_start + WINDOW_HOURS)
    opens, closes, volumes = open_[window], close[window], volume[window]
    if len(opens) != WINDOW_HOURS or np.isnan(opens).any() or np.isnan(closes).any() or np.isnan(volumes).any():
        return None
    returns = np.log(closes / opens)
    split = CONTEXT_DAYS * HOURS_PER_DAY
    context_volatility = float(np.std(returns[:split], ddof=1))
    day_volatility = float(np.std(returns[split:], ddof=1))
    context_daily_volume = float(volumes[:split].sum()) / CONTEXT_DAYS
    if context_volatility <= 0.0 or context_daily_volume <= 0.0:
        return None
    day_log_return = math.log(closes[-1] / opens[split])
    day_volume = float(volumes[split:].sum())
    return DayEvent(
        day_return=math.expm1(day_log_return),
        move_z=day_log_return / (context_volatility * math.sqrt(HOURS_PER_DAY)),
        volatility_ratio=day_volatility / context_volatility,
        volume_ratio=day_volume / context_daily_volume,
        context_hourly_volatility=context_volatility,
        day_hourly_volatility=day_volatility,
        context_daily_volume=context_daily_volume,
        day_volume=day_volume,
    )


_MOVE_PHRASES = {
    "sharp_selloff": "a sharp sell-off",
    "selloff": "a sell-off",
    "flat": "a quiet day for price",
    "rally": "a rally",
    "sharp_rally": "a sharp rally",
}
_VOLATILITY_PHRASES = {
    "calm": "calmer than usual",
    "normal": "about as choppy as usual",
    "spike": "a volatility spike",
}
_VOLUME_PHRASES = {
    "drought": "unusually thin trading",
    "normal": "ordinary trading activity",
    "surge": "a volume surge",
}


def explain(event: DayEvent, base_asset: str) -> str:
    """Describe the day in plain English, from the labels and measurements alone.

    Args:
        event: The day's measurements.
        base_asset: The traded asset, such as ``BTC``.

    Returns:
        A short paragraph that a new investor can read.
    """
    direction = "rose" if event.day_return >= 0 else "fell"
    size = abs(event.move_z)
    if event.move == "flat":
        price = (
            f"{base_asset} {direction} {abs(event.day_return):.2%} over the last 24 hours, within one "
            f"typical daily move for the previous six days: {_MOVE_PHRASES['flat']}."
        )
    else:
        price = (
            f"{base_asset} {direction} {abs(event.day_return):.2%} over the last 24 hours, about "
            f"{size:.1f} times a typical daily move for the previous six days: "
            f"{_MOVE_PHRASES[event.move]}."
        )
    swings = (
        f"Hourly swings were {event.volatility_ratio:.1f} times those of the previous six days "
        f"({_VOLATILITY_PHRASES[event.volatility]}), and volume was {event.volume_ratio:.1f} times "
        f"the daily average ({_VOLUME_PHRASES[event.volume]})."
    )
    return f"{price} {swings}"


def rationale(event: DayEvent, base_asset: str) -> str:
    """Show the numbers behind the labels, step by step.

    Args:
        event: The day's measurements.
        base_asset: The traded asset, such as ``BTC``.

    Returns:
        The reasoning that leads from the window to the labels.
    """
    typical_move = event.context_hourly_volatility * math.sqrt(HOURS_PER_DAY)
    return (
        f"Previous six days: hourly log returns had a standard deviation of "
        f"{event.context_hourly_volatility:.3%}, so a typical daily move was about {typical_move:.2%}, "
        f"and {event.context_daily_volume:,.2f} {base_asset} traded per day on average. "
        f"Last 24 hours: the price changed {event.day_return:+.2%}, which is {event.move_z:+.2f} "
        f"typical daily moves ({event.move}); hourly returns had a standard deviation of "
        f"{event.day_hourly_volatility:.3%}, {event.volatility_ratio:.2f} times the previous six days "
        f"({event.volatility}); {event.day_volume:,.2f} {base_asset} traded, "
        f"{event.volume_ratio:.2f} times the daily average ({event.volume})."
    )
