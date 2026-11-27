"""Chronological partitions and scoped forecast windows for Hill of Towie SCADA."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray
import pyarrow as pa

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import RegularAxis
from timenet.types import (
    ForecastingTask,
    MissingReason,
    MissingValues,
    Split,
    TimeInterval,
    TimeOrigin,
    TimeSeriesSpec,
    ureg,
)


STEP = timedelta(minutes=10)
STEP_US = 600_000_000
LAST_MONTH_END = datetime(2026, 5, 1, tzinfo=UTC)
_CONTEXT = 36
_HORIZON = 6
_AXIS = RegularAxis.from_rate_hz(Fraction(1, 600))
_MISSING = MissingValues(reason=MissingReason.NOT_RECORDED, description="No SCADA value for this turbine and interval.")
_POWER = TimeSeriesSpec(
    spec_type="active_power", name="Active power", unit_value=ureg.kilowatt, dtype="float64", missing=_MISSING
)
_WIND = TimeSeriesSpec(
    spec_type="wind_speed",
    name="Active wind speed",
    unit_value=ureg.meter / ureg.second,
    dtype="float64",
    missing=_MISSING,
)
_RPM = TimeSeriesSpec(
    spec_type="generator_speed", name="Generator speed", unit_value=ureg.rpm, dtype="float64", missing=_MISSING
)
_SPECS = (_POWER, _WIND, _RPM)


@dataclass(frozen=True)
class Period:
    """One contiguous split or evaluation cohort, with an exclusive end."""

    start: datetime
    end: datetime
    split: Split
    cohort: str


_PERIOD_STARTS = (
    (datetime(2016, 1, 1, tzinfo=UTC), Split.TRAIN, "historical_train"),
    (datetime(2024, 10, 1, tzinfo=UTC), Split.VALIDATION, "post_tuneup_validation"),
    (datetime(2025, 1, 1, tzinfo=UTC), Split.TEST, "pre_commissioning"),
    (datetime(2025, 10, 28, 13, tzinfo=UTC), Split.TEST, "dynamic_yaw_commissioning"),
    (datetime(2026, 1, 7, 13, tzinfo=UTC), Split.TEST, "dynamic_yaw_west_trial"),
    (datetime(2026, 2, 3, 18, tzinfo=UTC), Split.TEST, "dynamic_yaw_west_and_east_trial"),
)
_PERIODS = tuple(
    Period(start, end, split, cohort)
    for (start, split, cohort), end in zip(
        _PERIOD_STARTS, (*(row[0] for row in _PERIOD_STARTS[1:]), LAST_MONTH_END), strict=True
    )
)


def month_periods(start: datetime, end: datetime) -> Iterator[Period]:
    """Intersect each release period with the UTC month being read."""
    for period in _PERIODS:
        lower, upper = max(start, period.start), min(end, period.end)
        if lower < upper:
            yield Period(lower, upper, period.split, period.cohort)


@dataclass(frozen=True)
class Turbine:
    """Published identifiers and rated capacity of a SCADA station."""

    station: int
    name: str
    rated_power_kw: int


@dataclass(frozen=True)
class WindowSeries:
    """One turbine record and the valid hourly cutoffs for its month and cohort."""

    record: Record
    power: Signal
    period: Period
    cutoffs: tuple[int, ...]


def prepare_windows(values: NDArray[np.float64], turbine: Turbine, period: Period) -> WindowSeries:
    """Store source nulls and select only windows with complete inputs and power targets."""
    history_valid = np.isfinite(values).all(axis=0)
    power_values = values[0]
    total = power_values.size
    cutoffs = np.arange(_CONTEXT, total - _HORIZON + 1, _HORIZON, dtype=np.int64)
    history_counts = np.concatenate(([0], np.cumsum(history_valid)))
    power_counts = np.concatenate(([0], np.cumsum(np.isfinite(power_values))))
    valid = (history_counts[cutoffs] - history_counts[cutoffs - _CONTEXT] == _CONTEXT) & (
        power_counts[cutoffs + _HORIZON] - power_counts[cutoffs] == _HORIZON
    )
    signals = tuple(
        Signal.from_values(pa.array(channel, mask=np.isnan(channel)), spec=spec, name=spec.spec_type, time_axis=_AXIS)
        for spec, channel in zip(_SPECS, values, strict=True)
    )
    record = Record(
        start_time=TimeOrigin(period.start + STEP),
        metadata={
            "station_id": turbine.station,
            "turbine_id": turbine.name,
            "rated_power_kw": turbine.rated_power_kw,
            "evaluation_cohort": period.cohort,
        },
        sources=(Source(name="SCADA", signals=signals),),
    )
    return WindowSeries(record, signals[0], period, tuple(int(cutoff) for cutoff in cutoffs[valid]))


def forecasting_tasks(series: list[WindowSeries]) -> Iterator[ForecastingTask]:
    """Stream tasks that point to disjoint history and target intervals on their source record."""
    for entry in series:
        for cutoff in entry.cutoffs:
            start = cutoff - _CONTEXT
            stop = cutoff + _HORIZON
            yield ForecastingTask(
                inputs=(entry.record,),
                scope=TimeInterval(start_us=start * STEP_US, end_us=cutoff * STEP_US),
                targets=(TimeInterval(start_us=cutoff * STEP_US, end_us=stop * STEP_US, time_series=(entry.power,)),),
                prompt="Forecast mean active power in kW for the next six ten-minute intervals.",
                split=entry.period.split,
                metadata={
                    "evaluation_cohort": entry.period.cohort,
                    "window_start_utc": (entry.period.start + start * STEP).isoformat(),
                    "forecast_cutoff_utc": (entry.period.start + cutoff * STEP).isoformat(),
                },
            )
