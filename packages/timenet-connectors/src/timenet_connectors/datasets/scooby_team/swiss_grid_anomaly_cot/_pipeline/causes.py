"""Attribute a cause to each anomalous day with explicit, ordered rules. No model, no LLM.

Below expectation (z <= -3), first matching rule wins:
  1. covid_lockdown     date in the 2020 lockdown (16 Mar - 10 May 2020)
  2. christmas_newyear  22 Dec - 6 Jan (Christmas school holidays)
  3. public_holiday     holiday for >= 20 % of the population (the share is stated in the facts)
  4. bridge_day         Monday before a Tuesday holiday / Friday after a Thursday holiday
  5. summer_holidays    14 Jul - 15 Aug (overlap of the cantonal summer school holidays)
  6. unexplained

Above expectation (z >= +3):
  1. cold_snap          daily mean temperature >= 4 °C below the train-years normal for the date
  2. heatwave           daily mean temperature >= 4 °C above normal, or >= 24 °C, from May to September
  3. unexplained

The temperature normal is the train-years (2020-2023) mean for each day of the year, smoothed with a
centred 31-day circular window. It uses no validation or test data.

At |z| >= 3, no day of 2020-2025 lies above expectation: the temperature terms of the baseline absorb
the weather (the largest positive z is 2.68, during the cold spell of 21-23 November 2025). The two
weather causes are kept for completeness. Each rationale states the weather effect separately (see
``weather_effect``).
"""

import numpy as np
import pandas as pd

from .baseline import PARTIAL_HOLIDAY_SHARE
from .timefields import field


CAUSES = (
    "none",
    "public_holiday",
    "bridge_day",
    "christmas_newyear",
    "summer_holidays",
    "covid_lockdown",
    "cold_snap",
    "heatwave",
    "unexplained",
)
CAUSE_DESCRIPTIONS = {
    "none": "No day of the window deviates from the weather- and calendar-adjusted expectation.",
    "public_holiday": "A public holiday observed by at least 20 % of the Swiss population.",
    "bridge_day": "A working day between a public holiday and the weekend.",
    "christmas_newyear": "The Christmas and New Year holiday period, 22 December to 6 January.",
    "summer_holidays": "The core summer holiday weeks, 14 July to 15 August.",
    "covid_lockdown": "The COVID-19 lockdown in Switzerland, 16 March to 10 May 2020.",
    "cold_snap": "Consumption above expectation during unusually cold weather.",
    "heatwave": "Consumption above expectation during unusually hot weather.",
    "unexplained": "A deviation that no calendar or weather rule explains.",
}
TEMP_ANOMALY_C = 4.0
DAYS_IN_YEAR = 365
"""29 February shares the normal of 31 December's position: day 366 maps to day 365."""
HOT_DAY_C = 24.0


def temperature_normal(d: pd.DataFrame) -> pd.Series:
    """Train-years mean temperature for each day of the year, circular 31-day smoothing.

    Returns:
        The normal temperature (°C) of each day of ``d``.
    """
    train = d[d.split == "train"]
    doy = np.minimum(field(train.index, "day_of_year"), DAYS_IN_YEAR)
    by_doy = train.temp_c.groupby(doy).mean().reindex(range(1, DAYS_IN_YEAR + 1))
    padded = pd.concat([by_doy.iloc[-15:], by_doy, by_doy.iloc[:15]])
    smooth = padded.rolling(31, center=True, min_periods=20).mean().iloc[15:-15]
    smooth.index = range(1, DAYS_IN_YEAR + 1)
    all_doy = np.minimum(field(d.index, "day_of_year"), DAYS_IN_YEAR)
    return pd.Series(smooth.reindex(all_doy).to_numpy(), index=d.index, name="temp_normal_c")


def attribute(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["temp_normal_c"] = temperature_normal(out)
    out["temp_anomaly_c"] = out.temp_c - out.temp_normal_c
    summer = np.isin(field(out.index, "month"), [5, 6, 7, 8, 9])

    low = out.is_anomaly & (out.z < 0)
    high = out.is_anomaly & (out.z > 0)
    conditions = [
        low & out.lockdown,
        low & out.christmas_period,
        low & (out.holiday_pop_share >= PARTIAL_HOLIDAY_SHARE),
        low & out.is_bridge_day,
        low & out.summer_holiday_core,
        low,
        high & (out.temp_anomaly_c <= -TEMP_ANOMALY_C),
        high & summer & ((out.temp_anomaly_c >= TEMP_ANOMALY_C) | (out.temp_c >= HOT_DAY_C)),
        high,
    ]
    choices = [
        "covid_lockdown",
        "christmas_newyear",
        "public_holiday",
        "bridge_day",
        "summer_holidays",
        "unexplained",
        "cold_snap",
        "heatwave",
        "unexplained",
    ]
    out["cause"] = np.select(conditions, choices, default="none")
    return out


def weather_effect(scored: pd.DataFrame, baseline) -> pd.Series:
    """MW that the day's weather adds to expected consumption, against normal temperature.

    The baseline evaluated with today's and yesterday's temperature, minus the same baseline with
    the train-years normal temperature for both days. Radiation is kept as observed.

    Returns:
        The weather effect in MW, one value per day.
    """
    normal = scored.copy()
    tn = temperature_normal(scored)
    normal["hdd"] = np.maximum(0.0, 16.0 - tn)
    normal["hdd_lag1"] = normal.hdd.shift(1)
    normal["cdd"] = np.maximum(0.0, tn - 20.0)
    return (baseline.predict(scored) - baseline.predict(normal)).rename("weather_effect_mw")
