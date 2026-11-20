"""Assemble the aligned hourly national table from parsed sources. No file or network access."""

import pandas as pd

from . import calendar_ch, meteoswiss, swissgrid
from .timefields import field, normalize


TZ = swissgrid.TZ


def hourly_dataset(quarters: pd.DataFrame, stations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join Swissgrid hourly power, national weather and the holiday calendar.

    Args:
        quarters: The Swissgrid quarter hours of every year, indexed by UTC start, in kWh.
        stations: The hourly frame of each SwissMetNet station, indexed by UTC start.

    Returns:
        One row per UTC hour with power in MW, weather, local calendar fields and holiday flags.

    Raises:
        SwissgridFormatError: If the yearly files overlap.
    """
    if quarters.index.has_duplicates:
        raise swissgrid.SwissgridFormatError("the yearly files overlap")
    hourly = swissgrid.to_hourly_mw(quarters.sort_index())
    population = meteoswiss.canton_population()
    weather = meteoswiss.national_series(
        stations, meteoswiss.station_weights(population), pd.DatetimeIndex(hourly.index)
    )
    df = hourly.join(weather)
    local = pd.DatetimeIndex(df.index).tz_convert(TZ)
    df["local_date"] = normalize(local.tz_localize(None))
    df["local_hour"] = field(local, "hour")
    df["dow"] = field(local, "day_of_week")
    first, last = df.local_date.min(), df.local_date.max()
    cal = calendar_ch.holiday_calendar(str(first.date()), str(last.date()), population)
    df = df.join(cal, on="local_date")
    df["year"] = df.local_date.dt.year
    return df
