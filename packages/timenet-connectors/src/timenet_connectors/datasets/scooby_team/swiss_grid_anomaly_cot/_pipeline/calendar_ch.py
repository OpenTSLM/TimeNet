"""Swiss public holidays per local day, weighted by the population of the cantons that observe them.

Only the federal 1 August is a holiday everywhere. Every other public holiday is cantonal. A day's
``holiday_pop_share`` is the share of the Swiss population living in a canton where the day is a
public holiday (``holidays`` package, category ``public``).

Rules:

* ``is_holiday``: ``holiday_pop_share >= 0.5``.
* ``is_bridge_day``: a Monday before a Tuesday holiday, or a Friday after a Thursday holiday, that is
  not itself a holiday. This is the usual case of Ascension Friday.
"""

import holidays
import pandas as pd

from .timefields import field


HOLIDAY_SHARE = 0.5
MONDAY, FRIDAY = 0, 4


def _most_shared(shares: dict[str, float] | None) -> str | None:
    if not shares:
        return None
    return max(shares.items(), key=lambda item: item[1])[0]


def holiday_calendar(start: str, end: str, population: pd.Series) -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D")
    years = sorted({int(y) for y in field(days, "year")})
    total = float(population.sum())
    share = pd.Series(0.0, index=days)
    names: dict[str, dict[str, float]] = {}
    for canton, pop in population.items():
        cal = holidays.Switzerland(subdiv=canton, years=years, categories=("public",), language="en_US")
        for day, name in cal.items():
            ts = pd.Timestamp(day)
            if ts in share.index:
                share[ts] += pop / total
                by_name = names.setdefault(str(day), {})
                by_name[name] = by_name.get(name, 0.0) + pop / total

    cal = pd.DataFrame(index=pd.DatetimeIndex(days, name="local_date"))
    cal["holiday_pop_share"] = share.round(4)
    cal["holiday_name"] = [_most_shared(names.get(str(d.date()))) for d in days]
    cal["is_holiday"] = cal.holiday_pop_share >= HOLIDAY_SHARE
    dow = field(cal.index, "day_of_week")
    hol = cal.is_holiday
    next_is_hol = hol.shift(-1, fill_value=False)
    prev_is_hol = hol.shift(1, fill_value=False)
    cal["is_bridge_day"] = ~hol & (((dow == MONDAY) & next_is_hol) | ((dow == FRIDAY) & prev_is_hol))
    return cal
