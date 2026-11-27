"""Expected daily end-user consumption: a transparent linear model, fitted on train years only.

    load_d = profile[daytype(d), month(d)]
             + b_hdd * HDD(T_d) + b_hdd_lag * HDD(T_{d-1}) + b_cdd * CDD(T_d)
             + b_rad * radiation_d + b_trend * years_since_2020

* daytype: Monday, Tuesday-Thursday, Friday, Saturday, Sunday (5 types) x 12 months = 60 levels.
* HDD(T) = max(0, 16 - T), CDD(T) = max(0, T - 20), on the daily mean population-weighted temperature.
  Yesterday's HDD captures the thermal inertia of buildings.
* radiation: daily mean global radiation (W/m²). Behind-the-meter PV lowers metered consumption.
* The trend absorbs slow drift (efficiency, PV growth). It is linear in time, so its extrapolation
  to the validation and test years is part of what the evaluation measures.

Days excluded from the fit, because they are exactly what the model must NOT learn as normal:
public holidays (population share >= 0.2), bridge days, 22 Dec - 6 Jan (the school holidays of
all cantons) and the 2020 COVID-19 lockdown. Fitted by ordinary least squares (numpy.linalg.lstsq). No other tuning.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .timefields import field


TRAIN_YEARS = (2020, 2021, 2022, 2023)
VALIDATION_YEARS = (2024,)
TEST_YEARS = (2025,)
HDD_BASE_C = 16.0
CDD_BASE_C = 20.0
PARTIAL_HOLIDAY_SHARE = 0.2

# Swiss Federal Council: "extraordinary situation" from 16 March 2020. Step 1 of the reopening on
# 27 April 2020, step 2 (shops, schools) on 11 May 2020. The lockdown period ends the day before step 2.
LOCKDOWN = (pd.Timestamp("2020-03-16"), pd.Timestamp("2020-05-10"))


def split_of_year(year: int) -> str:
    if year in TRAIN_YEARS:
        return "train"
    if year in VALIDATION_YEARS:
        return "validation"
    if year in TEST_YEARS:
        return "test"
    raise ValueError(f"year {year} is in no split")


def daytype(dow: pd.Series) -> pd.Series:
    return dow.map({0: "mon", 1: "tue-thu", 2: "tue-thu", 3: "tue-thu", 4: "fri", 5: "sat", 6: "sun"})


CHRISTMAS_START = (12, 22)
"""(month, day) of the first day of the Christmas period."""
CHRISTMAS_END = (1, 6)
"""(month, day) of the last day of the Christmas period, in the next year."""
SUMMER_HOLIDAY_START = (7, 14)
SUMMER_HOLIDAY_END = (8, 15)


def _from(dates: pd.DatetimeIndex, month_day: tuple[int, int]) -> np.ndarray:
    month, day = month_day
    return (field(dates, "month") == month) & (field(dates, "day") >= day)


def _until(dates: pd.DatetimeIndex, month_day: tuple[int, int]) -> np.ndarray:
    month, day = month_day
    return (field(dates, "month") == month) & (field(dates, "day") <= day)


def is_christmas_period(dates: pd.DatetimeIndex) -> np.ndarray:
    """Flag 22 December - 6 January, the Christmas school holidays, which every canton has in this window.

    Returns:
        A boolean array, one value per date.
    """
    return _from(dates, CHRISTMAS_START) | _until(dates, CHRISTMAS_END)


def is_summer_holiday_core(dates: pd.DatetimeIndex) -> np.ndarray:
    """Flag 14 July - 15 August, when the summer school holidays of nearly all cantons overlap.

    Many firms close in these weeks too (construction-industry holidays). The period is not excluded
    from the fit, since it is a month of ordinary days as well; it only serves to attribute a cause.

    Returns:
        A boolean array, one value per date.
    """
    return _from(dates, SUMMER_HOLIDAY_START) | _until(dates, SUMMER_HOLIDAY_END)


def is_lockdown(dates: pd.DatetimeIndex) -> np.ndarray:
    return (dates >= LOCKDOWN[0]) & (dates <= LOCKDOWN[1])


def daily_frame(hourly: pd.DataFrame) -> pd.DataFrame:
    """One row per local day with the inputs of the model and the facts used downstream."""
    g = hourly.groupby("local_date")
    d = g.agg(
        load_mw=("load_enduser_mw", "mean"),
        load_total_mw=("load_total_mw", "mean"),
        production_mw=("production_mw", "mean"),
        net_import_mw=("net_import_mw", "mean"),
        temp_c=("temp_c", "mean"),
        temp_min_c=("temp_c", "min"),
        temp_max_c=("temp_c", "max"),
        radiation_wm2=("radiation_wm2", "mean"),
        dow=("dow", "first"),
        holiday_pop_share=("holiday_pop_share", "first"),
        holiday_name=("holiday_name", "first"),
        is_holiday=("is_holiday", "first"),
        is_bridge_day=("is_bridge_day", "first"),
        n_hours=("load_enduser_mw", "size"),
    )
    for b in ("at", "de", "fr", "it"):
        d[f"net_import_{b}_mw"] = g[f"net_import_{b}_mw"].mean()
    d.index = pd.DatetimeIndex(d.index, name="date")
    d["year"] = field(d.index, "year")
    d["split"] = d.year.map(split_of_year)
    d["daytype"] = daytype(d.dow)
    d["month"] = field(d.index, "month")
    d["hdd"] = np.maximum(0.0, HDD_BASE_C - d.temp_c)
    d["hdd_lag1"] = d.hdd.shift(1)
    d["cdd"] = np.maximum(0.0, d.temp_c - CDD_BASE_C)
    d["trend_years"] = (d.index - pd.Timestamp("2020-01-01")).days / 365.25
    d["christmas_period"] = is_christmas_period(d.index)
    d["lockdown"] = is_lockdown(d.index)
    d["summer_holiday_core"] = is_summer_holiday_core(d.index)
    d["special_day"] = (
        (d.holiday_pop_share >= PARTIAL_HOLIDAY_SHARE) | d.is_bridge_day | d.christmas_period | d.lockdown
    )
    return d


@dataclass(frozen=True)
class Baseline:
    columns: tuple[str, ...]
    coef: np.ndarray
    n_fit_days: int

    def design(self, d: pd.DataFrame) -> pd.DataFrame:
        levels = pd.get_dummies(d.daytype + "_" + d.month.astype(str).str.zfill(2), dtype=float)
        design = pd.concat([levels, d[["hdd", "hdd_lag1", "cdd", "radiation_wm2", "trend_years"]]], axis=1)
        return design.reindex(columns=list(self.columns), fill_value=0.0)

    def predict(self, d: pd.DataFrame) -> pd.Series:
        return pd.Series(self.design(d).to_numpy() @ self.coef, index=d.index, name="expected_mw")

    def named_coefficients(self) -> dict[str, float]:
        return dict(zip(self.columns, self.coef.round(3).tolist(), strict=True))


def fit_days(d: pd.DataFrame) -> pd.Series:
    return (d.split == "train") & ~d.special_day & d[["temp_c", "hdd_lag1", "radiation_wm2"]].notna().all(axis=1)


def fit_baseline(d: pd.DataFrame) -> Baseline:
    mask = fit_days(d)
    train = d[mask]
    levels = sorted((train.daytype + "_" + train.month.astype(str).str.zfill(2)).unique())
    columns = (*levels, "hdd", "hdd_lag1", "cdd", "radiation_wm2", "trend_years")
    proto = Baseline(columns=columns, coef=np.zeros(len(columns)), n_fit_days=int(mask.sum()))
    design = proto.design(train).to_numpy()
    coef, *_ = np.linalg.lstsq(design, train.load_mw.to_numpy(), rcond=None)
    return Baseline(columns=columns, coef=coef, n_fit_days=int(mask.sum()))
