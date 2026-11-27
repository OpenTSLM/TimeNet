"""Weekly windows: facts, prompts, targets and rationales for the three tasks of each window.

A window is one ISO week, Monday 00:00 to Sunday 24:00 Swiss civil time (167, 168 or 169 hours).
Its split is the split of the year that holds its Thursday, as ISO weeks are numbered.

Tasks per window:
  * anomaly_cause (classification): the cause of the largest anomaly, or "none".
  * anomaly_days (temporal localization, exhaustive): the anomalous days; possibly none.
  * exchange (answer): net importer or exporter, and through which borders.

The prompts give the week's dates and the series, never the holidays: a holiday has to be read from
the curve (a weekday shaped like a Sunday). The facts and the rationales keep the calendar.
"""

import json

import numpy as np
import pandas as pd

from . import rationales
from .anomalies import Z_THRESHOLD
from .baseline import PARTIAL_HOLIDAY_SHARE, split_of_year
from .timefields import field


BORDERS = {"at": "Austria", "de": "Germany", "fr": "France", "it": "Italy"}

ANOMALY_SIGNALS = ("load_enduser_mw", "temp_c", "radiation_wm2")
EXCHANGE_SIGNALS = ("net_import_at_mw", "net_import_de_mw", "net_import_fr_mw", "net_import_it_mw",
                    "production_mw", "load_total_mw")  # fmt: skip


def iso_weeks(days: pd.DataFrame) -> list[pd.DatetimeIndex]:
    """Complete ISO weeks (seven local days present) of the daily frame."""
    mondays = days.index[field(days.index, "day_of_week") == 0]
    weeks = []
    for m in mondays:
        span = pd.date_range(m, periods=7, freq="D")
        if span.isin(days.index).all():
            weeks.append(span)
    return weeks


def rolling_weeks(days: pd.DataFrame, split: str) -> list[pd.DatetimeIndex]:
    """Seven-day windows starting on every day, kept only when all seven days belong to ``split``.

    Used to augment the train split. A window never reaches into another split's days.

    Returns:
        The windows, in time order.
    """
    inside = days.index[days.split == split]
    weeks = []
    for start in inside:
        span = pd.date_range(start, periods=7, freq="D")
        if span.isin(inside).all():
            weeks.append(span)
    return weeks


def _window_id(week: pd.DatetimeIndex) -> str:
    if week[0].day_of_week == 0:
        iso = week[0].isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    return f"{week[0].date()}+7d"


def _day_facts(row: pd.Series, date: pd.Timestamp) -> dict:
    f = {
        "date": str(date.date()),
        "day": rationales.day_name(date),
        "load": rationales.mw(row.load_mw),
        "exp": rationales.mw(row.expected_mw),
        "pct": rationales.pct(row.rel_dev),
        "dir": "below" if row.rel_dev < 0 else "above",
        "z": rationales.one(row.z),
        "is_anomaly": bool(row.is_anomaly),
        "cause": row.cause if row.is_anomaly else "none",
    }
    if row.holiday_pop_share >= PARTIAL_HOLIDAY_SHARE:
        f["holiday"] = row.holiday_name
        f["share"] = rationales.one(row.holiday_pop_share * 100)
    return f


def window_facts(week: pd.DatetimeIndex, days: pd.DataFrame, hourly: pd.DataFrame) -> dict:
    d = days.loc[week]
    h = hourly[hourly.local_date.isin(week)]
    day_facts = [_day_facts(d.loc[t], t) for t in week]
    anomalous = [f for f in day_facts if f["is_anomaly"]]
    largest = max(day_facts, key=lambda f: abs(float(f["z"])))
    main = max(anomalous, key=lambda f: abs(float(f["z"]))) if anomalous else None

    thursday = week[3]
    return {
        "window_id": _window_id(week),
        "start_local": str(week[0].date()),
        "end_local": str(week[-1].date()),
        "split": split_of_year(thursday.year),
        "n_hours": len(h),
        "z_threshold": Z_THRESHOLD,
        "weather": _weather_facts(d),
        "days": day_facts,
        "anomalous_days": anomalous,
        "largest_deviation": {
            "day": largest["day"],
            "deviation_pct": largest["pct"],
            "direction": largest["dir"],
            "z": largest["z"],
        },
        "main_anomaly": main,
        "label": main["cause"] if main else "none",
        "exchange": _exchange_facts(h),
    }


def _weather_facts(d: pd.DataFrame) -> dict:
    temp_anom = float(d.temp_c.mean() - d.temp_normal_c.mean())
    effect = float(d.weather_effect_mw.mean())
    return {
        "temp_mean_c": rationales.one(d.temp_c.mean()),
        "temp_normal_c": rationales.one(d.temp_normal_c.mean()),
        "temp_anomaly_abs_c": rationales.one(abs(temp_anom)),
        "direction": "below" if temp_anom < 0 else "above",
        "weather_effect_abs_mw": rationales.mw(abs(effect)),
        "effect_sign": int(np.sign(round(effect))),
        "radiation_mean_wm2": rationales.mw(d.radiation_wm2.mean()),
    }


def _exchange_facts(h: pd.DataFrame) -> dict:
    net = float(h.net_import_mw.mean())
    prod, cons = float(h.production_mw.mean()), float(h.load_total_mw.mean())
    borders = []
    for code, name in BORDERS.items():
        v = float(h[f"net_import_{code}_mw"].mean())
        borders.append({
            "code": code, "name": name, "net_import_mw_signed": round(v, 1),
            "net_import_mw": rationales.mw(max(v, 0.0)), "net_export_mw": rationales.mw(max(-v, 0.0)),
        })  # fmt: skip
    return {
        "role": "importer" if net > 0 else "exporter",
        "net_import_mw": rationales.mw(net),
        "net_flow_abs_mw": rationales.mw(abs(net)),
        "net_flow_noun": "imports" if net > 0 else "exports",
        "import_hours_pct": rationales.one(100 * float((h.net_import_mw > 0).mean())),
        "production_mw": rationales.mw(prod),
        "consumption_mw": rationales.mw(cons),
        "production_short": cons > prod,
        "gap_mw": rationales.mw(abs(cons - prod)),
        "borders": borders,
    }


def prompts(facts: dict) -> dict[str, str]:
    start = rationales.day_name(pd.Timestamp(facts["start_local"]))
    end = rationales.day_name(pd.Timestamp(facts["end_local"]))
    head = f"Hourly data for Switzerland from {start} to {end} (Swiss local time)."
    anomaly_inputs = ("The series are end-user electricity consumption (MW), population-weighted air "
                      "temperature (°C) and global solar radiation (W/m²).")  # fmt: skip
    return {
        "anomaly_cause": f"{head} {anomaly_inputs} Did consumption deviate from what the weekday, "
        "weather and season predict? Explain, then name the cause of the largest deviation.",
        "anomaly_days": f"{head} {anomaly_inputs} Which days of this week had abnormal "
        "consumption, given the weekday, weather and season?",
        "exchange": f"{head} The series are net imports from Austria, Germany, France and Italy (MW, "
        "positive means import into Switzerland), Swiss production and total consumption (MW). Was "
        "Switzerland a net importer or exporter of electricity this week, and through which borders?",
    }


def build_window(facts: dict) -> dict:
    key = facts["window_id"]
    texts = {
        "anomaly_cause": rationales.anomaly_rationale(facts, key),
        "anomaly_days": rationales.localization_rationale(facts, key),
        "exchange": rationales.exchange_rationale(facts, key),
    }
    facts_text = json.dumps(facts, ensure_ascii=False)
    for name, text in texts.items():
        bad = rationales.check_numbers(text, facts_text)
        if bad:
            raise ValueError(f"{key} {name}: numbers not in the facts: {sorted(bad)}\n{text}")
    return {
        "window_id": key,
        "split": facts["split"],
        "label": facts["label"],
        "anomalous_dates": [d["date"] for d in facts["anomalous_days"]],
        "exchange_answer": f"net {facts['exchange']['role']}",
        "prompts": prompts(facts),
        "rationales": texts,
        "facts": facts,
    }
