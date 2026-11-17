"""Flag anomalous days from the baseline residual with a robust z-score.

    rel_d = (load_d - expected_d) / expected_d
    z_d   = (rel_d - median_fit) / (1.4826 * MAD_fit)

The median and MAD come from the baseline's own fit days (train years, ordinary days only), so the
scale is that of normal days and nothing is learned from validation or test.

Threshold |z| >= 3.0. Chosen on train years (see reports/phase3/threshold_table.md): it flags about
25 days a year, 91 % of public holidays (share >= 0.5) and 0.6 % of ordinary days.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baseline import Baseline, fit_days


Z_THRESHOLD = 3.0


@dataclass(frozen=True)
class ResidualScale:
    median: float
    robust_sd: float


def residual_scale(d: pd.DataFrame, rel: pd.Series) -> ResidualScale:
    r = rel[fit_days(d)]
    med = float(r.median())
    return ResidualScale(median=med, robust_sd=float(1.4826 * (r - med).abs().median()))


def score_days(d: pd.DataFrame, baseline: Baseline) -> tuple[pd.DataFrame, ResidualScale]:
    out = d.copy()
    out["expected_mw"] = baseline.predict(d)
    out["rel_dev"] = (out.load_mw - out.expected_mw) / out.expected_mw
    scale = residual_scale(out, out.rel_dev)
    out["z"] = (out.rel_dev - scale.median) / scale.robust_sd
    out["is_anomaly"] = out.z.abs() >= Z_THRESHOLD
    out.loc[out.z.isna(), "is_anomaly"] = False
    return out, scale


def threshold_table(scored: pd.DataFrame, thresholds=(2.0, 2.5, 3.0, 3.5, 4.0)) -> pd.DataFrame:
    rows = []
    train = scored[scored.split == "train"]
    for t in thresholds:
        a = train.z.abs() >= t
        rows.append({
            "threshold": t,
            "flagged_days_per_year": round(float(a.mean() * 365.25), 1),
            "holiday_recall": round(float(a[train.is_holiday].mean()), 3),
            "ordinary_day_flag_rate": round(float(a[~train.special_day].mean()), 4),
            "christmas_period_recall": round(float(a[train.christmas_period].mean()), 3),
        })  # fmt: skip
    return pd.DataFrame(rows)


def events(scored: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive anomalous days of the same sign into events."""
    a = scored[scored.is_anomaly].copy()
    if a.empty:
        return pd.DataFrame(columns=["start", "end", "n_days", "sign", "peak_z"])
    sign = np.sign(a.z)
    new = (a.index.to_series().diff() != pd.Timedelta("1D")) | (sign != sign.shift())
    a["event_id"] = new.cumsum().to_numpy()
    return a.groupby("event_id").agg(
        start=("z", lambda s: s.index.min()),
        end=("z", lambda s: s.index.max()),
        n_days=("z", "size"),
        sign=("z", lambda s: int(np.sign(s.iloc[0]))),
        peak_z=("z", lambda s: float(s.loc[s.abs().idxmax()])),
    )
