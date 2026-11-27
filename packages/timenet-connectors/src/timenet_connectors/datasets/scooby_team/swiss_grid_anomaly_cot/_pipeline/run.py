"""Run the whole annotation on an hourly table. No file or network access."""

from dataclasses import dataclass

import pandas as pd

from .anomalies import ResidualScale, score_days
from .baseline import Baseline, daily_frame, fit_baseline
from .causes import attribute, weather_effect
from .qa import build_window, iso_weeks, window_facts


@dataclass(frozen=True)
class Annotation:
    """The scored days, the fitted model and the weekly windows."""

    days: pd.DataFrame
    baseline: Baseline
    scale: ResidualScale
    windows: list[dict]


def annotate(hourly: pd.DataFrame) -> Annotation:
    """Fit the baseline on train years, score and attribute every day, and build the weekly windows.

    Returns:
        The annotation.
    """
    days = daily_frame(hourly)
    baseline = fit_baseline(days)
    scored, scale = score_days(days, baseline)
    scored = attribute(scored)
    scored["weather_effect_mw"] = weather_effect(scored, baseline)
    windows = [build_window(window_facts(week, scored, hourly)) for week in iso_weeks(scored)]
    return Annotation(days=scored, baseline=baseline, scale=scale, windows=windows)
