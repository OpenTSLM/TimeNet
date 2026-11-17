"""Parse SwissMetNet hourly files and build population-weighted national series.

* ``reference_timestamp`` is ``dd.mm.yyyy HH:MM`` in UTC and labels the END of the hour. The
  radiation-weighted mean label hour at Zürich is 11.99 UTC, against 11.93 expected for end labels
  (solar noon ~11:26 UTC). See docs/phase1_exploration.md.
* ``tre200h0``: air temperature 2 m, hourly mean, °C. ``gre000h0``: global radiation, hourly mean, W/m².
"""

from io import StringIO
from pathlib import Path

import pandas as pd

from .stations import CANTON_POPULATION_2023, STATION_CANTONS


PARAMETERS = {"tre200h0": "temp_c", "gre000h0": "radiation_wm2"}
MIN_WEIGHT_COVERAGE = 0.8
"""A national value needs stations covering at least this share of the population weight. Below
that, the value stays null. Missing station values are never filled."""

# German canton names as the BFS table writes them -> canton code.
_BFS_NAMES = {
    "Zürich": "ZH", "Bern / Berne": "BE", "Luzern": "LU", "Uri": "UR", "Schwyz": "SZ",
    "Obwalden": "OW", "Nidwalden": "NW", "Glarus": "GL", "Zug": "ZG", "Fribourg / Freiburg": "FR",
    "Solothurn": "SO", "Basel-Stadt": "BS", "Basel-Landschaft": "BL", "Schaffhausen": "SH",
    "Appenzell Ausserrhoden": "AR", "Appenzell Innerrhoden": "AI", "St. Gallen": "SG",
    "Graubünden / Grigioni / Grischun": "GR", "Aargau": "AG", "Thurgau": "TG", "Ticino": "TI",
    "Vaud": "VD", "Valais / Wallis": "VS", "Neuchâtel": "NE", "Genève": "GE", "Jura": "JU",
}  # fmt: skip


class MeteoSwissFormatError(ValueError):
    """A station file does not have the expected shape."""


def read_canton_population(path: Path) -> pd.Series:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    df = pd.read_csv(StringIO(text))
    names = df.iloc[:, 1].str.removeprefix("- ").str.strip()
    unknown = sorted(set(names) - set(_BFS_NAMES))
    if unknown:
        raise MeteoSwissFormatError(f"unknown canton names in {path.name}: {unknown}")
    pop = pd.Series(df.iloc[:, -1].astype("int64").to_numpy(), index=names.map(_BFS_NAMES), name="population")
    if set(pop.index) != set(_BFS_NAMES.values()):
        raise MeteoSwissFormatError(f"{path.name} lacks cantons {sorted(set(_BFS_NAMES.values()) - set(pop.index))}")
    return pop


def canton_population() -> pd.Series:
    """The 2023 BFS population per canton, as a Series indexed by canton code."""
    return pd.Series(CANTON_POPULATION_2023, name="population", dtype="int64")


def station_weights(population: pd.Series) -> pd.Series:
    w = pd.Series({s: int(population[list(c)].sum()) for s, c in STATION_CANTONS.items()}, dtype="float64")
    return w / w.sum()


def read_station(path: Path, station: str) -> pd.DataFrame:
    """Read one station, indexed by the UTC start of each hour."""
    df = pd.read_csv(path, sep=";", usecols=["station_abbr", "reference_timestamp", *PARAMETERS])
    if set(df.station_abbr) != {station}:
        raise MeteoSwissFormatError(f"{path.name}: station_abbr {sorted(set(df.station_abbr))}, expected {station}")
    end = pd.to_datetime(df.reference_timestamp, format="%d.%m.%Y %H:%M").dt.tz_localize("UTC")
    start = pd.DatetimeIndex(end - pd.Timedelta("1h"), name="utc_start")
    if start.has_duplicates:
        raise MeteoSwissFormatError(f"{path.name}: duplicated timestamps")
    return df[list(PARAMETERS)].rename(columns=PARAMETERS).set_index(start).astype("float64")


def national_series(stations: dict[str, pd.DataFrame], weights: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Population-weighted mean over the stations that report, if they cover enough weight."""
    out = {}
    for col in PARAMETERS.values():
        values = pd.DataFrame({s: df[col].reindex(index) for s, df in stations.items()})
        present = values.notna()
        w = present.mul(weights[values.columns], axis=1)
        coverage = w.sum(axis=1)
        # A missing value has zero weight, and sum() skips it: nothing is filled in.
        mean = values.mul(weights[values.columns], axis=1).sum(axis=1, min_count=1) / coverage.where(coverage > 0)
        out[col] = mean.where(coverage >= MIN_WEIGHT_COVERAGE - 1e-9)
        out[f"{col}_coverage"] = coverage
    return pd.DataFrame(out, index=index)
