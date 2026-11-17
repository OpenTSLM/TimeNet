r"""Parse the Swissgrid "Energy Statistic Switzerland" yearly workbooks.

Facts this module relies on, all checked on the real files (see docs/phase1_exploration.md):

* Sheet ``Zeitreihen0h15``. Row 1 holds bilingual headers ("German\nEnglish"), row 2 the units,
  and then one row per quarter hour follows. Energies are in kWh per quarter hour.
* Timestamps are Swiss civil time (Europe/Zurich). On the spring-forward day the 02:xx labels are
  missing. On the fall-back day the four 02:xx labels appear twice, CEST first.
* Each label marks the END of its quarter hour in 2020-2024 (``01.01. 00:15`` ...
  ``01.01.(Y+1) 00:00``), and its START in 2025 (``01.01. 00:00`` ... ``31.12. 23:45``). The
  convention is detected from the first label of each file, never assumed.
* 2020 stores Excel datetimes. 2021-2025 store strings ``dd.mm.yyyy HH:MM``.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from .timefields import floor_hour


TZ = "Europe/Zurich"
QUARTER = pd.Timedelta("15min")
QUARTERS_PER_HOUR = 4
SHEET = "Zeitreihen0h15"

# English header part -> output column name. Only these columns are read.
COLUMNS = {
    "Total energy consumed by end users in the Swiss controlblock": "load_enduser",
    "Total energy consumption Swiss controlblock": "load_total",
    "Total energy production Swiss controlblock": "production",
    "Cross Border Exchange CH->AT": "exp_at",
    "Cross Border Exchange AT->CH": "imp_at",
    "Cross Border Exchange CH->DE": "exp_de",
    "Cross Border Exchange DE->CH": "imp_de",
    "Cross Border Exchange CH->FR": "exp_fr",
    "Cross Border Exchange FR->CH": "imp_fr",
    "Cross Border Exchange CH->IT": "exp_it",
    "Cross Border Exchange IT->CH": "imp_it",
    "Transit": "transit",
    "Import": "import",
    "Export": "export",
}
BORDERS = ("at", "de", "fr", "it")


class SwissgridFormatError(ValueError):
    """The workbook does not have the shape this parser was written for."""


class LabelConvention(Enum):
    PERIOD_END = "end"
    PERIOD_START = "start"


def detect_convention(first_label: pd.Timestamp, year: int) -> LabelConvention:
    """Tell the label convention of a year from its first label.

    Returns:
        PERIOD_END if the year starts at 00:15, PERIOD_START if it starts at 00:00.

    Raises:
        SwissgridFormatError: If the first label is neither.
    """
    if first_label == pd.Timestamp(year, 1, 1, 0, 15):
        return LabelConvention.PERIOD_END
    if first_label == pd.Timestamp(year, 1, 1, 0, 0):
        return LabelConvention.PERIOD_START
    raise SwissgridFormatError(f"{year}: first label {first_label} is neither 00:00 nor 00:15 on 1 January")


def parse_labels(raw: pd.Series) -> pd.Series:
    """Turn the timestamp column (strings or datetimes) into naive local Timestamps.

    Returns:
        The labels as naive Timestamps, indexed from 0.

    Raises:
        SwissgridFormatError: If the column mixes types or a label is off the 15-minute grid.
    """
    if all(isinstance(v, datetime) for v in raw):
        labels = pd.to_datetime(raw)
    elif all(isinstance(v, str) for v in raw):
        labels = pd.to_datetime(raw, format="%d.%m.%Y %H:%M")
    else:
        kinds = sorted({type(v).__name__ for v in raw})
        raise SwissgridFormatError(f"timestamp column mixes types {kinds}")
    # Excel datetimes can carry float noise. A label must be on a whole quarter hour.
    off_grid = labels[labels != labels.dt.round("15min")]
    if len(off_grid):
        raise SwissgridFormatError(f"{len(off_grid)} labels are off the 15-minute grid, first {off_grid.iloc[0]}")
    return labels.reset_index(drop=True)


def _wall_clock(instants: pd.DatetimeIndex, *, left: bool) -> pd.DatetimeIndex:
    """Naive Swiss wall-clock time of each UTC instant.

    With ``left``, use the UTC offset in force just before the instant. An end label at the
    fall-back instant 01:00 UTC can read 03:00 (CEST, the offset during the quarter it closes),
    as in 2020, or 02:00 (CET), as in 2021-2024. Both are correct renderings of the same instant.

    Returns:
        The naive wall-clock times.
    """
    probe = instants - pd.Timedelta("1us") if left else instants
    offsets = pd.to_timedelta([t.utcoffset() for t in probe.tz_convert(TZ)])
    return (instants.tz_localize(None) + offsets).as_unit("us")


def labels_to_utc_start(labels: pd.Series, year: int) -> pd.DatetimeIndex:
    """Map naive local labels to the UTC start of each quarter hour, and check the whole year is there.

    The rows are taken as consecutive quarter hours from 1 January 00:00 local time. Each label is
    then checked against the wall-clock reading of its instant, so a missing, extra or shuffled
    row shifts every later label and fails the check. Repeated fall-back labels need no guessing.

    Returns:
        The UTC start of each quarter hour, in row order.

    Raises:
        SwissgridFormatError: If the row count or any label does not match a complete year.
    """
    convention = detect_convention(labels.iloc[0], year)
    expected = pd.date_range(
        pd.Timestamp(year, 1, 1).tz_localize(TZ).tz_convert("UTC"),
        pd.Timestamp(year + 1, 1, 1).tz_localize(TZ).tz_convert("UTC"),
        freq=QUARTER,
        inclusive="left",
    )
    if len(labels) != len(expected):
        raise SwissgridFormatError(f"{year}: {len(labels)} rows, a complete year has {len(expected)} quarter hours")
    labelled = expected + QUARTER if convention is LabelConvention.PERIOD_END else expected
    lab = pd.DatetimeIndex(labels).as_unit("us")
    ok = (lab == _wall_clock(labelled, left=False)) | (lab == _wall_clock(labelled, left=True))
    if not ok.all():
        i = int(np.argmin(ok))
        raise SwissgridFormatError(
            f"{year}: row {i} is labelled {lab[i]}, but the quarter hour at that position "
            f"({convention.value} label {labelled[i]}) reads {_wall_clock(labelled[i : i + 1], left=False)[0]} locally"
        )
    return expected.rename("utc_start")


def read_year(path: Path, year: int) -> pd.DataFrame:
    """Read one workbook into a frame indexed by the UTC start of each quarter hour, values in kWh."""
    header = pd.read_excel(path, sheet_name=SHEET, header=None, nrows=2, engine="openpyxl")
    english = [str(h).split("\n")[-1].strip() for h in header.iloc[0]]
    units = list(header.iloc[1])
    positions = {}
    for name, out in COLUMNS.items():
        hits = [i for i, h in enumerate(english) if h == name]
        if len(hits) != 1:
            raise SwissgridFormatError(f"{path.name}: header {name!r} found {len(hits)} times")
        if units[hits[0]] != "kWh":
            raise SwissgridFormatError(f"{path.name}: {name!r} is in {units[hits[0]]!r}, expected kWh")
        positions[out] = hits[0]
    # read_excel returns usecols in sheet order, whatever order they are passed in.
    positions = dict(sorted(positions.items(), key=lambda kv: kv[1]))

    data = pd.read_excel(
        path, sheet_name=SHEET, header=None, skiprows=2, usecols=[0, *positions.values()], engine="openpyxl"
    )
    data.columns = ["label", *positions.keys()]
    index = labels_to_utc_start(parse_labels(data["label"]), year)
    frame = data.drop(columns="label").set_index(index).rename_axis("utc_start")
    if frame.isna().any().any():
        raise SwissgridFormatError(f"{path.name}: null values in {frame.columns[frame.isna().any()].tolist()}")
    return frame.astype("float64")


def to_hourly_mw(quarters: pd.DataFrame) -> pd.DataFrame:
    """Sum four quarter hours per UTC hour, kWh -> MWh, which is the hour's mean power in MW.

    Swiss civil time is UTC+1 or UTC+2, so UTC hours are also whole local hours.

    Returns:
        One row per UTC hour, in MW, with the net import per border and in total.

    Raises:
        SwissgridFormatError: If an hour does not hold exactly four quarter hours.
    """
    hour = floor_hour(quarters.index)
    counts = quarters.groupby(hour).size()
    incomplete = counts[counts != QUARTERS_PER_HOUR]
    if len(incomplete):
        raise SwissgridFormatError(f"hours without four quarters: {incomplete.index[:3].tolist()}")
    hourly = quarters.groupby(hour).sum() / 1000.0
    hourly.index.name = "utc_start"
    out = hourly.add_suffix("_mw")
    for b in BORDERS:
        out[f"net_import_{b}_mw"] = out[f"imp_{b}_mw"] - out[f"exp_{b}_mw"]
    out["net_import_mw"] = out[[f"net_import_{b}_mw" for b in BORDERS]].sum(axis=1)
    return out
