"""Convert the ``sethlei/oil-fundamentals`` snapshot (Hub repo ``sethlei/oil-fundamentals``) to TimeF.

The snapshot is eight Parquet tables. The connector builds one panel Record on a shared
1920-01-01 UTC origin with one Source per publisher and one Signal per upstream series, plus one
Record per EIA STEO publication holding that vintage's forward WTI forecast path. Months are
stamped at month start. The panel has no price series and the dataset has no tasks: see the card.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.types import Annotation, ScalarPredictionTask, Split, TimeInterval, TimeSeriesSpec
from timenet.types.clock import TimeOrigin, offset_us
from timenet.types.specs import MissingReason, MissingValues
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef


ORIGIN = TimeOrigin(datetime(1920, 1, 1, tzinfo=UTC))
"""The shared relative zero: US crude production, the oldest series, starts in 1920-01."""
ACCESS_DATE = "2026-11-11"
US_PER_DAY = 86_400 * 10**6
WEEK_US = Fraction(7 * US_PER_DAY)

# Publication lags measured from the 2026-11-11 pull as pull month minus last reported month.
LAG_MONTHS = {
    "eia_stocks": 1,
    "eia_production": 3,
    "eia_international": 4,
    "jodi": 3,
    "cftc": 0,
    "frb": 0,
    "gpr": 1,
}
CITATIONS = {
    "EIA": "U.S. Energy Information Administration, https://www.eia.gov, accessed 2026-11-11.",
    "JODI": (
        "International Energy Forum (IEF). Joint Organizations Data Initiative (JODI) Database. "
        "www.jodidata.org, accessed 2026-11-11."
    ),
    "CFTC": (
        "U.S. Commodity Futures Trading Commission, Commitments of Traders, Disaggregated Futures "
        "Only, accessed 2026-11-11."
    ),
    "FRB": (
        "Board of Governors of the Federal Reserve System (US), Nominal Broad U.S. Dollar Index, "
        "retrieved from FRED, Federal Reserve Bank of St. Louis, 2026-11-11."
    ),
    "GPR": (
        "Caldara, D. and Iacoviello, M. (2022). Measuring Geopolitical Risk. American Economic "
        "Review 112(4), 1194-1225. Data downloaded from https://www.matteoiacoviello.com/gpr.htm "
        "on 2026-11-11."
    ),
}
JODI_FLOWS = {
    "INDPROD": ("production", "KBD", "kilooil_barrel/day"),
    "REFINOBS": ("refinery_intake", "KBD", "kilooil_barrel/day"),
    "TOTIMPSB": ("imports", "KBD", "kilooil_barrel/day"),
    "TOTEXPSB": ("exports", "KBD", "kilooil_barrel/day"),
    "CLOSTLV": ("closing_stocks", "KBBL", "kilooil_barrel"),
}
CFTC_COLUMNS = (
    "open_interest",
    "prod_merc_long",
    "prod_merc_short",
    "swap_long",
    "swap_short",
    "swap_spread",
    "m_money_long",
    "m_money_short",
    "m_money_spread",
    "other_rept_long",
    "other_rept_short",
    "other_rept_spread",
    "nonrept_long",
    "nonrept_short",
)

HORIZONS_MONTHS = (1, 3, 6, 12)
TASK_ORIGIN_MONTHS = ("2000-01", "2025-12")
"""First and last origin month of the production forecasting tasks."""
SLOWEST_LAG_MONTHS = 4
"""The stored scope is sealed at the slowest publication lag, so an unaware consumer never sees a leak."""
SPLITS = ((Split("train"), "2018-12"), (Split("validation"), "2021-12"), (Split("test"), "2026-10"))
"""Split by TARGET month: train through 2018-12, val through 2021-12, test through 2026-10."""

NOT_RECORDED = MissingValues(
    reason=MissingReason.NOT_RECORDED, description="The source reported no value for the step."
)


def _us(timestamps: pa.Array) -> np.ndarray:
    """Convert a timestamp column to microsecond offsets from :data:`ORIGIN`.

    Returns:
        Offsets in whole microseconds, one per row.
    """
    moments = timestamps.cast(pa.timestamp("us", tz="UTC")).to_pylist()
    return np.array([offset_us(moment, ORIGIN.timestamp) for moment in moments], dtype=np.int64)


def _read(path: Path, columns: list[str] | None = None) -> pa.Table:
    """Read one snapshot table.

    Returns:
        The table, with the requested columns.

    Raises:
        TimeNetConnectorError: If the file cannot be read.
    """
    try:
        return pq.read_table(path, columns=columns)
    except (OSError, pa.ArrowInvalid) as exc:
        raise TimeNetConnectorError(f"{path}: {exc}") from exc


def _sorted_by(table: pa.Table, column: str) -> pa.Table:
    """Return the table sorted ascending by one column."""
    return table.sort_by([(column, "ascending")])


def _irregular(
    values: np.ndarray,
    offsets: np.ndarray,
    *,
    spec: TimeSeriesSpec,
    name: str,
    annotations: Mapping[str, str | int | float],
) -> Signal | None:
    """Build an irregular signal from non-null values and annotate it.

    Rows whose value is null are dropped: an irregular axis has no step for them.

    Returns:
        The signal, or ``None`` when no value is finite.
    """
    keep = np.isfinite(values)
    if not keep.any():
        return None
    values, offsets = values[keep], offsets[keep]
    order = np.argsort(offsets, kind="stable")
    values, offsets = values[order], offsets[order]
    unique_offsets, first_index = np.unique(offsets, return_index=True)
    if len(unique_offsets) != len(offsets):
        values, offsets = values[first_index], unique_offsets
    signal = Signal.from_irregular(values.astype(np.float64), time_offsets_us=offsets, spec=spec, name=name)
    for key, value in annotations.items():
        signal.annotate(Annotation(key=key, value=value))
    return signal


def _common(publisher: str, upstream_id: str, frequency: str, unit_label: str, lag: int, **extra: str) -> dict:
    """Return the annotations every panel signal carries."""
    return {
        "publisher": publisher,
        "upstream_series_id": upstream_id,
        "frequency": frequency,
        "unit_label": unit_label,
        "access_date": ACCESS_DATE,
        "publication_lag_months": lag,
        "citation": CITATIONS[publisher],
        **extra,
    }


def _weekly_stocks(path: Path) -> Signal:
    """Build the weekly crude-stocks signal on a 7-day grid, with missing weeks declared.

    Returns:
        The stocks signal.

    Raises:
        TimeNetConnectorError: If the week-ending dates are not all seven days apart.
    """
    stocks = _sorted_by(_read(path, ["period_end", "value"]), "period_end")
    offsets = _us(stocks["period_end"])
    values = stocks["value"].cast(pa.float64()).to_numpy(zero_copy_only=False)
    step = int(WEEK_US)
    if np.any((offsets[1:] - offsets[:-1]) % step != 0):
        raise TimeNetConnectorError("weekly stocks are not on a 7-day grid")
    n_steps = int((offsets[-1] - offsets[0]) // step) + 1
    grid = np.full(n_steps, np.nan)
    grid[(offsets - offsets[0]) // step] = values
    signal = Signal.from_values(
        pa.array(grid, mask=np.isnan(grid)),
        spec=TimeSeriesSpec(
            spec_type="crude_stocks_ex_spr",
            name="US commercial crude oil stocks, excluding SPR",
            unit_value="kilooil_barrel",
            dtype="float64",
            missing=NOT_RECORDED,
        ),
        name="eia_WCESTUS1",
        time_axis=RegularAxis(period_us=WEEK_US, offset_us=int(offsets[0])),
    )
    for key, value in _common(
        "EIA", "WCESTUS1", "weekly", "thousand barrels", LAG_MONTHS["eia_stocks"], geography="USA"
    ).items():
        signal.annotate(Annotation(key=key, value=value))
    return signal


def _eia_signals(paths: Mapping[str, Path]) -> tuple[Signal, ...]:
    """Build the EIA survey signals: weekly stocks on a 7-day grid, monthly production, international production."""
    signals: list[Signal] = [_weekly_stocks(paths["stocks"])]

    production = _read(paths["production"], ["period_start", "value"])
    signal = _irregular(
        production["value"].cast(pa.float64()).to_numpy(zero_copy_only=False),
        _us(production["period_start"]),
        spec=TimeSeriesSpec(
            spec_type="us_crude_production",
            name="US field production of crude oil",
            unit_value="kilooil_barrel/day",
            dtype="float64",
        ),
        name="eia_MCRFPUS2",
        annotations=_common(
            "EIA", "MCRFPUS2", "monthly", "thousand barrels per day", LAG_MONTHS["eia_production"], geography="USA"
        ),
    )
    if signal is not None:
        signals.append(signal)

    international = _read(
        paths["international"],
        ["period_start", "countryRegionId", "countryRegionName", "is_aggregate", "value", "missing_reason"],
    )
    spec = TimeSeriesSpec(
        spec_type="petroleum_liquids_production",
        name="Total petroleum and other liquids production",
        unit_value="kilooil_barrel/day",
        dtype="float64",
    )
    ids = international["countryRegionId"].unique().to_pylist()
    for region_id in ids:
        rows = international.filter(pc.field("countryRegionId") == region_id)
        region_name = rows["countryRegionName"][0].as_py()
        aggregate = bool(rows["is_aggregate"][0].as_py())
        signal = _irregular(
            rows["value"].cast(pa.float64()).to_numpy(zero_copy_only=False),
            _us(rows["period_start"]),
            spec=spec,
            name=f"eia_intl_{region_id}",
            annotations=_common(
                "EIA",
                f"international/{region_id}",
                "monthly",
                "thousand barrels per day",
                LAG_MONTHS["eia_international"],
                geography=str(region_id),
                geography_name=str(region_name),
                aggregate="true" if aggregate else "false",
                product="Total petroleum and other liquids",
            ),
        )
        if signal is not None:
            signals.append(signal)
    return tuple(signals)


def _jodi_signals(path: Path) -> tuple[Signal, ...]:
    """Build one signal per (area, flow) of JODI crude oil, in the flow's own unit."""
    table = _read(path, ["ref_area", "period_start", "flow_breakdown", "unit_measure", "value"])
    signals: list[Signal] = []
    for flow, (flow_name, unit_code, unit) in JODI_FLOWS.items():
        rows = table.filter((pc.field("flow_breakdown") == flow) & (pc.field("unit_measure") == unit_code))
        spec = TimeSeriesSpec(
            spec_type=f"crude_{flow_name}",
            name=f"Crude oil {flow_name.replace('_', ' ')}",
            unit_value=unit,
            dtype="float64",
        )
        for area in rows["ref_area"].unique().to_pylist():
            area_rows = rows.filter(pc.field("ref_area") == area)
            signal = _irregular(
                area_rows["value"].cast(pa.float64()).to_numpy(zero_copy_only=False),
                _us(area_rows["period_start"]),
                spec=spec,
                name=f"jodi_{area}_{flow}",
                annotations=_common(
                    "JODI",
                    f"CRUDEOIL/{flow}/{unit_code}/{area}",
                    "monthly",
                    "thousand barrels" if unit_code == "KBBL" else "thousand barrels per day",
                    LAG_MONTHS["jodi"],
                    geography=str(area),
                    product="CRUDEOIL",
                    flow=flow,
                ),
            )
            if signal is not None:
                signals.append(signal)
    return tuple(signals)


def _cftc_signals(path: Path) -> tuple[Signal, ...]:
    """Build one signal per positioning column of the WTI Commitments of Traders report."""
    table = _sorted_by(_read(path), "date")
    offsets = _us(table["date"])
    signals: list[Signal] = []
    for column in CFTC_COLUMNS:
        if column not in table.column_names:
            continue
        signal = _irregular(
            table[column].cast(pa.float64()).to_numpy(zero_copy_only=False),
            offsets,
            spec=TimeSeriesSpec(
                spec_type=f"cot_{column}",
                name=f"COT {column.replace('_', ' ')}",
                unit_value="dimensionless",
                dtype="float64",
            ),
            name=f"cftc_067651_{column}",
            annotations=_common(
                "CFTC",
                f"067651/{column}",
                "weekly (Tuesday as-of)",
                "contracts",
                LAG_MONTHS["cftc"],
                market="WTI crude oil, NYMEX, code 067651",
            ),
        )
        if signal is not None:
            signals.append(signal)
    return tuple(signals)


def _frb_signals(path: Path) -> tuple[Signal, ...]:
    """Build the broad dollar index signals, DTWEXBGS and its predecessor DTWEXM."""
    table = _read(path, ["series_id", "date", "value"])
    signals: list[Signal] = []
    for series_id, spec_type, label in (
        ("DTWEXBGS", "dollar_index_broad", "Nominal broad US dollar index, Jan 2006 = 100"),
        ("DTWEXM", "dollar_index_major", "Trade-weighted US dollar index, major currencies, Mar 1973 = 100"),
    ):
        rows = table.filter(pc.field("series_id") == series_id)
        signal = _irregular(
            rows["value"].cast(pa.float64()).to_numpy(zero_copy_only=False),
            _us(rows["date"]),
            spec=TimeSeriesSpec(spec_type=spec_type, name=label, unit_value="dimensionless", dtype="float64"),
            name=f"frb_{series_id}",
            annotations=_common("FRB", series_id, "daily (business days)", "index", LAG_MONTHS["frb"]),
        )
        if signal is not None:
            signals.append(signal)
    return tuple(signals)


def _gpr_signals(path: Path) -> tuple[Signal, ...]:
    """Build the geopolitical risk index and its threats and acts components."""
    table = _sorted_by(_read(path, ["month", "gpr", "gpr_threats", "gpr_acts"]), "month")
    offsets = _us(table["month"])
    signals: list[Signal] = []
    for column, label in (
        ("gpr", "Geopolitical risk index"),
        ("gpr_threats", "Geopolitical threats index"),
        ("gpr_acts", "Geopolitical acts index"),
    ):
        signal = _irregular(
            table[column].cast(pa.float64()).to_numpy(zero_copy_only=False),
            offsets,
            spec=TimeSeriesSpec(spec_type=column, name=label, unit_value="dimensionless", dtype="float64"),
            name=f"gpr_{column}",
            annotations=_common("GPR", column, "monthly", "index (1985-2019 = 100)", LAG_MONTHS["gpr"]),
        )
        if signal is not None:
            signals.append(signal)
    return tuple(signals)


def _steo_records(path: Path) -> list[Record]:
    """Build one Record per STEO publication month with its forward WTI forecast path."""
    table = _read(path, ["pub_year", "pub_month", "target_month_start", "horizon_months", "wti"])
    spec = TimeSeriesSpec(
        spec_type="wti_forecast",
        name="STEO WTI spot price forecast (USD per barrel)",
        unit_value=None,
        dtype="float64",
    )
    years = table["pub_year"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    months = table["pub_month"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    table = table.append_column("pub_key", pa.array(years * 100 + months))
    records: list[Record] = []
    for key in sorted(table["pub_key"].unique().to_pylist()):
        rows = table.filter(pc.field("pub_key") == key)
        year, month = divmod(int(key), 100)
        signal = _irregular(
            rows["wti"].cast(pa.float64()).to_numpy(zero_copy_only=False),
            _us(rows["target_month_start"]),
            spec=spec,
            name="wti_forecast",
            annotations=_common(
                "EIA",
                f"STEO/{year}-{month:02d}/WTIPUUS",
                "monthly",
                "USD per barrel",
                0,
                information_cutoff=f"end of {year}-{month - 1:02d}" if month > 1 else f"end of {year - 1}-12",
            ),
        )
        if signal is None:
            continue
        record = Record(
            sources=(Source(name=f"EIA STEO {year}-{month:02d}", signals=(signal,)),),
            start_time=ORIGIN,
            metadata={"kind": "steo_vintage", "pub_month": f"{year}-{month:02d}", "forward_months": len(rows)},
        )
        records.append(record)
    return records


def _month(year: int, month: int, delta: int = 0) -> tuple[int, int]:
    """Return (year, month) shifted by ``delta`` months."""
    index = year * 12 + (month - 1) + delta
    return index // 12, index % 12 + 1


def _month_key(year: int, month: int) -> str:
    """Return the ``YYYY-MM`` key of a month."""
    return f"{year}-{month:02d}"


def _split_of(month_key: str) -> Split | None:
    """Return the split a target month falls in, or ``None`` past the last test month."""
    for split, last in SPLITS:
        if month_key <= last:
            return split
    return None


def _month_start_us(year: int, month: int) -> int:
    """Return the offset of 00:00 UTC on the first day of a month."""
    return offset_us(datetime(year, month, 1, tzinfo=UTC), ORIGIN.timestamp)


def _production_tasks(dataset: TimeFDataset, panel: Record, production: pa.Table) -> int:
    """Add the keyless task family: US crude production ``h`` months ahead, from the sealed panel.

    One ``ScalarPredictionTask`` per (origin month, horizon). The scope ends, exclusive, at the first
    day of the month after the information month minus the slowest publication lag, so a consumer
    that honours ``scope`` and nothing else sees no value that was unpublished at the origin. The
    split is by target month; a task whose origin and target fall in different splits is dropped.

    Returns:
        The number of tasks added.
    """
    offsets = _us(production["period_start"])
    values = production["value"].cast(pa.float64()).to_numpy(zero_copy_only=False)
    by_month: dict[str, float] = {}
    for moment, value in zip(
        production["period_start"].cast(pa.timestamp("us", tz="UTC")).to_pylist(), values, strict=True
    ):
        if np.isfinite(value):
            by_month[_month_key(moment.year, moment.month)] = float(value)
    del offsets
    first_year, first_month = (int(part) for part in TASK_ORIGIN_MONTHS[0].split("-"))
    last_key = TASK_ORIGIN_MONTHS[1]
    source = dataset.annotate(
        Annotation(
            key="target_source", value="EIA MCRFPUS2, US field production of crude oil, thousand barrels per day"
        )
    )
    added = 0
    year, month = first_year, first_month
    while _month_key(year, month) <= last_key:
        origin_key = _month_key(year, month)
        origin_split = _split_of(origin_key)
        cutoff = _month_start_us(*_month(year, month, 1 - SLOWEST_LAG_MONTHS))
        for horizon in HORIZONS_MONTHS:
            target_key = _month_key(*_month(year, month, horizon))
            target = by_month.get(target_key)
            split = _split_of(target_key)
            if target is None or split is None or split != origin_split:
                continue
            dataset.add_task(
                task=ScalarPredictionTask(
                    inputs=(panel,),
                    scope=TimeInterval.micros(0, cutoff),
                    targets=(target,),
                    target_name="us_crude_production",
                    split=split,
                    input_annotations=(source,),
                    metadata={
                        "information_month": origin_key,
                        "target_month": target_key,
                        "horizon_months": horizon,
                        "sealed_through": _month_key(*_month(year, month, -SLOWEST_LAG_MONTHS)),
                    },
                )
            )
            added += 1
        year, month = _month(year, month, 1)
    return added


class OilFundamentalsConnector(BaseHuggingFaceConnector):
    """Connector for the ``sethlei/oil-fundamentals`` panel (Hub repo ``sethlei/oil-fundamentals``)."""

    HF_REPO = "sethlei/oil-fundamentals"
    SOURCE_REVISION = "f0a8c82103c17a1e76f518721d59d1654a316bfd"
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {
        "stocks": "data/eia_us_crude_stocks_weekly.parquet",
        "production": "data/eia_us_crude_production_monthly.parquet",
        "international": "data/eia_international_production_monthly.parquet",
        "steo": "data/eia_steo_wti_vintages_forward.parquet",
        "jodi": "data/jodi_crude_primary.parquet",
        "cftc": "data/cftc_cot_wti_weekly.parquet",
        "frb": "data/frb_h10_dollar_index_daily.parquet",
        "gpr": "data/gpr_monthly.parquet",
    }

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Build the panel Record and one Record per STEO vintage.

        Args:
            raw_refs: The eight Parquet tables from :meth:`download`.

        Returns:
            The populated :class:`~timenet.dataset.TimeFDataset`, with the production forecasting tasks.
        """
        paths = self.ref_paths(raw_refs, *self.SOURCE_FILES)
        dataset = TimeFDataset(metadata=self.metadata())
        sources = (
            Source(name="EIA", signals=_eia_signals(paths)),
            Source(name="JODI", signals=_jodi_signals(paths["jodi"])),
            Source(name="CFTC", signals=_cftc_signals(paths["cftc"])),
            Source(name="FRB", signals=_frb_signals(paths["frb"])),
            Source(name="GPR", signals=_gpr_signals(paths["gpr"])),
        )
        panel = dataset.add_record(record=Record(sources=sources, start_time=ORIGIN, metadata={"kind": "panel"}))
        panel.add_annotations(
            [
                Annotation(key="snapshot_date", value=ACCESS_DATE),
                Annotation(key="revision_status", value="latest-revised on the snapshot date, not point-in-time"),
                Annotation(key="month_convention", value="month start"),
            ]
        )
        for record in _steo_records(paths["steo"]):
            dataset.add_record(record=record)
        _production_tasks(dataset, panel, _read(paths["production"], ["period_start", "value"]))
        dataset.annotate(
            Annotation(
                key="price_series",
                value=(
                    "None. WTI/Brent spot and NYMEX settles are licensed to EIA by Refinitiv (LSEG) and CME; "
                    "sethlei/oil-fundamentals-prices fetches them from EIA with the user's own API key."
                ),
            )
        )
        return dataset


CONNECTOR = OilFundamentalsConnector
