"""Compose the open oil-fundamentals panel with EIA's licensed price series, fetched by the user.

The parent ``sethlei/oil-fundamentals`` is open. The six price series (WTI Cushing and Brent Europe
spot, NYMEX WTI contracts 1-4) are republished by EIA under licence, so this connector downloads
them from EIA's API v2 with the user's own key instead of redistributing them. ``compose`` builds
one new combined panel Record on the shared 1920 origin, imports the STEO vintage records, derives
the monthly WTI mean, and adds the two price task families.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.errors import TimeNetConnectorError, TimeNetDownloadError
from timenet.types import Annotation, ScalarPredictionTask, TimeInterval, TimeSeriesSpec
from timenet.types.clock import offset_us
from timenet.types.reference_keys import ParentAlias
from timenet_connectors.datasets.sethlei.oil_fundamentals.connector import (
    ACCESS_DATE,
    HORIZONS_MONTHS,
    ORIGIN,
    SLOWEST_LAG_MONTHS,
    TASK_ORIGIN_MONTHS,
    _month,
    _month_key,
    _month_start_us,
    _split_of,
)


if TYPE_CHECKING:
    from timenet.connectors.context import BuildContext
    from timenet.reader import TimeFReader

EIA_API = "https://api.eia.gov/v2/"
REGISTER_URL = "https://www.eia.gov/opendata/register.php"
WINDOW = ("1983-01-01", "2026-10-31")
"""The fixed fetch window. A new window is a new dataset version."""
PAGE = 5000
SATURDAY = 5
"""``date.weekday()`` of Saturday; Saturday and Sunday are not trading days."""
PINS_PATH = Path(__file__).with_name("pins.json")
PARENT = ParentAlias("fundamentals")

# series id -> (route, display name, licensor)
SERIES: dict[str, tuple[str, str, str]] = {
    "RWTC": ("petroleum/pri/spt/data/", "WTI Cushing spot price FOB", "Refinitiv, an LSEG business"),
    "RBRTE": ("petroleum/pri/spt/data/", "Brent Europe spot price FOB", "Refinitiv, an LSEG business"),
    "RCLC1": ("petroleum/pri/fut/data/", "NYMEX WTI futures contract 1", "CME Group (NYMEX)"),
    "RCLC2": ("petroleum/pri/fut/data/", "NYMEX WTI futures contract 2", "CME Group (NYMEX)"),
    "RCLC3": ("petroleum/pri/fut/data/", "NYMEX WTI futures contract 3", "CME Group (NYMEX)"),
    "RCLC4": ("petroleum/pri/fut/data/", "NYMEX WTI futures contract 4", "CME Group (NYMEX)"),
}
PRICE_SPECS = {
    series: TimeSeriesSpec(
        spec_type=f"price_{series.lower()}", name=f"{label} (USD per barrel)", unit_value=None, dtype="float64"
    )
    for series, (_, label, _) in SERIES.items()
}
MONTHLY_MEAN_SPEC = TimeSeriesSpec(
    spec_type="wti_monthly_mean",
    name="WTI Cushing spot, calendar-month unweighted mean of daily closes (USD per barrel)",
    unit_value=None,
    dtype="float64",
)


@dataclass(frozen=True)
class PriceSeriesRef:
    """One fetched series: its canonical rows on disk, their hash, and whether it drifted from the pin."""

    series: str
    path: Path
    sha256: str
    pinned: str | None


def _canonical_rows(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Reduce EIA rows to sorted ``(period, value)`` pairs, values as served.

    Returns:
        The pairs sorted by period, one per period.
    """
    pairs: dict[str, str] = {}
    for row in rows:
        period, value = row.get("period"), row.get("value")
        if period is None or value is None:
            continue
        pairs[str(period)] = str(value)
    return sorted(pairs.items())


def _sha256(pairs: list[tuple[str, str]]) -> str:
    """Hash the canonical pairs."""
    digest = hashlib.sha256()
    for period, value in pairs:
        digest.update(f"{period}\t{value}\n".encode())
    return digest.hexdigest()


def _fetch_series(client: httpx.Client, series: str, api_key: str) -> list[dict[str, Any]]:
    """Page through one EIA series over the fixed window.

    The key travels as a query parameter and never in a logged URL string.

    Returns:
        Every row EIA returned.

    Raises:
        TimeNetDownloadError: If EIA answers with an error or an unexpected shape.
    """
    route = SERIES[series][0]
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        params = {
            "api_key": api_key,
            "frequency": "daily",
            "data[0]": "value",
            "facets[series][]": series,
            "start": WINDOW[0],
            "end": WINDOW[1],
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": str(offset),
            "length": str(PAGE),
        }
        try:
            response = client.get(route, params=params)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise TimeNetDownloadError(
                f"EIA API v2 request for {series} failed: {type(exc).__name__} (key redacted)"
            ) from exc
        body = payload.get("response", {})
        page = body.get("data", [])
        if not isinstance(page, list):
            raise TimeNetDownloadError(f"EIA API v2 returned no data list for {series}")
        rows.extend(page)
        total = int(body.get("total", len(rows)))
        offset += len(page)
        if not page or offset >= total:
            return rows


def _load_pins() -> dict[str, str]:
    """Read the pinned canonical hashes beside the connector."""
    return json.loads(PINS_PATH.read_text()) if PINS_PATH.exists() else {}


class OilFundamentalsPricesConnector(BaseConnector[PriceSeriesRef]):
    """Connector for ``sethlei/oil-fundamentals-prices``: the open panel plus user-fetched EIA prices."""

    def __init__(self) -> None:
        # Never raise here: TimeNet's discovery instantiates every connector, with or without a key.
        self.api_key = os.environ.get("EIA_TOKEN") or os.environ.get("EIA_API_KEY")
        self.accept_revisions = os.environ.get("OIL_PRICES_ACCEPT_REVISIONS") == "1"

    def download(self, cache_dir: Path) -> list[PriceSeriesRef]:
        """Fetch the six price series from EIA with the user's key, canonicalise, hash and pin-check them.

        Returns:
            One reference per series.

        Raises:
            TimeNetConnectorError: If no key is set, or a series drifted from its pin and revisions
                are not accepted.
            TimeNetDownloadError: If EIA returns no rows for a series.
        """
        if not self.api_key:
            raise TimeNetConnectorError(
                "sethlei/oil-fundamentals-prices needs your own EIA API key: register (free) at "
                f"{REGISTER_URL} and set EIA_TOKEN. The price series are licensed to EIA and are fetched "
                "by you under EIA's terms, never redistributed."
            )
        pins = _load_pins()
        refs: list[PriceSeriesRef] = []
        with httpx.Client(base_url=EIA_API, timeout=120.0) as client:
            for series in SERIES:
                path = cache_dir / f"{series}.json"
                if path.exists():
                    pairs = [tuple(pair) for pair in json.loads(path.read_text())]
                else:
                    pairs = _canonical_rows(_fetch_series(client, series, self.api_key))
                    if not pairs:
                        raise TimeNetDownloadError(f"EIA returned no rows for {series} in {WINDOW[0]}..{WINDOW[1]}")
                    path.write_text(json.dumps(pairs))
                sha = _sha256([(str(period), str(value)) for period, value in pairs])
                pinned = pins.get(series)
                if pinned not in {None, sha} and not self.accept_revisions:
                    raise TimeNetConnectorError(
                        f"{series} differs from its pinned canonical hash (EIA revised history or changed "
                        "the window). Set OIL_PRICES_ACCEPT_REVISIONS=1 to build anyway and record the new hash."
                    )
                refs.append(PriceSeriesRef(series, path, sha, pinned))
        return refs

    # ruff: ignore[unused-method-argument, no-self-use]
    def convert(self, raw_refs: list[PriceSeriesRef]) -> TimeFDataset:
        """Unused: the card declares a parent, so the engine calls :meth:`compose`.

        Raises:
            TimeNetConnectorError: Always.
        """
        raise TimeNetConnectorError("sethlei/oil-fundamentals-prices composes over its parent; use compose()")

    def compose(self, raw_refs: list[PriceSeriesRef], context: BuildContext) -> TimeFDataset:
        """Build the combined panel, import the STEO vintages, derive the monthly mean, add the tasks.

        Returns:
            The populated dataset.
        """
        reader = context.parents[PARENT]
        dataset = TimeFDataset(metadata=self.metadata())
        parent_panel, vintage_ids = _find_parent_records(reader)
        sources = [_copy_source(source) for source in parent_panel.sources]
        prices, daily_wti = _price_signals(raw_refs)
        monthly = _monthly_means(daily_wti)
        sources.append(
            Source(name="EIA prices (fetched by the user)", signals=(*prices, _monthly_mean_signal(monthly)))
        )
        panel = dataset.add_record(record=Record(sources=tuple(sources), start_time=ORIGIN, metadata={"kind": "panel"}))
        for annotation in parent_panel.annotations:
            panel.annotate(Annotation(key=annotation.key, value=annotation.value))
        panel.annotate(Annotation(key="price_fetch_window", value=f"{WINDOW[0]}..{WINDOW[1]}"))
        for ref in raw_refs:
            panel.annotate(
                Annotation(
                    key=f"canonical_sha256_{ref.series}",
                    value=ref.sha256
                    if ref.pinned in {None, ref.sha256}
                    else f"{ref.sha256} (REVISED from pin {ref.pinned})",
                )
            )
        imported = dataset.import_from(reader, alias=PARENT, records=vintage_ids)
        vintages = list(imported.records.values())
        target_source = dataset.annotate(
            Annotation(
                key="target_source",
                value="Derived: calendar-month unweighted mean of the WTI Cushing daily close (Refinitiv, an LSEG business, "
                "via EIA, fetched by the user); complete months only",
            )
        )
        _t1_tasks(dataset, panel, monthly, target_source)
        _t2_tasks(dataset, panel, monthly, vintages, target_source)
        return dataset


def _find_parent_records(reader: TimeFReader) -> tuple[Record, list[int]]:
    """Return the parent's panel Record (values loaded lazily) and the ids of its STEO vintage records.

    Raises:
        TimeNetConnectorError: If the parent has no panel.
    """
    panel: Record | None = None
    vintages: list[int] = []
    for record in reader.iter_records():
        kind = record.metadata.get("kind")
        if kind == "panel":
            panel = record
        elif kind == "steo_vintage":
            vintages.append(record.id)
    if panel is None:
        raise TimeNetConnectorError("the sethlei/oil-fundamentals parent has no panel record")
    return panel, vintages


def _copy_source(source: Source) -> Source:
    """Rebuild a parent Source with fresh Signals holding the same values, axes, specs and annotations."""
    signals: list[Signal] = []
    for signal in source.signals:
        values = signal.to_arrow()
        axis = signal.time_axis
        if axis.__class__.__name__ == "RegularAxis":
            fresh = Signal.from_values(values, spec=signal.spec, name=signal.name, time_axis=axis)
        else:
            fresh = Signal.from_irregular(
                signal.to_numpy(), time_offsets_us=signal.time_offsets_us(), spec=signal.spec, name=signal.name
            )
        for annotation in signal.annotations:
            fresh.annotate(Annotation(key=annotation.key, value=annotation.value))
        signals.append(fresh)
    return Source(
        name=source.name, sources=tuple(_copy_source(child) for child in source.sources), signals=tuple(signals)
    )


def _price_signals(raw_refs: list[PriceSeriesRef]) -> tuple[list[Signal], dict[date, float]]:
    """Build the six daily price signals and return the WTI daily closes by date."""
    signals: list[Signal] = []
    wti: dict[date, float] = {}
    for ref in raw_refs:
        pairs = json.loads(ref.path.read_text())
        dates, values = [], []
        for period, value in pairs:
            try:
                day, number = date.fromisoformat(str(period)), float(value)
            except ValueError:
                continue
            if not np.isfinite(number):
                continue
            dates.append(day)
            values.append(number)
        if not dates:
            continue
        offsets = np.array(
            [offset_us(datetime(d.year, d.month, d.day, tzinfo=UTC), ORIGIN.timestamp) for d in dates], dtype=np.int64
        )
        signal = Signal.from_irregular(
            np.array(values, dtype=np.float64),
            time_offsets_us=offsets,
            spec=PRICE_SPECS[ref.series],
            name=f"eia_{ref.series}",
        )
        licensor = SERIES[ref.series][2]
        for key, value in {
            "publisher": "EIA",
            "licensor": licensor,
            "upstream_series_id": ref.series,
            "frequency": "daily (trading days)",
            "unit_label": "USD per barrel",
            "fetched_by_user": "true",
            "fetch_window": f"{WINDOW[0]}..{WINDOW[1]}",
            "canonical_sha256": ref.sha256,
            "publication_lag_months": 0,
            "citation": f"{licensor}, as published by the U.S. Energy Information Administration ({ref.series}), fetched by the user",
        }.items():
            signal.annotate(Annotation(key=key, value=value))
        signals.append(signal)
        if ref.series == "RWTC":
            wti = dict(zip(dates, values, strict=True))
    if not wti:
        raise TimeNetConnectorError("RWTC is required to derive the monthly WTI mean")
    return signals, wti


def _last_business_day(year: int, month: int) -> date:
    """Return the last weekday of a month."""
    next_year, next_month = _month(year, month, 1)
    day = date(next_year, next_month, 1) - timedelta(days=1)
    while day.weekday() >= SATURDAY:
        day -= timedelta(days=1)
    return day


def _monthly_means(daily: dict[date, float]) -> dict[str, float]:
    """Calendar-month unweighted means of the daily closes, complete months only.

    A month is complete when the series reaches its last weekday.

    Returns:
        Means keyed by ``YYYY-MM``.
    """
    sums: dict[str, list[float]] = {}
    for day, value in daily.items():
        sums.setdefault(_month_key(day.year, day.month), []).append(value)
    last = max(daily)
    means: dict[str, float] = {}
    for key, values in sums.items():
        year, month = (int(part) for part in key.split("-"))
        if (year, month) < (last.year, last.month) or last >= _last_business_day(year, month):
            means[key] = float(np.mean(values))
    return means


def _monthly_mean_signal(monthly: dict[str, float]) -> Signal:
    """Build the derived monthly-mean signal, stamped at month start."""
    keys = sorted(monthly)
    offsets = np.array([_month_start_us(*(int(part) for part in key.split("-"))) for key in keys], dtype=np.int64)
    signal = Signal.from_irregular(
        np.array([monthly[key] for key in keys], dtype=np.float64),
        time_offsets_us=offsets,
        spec=MONTHLY_MEAN_SPEC,
        name="wti_monthly_mean",
    )
    for key, value in {
        "publisher": "derived",
        "derived_from": "eia_RWTC",
        "frequency": "monthly (complete months)",
        "unit_label": "USD per barrel",
        "publication_lag_months": 0,
        "access_date": ACCESS_DATE,
        "citation": "Derived from Refinitiv (LSEG) data as published by EIA; no licence from Refinitiv is claimed",
    }.items():
        signal.annotate(Annotation(key=key, value=value))
    return signal


def _sealed_cutoff_us(info_year: int, info_month: int) -> int:
    """Exclusive scope end for an information month: first day of (info month - slowest lag + 1)."""
    return _month_start_us(*_month(info_year, info_month, 1 - SLOWEST_LAG_MONTHS))


def _t1_tasks(dataset: TimeFDataset, panel: Record, monthly: dict[str, float], target_source: Annotation) -> int:
    """Add T1: WTI monthly mean ``h`` months ahead of each origin month, split by target month.

    Returns:
        The number of tasks added.
    """
    first_year, first_month = (int(part) for part in TASK_ORIGIN_MONTHS[0].split("-"))
    year, month, added = first_year, first_month, 0
    while _month_key(year, month) <= TASK_ORIGIN_MONTHS[1]:
        origin_key = _month_key(year, month)
        origin_split = _split_of(origin_key)
        for horizon in HORIZONS_MONTHS:
            target_key = _month_key(*_month(year, month, horizon))
            target, split = monthly.get(target_key), _split_of(target_key)
            if target is None or split is None or split != origin_split:
                continue
            dataset.add_task(
                task=ScalarPredictionTask(
                    inputs=(panel,),
                    scope=TimeInterval.micros(0, _sealed_cutoff_us(year, month)),
                    targets=(target,),
                    target_name="wti_monthly_mean",
                    split=split,
                    input_annotations=(target_source,),
                    metadata={
                        "family": "T1",
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


def _t2_tasks(
    dataset: TimeFDataset, panel: Record, monthly: dict[str, float], vintages: list[Record], target_source: Annotation
) -> int:
    """Add T2: for each STEO vintage and horizon, the realized WTI monthly mean, with EIA's forecast as input.

    The information month is the month before publication (the release day is not on disk).

    Returns:
        The number of tasks added.
    """
    return sum(_vintage_tasks(dataset, panel, monthly, vintage, target_source) for vintage in vintages)


def _vintage_tasks(
    dataset: TimeFDataset, panel: Record, monthly: dict[str, float], vintage: Record, target_source: Annotation
) -> int:
    """Add the T2 tasks of one STEO vintage.

    Returns:
        The number of tasks added.
    """
    pub_key = str(vintage.metadata.get("pub_month", ""))
    if not pub_key:
        return 0
    info_year, info_month = _month(*(int(part) for part in pub_key.split("-")), -1)
    info_split = _split_of(_month_key(info_year, info_month))
    signal = vintage.sources[0].signals[0]
    forecasts = dict(zip(signal.time_offsets_us().tolist(), signal.to_numpy().tolist(), strict=True))
    added = 0
    for horizon in HORIZONS_MONTHS:
        target_year, target_month = _month(info_year, info_month, horizon)
        target_key = _month_key(target_year, target_month)
        target, split = monthly.get(target_key), _split_of(target_key)
        forecast = forecasts.get(_month_start_us(target_year, target_month))
        if target is None or split is None or split != info_split or forecast is None or not np.isfinite(forecast):
            continue
        steo = dataset.annotate(Annotation(key="steo_forecast", value=float(forecast)))
        dataset.add_task(
            task=ScalarPredictionTask(
                inputs=(panel,),
                scope=TimeInterval.micros(0, _sealed_cutoff_us(info_year, info_month)),
                targets=(target,),
                target_name="wti_monthly_mean",
                split=split,
                input_annotations=(target_source, steo),
                metadata={
                    "family": "T2",
                    "pub_month": pub_key,
                    "information_month": _month_key(info_year, info_month),
                    "target_month": target_key,
                    "horizon_months": horizon,
                    "steo_forecast": float(forecast),
                    "sealed_through": _month_key(*_month(info_year, info_month, -SLOWEST_LAG_MONTHS)),
                },
            )
        )
        added += 1
    return added


CONNECTOR = OilFundamentalsPricesConnector
