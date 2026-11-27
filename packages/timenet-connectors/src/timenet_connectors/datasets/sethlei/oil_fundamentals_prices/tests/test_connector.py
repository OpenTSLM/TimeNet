"""Synthetic test: a tiny parent in a temporary registry, synthetic EIA rows, no network."""

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset
from timenet.engine import open_parents, store_dataset
from timenet.registry import LocalRegistry
from timenet.types import Split, TimeInterval
from timenet.types.clock import offset_us
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.sethlei.oil_fundamentals import OilFundamentalsConnector
from timenet_connectors.datasets.sethlei.oil_fundamentals.connector import ORIGIN
from timenet_connectors.datasets.sethlei.oil_fundamentals_prices import OilFundamentalsPricesConnector
from timenet_connectors.datasets.sethlei.oil_fundamentals_prices.connector import (
    SERIES,
    PriceSeriesRef,
    _canonical_rows,
    _monthly_means,
    _sha256,
)


def _ts(*days: str) -> pa.Array:
    return pa.array([datetime.fromisoformat(day) for day in days], type=pa.timestamp("ns"))


def _parent_refs(root: Path) -> list[HubFileRef]:
    """Write the smallest parent snapshot that yields a panel and one STEO vintage (pub 2020-01)."""
    tables = {
        "stocks": pa.table({"period_end": _ts("2019-12-06", "2019-12-13"), "value": [440.0, 441.0]}),
        "production": pa.table({"period_start": _ts("2019-12-01", "2020-01-01"), "value": [12800.0, 12700.0]}),
        "international": pa.table(
            {
                "period_start": _ts("2019-12-01"),
                "countryRegionId": ["USA"],
                "countryRegionName": ["United States"],
                "is_aggregate": [False],
                "value": [19000.0],
                "missing_reason": [None],
            }
        ),
        "steo": pa.table(
            {
                "pub_year": [2020, 2020],
                "pub_month": [1, 1],
                "target_month_start": _ts("2020-01-01", "2020-03-01"),
                "horizon_months": [0, 2],
                "wti": [59.0, 55.0],
            }
        ),
        "jodi": pa.table(
            {
                "ref_area": ["US"],
                "period_start": _ts("2019-12-01"),
                "flow_breakdown": ["INDPROD"],
                "unit_measure": ["KBD"],
                "value": [12800.0],
            }
        ),
        "cftc": pa.table({"date": _ts("2019-12-03"), "open_interest": [2_100_000.0]}),
        "frb": pa.table({"series_id": ["DTWEXBGS"], "date": _ts("2019-12-02"), "value": [114.1]}),
        "gpr": pa.table({"month": _ts("2019-12-01"), "gpr": [120.0], "gpr_threats": [130.0], "gpr_acts": [100.0]}),
    }
    refs = []
    for part, table in tables.items():
        path = root / OilFundamentalsConnector.SOURCE_FILES[part]
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        refs.append(HubFileRef(path, part))
    return refs


def _price_refs(root: Path) -> list[PriceSeriesRef]:
    """Synthetic EIA rows in the API's ``period``/``value`` shape: every weekday of Dec 2019 to Mar 2020."""
    root.mkdir(parents=True, exist_ok=True)
    refs = []
    for index, series in enumerate(SERIES):
        rows = []
        day = datetime(2019, 12, 2)
        while day <= datetime(2020, 3, 31):
            if day.weekday() < 5:
                rows.append({"period": day.strftime("%Y-%m-%d"), "value": f"{50 + index + (day.month - 12) % 12:.2f}"})
            day += timedelta(days=1)
        pairs = _canonical_rows(rows)
        path = root / f"{series}.json"
        path.write_text(json.dumps(pairs))
        refs.append(PriceSeriesRef(series, path, _sha256(pairs), None))
    return refs


@pytest.fixture
def registry(tmp_path: Path) -> LocalRegistry:
    parent = OilFundamentalsConnector().convert(_parent_refs(tmp_path / "parent"))
    store_dataset(parent, tmp_path / "registry")
    return LocalRegistry(tmp_path / "registry")


def test_monthly_means_use_complete_months_only() -> None:
    daily = {datetime(2020, 1, d).date(): 60.0 for d in range(2, 32) if datetime(2020, 1, d).weekday() < 5}
    daily[datetime(2020, 2, 3).date()] = 70.0  # February is incomplete
    assert _monthly_means(daily) == {"2020-01": 60.0}


def test_compose_builds_combined_panel_and_price_tasks(tmp_path: Path, registry: LocalRegistry) -> None:
    connector = OilFundamentalsPricesConnector()
    with open_parents(connector.metadata(), registry) as context:
        dataset = connector.compose(_price_refs(tmp_path / "prices"), context)
        assert isinstance(dataset, TimeFDataset)
        panel = dataset.records[0]
        assert panel.start_time == ORIGIN
        names = {signal.name for source in panel.sources for signal in source.signals}
        assert {"eia_WCESTUS1", "jodi_US_INDPROD", "eia_RWTC", "eia_RCLC4", "wti_monthly_mean"} <= names
        # The STEO vintage is imported, not rebuilt, and no second panel exists.
        kinds = [record.metadata.get("kind") for record in dataset.records]
        assert kinds.count("panel") == 1
        assert kinds.count("steo_vintage") == 1
        families = {task.metadata["family"] for task in dataset.tasks}
        assert families == {"T1", "T2"}
        # pub 2020-01 -> information month 2019-12. The vintage forecasts 2020-01 (h=1) and 2020-03 (h=3),
        # and the synthetic prices complete both months, so T2 holds exactly those two tasks.
        t2_tasks = sorted(
            (task for task in dataset.tasks if task.metadata["family"] == "T2"),
            key=lambda task: int(str(task.metadata["horizon_months"])),
        )
        assert [task.metadata["target_month"] for task in t2_tasks] == ["2020-01", "2020-03"]
        t2 = t2_tasks[1]
        # horizon 3 -> target 2020-03; sealed through 2019-08.
        assert t2.metadata["information_month"] == "2019-12"
        assert t2.metadata["target_month"] == "2020-03"
        assert t2.metadata["steo_forecast"] == pytest.approx(55.0)
        assert t2.split == Split("validation")
        assert isinstance(t2.scope, TimeInterval)
        assert t2.scope.end_us == offset_us(datetime(2019, 9, 1, tzinfo=UTC), ORIGIN.timestamp)
        store_dataset(dataset, registry.root)
