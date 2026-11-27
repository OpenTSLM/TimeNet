"""Fixture test: tiny synthetic tables shaped like the snapshot, converted without network."""

from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset.axis import RegularAxis
from timenet.types import Split, TimeInterval
from timenet.types.clock import offset_us
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.sethlei.oil_fundamentals import OilFundamentalsConnector
from timenet_connectors.datasets.sethlei.oil_fundamentals.connector import ORIGIN


def _ts(*days: str) -> pa.Array:
    return pa.array([datetime.fromisoformat(day) for day in days], type=pa.timestamp("ns"))


def _write(root: Path, part: str, table: pa.Table) -> HubFileRef:
    path = root / OilFundamentalsConnector.SOURCE_FILES[part]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return HubFileRef(path, part)


@pytest.fixture
def refs(tmp_path: Path) -> list[HubFileRef]:
    # Weekly stocks with one missing week (2020-01-17), so the 7-day grid gets a declared null.
    stocks = pa.table({"period_end": _ts("2020-01-03", "2020-01-10", "2020-01-24"), "value": [430.0, 431.5, 428.0]})
    production = pa.table({"period_start": _ts("2020-01-01", "2020-02-01"), "value": [12800.0, 12700.0]})
    international = pa.table(
        {
            "period_start": _ts("2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01", "2020-01-01"),
            "countryRegionId": ["USA", "USA", "WORL", "WORL", "GONE"],
            "countryRegionName": ["United States", "United States", "World", "World", "Gone"],
            "is_aggregate": [False, False, True, True, False],
            "value": [19000.0, 18900.0, 100000.0, 99000.0, None],
            "missing_reason": [None, None, None, None, "NOT_APPLICABLE"],
        }
    )
    steo = pa.table(
        {
            "pub_year": [2020, 2020, 2020],
            "pub_month": [1, 1, 2],
            "target_month_start": _ts("2020-01-01", "2020-02-01", "2020-02-01"),
            "horizon_months": [0, 1, 0],
            "wti": [59.0, 60.0, 50.0],
        }
    )
    jodi = pa.table(
        {
            "ref_area": ["US", "US", "US", "XX"],
            "period_start": _ts("2020-01-01", "2020-02-01", "2020-01-01", "2020-01-01"),
            "flow_breakdown": ["INDPROD", "INDPROD", "CLOSTLV", "INDPROD"],
            "unit_measure": ["KBD", "KBD", "KBBL", "KBD"],
            "value": [12800.0, 12700.0, 430000.0, None],
        }
    )
    cftc = pa.table(
        {
            "date": _ts("2020-01-07", "2020-01-14"),
            "open_interest": [2_100_000.0, 2_150_000.0],
            "prod_merc_long": [500_000.0, 510_000.0],
            "prod_merc_short": [700_000.0, 690_000.0],
        }
    )
    frb = pa.table(
        {
            "series_id": ["DTWEXBGS", "DTWEXBGS", "DTWEXM"],
            "date": _ts("2020-01-02", "2020-01-03", "2019-12-31"),
            "value": [114.1, 114.3, 92.0],
        }
    )
    gpr = pa.table(
        {
            "month": _ts("2020-01-01", "2020-02-01"),
            "gpr": [120.0, 110.0],
            "gpr_threats": [130.0, 115.0],
            "gpr_acts": [100.0, 95.0],
        }
    )
    tables = {
        "stocks": stocks,
        "production": production,
        "international": international,
        "steo": steo,
        "jodi": jodi,
        "cftc": cftc,
        "frb": frb,
        "gpr": gpr,
    }
    return [_write(tmp_path, part, table) for part, table in tables.items()]


def test_convert_builds_panel_and_vintages(refs: list[HubFileRef]) -> None:
    dataset = OilFundamentalsConnector().convert(refs)
    records = dataset.records
    assert len(records) == 1 + 2  # the panel and two STEO vintages
    panel = records[0]
    assert panel.start_time == ORIGIN
    by_source = {source.name: source.signals for source in panel.sources}
    assert set(by_source) == {"EIA", "JODI", "CFTC", "FRB", "GPR"}
    # EIA: weekly stocks, US production, and two international series (the all-null one is dropped).
    assert [signal.name for signal in by_source["EIA"]] == [
        "eia_WCESTUS1",
        "eia_MCRFPUS2",
        "eia_intl_USA",
        "eia_intl_WORL",
    ]
    stocks = by_source["EIA"][0]
    assert stocks.spec.missing is not None
    assert stocks.n_values == 4  # 2020-01-03 .. 2020-01-24 on a 7-day grid, one declared null
    assert stocks.to_arrow().null_count == 1
    # JODI: the all-null XX series is dropped; US has production and closing stocks in their own units.
    jodi_names = sorted(signal.name for signal in by_source["JODI"])
    assert jodi_names == ["jodi_US_CLOSTLV", "jodi_US_INDPROD"]
    units = {signal.name: str(signal.spec.unit_value) for signal in by_source["JODI"]}
    assert units["jodi_US_CLOSTLV"] == "kilooil_barrel"
    assert units["jodi_US_INDPROD"] == "kilooil_barrel / day"
    assert len(by_source["CFTC"]) == 3
    assert len(by_source["FRB"]) == 2
    assert len(by_source["GPR"]) == 3
    # STEO vintages: forward rows only, one record per publication month.
    vintages = records[1:]
    assert [record.metadata["pub_month"] for record in vintages] == ["2020-01", "2020-02"]
    assert vintages[0].sources[0].signals[0].n_values == 2
    # The fixture holds production for 2020-01 and 2020-02, so every origin whose target lands on one
    # of those months gets a task: origins 2019-01/02 (h=12), 2019-07/08 (h=6), 2019-10/11 (h=3),
    # 2019-12 and 2020-01 (h=1). All origins and targets are in the validation split, so none is dropped.
    tasks = dataset.tasks
    assert len(tasks) == 8
    assert {task.split for task in tasks} == {Split("validation")}
    (task,) = [task for task in tasks if task.metadata["information_month"] == "2020-01"]
    assert task.split == Split("validation")
    assert task.targets == (12700.0,)
    assert task.metadata["information_month"] == "2020-01"
    assert task.metadata["target_month"] == "2020-02"
    assert task.metadata["sealed_through"] == "2019-09"
    assert isinstance(task.scope, TimeInterval)
    assert task.scope.end_us == offset_us(datetime(2019, 10, 1, tzinfo=UTC), ORIGIN.timestamp)


def test_every_signal_carries_provenance(refs: list[HubFileRef]) -> None:
    dataset = OilFundamentalsConnector().convert(refs)
    panel = dataset.records[0]
    required = {
        "publisher",
        "upstream_series_id",
        "frequency",
        "unit_label",
        "access_date",
        "publication_lag_months",
        "citation",
    }
    for source in panel.sources:
        for signal in source.signals:
            keys = {annotation.key for annotation in signal.annotations}
            assert required <= keys, (signal.name, required - keys)


def test_weekly_grid_offset_is_the_first_friday(refs: list[HubFileRef]) -> None:
    dataset = OilFundamentalsConnector().convert(refs)
    stocks = dataset.records[0].sources[0].signals[0]
    first = offset_us(datetime(2020, 1, 3, tzinfo=UTC), ORIGIN.timestamp)
    axis = stocks.time_axis
    assert isinstance(axis, RegularAxis)
    assert axis.time_offset_us(0) == first
