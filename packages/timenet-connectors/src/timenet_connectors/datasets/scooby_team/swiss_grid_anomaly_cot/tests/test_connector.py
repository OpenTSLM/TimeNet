from datetime import UTC, datetime
import json

import numpy as np
import openpyxl
import pandas as pd
import pytest

from timenet.dataset import TimeFDataset
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import assert_datasets_equal
from timenet.types import AnswerTask, ClassificationTask, Split, TemporalLocalizationTask
from timenet.types.text import read_value
from timenet_connectors.datasets.scooby_team.swiss_grid_anomaly_cot._pipeline import swissgrid
from timenet_connectors.datasets.scooby_team.swiss_grid_anomaly_cot._pipeline.swissgrid import (
    SwissgridFormatError,
    labels_to_utc_start,
    parse_labels,
)
from timenet_connectors.datasets.scooby_team.swiss_grid_anomaly_cot._pipeline.timefields import field, normalize
from timenet_connectors.datasets.scooby_team.swiss_grid_anomaly_cot.connector import (
    CAUSE,
    GRID_SIGNALS,
    SwissGridAnomalyCotConnector,
    build_dataset,
)


TZ = "Europe/Zurich"
_HOLIDAY = pd.Timestamp("2025-05-29")  # a Thursday of the test year


def _hourly() -> pd.DataFrame:
    """Six years in the shape of the connector's hourly table: a heating response, a weekend dip,
    noise, and one public holiday with a deep dip in the test year."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2019-12-31 23:00", "2025-12-31 22:00", freq="h", tz="UTC")
    local = idx.tz_convert(TZ)
    date = normalize(local.tz_localize(None))
    temp = 10 - 10 * np.cos(2 * np.pi * (field(local, "day_of_year") - 15) / 365) + rng.normal(0, 2, len(idx))
    weekend = field(local, "day_of_week") >= 5
    load = 6000 + 60 * np.maximum(0, 16 - temp) - 900 * weekend + rng.normal(0, 60, len(idx))
    holiday = date == _HOLIDAY
    load = np.where(holiday, load * 0.75, load)
    df = pd.DataFrame(index=idx)
    df["load_enduser_mw"] = load
    df["load_total_mw"] = load + 700
    df["production_mw"] = load + 900
    for border, value in {"at": 100.0, "de": 300.0, "fr": 400.0, "it": -1000.0}.items():
        df[f"net_import_{border}_mw"] = value
    df["net_import_mw"] = -200.0
    df["temp_c"] = temp
    df["radiation_wm2"] = 150.0
    df["local_date"] = date
    df["local_hour"] = field(local, "hour")
    df["dow"] = field(local, "day_of_week")
    df["holiday_pop_share"] = np.where(holiday, 1.0, 0.0)
    df["holiday_name"] = ["Ascension Day" if flag else None for flag in holiday]
    df["is_holiday"] = holiday
    df["is_bridge_day"] = False
    df["year"] = field(date, "year")
    return df


@pytest.fixture(scope="module")
def dataset() -> TimeFDataset:
    return build_dataset(TimeFDataset(metadata=SwissGridAnomalyCotConnector().metadata()), _hourly())


def test_each_iso_week_becomes_a_record_with_three_tasks(dataset):
    records = dataset.records
    # ISO weeks with all seven days in 2020-2025: 2020-W02 to 2025-W52.
    assert len(records) == 312
    assert len(dataset.tasks_of(ClassificationTask)) == 312
    assert len(dataset.tasks_of(TemporalLocalizationTask)) == 312
    assert len(dataset.tasks_of(AnswerTask)) == 312
    splits = [task.split for task in dataset.tasks_of(ClassificationTask)]
    assert (splits.count(Split.TRAIN), splits.count(Split.VALIDATION), splits.count(Split.TEST)) == (208, 52, 52)
    first = records[0]
    assert first.metadata == {"window_id": "2020-W02"}
    grid, weather = first.sources
    assert [signal.name for signal in grid.signals] == list(GRID_SIGNALS.values())
    assert [signal.name for signal in weather.signals] == ["Temperature", "Global radiation"]


def test_records_carry_utc_start_and_dst_lengths(dataset):
    by_week = {record.metadata["window_id"]: record for record in dataset.records}
    spring = by_week["2025-W13"]  # contains 30 March 2025
    assert spring.start_time.timestamp == int(datetime(2025, 3, 23, 23, tzinfo=UTC).timestamp() * 1_000_000)
    assert len(spring.signals[0].to_numpy()) == 167
    assert len(by_week["2025-W43"].signals[0].to_numpy()) == 169  # contains 26 October 2025
    assert len(by_week["2025-W20"].signals[0].to_numpy()) == 168


def test_the_holiday_dip_is_found_explained_and_localized(dataset):
    week = next(r for r in dataset.records if r.metadata["window_id"] == "2025-W22")
    cause = next(t for t in dataset.tasks_of(ClassificationTask) if t.inputs[0] is week)
    assert cause.label_set == CAUSE
    assert cause.targets is not None
    assert read_value(cause.targets[0]) == "public_holiday"
    assert "Ascension Day" not in read_value(cause.prompt)  # the holiday is read from the curve, not the prompt
    rationale = read_value(cause.rationale)
    assert "Ascension Day" in rationale
    assert rationale.endswith("Answer: public_holiday")
    days = next(t for t in dataset.tasks_of(TemporalLocalizationTask) if t.inputs[0] is week)
    assert days.targets is not None
    (span,) = days.targets
    # Thursday is the fourth local day of the week: hours 72 to 96 of the record.
    assert (span.start_us, span.end_us) == (72 * 3_600_000_000, 96 * 3_600_000_000)
    facts = json.loads(read_value(next(a.value for a in week.annotations if a.key == "facts")))
    assert facts["label"] == "public_holiday"


def test_exchange_answer(dataset):
    task = dataset.tasks_of(AnswerTask)[0]
    assert task.targets is not None
    assert read_value(task.targets[0]) == "net exporter"
    assert "Italy (1,000 MW)" in read_value(task.rationale)


def test_the_dataset_round_trips_through_storage(dataset, tmp_path):
    version_dir = dataset.write(path=tmp_path / "registry")
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        assert_datasets_equal(dataset, reader.read())


def _render(instants: pd.DatetimeIndex) -> list[str]:
    return [(t.tz_localize(None) + t.tz_convert(TZ).utcoffset()).strftime("%d.%m.%Y %H:%M") for t in instants]


def _year(year: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(year, 1, 1).tz_localize(TZ).tz_convert("UTC")
    end = pd.Timestamp(year + 1, 1, 1).tz_localize(TZ).tz_convert("UTC")
    return pd.date_range(start, end, freq="15min", inclusive="left")


def test_both_label_conventions_parse_to_the_same_quarters():
    end_labels = parse_labels(pd.Series(_render(_year(2024) + pd.Timedelta("15min"))))
    start_labels = parse_labels(pd.Series(_render(_year(2025))))
    assert labels_to_utc_start(end_labels, 2024).equals(_year(2024).rename("utc_start"))
    assert labels_to_utc_start(start_labels, 2025).equals(_year(2025).rename("utc_start"))
    with pytest.raises(SwissgridFormatError, match="rows"):
        labels_to_utc_start(end_labels.drop(index=28810).reset_index(drop=True), 2024)


def test_the_workbook_columns_are_read_by_header(tmp_path):
    headers = list(swissgrid.COLUMNS)
    order = [headers[2], "Decoy", headers[1], *headers[3:], headers[0]]
    path = tmp_path / "EnergieUebersichtCH-2021.xlsx"
    workbook = openpyxl.Workbook(write_only=True)
    sheet = workbook.create_sheet(swissgrid.SHEET)
    sheet.append([None, *(f"Deutsch\n{header}" for header in order)])
    sheet.append(["Zeitstempel", *(["kWh"] * len(order))])
    for label in _render(_year(2021) + pd.Timedelta("15min")):
        sheet.append([label, *range(1, len(order) + 1)])
    workbook.save(path)
    frame = swissgrid.read_year(path, 2021)
    for position, header in enumerate(order, start=1):
        if header in swissgrid.COLUMNS:
            assert (frame[swissgrid.COLUMNS[header]] == position).all(), header
