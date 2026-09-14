import csv
from datetime import UTC, date, datetime, timedelta
import hashlib
from pathlib import Path
import shutil

import pytest

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.engine import store_dataset
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import assert_datasets_equal
from timenet.types import AnswerTask, ForecastingTask, TimeInterval
from timenet_connectors.datasets.adityalab.time_mmd import connector as connector_module
from timenet_connectors.datasets.adityalab.time_mmd.connector import (
    RELEASE_DIGESTS,
    TIME_MMD_COMMIT,
    TimeMmdConnector,
    TimeMmdSource,
)
from timenet_connectors.datasets.adityalab.time_mmd.domains import DOMAINS, ENERGY, ENVIRONMENT
from timenet_connectors.datasets.adityalab.time_mmd.keys import AnnotationKey, TextKind
from timenet_connectors.datasets.adityalab.time_mmd.timeline import US_PER_DAY


# A synthetic release of the two domains. These tests write it, and it holds no byte of the
# real release: every value, date and text below is invented. What is kept is the shape: the
# headers, the cadences, the row index the textual files carry, and the cases the connector
# decides on. A text before the first value, one after the last, a fact that reads "NA", a
# fact that starts with "NASA", an empty field, and two reports on one day.
#
# The series are long enough for one forecasting task at the smallest horizon of each domain
# and none at the larger ones. The test origin is n - n // 5, and a horizon fits when
# origin + horizon <= n. Sixty weeks give origin 48, so a 12-week horizon fits once. Two
# hundred and forty days give origin 192, so a 48-day horizon fits once.
_ENERGY_WEEKS = 60
_ENERGY_FIRST = date(2020, 1, 6)  # a Monday, as every date of the real file is
_ENERGY_LAST = _ENERGY_FIRST + timedelta(days=7 * (_ENERGY_WEEKS - 1))

_ENVIRONMENT_DAYS = 240
_ENVIRONMENT_FIRST = date(2021, 1, 1)
_ENVIRONMENT_LAST = _ENVIRONMENT_FIRST + timedelta(days=_ENVIRONMENT_DAYS - 1)

_CATEGORIES = ("Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy")
_POLLUTANTS = ("Ozone", "PM2.5", "NO2", "CO")

_TEXT_HEADER = ("", "start_date", "end_date", "fact", "preds")

_ENERGY_REPORTS = [
    # A week inside the series. Its fact quotes the following Monday, as the real reports do.
    (
        "2020-01-13",
        "2020-01-17",
        "Synthetic report: the price rose to $2.62 per gallon on January 20, 2020.",
        "Long term: a synthetic outlook.;Short term: a synthetic outlook.",
    ),
    # Nothing found, in the form the prompt allowed.
    ("2020-02-03", "2020-02-07", "NA (no relevant information found)", "NA;NA"),
    # Filed after the last value of the series. Its fact stays, and it gets no caption task.
    ("2021-03-01", "2021-03-05", "Synthetic report filed after the last value.", ""),
]

_ENERGY_SEARCHES = [
    # Two weeks before the first value. The record's zero moves back to cover it.
    (
        "2019-12-23",
        "2019-12-29",
        "Synthetic search fact before the series began.",
        "NA;Short term: a synthetic outlook.",
    ),
    ("2020-01-06", "2020-01-12", "", "NA;NA"),
    # A fact that starts with the letters NA and states something.
    ("2020-02-10", "2020-02-16", "NASA launched a synthetic satellite.", "Long term: a synthetic outlook.;NA"),
]

_ENVIRONMENT_REPORTS = [
    ("2021-06-03", "2021-06-03", "Synthetic air quality report.", "NA;NA"),
    # A second report on the same day.
    ("2021-06-03", "2021-06-03", "Synthetic second report of the day.", "Long term: synthetic.;Short term: synthetic."),
    # The day after the last value.
    ("2021-08-29", "2021-08-29", "Synthetic late report.", "NA"),
]

_ENVIRONMENT_SEARCHES = [
    ("2021-05-31", "2021-06-06", "Synthetic environment search fact.", "Long term: synthetic.;Short term: synthetic."),
    # A week that straddles the last value. Its predictions hold one part and no separator.
    ("2021-08-23", "2021-08-29", "NA", "Long term only, in one part."),
]


def _energy_rows() -> list[dict[str, str]]:
    rows = []
    for index in range(_ENERGY_WEEKS):
        day = _ENERGY_FIRST + timedelta(days=7 * index)
        us = 2.0 + 0.01 * index
        row = {"date": day.isoformat(), "OT": f"{us:.3f}"}
        for offset, column in enumerate(one.column for one in ENERGY.signals[1:]):
            row[column] = f"{us + 0.05 * (offset + 1):.3f}"
        row["start_date"] = day.isoformat()
        row["end_date"] = (day + timedelta(days=6)).isoformat()
        rows.append(row)
    return rows


def _environment_rows() -> list[dict[str, str]]:
    rows = []
    for index in range(_ENVIRONMENT_DAYS):
        day = _ENVIRONMENT_FIRST + timedelta(days=index)
        rows.append(
            {
                "CBSA": "Synthetic City, ZZ",
                "CBSA Code": "99999",
                "date": day.isoformat(),
                "OT": str(20 + (index * 7) % 160),
                "Category": _CATEGORIES[index % len(_CATEGORIES)],
                "Defining Parameter": _POLLUTANTS[index % len(_POLLUTANTS)],
                "Defining Site": f"99-999-{index % 3:04d}",
                "Number of Sites Reporting": str(10 + index % 3),
                "start_date": day.isoformat(),
                "end_date": day.isoformat(),
            }
        )
    return rows


def _write_csv(path: Path, header: list[str] | tuple[str, ...], rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_numerical(root: Path, shape, rows: list[dict[str, str]]) -> None:
    header = list(rows[0].keys())
    _write_csv(shape.numerical_path(root), header, [[row[column] for column in header] for row in rows])


def _write_textual(root: Path, shape, kind: TextKind, rows: list[tuple[str, str, str, str]]) -> None:
    _write_csv(shape.textual_path(root, kind), _TEXT_HEADER, [[index, *row] for index, row in enumerate(rows)])


def _write_release(root: Path, *, energy_rows=None, environment_rows=None, energy_reports=None) -> Path:
    _write_numerical(root, ENERGY, energy_rows if energy_rows is not None else _energy_rows())
    _write_textual(root, ENERGY, TextKind.REPORT, energy_reports if energy_reports is not None else _ENERGY_REPORTS)
    _write_textual(root, ENERGY, TextKind.SEARCH, _ENERGY_SEARCHES)
    _write_numerical(root, ENVIRONMENT, environment_rows if environment_rows is not None else _environment_rows())
    _write_textual(root, ENVIRONMENT, TextKind.REPORT, _ENVIRONMENT_REPORTS)
    _write_textual(root, ENVIRONMENT, TextKind.SEARCH, _ENVIRONMENT_SEARCHES)
    return root


@pytest.fixture(scope="session")
def release(tmp_path_factory) -> Path:
    return _write_release(tmp_path_factory.mktemp("time_mmd_release"))


@pytest.fixture(scope="session")
def dataset(release: Path) -> TimeFDataset:
    return _convert(release)


def _convert(root: Path) -> TimeFDataset:
    return TimeMmdConnector().convert([TimeMmdSource(root=root)])


def _record(dataset: TimeFDataset, record_id: str):
    return next(one for one in dataset.records if one.record_id == record_id)


def _series(dataset: TimeFDataset, record_id: str, signal: str):
    return next(one for one in _record(dataset, record_id).time_series if one.signal == signal)


def _annotations(dataset: TimeFDataset, record_id: str, key: str) -> list:
    return [one for one in _record(dataset, record_id).annotations if one.key == key]


def _streamed(dataset: TimeFDataset) -> list:
    return list(dataset.iter_tasks())


def _day(record, offset_us: int) -> date:
    return (datetime.fromtimestamp(record.start_time / 1_000_000, tz=UTC) + timedelta(microseconds=offset_us)).date()


def test_is_a_connector():
    assert isinstance(TimeMmdConnector(), BaseConnector)


def test_metadata():
    metadata = TimeMmdConnector().metadata()
    assert metadata.dataset_id == "adityalab/time-mmd"
    assert str(metadata.license) == "ODC-By-1.0"
    assert {str(domain) for domain in metadata.domains} == {"energy", "environment"}


def test_the_digests_name_the_three_files_of_each_domain():
    assert set(RELEASE_DIGESTS) == {path for shape in DOMAINS for path in shape.relative_paths}


def _synthetic_digests(release: Path) -> dict[str, str]:
    return {relative: hashlib.sha256((release / relative).read_bytes()).hexdigest() for relative in RELEASE_DIGESTS}


def _fake_download_files(release: Path, cache_dir: Path, seen: list):
    # Stands in for the download helper: it records the artifacts and copies the synthetic release
    # into place, and like the real helper it leaves a file that is already there alone.
    async def fake(artifacts):
        seen.extend(artifacts)
        for artifact in artifacts:
            if not artifact.dest.exists():
                artifact.dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(release / artifact.dest.relative_to(cache_dir), artifact.dest)
        return [artifact.dest for artifact in artifacts]

    return fake


def test_download_fetches_the_pinned_files_and_keeps_their_layout(release, monkeypatch, tmp_path: Path):
    seen: list = []
    digests = _synthetic_digests(release)
    monkeypatch.setattr(connector_module, "RELEASE_DIGESTS", digests)
    monkeypatch.setattr(connector_module, "download_files", _fake_download_files(release, tmp_path, seen))
    sources = TimeMmdConnector().download(tmp_path)

    assert sources == [TimeMmdSource(root=tmp_path)]
    assert {artifact.url for artifact in seen} == {
        f"https://raw.githubusercontent.com/AdityaLab/Time-MMD/{TIME_MMD_COMMIT}/{relative}" for relative in digests
    }
    assert {artifact.dest for artifact in seen} == {tmp_path / relative for relative in digests}
    assert all(artifact.sha256 == digests[artifact.dest.relative_to(tmp_path).as_posix()] for artifact in seen)


def test_a_cached_file_with_other_bytes_stops_the_download(release, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(connector_module, "RELEASE_DIGESTS", _synthetic_digests(release))
    monkeypatch.setattr(connector_module, "download_files", _fake_download_files(release, tmp_path, []))
    TimeMmdConnector().download(tmp_path)
    (tmp_path / "numerical" / "Energy" / "Energy.csv").write_text("other bytes", encoding="utf-8")
    with pytest.raises(TimeNetDownloadError, match=r"Energy\.csv holds bytes whose SHA-256"):
        TimeMmdConnector().download(tmp_path)


def test_a_file_the_download_did_not_write_stops_the_build(monkeypatch, tmp_path: Path):
    async def nothing(artifacts):
        return [artifact.dest for artifact in artifacts]

    monkeypatch.setattr(connector_module, "download_files", nothing)
    with pytest.raises(TimeNetDownloadError, match="is missing"):
        TimeMmdConnector().download(tmp_path)


def test_one_record_for_each_domain(dataset):
    assert [record.record_id for record in dataset.records] == ["time-mmd-energy", "time-mmd-environment"]


def test_convert_is_deterministic(release):
    assert_datasets_equal(_convert(release), _convert(release))


def test_energy_carries_the_nine_prices_as_float_signals(dataset):
    record = _record(dataset, "time-mmd-energy")
    assert [one.signal for one in record.time_series] == [one.signal for one in ENERGY.signals]
    assert {one.spec.spec_type for one in record.time_series} == {"retail_gasoline_price"}
    values = _series(dataset, "time-mmd-energy", "us").to_numpy()
    assert len(values) == _ENERGY_WEEKS
    assert float(values[0]) == pytest.approx(2.0)
    assert float(values[-1]) == pytest.approx(2.0 + 0.01 * (_ENERGY_WEEKS - 1), abs=1e-5)


def test_environment_carries_typed_signals(dataset):
    aqi = _series(dataset, "time-mmd-environment", "aqi")
    assert aqi.spec.dtype == "int16"
    assert [int(value) for value in aqi.to_numpy()[:3]] == [20, 27, 34]
    category = _series(dataset, "time-mmd-environment", "category")
    assert category.spec.dtype == "enum"
    assert category.to_arrow().to_pylist()[:2] == ["Good", "Moderate"]
    site = _series(dataset, "time-mmd-environment", "defining_site")
    assert site.spec.dtype == "str"
    assert site.to_arrow().to_pylist()[0] == "99-999-0000"
    assert int(_series(dataset, "time-mmd-environment", "sites_reporting").to_numpy()[1]) == 11


def test_the_axis_states_the_cadence_of_the_domain(dataset):
    weekly = _series(dataset, "time-mmd-energy", "us").time_axis
    daily = _series(dataset, "time-mmd-environment", "aqi").time_axis
    assert isinstance(weekly, RegularAxis) and isinstance(daily, RegularAxis)
    assert weekly.period_us == 7 * US_PER_DAY
    assert daily.period_us == US_PER_DAY


def test_the_record_zero_moves_back_to_cover_the_earliest_text(dataset):
    record = _record(dataset, "time-mmd-energy")
    # The earliest search row starts two weeks before the first value.
    assert record.start_time == int(datetime(2019, 12, 23, tzinfo=UTC).timestamp()) * 1_000_000
    axis = _series(dataset, "time-mmd-energy", "us").time_axis
    assert isinstance(axis, RegularAxis)
    assert axis.start_index == 2
    assert _day(record, _series(dataset, "time-mmd-energy", "us").span_us[0]) == _ENERGY_FIRST


def test_the_record_zero_stays_on_the_first_value_when_no_text_precedes_it(dataset):
    record = _record(dataset, "time-mmd-environment")
    assert record.start_time == int(datetime(2021, 1, 1, tzinfo=UTC).timestamp()) * 1_000_000
    axis = _series(dataset, "time-mmd-environment", "aqi").time_axis
    assert isinstance(axis, RegularAxis)
    assert axis.start_index == 0


def test_the_session_reaches_the_later_of_the_last_value_and_the_last_text(dataset):
    energy = _record(dataset, "time-mmd-energy")
    assert energy.time_span == TimeInterval.micros(0, energy.time_span.end_us)
    assert _day(energy, energy.time_span.end_us) == date(2021, 3, 6)  # the day after the late report
    environment = _record(dataset, "time-mmd-environment")
    assert _day(environment, environment.time_span.end_us) == date(2021, 8, 30)


def test_each_record_states_its_target_signal_and_source_columns(dataset):
    assert _annotations(dataset, "time-mmd-energy", AnnotationKey.TARGET_SIGNAL)[0].value == "us"
    assert _annotations(dataset, "time-mmd-environment", AnnotationKey.TARGET_SIGNAL)[0].value == "aqi"
    columns = _annotations(dataset, "time-mmd-energy", AnnotationKey.SOURCE_COLUMNS)[0].value
    assert columns["us"] == "OT"
    assert columns["west_coast"].startswith("Weekly West Coast")


def test_the_constant_columns_become_static_annotations(dataset):
    assert _annotations(dataset, "time-mmd-environment", AnnotationKey.CBSA)[0].value == "Synthetic City, ZZ"
    assert _annotations(dataset, "time-mmd-environment", AnnotationKey.CBSA_CODE)[0].value == 99999
    assert _annotations(dataset, "time-mmd-energy", AnnotationKey.CBSA) == []


def test_a_text_becomes_an_annotation_over_the_days_it_states(dataset):
    record = _record(dataset, "time-mmd-energy")
    fact = _annotations(dataset, "time-mmd-energy", AnnotationKey.REPORT_FACT)[0]
    assert fact.id == "time-mmd-energy-report-fact-0"
    assert fact.value == _ENERGY_REPORTS[0][2]
    assert fact.source == "Llama3-70B"
    assert isinstance(fact.span, TimeInterval)
    assert (_day(record, fact.span.start_us), _day(record, fact.span.end_us)) == (date(2020, 1, 13), date(2020, 1, 18))
    assert fact.span.time_series_ids is None


def test_a_field_that_states_nothing_becomes_no_annotation(dataset):
    assert [one.id for one in _annotations(dataset, "time-mmd-energy", AnnotationKey.REPORT_FACT)] == [
        "time-mmd-energy-report-fact-0",
        "time-mmd-energy-report-fact-2",
    ]
    assert [one.id for one in _annotations(dataset, "time-mmd-energy", AnnotationKey.REPORT_PREDICTION)] == [
        "time-mmd-energy-report-prediction-0",
    ]


def test_a_prediction_with_one_part_stated_is_kept_whole(dataset):
    kept = _annotations(dataset, "time-mmd-energy", AnnotationKey.SEARCH_PREDICTION)
    assert [one.value for one in kept] == ["NA;Short term: a synthetic outlook.", "Long term: a synthetic outlook.;NA"]


def test_a_fact_that_starts_with_the_letters_na_is_kept(dataset):
    facts = _annotations(dataset, "time-mmd-energy", AnnotationKey.SEARCH_FACT)
    assert "NASA launched a synthetic satellite." in [one.value for one in facts]


def test_text_outside_the_values_is_kept_in_the_record(dataset):
    record = _record(dataset, "time-mmd-energy")
    before = _annotations(dataset, "time-mmd-energy", AnnotationKey.SEARCH_FACT)[0]
    assert before.span is not None and before.span.start_us == 0
    after = _annotations(dataset, "time-mmd-energy", AnnotationKey.REPORT_FACT)[1]
    assert after.span is not None and _day(record, after.span.start_us) == date(2021, 3, 1)
    assert after.span.end_us == record.time_span.end_us


def test_two_reports_on_one_day_are_two_annotations(dataset):
    facts = _annotations(dataset, "time-mmd-environment", AnnotationKey.REPORT_FACT)
    assert [one.id for one in facts[:2]] == [
        "time-mmd-environment-report-fact-0",
        "time-mmd-environment-report-fact-1",
    ]
    assert facts[0].span == facts[1].span


def test_the_tasks_stream_and_cover_both_types(dataset):
    assert dataset.has_task_stream
    tasks = _streamed(dataset)
    assert {type(task) for task in tasks} == {ForecastingTask, AnswerTask}
    assert len(tasks) == 5


def test_one_forecasting_task_at_the_smallest_horizon(dataset):
    record = _record(dataset, "time-mmd-energy")
    forecasts = [
        task
        for task in _streamed(dataset)
        if isinstance(task, ForecastingTask) and task.record_ids == (record.record_id,)
    ]
    assert [task.id for task in forecasts] == ["time-mmd-energy-forecast-h12-o48"]
    task = forecasts[0]
    assert isinstance(task.scope, TimeInterval) and isinstance(task.target_span, TimeInterval)
    assert _day(record, task.scope.start_us) == _ENERGY_FIRST  # not the record's zero, two weeks earlier
    assert task.scope.end_us == task.target_span.start_us
    assert _day(record, task.target_span.start_us) == _ENERGY_FIRST + timedelta(days=7 * 48)
    assert _day(record, task.target_span.end_us) == _ENERGY_LAST + timedelta(days=7)
    assert task.target_span.time_series_ids == ("time-mmd-energy-us",)
    assert task.prompt is None


def test_a_caption_asks_for_a_report_fact_by_reference(dataset):
    record = _record(dataset, "time-mmd-energy")
    captions = [
        task for task in _streamed(dataset) if isinstance(task, AnswerTask) and task.record_ids == (record.record_id,)
    ]
    assert [task.id for task in captions] == ["time-mmd-energy-report-fact-0-caption"]
    task = captions[0]
    assert task.target_annotation_ids == ("time-mmd-energy-report-fact-0",)
    assert task.target is None
    assert task.prompt is None
    assert isinstance(task.scope, TimeInterval)
    assert (_day(record, task.scope.start_us), _day(record, task.scope.end_us)) == (_ENERGY_FIRST, date(2020, 1, 18))


def test_a_report_past_the_last_value_gets_no_caption(dataset):
    captions = [task for task in _streamed(dataset) if isinstance(task, AnswerTask)]
    assert {task.target_annotation_ids[0] for task in captions} == {
        "time-mmd-energy-report-fact-0",
        "time-mmd-environment-report-fact-0",
        "time-mmd-environment-report-fact-1",
    }


def test_search_facts_get_no_caption(dataset):
    captions = [task for task in _streamed(dataset) if isinstance(task, AnswerTask)]
    assert not any("search" in task.target_annotation_ids[0] for task in captions)


def test_convert_round_trips_through_the_writer(release, tmp_path: Path):
    dataset = _convert(release)
    dataset.derive_schema()
    version_dir = store_dataset(dataset, tmp_path / "out")
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
    assert [record.record_id for record in restored.records] == ["time-mmd-energy", "time-mmd-environment"]
    assert len(restored.tasks) == 5
    schema = restored.schema
    assert schema is not None
    assert {annotation.key for annotation in schema.annotations} >= {
        "report_fact",
        "report_prediction",
        "search_fact",
        "search_prediction",
        "target_signal",
        "source_columns",
        "cbsa",
        "cbsa_code",
    }
    energy = _record(restored, "time-mmd-energy")
    assert energy.start_time == _record(dataset, "time-mmd-energy").start_time
    assert energy.time_span == _record(dataset, "time-mmd-energy").time_span
    restored_aqi = _series(restored, "time-mmd-environment", "aqi").to_numpy()
    assert [int(value) for value in restored_aqi[:3]] == [20, 27, 34]
    assert _series(restored, "time-mmd-environment", "category").to_arrow().to_pylist()[0] == "Good"


def test_a_week_missing_from_the_series_stops_the_build(tmp_path: Path):
    rows = _energy_rows()
    del rows[10]
    with pytest.raises(TimeFFormatError, match="cadence"):
        _convert(_write_release(tmp_path, energy_rows=rows))


def test_a_missing_value_stops_the_build(tmp_path: Path):
    rows = _energy_rows()
    rows[3]["OT"] = ""
    with pytest.raises(TimeFFormatError, match="holds no value"):
        _convert(_write_release(tmp_path, energy_rows=rows))


def test_a_period_that_disagrees_with_its_date_stops_the_build(tmp_path: Path):
    rows = _energy_rows()
    rows[3]["end_date"] = rows[3]["date"]
    with pytest.raises(TimeFFormatError, match="states the period"):
        _convert(_write_release(tmp_path, energy_rows=rows))


def test_a_category_outside_the_codebook_stops_the_build(tmp_path: Path):
    rows = _environment_rows()
    rows[0]["Category"] = "Pleasant"
    with pytest.raises(TimeFFormatError, match="does not name"):
        _convert(_write_release(tmp_path, environment_rows=rows))


def test_a_constant_column_that_varies_stops_the_build(tmp_path: Path):
    rows = _environment_rows()
    rows[5]["CBSA"] = "Another City, ZZ"
    with pytest.raises(TimeFFormatError, match="distinct values"):
        _convert(_write_release(tmp_path, environment_rows=rows))


def test_a_text_that_ends_before_it_starts_stops_the_build(tmp_path: Path):
    reports = [*_ENERGY_REPORTS, ("2020-03-02", "2020-02-24", "Synthetic reversed row.", "NA;NA")]
    with pytest.raises(TimeFFormatError, match="before it starts"):
        _convert(_write_release(tmp_path, energy_reports=reports))


def test_a_date_that_does_not_parse_stops_the_build(tmp_path: Path):
    reports = [*_ENERGY_REPORTS, ("2020/03/02", "2020-03-06", "Synthetic row.", "NA;NA")]
    with pytest.raises(TimeFFormatError, match="YYYY-MM-DD"):
        _convert(_write_release(tmp_path, energy_reports=reports))


def test_a_missing_column_stops_the_build(tmp_path: Path):
    rows = [{column: value for column, value in row.items() if column != "OT"} for row in _energy_rows()]
    with pytest.raises(TimeFFormatError, match="lacks the columns"):
        _convert(_write_release(tmp_path, energy_rows=rows))
