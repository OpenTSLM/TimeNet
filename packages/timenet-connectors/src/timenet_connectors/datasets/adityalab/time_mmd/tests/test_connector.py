from collections import Counter
import csv
from datetime import UTC, date, datetime, timedelta
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from timenet.dataset import TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import Annotation, ForecastingTask, Split, TimeInterval
from timenet_connectors.datasets.adityalab.time_mmd import TimeMmdConnector, sources
from timenet_connectors.datasets.adityalab.time_mmd.connector import TimeMmdDomainFiles
from timenet_connectors.datasets.adityalab.time_mmd.sources import ENERGY, ENVIRONMENT
from timenet_connectors.datasets.adityalab.time_mmd.tasks import ContextIndex, iter_windows, split_stretches


# Synthetic files shaped exactly like the release's CSVs. The values are made up for the tests, so
# the repo ships no dataset bytes. The column headers come from the domain declarations, because a
# header is part of the source shape the connector must match.
_ENERGY_FIRST = date(2000, 1, 3)
_ENERGY_ROWS = 120
_ENVIRONMENT_FIRST = date(2010, 1, 1)
_ENVIRONMENT_ROWS = 2000
_DAY = timedelta(days=1)
_DAY_US = 86_400_000_000

_CATEGORIES = ("Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy")
_PARAMETERS = ("Ozone", "PM2.5", "NO2", "CO", "PM10")
_CBSA = "New York-Newark-Jersey City, NY-NJ-PA"

# Text rows: (row index, start, end, fact, preds). The first starts before the numerical rows do.
_ENERGY_REPORT = [
    (0, date(1999, 12, 27), date(1999, 12, 31), "Prices rose before the series starts.", "Long one;Short one"),
    (1, date(2000, 1, 3), date(2000, 1, 9), "Prices held in the first week.", "NA;NA"),
    (2, date(2000, 6, 5), date(2000, 6, 11), "A fact; with a semicolon.", ""),
]
_ENERGY_SEARCH = [
    (0, date(2000, 2, 7), date(2000, 2, 13), "A search result.", "NA;Short two"),
    # A lone prediction, as the release has twice: no ";", so it does not say which half it holds.
    (1, date(2000, 3, 6), date(2000, 3, 12), "Another result.", "In the short-term, prices hold."),
]
_ENVIRONMENT_REPORT = [
    (0, date(2012, 6, 7), date(2012, 6, 7), "Smoke reached the city.", "Worse;Better"),
]
_ENVIRONMENT_SEARCH = [
    (0, date(2009, 12, 28), date(2010, 1, 3), "NA (nothing found)", "NA;NA"),
    (1, date(2015, 6, 1), date(2015, 6, 7), "Good air.", "NA;NA"),
]


def _energy_ot(index):
    return round(1.0 + 0.01 * index, 3)


def _write_energy(path, n=_ENERGY_ROWS, first=_ENERGY_FIRST, *, skip_row=None):
    headers = ["date", *(column.header for column in ENERGY.columns), "start_date", "end_date"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for index in range(n):
            if index == skip_row:
                continue
            start = first + index * ENERGY.period
            values = [_energy_ot(index), *(round(_energy_ot(index) + 0.1 * k, 3) for k in range(1, 9))]
            writer.writerow([start.isoformat(), *values, start.isoformat(), (start + 6 * _DAY).isoformat()])


def _write_environment(path, n=_ENVIRONMENT_ROWS, first=_ENVIRONMENT_FIRST, *, category=None):
    headers = ["CBSA", "CBSA Code", "date", "OT", "Category", "Defining Parameter", "Defining Site"]
    headers += ["Number of Sites Reporting", "start_date", "end_date"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for index in range(n):
            day = (first + index * _DAY).isoformat()
            writer.writerow(
                [
                    _CBSA,
                    "35620",
                    day,
                    20 + index % 50,
                    category or _CATEGORIES[index % len(_CATEGORIES)],
                    _PARAMETERS[index % len(_PARAMETERS)],
                    f"36-{index % 3:03d}-0001",
                    10 + index % 4,
                    day,
                    day,
                ]
            )


def _write_text(path, rows, header=("", "start_date", "end_date", "fact", "preds")):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, start, end, fact, preds in rows:
            writer.writerow([index, start.isoformat(), end.isoformat(), fact, preds])


def _refs(tmp_path, *, domains=("Energy", "Environment")):
    refs = []
    for domain in domains:
        root = tmp_path / domain
        root.mkdir()
        numerical, report, search = root / f"{domain}.csv", root / "report.csv", root / "search.csv"
        if domain == "Energy":
            _write_energy(numerical)
            _write_text(report, _ENERGY_REPORT)
            _write_text(search, _ENERGY_SEARCH)
        else:
            _write_environment(numerical)
            _write_text(report, _ENVIRONMENT_REPORT)
            _write_text(search, _ENVIRONMENT_SEARCH)
        refs.append(TimeMmdDomainFiles(domain=domain, numerical=numerical, report=report, search=search))
    return refs


def _convert(tmp_path, **kwargs) -> TimeFDataset:
    return TimeMmdConnector().convert(_refs(tmp_path, **kwargs))


def _record(dataset, domain):
    return next(record for record in dataset.records if record.metadata["domain"] == domain)


def _unix_us(day):
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()) * 1_000_000


def test_metadata():
    """The dataset.yaml declares the id, the ODC-By license, and the two domains."""
    metadata = TimeMmdConnector().metadata()
    assert metadata.dataset_id == "adityalab/time-mmd"
    assert str(metadata.license) == "ODC-By-1.0"
    assert {str(domain) for domain in metadata.domains} == {"energy", "environment"}


def test_convert_builds_one_record_per_domain(tmp_path):
    """Converting both domains gives two records, Energy then Environment, and the dataset-level
    annotations name the pinned source repository and revision and the target signal."""
    dataset = _convert(tmp_path)
    assert [record.metadata["domain"] for record in dataset.records] == ["Energy", "Environment"]
    assert {annotation.key: annotation.value for annotation in dataset.annotations} == {
        "source_repository": "AdityaLab/Time-MMD",
        "source_revision": "00281e2d86058286d5548b15a7670e8eda57ef62",
        "target_signal": "OT",
    }


def test_energy_signals_sit_on_a_weekly_axis_offset_from_the_earliest_text(tmp_path):
    """The Energy record's origin is the earliest text span, which starts a week before the first
    numerical row, so the weekly axis carries that week as its offset. The signals are ordered as
    the declaration lists them with OT first, hold float32 values, and the record's time span runs
    from the origin to the end of the last week. Energy has no static annotations."""
    record = _record(_convert(tmp_path), "Energy")
    assert record.start_time.timestamp == _unix_us(date(1999, 12, 27))
    names = [signal.name for signal in record.sources[0].signals]
    assert names == [column.signal_name for column in ENERGY.columns]
    assert names[0] == "OT"
    ot = record.sources[0].signals[0]
    assert isinstance(ot.time_axis, RegularAxis)
    assert ot.time_axis.period_us == Fraction(7 * _DAY_US)
    assert ot.time_axis.offset_us == 7 * _DAY_US  # the numerical rows start one week after the origin
    assert ot.n_values == _ENERGY_ROWS
    values = ot.to_numpy()
    assert values.dtype == np.float32
    assert values[3] == pytest.approx(_energy_ot(3))
    assert record.sources[0].signals[1].to_numpy()[3] == pytest.approx(_energy_ot(3) + 0.1)
    assert record.time_span == TimeInterval.micros(0, (_ENERGY_ROWS + 1) * 7 * _DAY_US)
    static = {annotation.key: annotation.value for annotation in record.annotations if annotation.span is None}
    assert static == {}


def test_environment_signals_keep_their_dtypes(tmp_path):
    """The Environment record keeps each column's declared dtype on a daily axis: int16 for the AQI
    and the site count, enum for the category, and str for the site id. The CBSA name and code,
    constant through the file, become the record's only static annotations."""
    record = _record(_convert(tmp_path), "Environment")
    signals = {signal.name: signal for signal in record.signals}
    assert set(signals) == {"OT", "category", "defining_parameter", "defining_site", "sites_reporting"}
    assert signals["OT"].time_axis.period_us == Fraction(_DAY_US)
    assert signals["OT"].to_numpy().dtype == np.int16
    assert int(signals["OT"].to_numpy()[7]) == 27
    assert signals["category"].to_arrow().to_pylist()[:3] == ["Good", "Moderate", "Unhealthy for Sensitive Groups"]
    assert signals["defining_site"].to_arrow().to_pylist()[1] == "36-001-0001"
    assert int(signals["sites_reporting"].to_numpy()[2]) == 12
    static = {annotation.key: annotation.value for annotation in record.annotations if annotation.span is None}
    assert static == {"cbsa": _CBSA, "cbsa_code": "35620"}


def test_text_rows_become_interval_annotations_verbatim(tmp_path):
    """Each text row yields a fact annotation and, per non-NA half of its prediction, a long-term or
    short-term one, keyed by file kind and stored verbatim. The span covers the row's dates
    inclusive of the end day, relative to the record origin, and the occurrence metadata names the
    source row. A prediction without a ";" yields no annotation."""
    record = _record(_convert(tmp_path), "Energy")
    spanned = [annotation for annotation in record.annotations if annotation.span is not None]
    by_key = Counter(annotation.key for annotation in spanned)
    # Report row 1 has an "NA;NA" prediction and row 2 an empty one, so each gets a fact only.
    # Search row 0 states a short-term prediction only; row 1 has no ";" and so no prediction.
    assert by_key == {
        "report_fact": 3,
        "report_prediction_long": 1,
        "report_prediction_short": 1,
        "search_fact": 2,
        "search_prediction_short": 1,
    }
    first = next(annotation for annotation in spanned if annotation.key == "report_fact")
    assert first.value == "Prices rose before the series starts."
    assert first.span == TimeInterval.micros(0, 5 * _DAY_US)
    assert first.occurrence_metadata == {"source_row": 0}
    halves = [(a.key, a.value) for a in spanned if a.key.startswith("report_prediction")]
    assert halves == [("report_prediction_long", "Long one"), ("report_prediction_short", "Short one")]
    report_row_one = [a.key for a in spanned if a.key.startswith("report") and a.occurrence_metadata["source_row"] == 1]
    assert report_row_one == ["report_fact"]
    shorts = [a.value for a in spanned if a.key == "search_prediction_short"]
    assert shorts == ["Short two"]
    semicolon = next(annotation for annotation in spanned if annotation.value == "A fact; with a semicolon.")
    assert semicolon.span.start_us == (7 + 22 * 7) * _DAY_US


def test_a_prediction_with_two_separators_raises(tmp_path):
    """A prediction cell with more than one ";" cannot be split into two halves and is rejected."""
    refs = _refs(tmp_path, domains=("Energy",))
    _write_text(refs[0].search, [(0, date(2000, 2, 7), date(2000, 2, 13), "A fact.", "Long;Short;Extra")])
    with pytest.raises(TimeFFormatError, match="holds one ';'"):
        TimeMmdConnector().convert(refs)


def test_split_stretches_follow_the_time_series_library_loader():
    """The stretches are the 70/10/20 chronological split of the Time-Series-Library loader, with
    validation and test each starting one lookback early. A training share shorter than the
    lookback is rejected."""
    assert split_stretches(120, 36) == {Split.TRAIN: (0, 84), Split.VALIDATION: (48, 96), Split.TEST: (60, 120)}
    with pytest.raises(TimeFValidationError, match="fewer than the 36 steps"):
        split_stretches(40, 36)


def test_windows_cover_each_stretch_with_stride_one():
    """Windows slide with stride one over each stretch, and the last window of a stretch ends on
    its last step. A horizon so long that only one window fits yields exactly that window."""
    windows = list(iter_windows(120, 36, 12))
    assert Counter(split for split, _ in windows) == {Split.TRAIN: 37, Split.VALIDATION: 1, Split.TEST: 13}
    assert windows[0] == (Split.TRAIN, 0)
    assert windows[36] == (Split.TRAIN, 36)
    assert windows[37] == (Split.VALIDATION, 48)
    assert windows[-1] == (Split.TEST, 72)
    assert list(iter_windows(120, 36, 48)) == [(Split.TRAIN, 0)]


def test_context_index_keeps_text_that_ends_inside_the_window():
    """A window may use an annotation that ends after the context starts and at or before the
    forecast boundary. One that ends on the context start, or runs past the boundary, is left out,
    and a static annotation never qualifies."""

    def spanned(start, end):
        return Annotation(key="t", value=f"{start}-{end}", span=TimeInterval.micros(start, end))

    before, on_start, inside, on_boundary, straddling = (
        spanned(0, 5),
        spanned(5, 10),
        spanned(12, 20),
        spanned(30, 40),
        spanned(35, 45),
    )
    index = ContextIndex([straddling, on_boundary, before, inside, on_start, Annotation(key="s", value="static")])
    assert index.within(10, 40) == (inside, on_boundary)
    assert index.within(0, 10) == (before, on_start)
    assert index.within(40, 100) == (straddling,)
    assert index.within(50, 60) == ()


def test_forecasting_tasks_follow_the_paper_protocol(tmp_path):
    """Every task is a ForecastingTask, one per window and horizon. The scope is the lookback
    interval over the whole record, the target the horizon interval over OT alone, and both map
    back to the expected step ranges. The input annotations are the text that falls in the
    context window, in order of its end, and the task metadata states the window's steps."""
    dataset = _convert(tmp_path)
    tasks = list(dataset.iter_tasks())
    assert all(isinstance(task, ForecastingTask) for task in tasks)
    energy = _record(dataset, "Energy")
    ot = energy.sources[0].signals[0]
    energy_tasks = [task for task in tasks if task.inputs == (energy,)]
    expected = Counter()
    for horizon in ENERGY.horizons:
        for split, _ in iter_windows(_ENERGY_ROWS, ENERGY.lookback, horizon):
            expected[split, horizon] += 1
    assert Counter((task.split, task.metadata["horizon_steps"]) for task in energy_tasks) == expected
    first = energy_tasks[0]
    assert first.metadata == {"context_steps": 36, "horizon_steps": 12, "first_context_step": 0}
    week = 7 * _DAY_US
    assert first.scope == TimeInterval.micros(week, week + 36 * week)
    assert first.targets == (TimeInterval.micros(week + 36 * week, week + 48 * week, time_series=(ot,)),)
    assert ot.step_range(first.scope) == (0, 36)
    assert ot.step_range(first.targets[0]) == (36, 48)
    # Report row 0 ends before the numerical rows start, so the first window does not see it.
    assert [(a.key, a.occurrence_metadata["source_row"]) for a in first.input_annotations] == [
        ("report_fact", 1),
        ("search_fact", 0),
        ("search_prediction_short", 0),
        ("search_fact", 1),
        ("report_fact", 2),
    ]
    assert all(any(a is attached for attached in energy.annotations) for a in first.input_annotations)
    assert first.target_annotations == ()
    last = energy_tasks[-1]
    assert last.split is Split.TRAIN
    assert last.metadata["horizon_steps"] == 48
    environment_tasks = [task for task in tasks if task.inputs != (energy,)]
    assert Counter(task.metadata["horizon_steps"] for task in environment_tasks).keys() == set(ENVIRONMENT.horizons)


def test_round_trip_through_storage(tmp_path):
    """Storing the dataset in a registry and reading it back preserves the records, the origin,
    the annotation counts, the typed signal values, and the tasks with their splits, metadata, and
    input annotations."""
    original = _convert(tmp_path)
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("adityalab/time-mmd", "1.0.0")) as reader:
        restored = reader.read()
    assert [record.metadata["domain"] for record in restored.records] == ["Energy", "Environment"]
    energy = _record(restored, "Energy")
    assert energy.start_time.timestamp == _unix_us(date(1999, 12, 27))
    spanned = [annotation for annotation in energy.annotations if annotation.span is not None]
    assert Counter(annotation.key for annotation in spanned) == {
        "report_fact": 3,
        "report_prediction_long": 1,
        "report_prediction_short": 1,
        "search_fact": 2,
        "search_prediction_short": 1,
    }
    environment = _record(restored, "Environment")
    signals = {signal.name: signal for signal in environment.signals}
    assert signals["category"].to_arrow().to_pylist()[:2] == ["Good", "Moderate"]
    assert int(signals["OT"].to_numpy()[7]) == 27
    original_tasks = list(original.iter_tasks())
    assert len(restored.tasks) == len(original_tasks)
    assert [(task.split, task.metadata) for task in restored.tasks] == [
        (task.split, task.metadata) for task in original_tasks
    ]
    ot = energy.sources[0].signals[0]
    assert ot.step_range(restored.tasks[0].scope) == (0, 36)
    assert [(a.key, a.value) for a in restored.tasks[0].input_annotations] == [
        (a.key, a.value) for a in original_tasks[0].input_annotations
    ]
    assert len(restored.tasks[0].input_annotations) == 5


def test_a_gap_in_the_numerical_rows_raises(tmp_path):
    """A numerical row whose date does not continue the cadence is rejected rather than filled."""
    refs = _refs(tmp_path, domains=("Energy",))
    _write_energy(refs[0].numerical, skip_row=10)
    with pytest.raises(TimeFFormatError, match="fills no gap"):
        TimeMmdConnector().convert(refs)


def test_a_missing_column_raises(tmp_path):
    """A numerical file that lacks one of the declared columns is rejected."""
    refs = _refs(tmp_path, domains=("Energy",))
    with refs[0].numerical.open() as handle:
        lines = handle.readlines()
    with refs[0].numerical.open("w") as handle:
        handle.writelines(line.replace("OT,", "target,", 1) for line in lines)
    with pytest.raises(TimeFFormatError, match="lacks the columns"):
        TimeMmdConnector().convert(refs)


def test_a_category_outside_the_codebook_raises(tmp_path):
    """A category value outside the enum's declared codebook is rejected."""
    refs = _refs(tmp_path, domains=("Environment",))
    _write_environment(refs[0].numerical, category="Apocalyptic")
    with pytest.raises(TimeFValidationError, match="Apocalyptic"):
        TimeMmdConnector().convert(refs)


def test_a_text_file_with_another_header_raises(tmp_path):
    """A text file whose header differs from the release's is rejected."""
    refs = _refs(tmp_path, domains=("Energy",))
    _write_text(refs[0].report, _ENERGY_REPORT, header=("start_date", "end_date", "fact", "preds", "extra"))
    with pytest.raises(TimeFFormatError, match="expected the header"):
        TimeMmdConnector().convert(refs)


def test_an_unknown_or_repeated_domain_raises(tmp_path):
    """The connector refuses a domain it does not declare, a domain given twice, and no domain."""
    refs = _refs(tmp_path, domains=("Energy",))
    other = Path(refs[0].numerical)
    with pytest.raises(TimeFValidationError, match="unknown: \\['Traffic'\\]"):
        TimeMmdConnector().convert([TimeMmdDomainFiles(domain="Traffic", numerical=other, report=other, search=other)])
    with pytest.raises(TimeFValidationError, match="at most once"):
        TimeMmdConnector().convert([*refs, *refs])
    with pytest.raises(TimeFValidationError, match="at least one domain"):
        TimeMmdConnector().convert([])


def test_source_is_pinned():
    """The source repository, commit, and file paths are the literal values the connector pins."""
    # Literal pins, so moving to another commit or file is a deliberate test edit.
    assert sources.SOURCE_REPOSITORY == "AdityaLab/Time-MMD"
    assert sources.SOURCE_REVISION == "00281e2d86058286d5548b15a7670e8eda57ef62"
    assert ENERGY.numerical_file == "numerical/Energy/Energy.csv"
    assert ENVIRONMENT.text_file("search") == "textual/Environment/Environment_search.csv"
