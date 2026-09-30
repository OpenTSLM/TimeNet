"""Offline tests for the ARFBench connector.

The fixture is hand-written and shaped like the release: a QA table and long-format Parquet files
named ``{incident}_{index}_{interval}.parquet``. It ships no ARFBench bytes. Each file pins one
hazard the real release carries: shuffled rows, a pandas index column, a query_name that names
another series, a null value, an empty tag label, a gap, and a metric published at two intervals but
not at every interval.
"""

import csv
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.connectors import BaseConnector
from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.engine import store_dataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import AnswerTask
from timenet_connectors.datasets.datadog.arfbench import ARFBenchConnector, connector as arfbench


_ANCHOR_S = arfbench.ANCHOR_US // 1_000_000
_YES = ["Yes, there is an anomaly", "No, there is no anomaly"]
_FIVE = ["Both", "Neither", "Only the first", "Only the second", "Cannot tell"]

# One row per file: (name, [(group, offset seconds, value)], pandas index values).
_FILES = {
    # Two tag groups on a complete 10 s grid, rows shuffled, index column out of order.
    "900_0_10": (
        [("dc:1", 10 * (step + 1), 1.0 + step) for step in range(6)]
        + [("dc:2", 10 * (step + 1), 10.0 + step) for step in range(6)],
        [7, 3, 11, 2, 5, 9, 1, 8, 4, 12, 6, 10],
    ),
    # One unnamed tag group with a gap (no row at 20 s or 30 s) and a null value at 40 s.
    "900_1_10": (
        [("", 0, 0.5), ("", 10, 1.5), ("", 40, None), ("", 50, 5.5)],
        [100, 101, 104, 105],
    ),
    # The same metric at 60 s, complete, starting one step into the grid.
    "900_1_60": ([("", 60, 6.5), ("", 120, 7.5), ("", 180, 8.5)], [1, 2, 3]),
    # A single observation, on the 60 s grid.
    "901_0_60": ([("web", 120, 42.0)], [9]),
    # Published but never opened: 60 s is finer, so the interval rule never reaches this file.
    "901_0_3600": ([("web", 3600, 1.0), ("web", 7200, 2.0)], [1, 2]),
}

_ROWS = [
    {
        "question": "Does the series show an anomaly?\nTime-series: a queue depth.",
        "task_category": "Anomaly Presence",
        "difficulty": "Tier 1",
        "options_str": json.dumps(_YES),
        "correct_answer": _YES[0],
        "query_group": "900_0",
        "interpolate_1": "1",
        "interpolate_2": "0",
    },
    {
        "question": "Are the two series correlated?",
        "task_category": "Anomaly Correlation",
        "difficulty": "Tier 3",
        "options_str": json.dumps(_FIVE),
        "correct_answer": _FIVE[2],
        "query_group": "900_0,900_1",
        "interpolate_1": "0",
        "interpolate_2": "0",
    },
    {
        "question": "Does either series show an anomaly?",
        "task_category": "Anomaly Presence",
        "difficulty": "Tier 2",
        # The same option list as row 0, so the two share one dataset annotation.
        "options_str": json.dumps(_YES),
        "correct_answer": _YES[1],
        "query_group": "900_1,901_0",
        "interpolate_1": "0",
        "interpolate_2": "0",
    },
]

# The four (metric, interval) pairs the three questions resolve to, which is the record list.
_RECORD_IDS = ["arfbench-900_0-10", "arfbench-900_1-10", "arfbench-900_1-60", "arfbench-901_0-60"]

_CSV_COLUMNS = [
    "Unnamed: 0",
    "question",
    "task_category",
    "difficulty",
    "options_str",
    "correct_answer",
    "query_group",
    "options",
    "interpolate_1",
    "interpolate_2",
]


def _write_source(root: Path) -> arfbench.ARFBenchSource:
    ts_dir = root / arfbench.TS_DIR
    ts_dir.mkdir(parents=True)
    for name, (observations, index) in _FILES.items():
        groups = [group for group, _, _ in observations]
        table = pa.table(
            {
                "epoch": pa.array(
                    np.array([_ANCHOR_S + offset for _, offset, _ in observations], dtype="datetime64[s]").astype(
                        "datetime64[ns]"
                    )
                ),
                "group": pa.array(groups, type=pa.string()),
                "value": pa.array([value for _, _, value in observations], type=pa.float64()),
                "num_groups": pa.array([len(set(groups))] * len(observations), type=pa.int64()),
                # The release's query_name names another series on most files. Nothing may read it.
                "query_name": pa.array(["900_7"] * len(observations), type=pa.string()),
                "__index_level_0__": pa.array(index, type=pa.int64()),
            }
        )
        pq.write_table(table, ts_dir / f"{name}.parquet")

    qa_csv = root / arfbench.QA_CSV
    with qa_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for index, row in enumerate(_ROWS):
            options = json.loads(row["options_str"])
            writer.writerow(
                {**row, "Unnamed: 0": index, "options": json.dumps([{"value": option} for option in options])}
            )

    published = arfbench._published_intervals([f"{arfbench.TS_DIR}/{name}.parquet" for name in _FILES])
    return arfbench.ARFBenchSource(qa_csv=qa_csv, ts_dir=ts_dir, published=published)


@pytest.fixture
def source(tmp_path):
    arfbench._read_file.cache_clear()
    return _write_source(tmp_path / "cache")


@pytest.fixture
def dataset(source):
    return ARFBenchConnector().convert([source])


def _record(dataset, record_id):
    return next(record for record in dataset.records if record.record_id == record_id)


def _signal(dataset, record_id, name):
    return next(signal for signal in _record(dataset, record_id).signals if signal.name == name)


def _annotation(annotated, key):
    return next(annotation for annotation in annotated.annotations if annotation.key == key)


def test_is_a_connector():
    assert isinstance(ARFBenchConnector(), BaseConnector)


def test_metadata():
    metadata = ARFBenchConnector().metadata()
    assert metadata.dataset_id == "datadog/arfbench"
    assert str(metadata.dataset_version) == "1.0.0"
    assert str(metadata.license) == "Apache-2.0"
    assert str(metadata.access) == "open"
    assert metadata.citation is not None
    assert "arXiv:2604.21199" in metadata.citation
    assert [str(domain) for domain in metadata.domains] == ["observability"]


def test_one_record_per_series_file_and_one_task_per_question(dataset):
    assert [record.record_id for record in dataset.records] == _RECORD_IDS
    tasks = list(dataset.iter_tasks())
    assert [task.id for task in tasks] == ["arfbench-000", "arfbench-001", "arfbench-002"]
    assert all(isinstance(task, AnswerTask) for task in tasks)
    # The tasks stream, so nothing writes them onto the records.
    assert all(record.task_ids == () for record in dataset.records)


def test_the_task_stream_answers_the_same_tasks_every_time_it_is_read(dataset):
    # iter_tasks is public, so a consumer may read the stream again, and a one-shot generator would
    # yield nothing the second time. The writer reads the stream more than once, so the ids have to
    # come from the rows rather than from a fresh uuid per read.
    first = list(dataset.iter_tasks())
    second = list(dataset.iter_tasks())
    assert len(first) == len(second) == len(_ROWS)
    assert [task.id for task in first] == [task.id for task in second]
    assert [[record.id for record in task.inputs] for task in first] == [
        [record.id for record in task.inputs] for task in second
    ]
    assert [task.prompt for task in first] == [task.prompt for task in second]


def test_interval_is_the_finest_one_every_cited_metric_publishes(source, dataset):
    assert source.published == {"900_0": (10,), "900_1": (10, 60), "901_0": (60, 3600)}
    tasks = list(dataset.iter_tasks())
    # 900_0 and 900_1 both publish 10 s; 901_0's finest is 60 s, so the last question drops to 60 and
    # reads 900_1 through its 60 s record, not its 10 s one.
    assert [[record.id for record in task.inputs] for task in tasks] == [
        ["arfbench-900_0-10"],
        ["arfbench-900_0-10", "arfbench-900_1-10"],
        ["arfbench-900_1-60", "arfbench-901_0-60"],
    ]
    assert [_annotation(record, "interval_s").value for record in dataset.records] == [10, 10, 60, 60]


def test_a_question_whose_metrics_share_no_interval_is_rejected():
    with pytest.raises(TimeFFormatError, match="no sampling interval in common"):
        arfbench._interval_for(("900_0", "901_0"), {"900_0": (10,), "901_0": (60,)})


def test_rows_come_back_sorted_by_group_then_epoch(dataset):
    # The fixture file stores these 12 rows shuffled and its pandas index does not restore the order.
    assert list(_signal(dataset, "arfbench-900_0-10", "dc:1").to_numpy()) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert list(_signal(dataset, "arfbench-900_0-10", "dc:2").to_numpy()) == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]


def test_the_pandas_index_column_reaches_nothing(dataset):
    names = {signal.name for record in dataset.records for signal in record.signals}
    annotations = [annotation for record in dataset.records for annotation in record.annotations] + list(
        dataset.annotations
    )
    assert not any("__index_level_0__" in name for name in names)
    assert not any("__index_level_0__" in f"{a.key}{a.value}" for a in annotations)
    assert not any("query_name" in name or "900_7" in name for name in names)


def test_a_null_value_becomes_a_missing_timestep(dataset):
    signal = _signal(dataset, "arfbench-900_1-10", "value")
    assert signal.n_values == 4
    assert signal.to_arrow().null_count == 1
    values, valid = signal.to_numpy_and_mask()
    assert list(valid) == [True, True, False, True]
    assert [values[index] for index in (0, 1, 3)] == [0.5, 1.5, 5.5]


def _write_hazard_file(path, observations):
    table = pa.table(
        {
            "epoch": pa.array(
                np.array([_ANCHOR_S + offset for _, offset, _ in observations], dtype="datetime64[s]").astype(
                    "datetime64[ns]"
                )
            ),
            "group": pa.array([group for group, _, _ in observations], type=pa.string()),
            "value": pa.array([value for _, _, value in observations], type=pa.float64()),
        }
    )
    pq.write_table(table, path)


def test_a_file_with_no_row_is_rejected(tmp_path):
    path = tmp_path / "900_2_10.parquet"
    _write_hazard_file(path, [])
    with pytest.raises(TimeFFormatError, match="holds no row"):
        arfbench._read_file(str(path))


def test_a_file_whose_values_are_all_null_keeps_its_signal(tmp_path):
    path = tmp_path / "900_5_10.parquet"
    _write_hazard_file(path, [("dc:1", 0, None), ("dc:1", 10, None)])
    signals = arfbench._signals_for(str(path), "900_5", 10)
    assert [signal.name for signal in signals] == ["dc:1"]
    _, valid = signals[0].to_numpy_and_mask()
    assert list(valid) == [False, False]


def test_a_null_tag_label_is_rejected(tmp_path):
    path = tmp_path / "900_3_10.parquet"
    _write_hazard_file(path, [("dc:1", 0, 1.0), (None, 10, 2.0)])
    with pytest.raises(TimeFFormatError, match="null tag label"):
        arfbench._read_file(str(path))


def test_an_empty_label_beside_a_real_one_spelled_value_is_rejected(tmp_path):
    # Both tag groups would be named "value", so one would silently replace the other.
    path = tmp_path / "900_4_10.parquet"
    _write_hazard_file(path, [("", 0, 1.0), (arfbench._EMPTY_SIGNAL, 0, 2.0)])
    with pytest.raises(TimeFFormatError, match="would collide"):
        arfbench._read_file(str(path))


def test_an_observation_before_the_shared_anchor_is_rejected(tmp_path):
    path = tmp_path / "900_6_10.parquet"
    _write_hazard_file(path, [("dc:1", -10, 1.0), ("dc:1", 0, 2.0)])
    with pytest.raises(TimeFValidationError, match="must be >= 0"):
        arfbench._signals_for(str(path), "900_6", 10)


def test_an_empty_tag_label_becomes_a_named_signal(dataset):
    assert [signal.name for signal in _record(dataset, "arfbench-900_1-10").signals] == ["value"]
    assert all(signal.name for record in dataset.records for signal in record.signals)


def test_a_complete_signal_gets_a_regular_axis(dataset):
    axis = _signal(dataset, "arfbench-900_0-10", "dc:1").time_axis
    assert isinstance(axis, RegularAxis)
    assert axis.period_us == Fraction(10_000_000)
    # The first observation is 10 s after the shared anchor, so it is step 1 of the grid.
    assert axis.start_index == 1


def test_a_gapped_signal_gets_an_irregular_axis_with_its_own_offsets(dataset):
    signal = _signal(dataset, "arfbench-900_1-10", "value")
    assert isinstance(signal.time_axis, IrregularAxis)
    assert (signal.time_axis.first_us, signal.time_axis.last_us) == (0, 50_000_000)
    assert list(signal.time_offsets_us()) == [0, 10_000_000, 40_000_000, 50_000_000]


def test_a_single_observation_stays_regular(dataset):
    signal = _signal(dataset, "arfbench-901_0-60", "web")
    assert signal.n_values == 1
    assert isinstance(signal.time_axis, RegularAxis)
    assert signal.time_axis.start_index == 2


def test_a_metric_two_questions_cite_is_stored_once(dataset):
    # Questions 0 and 1 both cite 900_0 at 10 s. They share the one record that owns its signals
    # rather than each carrying a copy, so the dataset holds four records for five citations.
    tasks = list(dataset.iter_tasks())
    shared = _record(dataset, "arfbench-900_0-10")
    assert tasks[0].inputs[0] is shared
    assert tasks[1].inputs[0] is shared
    assert len(dataset.records) == 4
    signal = _signal(dataset, "arfbench-900_0-10", "dc:1")
    assert signal.id == "arfbench-900_0-10-0"
    assert signal.source_id == "900_0@10s"


def test_every_record_is_one_metric_at_one_interval(dataset):
    record = _record(dataset, "arfbench-900_0-10")
    assert [source.name for source in record.sources] == ["metric 900_0 at 10 s"]
    assert [signal.name for signal in record.signals] == ["dc:1", "dc:2"]
    assert record.start_time == arfbench.ANCHOR_US
    assert [(a.key, a.value, a.unit) for a in record.annotations] == [
        ("metric_id", "900_0", None),
        ("incident_id", "900", None),
        ("interval_s", 10, "second"),
    ]


def test_tasks_carry_the_question_and_its_answer(dataset):
    task = next(iter(dataset.iter_tasks()))
    assert task.prompt == _ROWS[0]["question"]
    assert "\n" in task.prompt
    assert task.targets == (_YES[0],)
    assert [record.id for record in task.inputs] == ["arfbench-900_0-10"]
    (referenced,) = task.input_annotations
    assert referenced.key == "answer_options"
    assert task.targets[0] in referenced.value
    # The occurrence the task names is one attached to the dataset, which is what the writer checks.
    assert referenced.occurrence_id in {annotation.occurrence_id for annotation in dataset.annotations}


def test_one_dataset_annotation_per_distinct_option_list(dataset):
    options = [annotation for annotation in dataset.annotations if annotation.key == "answer_options"]
    tasks = list(dataset.iter_tasks())
    # Rows 0 and 2 offer the same two options, so three questions reference two lists.
    assert len(options) == len(dataset.annotations) == 2
    assert tasks[0].input_annotations[0].id == tasks[2].input_annotations[0].id
    assert tasks[1].input_annotations[0].id != tasks[0].input_annotations[0].id


def test_task_annotations(dataset):
    tasks = list(dataset.iter_tasks())
    assert [annotation.key for annotation in tasks[0].annotations] == ["task_category", "difficulty", "interpolate_1"]
    assert [(a.key, a.value) for a in tasks[1].annotations] == [
        ("task_category", "Anomaly Correlation"),
        ("difficulty", "Tier 3"),
    ]
    # Two questions in the same category share one annotation content and carry their own occurrence.
    first, third = _annotation(tasks[0], "task_category"), _annotation(tasks[2], "task_category")
    assert first.value == third.value == "Anomaly Presence"
    assert first.id == third.id
    assert first.occurrence_id != third.occurrence_id
    # The flags ride only on the rows that set them.
    assert _annotation(tasks[0], "interpolate_1").value is True
    assert not any(a.key.startswith("interpolate") for task in tasks[1:] for a in task.annotations)


def test_schema_has_one_spec_and_one_descriptor_per_record_key(dataset):
    schema = dataset.derive_schema()
    assert [spec.spec_type for spec in schema.time_series_specs] == ["metric"]
    assert schema.time_series_specs[0].dtype == "float64"
    # A null observation is stored as a missing timestep, which the spec has to allow.
    assert schema.time_series_specs[0].nullable
    assert {descriptor.key for descriptor in schema.annotations} >= {"metric_id", "incident_id", "interval_s"}
    assert schema.tasks == (AnswerTask,)


def test_round_trip_through_the_writer_and_the_reader(dataset, tmp_path):
    dataset.derive_schema()
    version_dir = store_dataset(dataset, tmp_path / "out")
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        read_back = reader.read()
    assert [record.record_id for record in read_back.records] == _RECORD_IDS
    tasks = sorted(read_back.tasks, key=lambda task: task.id)
    assert [task.id for task in tasks] == ["arfbench-000", "arfbench-001", "arfbench-002"]
    # A task's inputs come back as the read records, in citation order, and the reverse map is rebuilt.
    assert [record.id for record in tasks[1].inputs] == ["arfbench-900_0-10", "arfbench-900_1-10"]
    assert tasks[1].inputs[0] is _record(read_back, "arfbench-900_0-10")
    assert [task.id for task in read_back.tasks_for(_record(read_back, "arfbench-900_1-60"))] == ["arfbench-002"]
    assert {(a.key, a.value) for a in tasks[0].annotations} == {
        ("task_category", "Anomaly Presence"),
        ("difficulty", "Tier 1"),
        ("interpolate_1", True),
    }
    assert tasks[0].input_annotations[0].key == "answer_options"
    assert tasks[0].input_annotations[0].value == _YES
    gapped = _signal(read_back, "arfbench-900_1-10", "value")
    values, valid = gapped.to_numpy_and_mask()
    assert list(valid) == [True, True, False, True]
    assert [values[index] for index in (0, 1, 3)] == [0.5, 1.5, 5.5]
    assert list(gapped.time_offsets_us()) == [0, 10_000_000, 40_000_000, 50_000_000]
    regular = _signal(read_back, "arfbench-900_0-10", "dc:1")
    assert regular.time_axis == RegularAxis(period_us=Fraction(10_000_000), start_index=1)
    assert list(regular.to_numpy()) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_convert_reads_no_values_until_they_are_asked_for(source, dataset):
    signal = _signal(dataset, "arfbench-900_0-10", "dc:1")
    for parquet in source.ts_dir.glob("*.parquet"):
        parquet.unlink()
    arfbench._read_file.cache_clear()
    with pytest.raises((OSError, pa.ArrowInvalid)):
        signal.to_arrow()


def test_download_pins_the_revision_and_fetches_only_the_files_the_questions_use(monkeypatch, tmp_path, source):
    hub = pytest.importorskip("huggingface_hub")
    calls: list[list[str]] = []
    repo_files = [
        arfbench.QA_CSV,
        "arfbench-images/900_0.png",
        *(f"{arfbench.TS_DIR}/{name}.parquet" for name in _FILES),
    ]

    def fake_list(repo, repo_type, revision):
        assert (repo, repo_type, revision) == (arfbench.REPO, "dataset", arfbench.REVISION)
        return repo_files

    def fake_snapshot(repo, repo_type, revision, cache_dir, allow_patterns):
        assert (repo, repo_type, revision) == (arfbench.REPO, "dataset", arfbench.REVISION)
        assert cache_dir == str(tmp_path / "cache")
        calls.append(list(allow_patterns))
        return str(source.qa_csv.parent)

    monkeypatch.setattr(hub, "list_repo_files", fake_list)
    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot)
    fetched = ARFBenchConnector().download(tmp_path / "cache")

    # The QA table first, then only the four series files the three questions resolve to.
    assert calls == [
        [arfbench.QA_CSV],
        [f"{arfbench.TS_DIR}/{name}.parquet" for name in ("900_0_10", "900_1_10", "900_1_60", "901_0_60")],
    ]
    assert not any("arfbench-images" in path for call in calls for path in call)
    assert fetched[0].published["900_1"] == (10, 60)
    assert fetched[0].ts_dir == source.ts_dir
