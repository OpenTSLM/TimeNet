import json
from pathlib import Path

import numpy as np
import pytest

from timenet.config import settings
from timenet.dataset.axis import IrregularAxis
from timenet.engine import run_pipeline, store_dataset
from timenet.errors import TimeFValidationError
from timenet.format import control_writer
from timenet.format.control_audit import audit_control_database
from timenet.format.duckdb import connect_control
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import AnswerTask, InputModality, Split, TimeSeriesSpec
from timenet_connectors.datasets.bash_lab.sensorcaps import SensorCapsConnector, connector as module
from timenet_connectors.datasets.bash_lab.sensorcaps.connector import (
    DECLARED_SAMPLING_RATE_HZ,
    HF_REPO,
    SOURCE_REVISION,
    SOURCE_ROWS,
    SensorCapsFileRef,
)


_FIXTURES = Path(__file__).parent / "fixtures"


# Independent of the connector inventory and parser: fixtures have distinct values and text
# for every partition. Each partition adds 100 to the previous partition's sensor values.
_PARTITIONS = ("hhar_v2", "motion_v2", "pamap2_v2", "shoaib_v2", "uci_v2")
_CHANNELS = (
    ("accelerometer", "x", (1.1, -4.4, 7.7)),
    ("accelerometer", "y", (2.2, -5.5, 8.8)),
    ("accelerometer", "z", (3.3, -6.6, 9.9)),
    ("gyroscope", "x", (0.123456789012345, 0.4, 0.7)),
    ("gyroscope", "y", (-0.2, -0.5, 0.8)),
    ("gyroscope", "z", (0.3, 0.6, 0.9)),
)
# Expected physical identities in published array order, independent of connector constants.
_PAMAP2_CHANNELS = (
    ("accelerometer", "16g_y"),
    ("accelerometer", "16g_z"),
    ("accelerometer", "6g_x"),
    ("gyroscope", "y"),
    ("gyroscope", "z"),
    ("magnetometer", "x"),
)
_DATASET_ANNOTATIONS = {
    "source_repository": HF_REPO,
    "source_revision": SOURCE_REVISION,
    "declared_sampling_rate_hz": DECLARED_SAMPLING_RATE_HZ,
    "source_sampling_rate_hz": 20,
    "source_window_samples": 120,
    "time_axis_reconstruction": "round(linspace(0, 119, 60)) / 20 seconds relative to window start",
}
# Independent expected offsets: the upstream sampler skips index 60 at the midpoint.
_EXPECTED_OFFSETS_US = np.asarray([*range(0, 60, 2), *range(61, 120, 2)], dtype=np.int64) * 50_000


def _rows(partition="hhar_v2"):
    offset = _PARTITIONS.index(partition) * 100
    text = (_FIXTURES / "captions.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line.replace("hhar_v2", partition)) for line in text.splitlines()]
    for row in rows:
        lines = row["messages"][1]["content"].splitlines()
        for index, line in enumerate(lines):
            heading, _, value = line.partition(":")
            if heading in {"Accelerometer", "Gyroscope"}:
                samples = [[number + offset for number in sample] for sample in json.loads(value)]
                lines[index] = f"{heading}: {json.dumps(samples)}"
        if partition == "pamap2_v2":
            lines.append("Location: ankle")
            row["messages"][0]["content"] += " Location is supplied."
        row["messages"][1]["content"] = "\n".join(lines)
    return rows


def _refs(tmp_path):
    refs = []
    for partition in _PARTITIONS:
        path = tmp_path / f"{partition}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in _rows(partition)), encoding="utf-8"
        )
        refs.append(SensorCapsFileRef(path, partition))
    return refs


def _annotations(owner):
    return {annotation.key: annotation.value for annotation in owner.annotations}


def _write_row(tmp_path, row):
    path = tmp_path / "row.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return SensorCapsFileRef(path, "hhar_v2")


def test_downloads_pinned_source_inventory(monkeypatch, tmp_path):
    import huggingface_hub  # noqa: PLC0415

    seen = []

    def download(repo_id, **kwargs):
        seen.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    connector = SensorCapsConnector()
    downloaded = connector.download(tmp_path)
    assert sum(ref.expected_rows or 0 for ref in downloaded) == 35960
    assert seen == [
        (
            HF_REPO,
            {
                "revision": SOURCE_REVISION,
                "repo_type": "dataset",
                "cache_dir": str(tmp_path),
                "allow_patterns": [ref.filename for ref in downloaded],
            },
        )
    ]
    assert {ref.partition for ref in downloaded} == {"hhar_v2", "motion_v2", "pamap2_v2", "shoaib_v2", "uci_v2"}
    assert all(ref.path == tmp_path / ref.filename for ref in downloaded)


def _assert_records_and_tasks(records, tasks):
    assert len(records) == len(tasks) == 10
    for partition_index, partition in enumerate(_PARTITIONS):
        for row_index, row in enumerate(_rows(partition)):
            index = partition_index * 2 + row_index
            record, task = records[index], tasks[index]
            record_id = f"sensorcaps-{partition.replace('_', '-')}-{row_index:06d}"
            assert record.id == record_id
            assert record.start_time.timestamp is None
            assert len(record.sources) == 1
            source = record.sources[0]
            assert source.id == f"{record_id}-imu"
            assert source.name == "IMU"
            assert source.sources == ()
            assert len(source.signals) == len(record.signals) == 6
            expected_annotations = {
                "source_dataset": partition.removesuffix("_v2"),
                "source_partition": partition,
                "source_file": f"data/{partition}-00000-of-00001.jsonl",
                "source_row": row_index,
                "activity_summary": f"synthetic café {partition} movement {row_index}",
            }
            if partition == "pamap2_v2":
                expected_annotations["sensor_location"] = "ankle"
            assert _annotations(record) == expected_annotations
            identities = (
                _PAMAP2_CHANNELS if partition == "pamap2_v2" else tuple((sensor, axis) for sensor, axis, _ in _CHANNELS)
            )
            for signal, (sensor, axis), (_, _, values) in zip(source.signals, identities, _CHANNELS, strict=True):
                axis_id = axis.replace("_", "-")
                assert signal.id == f"{record_id}-{sensor}-{axis_id}"
                assert signal.name == f"{sensor}_{axis}"
                assert signal.spec == TimeSeriesSpec(
                    spec_type=sensor, name=sensor.capitalize(), unit_value=None, dtype="float32"
                )
                assert isinstance(signal.time_axis, IrregularAxis)
                assert signal.time_axis.first_us == 0
                assert signal.time_axis.last_us == 5_950_000
                np.testing.assert_array_equal(signal.time_offsets_us(), _EXPECTED_OFFSETS_US)
                assert signal.n_values == 60
                expected_values = np.resize(np.asarray(values[: row_index + 2], dtype=np.float64), 60)
                expected_values = (expected_values + partition_index * 100).astype(np.float32)
                actual = signal.to_numpy()
                assert actual.dtype == np.float32
                np.testing.assert_array_equal(actual, expected_values)
            assert type(task) is AnswerTask
            assert task.id == f"{record_id}-caption"
            assert len(task.inputs) == 1 and task.inputs[0] is record
            assert task.targets == (row["messages"][2]["content"],)
            expected_features = "Gyroscope moving average: [[0.1 0.2 0.3]]\nAccelerometer FFT: [1.+2.j 3.+4.j]" + (
                "\n" if partition == "pamap2_v2" else ""
            )
            assert task.prompt == row["messages"][0]["content"] + "\n\n" + expected_features
            assert task.metadata == {}
            assert task.split is Split.TRAIN
            expected_context = {
                "activity_summary": f"synthetic café {partition} movement {row_index}",
            }
            if partition == "pamap2_v2":
                expected_context["sensor_location"] = "ankle"
            assert {annotation.key: annotation.value for annotation in task.input_annotations} == expected_context
            for annotation in task.input_annotations:
                assert any(annotation is attached for attached in record.annotations)
            assert task.resolved_input_modalities == frozenset({InputModality.TEXT, InputModality.TIME_SERIES})


def test_conversion_preserves_source_semantics_and_reiterable_tasks(tmp_path):
    dataset = SensorCapsConnector().convert(_refs(tmp_path))
    assert _annotations(dataset) == _DATASET_ANNOTATIONS
    assert dataset.derive_schema().tasks == (AnswerTask,)
    # Verify complete content on both passes, including row-to-record associations.
    for _ in range(2):
        _assert_records_and_tasks(dataset.records, list(dataset.iter_tasks()))


def test_channel_reads_reuse_rows_and_reread_offsets(monkeypatch, tmp_path):
    dataset = SensorCapsConnector().convert(_refs(tmp_path)[:1])
    original = module._parse_row
    calls = []

    def parse(raw, context):
        calls.append(context)
        return original(raw, context)

    monkeypatch.setattr(module, "_parse_row", parse)
    first, second = dataset.records
    for signal in first.signals:
        assert len(signal.to_arrow()) == 60
    assert len(calls) == 1  # six channels decode only one source row
    np.testing.assert_array_equal(
        second.signals[0].to_numpy(), np.resize(np.asarray([1.1, -4.4, 7.7], dtype=np.float32), 60)
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(
        first.signals[3].to_numpy(), np.resize(np.asarray([0.123456789012345, 0.4], dtype=np.float32), 60)
    )
    assert len(calls) == 3  # revisiting evicted rows reads by byte offset, including UTF-8


@pytest.mark.parametrize("location", [None, "ankle"])
def test_multiline_feature_context_survives_storage(tmp_path, location):
    features = (
        "Gyroscope moving average: [[ 0.1  0.2  0.3]\n"
        " [ 0.4  0.5  0.6]]\n\n"
        "Accelerometer FFT: [[1.+2.j 3.-4.j]\n"
        " ...\n"
        " [5.+6.j 7.-8.j]]\n"
        "Gyroscope minimum: [-1. -2. -3.]\n"
        "Accelerometer variance: [0.1 0.2 0.3]  \n"
    )
    row = _rows()[0]
    user = row["messages"][1]["content"].split("Gyroscope moving average:")[0]
    if location is not None:
        user += f"Location: {location}\n"
    row["messages"][1]["content"] = user + features
    dataset = SensorCapsConnector().convert([_write_row(tmp_path, row)])

    def assert_context(task):
        context = {annotation.key: annotation.value for annotation in task.input_annotations}
        assert context["activity_summary"] == "synthetic café hhar_v2 movement 0"
        assert context.get("sensor_location") == location
        assert task.prompt == row["messages"][0]["content"] + "\n\n" + features
        assert task.targets == (row["messages"][2]["content"],)

    for _ in range(2):
        assert_context(next(dataset.iter_tasks()))
    # Large feature text belongs to streamed tasks, not the retained record graph.
    assert "derived_features" not in _annotations(dataset.records[0])
    version_dir = store_dataset(dataset, tmp_path / "registry")
    (tmp_path / "row.jsonl").unlink()
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        assert_context(reader.read().tasks[0])


@pytest.mark.parametrize("through_engine", [False, True], ids=["store-stream", "pipeline-read"])
def test_storage_preserves_source_semantics_after_source_removal(monkeypatch, tmp_path, through_engine):
    # Exercise task-batch boundaries and a final partial batch.
    monkeypatch.setattr(control_writer, "_TASK_BATCH_SIZE", 3)
    monkeypatch.setenv("TIMENET_CACHE", str(tmp_path / "cache"))
    cache = settings().cache_dir / "bash-lab/sensorcaps"

    def download(self, cache_dir):
        assert cache_dir == cache
        return _refs(cache_dir)

    connector = SensorCapsConnector()
    if through_engine:
        monkeypatch.setattr(SensorCapsConnector, "download", download)
        version_dir = run_pipeline(connector, tmp_path / "registry")
        assert not cache.exists()
    else:
        cache.mkdir(parents=True)
        refs = download(connector, cache)
        dataset = connector.convert(refs)
        version_dir = store_dataset(dataset, tmp_path / "registry")
        # The streamed reader must also work without any downloaded source files.
        for ref in refs:
            ref.path.unlink()

    expected_schema = SensorCapsConnector().convert(_refs(tmp_path)).derive_schema()
    with connect_control(version_dir / "control.duckdb", read_only=True) as connection:
        audit_control_database(connection)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        assert reader.metadata == connector.metadata()
        assert reader.schema == expected_schema
        if through_engine:
            restored = reader.read()
            _assert_records_and_tasks(restored.records, restored.tasks)
        else:
            records = list(reader.iter_records())
            _assert_records_and_tasks(records, list(reader.iter_tasks(records)))
        assert _annotations(reader.read()) == _DATASET_ANNOTATIONS
        reader.verify()


@pytest.mark.parametrize(
    ("sensor", "pattern"),
    [
        ("[]", "at least 1 item"),
        ("{}", "valid array"),
        ("[[1,2,3], [4,5]]", "at least 3 items"),
        ("[[1,2,3,4]]", "at most 3 items"),
        ('[[1,2,"3"], [4,5,6]]', "valid number"),
        ("[[true,2,3]]", "valid number"),
        ("[[NaN,2,3], [4,5,6]]", "finite number"),
        ("[[Infinity,2,3]]", "finite number"),
        ("not json", "Invalid JSON"),
        ("[[1,2,3]]", "lengths differ"),
    ],
)
def test_invalid_sensor_data_has_source_context(tmp_path, sensor, pattern):
    row = _rows()[0]
    user = row["messages"][1]["content"]
    lines = [f"Gyroscope: {sensor}" if line.startswith("Gyroscope:") else line for line in user.splitlines()]
    row["messages"][1]["content"] = "\n".join(lines)
    ref = _write_row(tmp_path, row)
    with pytest.raises(TimeFValidationError, match=pattern) as error:
        SensorCapsConnector().convert([ref])
    assert str(ref.path) in str(error.value)
    assert "hhar_v2, row 0" in str(error.value)


def test_integer_and_float_sensor_readings_are_accepted(tmp_path):
    row = _rows()[0]
    lines = row["messages"][1]["content"].splitlines()
    for index, line in enumerate(lines):
        heading, _, _ = line.partition(":")
        if heading in {"Accelerometer", "Gyroscope"}:
            lines[index] = f"{heading}: {json.dumps([[1, 2.5, -3]] * 60)}"
    row["messages"][1]["content"] = "\n".join(lines)
    dataset = SensorCapsConnector().convert([_write_row(tmp_path, row)])
    for signal, expected in zip(dataset.records[0].signals, [1, 2.5, -3] * 2, strict=True):
        np.testing.assert_array_equal(signal.to_numpy(), np.full(60, expected, dtype=np.float32))


def test_pinned_downloads_enforce_partition_counts():
    fixture = SensorCapsFileRef(_FIXTURES / "captions.jsonl", "hhar_v2")
    with pytest.raises(TimeFValidationError, match=r"hhar_v2.*expected 9166 rows, found 2"):
        SensorCapsConnector().convert([SensorCapsFileRef(fixture.path, fixture.partition, SOURCE_ROWS["hhar_v2"])])


@pytest.mark.parametrize("n_readings", [1, 59, 61])
def test_unexpected_sample_count_rejects_timing_reconstruction(tmp_path, n_readings):
    row = _rows()[0]
    lines = row["messages"][1]["content"].splitlines()
    for index, line in enumerate(lines):
        heading, _, _ = line.partition(":")
        if heading in {"Accelerometer", "Gyroscope"}:
            lines[index] = f"{heading}: {json.dumps([[1, 2, 3]] * n_readings)}"
    row["messages"][1]["content"] = "\n".join(lines)
    ref = _write_row(tmp_path, row)
    with pytest.raises(TimeFValidationError, match="expected 60 readings per sensor") as error:
        SensorCapsConnector().convert([ref])
    assert str(ref.path) in str(error.value)
