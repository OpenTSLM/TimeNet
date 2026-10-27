import json
from pathlib import Path

import numpy as np
import pytest

from timenet.config import settings
from timenet.dataset.axis import IrregularAxis
from timenet.engine import run_pipeline
from timenet.errors import TimeFValidationError
from timenet.format import control_writer
from timenet.format.control_audit import audit_control_database
from timenet.format.duckdb import connect_control
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import AnswerTask, InputModality, Split, TimeSeriesSpec, read_value
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.bash_lab.sensorcaps import SensorCapsConnector, connector as module
from timenet_connectors.datasets.bash_lab.sensorcaps.connector import DECLARED_SAMPLING_RATE_HZ


_FIXTURES = Path(__file__).parent / "fixtures"


# Independent of the connector inventory and parser: fixtures have distinct values and text
# for every partition. Each partition adds 100 to the previous partition's sensor values.
_PARTITIONS = ("hhar_v2", "motion_v2", "pamap2_v2", "shoaib_v2", "uci_v2")
_CHANNELS = (
    ("accelerometer", "x", 1.1),
    ("accelerometer", "y", 2.2),
    ("accelerometer", "z", 3.3),
    ("gyroscope", "x", 0.123456789012345),
    ("gyroscope", "y", -0.2),
    ("gyroscope", "z", 0.3),
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
# Literal pins, so moving to another repo or commit is a deliberate test edit.
_DATASET_ANNOTATIONS = {
    "source_repository": "BASH-Lab/SensorCaps",
    "source_revision": "4242e07feaacb0ee3aff86182a62914fa7abacef",
    "declared_sampling_rate_hz": DECLARED_SAMPLING_RATE_HZ,
    "source_sampling_rate_hz": 20,
    "source_window_samples": 120,
    "time_axis_reconstruction": "round(linspace(0, 119, 60)) / 20 seconds relative to window start",
}
# Independent expected offsets: the upstream sampler skips index 60 at the midpoint.
_EXPECTED_OFFSETS_US = np.asarray([*range(0, 60, 2), *range(61, 120, 2)], dtype=np.int64) * 50_000

_FEATURES = (
    "Gyroscope moving average: [[ 0.1  0.2  0.3]\n"
    " [ 0.4  0.5  0.6]]\n\n"
    "Accelerometer FFT: [[1.+2.j 3.-4.j]\n"
    " ...\n"
    " [5.+6.j 7.-8.j]]\n"
    "Gyroscope minimum: [-1. -2. -3.]\n"
    "Accelerometer variance: [0.1 0.2 0.3]  \n"
)


def _expected_values(partition_index, row_index, channel_index):
    steps = np.arange(60, dtype=np.float64)
    return (
        _CHANNELS[channel_index][2]
        + row_index * 10
        + steps * (channel_index + 1) / 10
        + steps**2 / 1000
        + partition_index * 100
    ).astype(np.float32)


def _rows(partition="hhar_v2"):
    offset = _PARTITIONS.index(partition) * 100
    text = (_FIXTURES / "captions.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line.replace("hhar_v2", partition)) for line in text.splitlines()]
    for row in rows:
        lines = row["messages"][1]["content"].split("Gyroscope moving average:")[0].splitlines()
        for index, line in enumerate(lines):
            heading, _, value = line.partition(":")
            if heading in {"Accelerometer", "Gyroscope"}:
                samples = [[number + offset for number in sample] for sample in json.loads(value)]
                lines[index] = f"{heading}: {json.dumps(samples)}"
        if partition == "pamap2_v2":
            lines.append("Location: ankle")
            row["messages"][0]["content"] += " Location is supplied."
        row["messages"][1]["content"] = "\n".join(lines) + "\n" + _FEATURES
    return rows


@pytest.fixture(autouse=True)
def _fixture_inventory(monkeypatch):
    """Pin the inventory to the fixtures: two rows per partition."""
    monkeypatch.setattr(SensorCapsConnector, "SOURCE_ROWS", dict.fromkeys(_PARTITIONS, 2))


def _refs(tmp_path):
    refs = []
    for partition in _PARTITIONS:
        path = tmp_path / f"{partition}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in _rows(partition)), encoding="utf-8"
        )
        refs.append(HubFileRef(path, partition))
    return refs


def _annotations(owner):
    return {annotation.key: read_value(annotation.value) for annotation in owner.annotations}


def _write_row(tmp_path, row):
    path = tmp_path / "row.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return HubFileRef(path, "hhar_v2")


def _assert_records_and_tasks(records, tasks):
    assert len(records) == len(tasks) == 10
    records_by_row = {
        (_annotations(record)["source_partition"], _annotations(record)["source_row"]): record for record in records
    }
    tasks_by_record = {task.inputs[0].id: task for task in tasks}
    for partition_index, partition in enumerate(_PARTITIONS):
        for row_index, row in enumerate(_rows(partition)):
            record = records_by_row[partition, row_index]
            task = tasks_by_record[record.id]
            assert record.start_time.timestamp is None
            assert len(record.sources) == 1
            source = record.sources[0]
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
            # Feature text is streamed with tasks, never retained on the record graph.
            assert _annotations(record) == expected_annotations
            identities = (
                _PAMAP2_CHANNELS if partition == "pamap2_v2" else tuple((sensor, axis) for sensor, axis, _ in _CHANNELS)
            )
            signals = {signal.name: signal for signal in source.signals}
            assert set(signals) == {f"{sensor}_{axis}" for sensor, axis in identities}
            for channel_index, (sensor, axis) in enumerate(identities):
                signal = signals[f"{sensor}_{axis}"]
                assert signal.spec == TimeSeriesSpec(
                    spec_type=sensor, name=sensor.capitalize(), unit_value=None, dtype="float32"
                )
                assert isinstance(signal.time_axis, IrregularAxis)
                assert signal.time_axis.first_us == 0
                assert signal.time_axis.last_us == 5_950_000
                np.testing.assert_array_equal(signal.time_offsets_us(), _EXPECTED_OFFSETS_US)
                assert signal.n_values == 60
                expected_values = _expected_values(partition_index, row_index, channel_index)
                actual = signal.to_numpy()
                assert actual.dtype == np.float32
                np.testing.assert_array_equal(actual, expected_values)
            assert type(task) is AnswerTask
            assert len(task.inputs) == 1 and task.inputs[0] is record
            assert task.targets == (row["messages"][2]["content"],)
            assert task.prompt == row["messages"][0]["content"] + "\n\n" + _FEATURES
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
    np.testing.assert_array_equal(second.signals[0].to_numpy(), _expected_values(0, 1, 0))
    assert len(calls) == 2
    np.testing.assert_array_equal(first.signals[3].to_numpy(), _expected_values(0, 0, 3))
    assert len(calls) == 3  # revisiting evicted rows reads by byte offset, including UTF-8


def test_pipeline_preserves_source_semantics_after_source_removal(monkeypatch, tmp_path):
    # Ten rows cover multiline features with/without location and a partial task batch.
    monkeypatch.setattr(control_writer, "_TASK_BATCH_SIZE", 3)
    monkeypatch.setenv("TIMENET_CACHE", str(tmp_path / "cache"))
    cache = settings().cache_dir / "bash-lab/sensorcaps"

    async def download_async(self, cache_dir):
        assert cache_dir == cache
        return _refs(cache_dir)

    connector = SensorCapsConnector()
    monkeypatch.setattr(SensorCapsConnector, "download_async", download_async)
    version_dir = run_pipeline(connector, tmp_path / "registry")
    assert not cache.exists()

    with connect_control(version_dir / "control.duckdb", read_only=True) as connection:
        audit_control_database(connection)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        assert reader.metadata == connector.metadata()
        assert reader.schema.tasks == (AnswerTask,)
        restored = reader.read()
        _assert_records_and_tasks(restored.records, restored.tasks)
        records = list(reader.iter_records())
        _assert_records_and_tasks(records, list(reader.iter_tasks(records)))
        assert _annotations(restored) == _DATASET_ANNOTATIONS
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


def test_integer_and_float_sensor_readings_are_accepted(monkeypatch, tmp_path):
    monkeypatch.setitem(SensorCapsConnector.SOURCE_ROWS, "hhar_v2", 1)
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


def test_pinned_inventory_mismatch_fails(monkeypatch, tmp_path):
    monkeypatch.setitem(SensorCapsConnector.SOURCE_ROWS, "hhar_v2", 3)
    with pytest.raises(TimeFValidationError, match=r"hhar_v2\): expected 3 rows, found 2"):
        SensorCapsConnector().convert(_refs(tmp_path)[:1])


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
