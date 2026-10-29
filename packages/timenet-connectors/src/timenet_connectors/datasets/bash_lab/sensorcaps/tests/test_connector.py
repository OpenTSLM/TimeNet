import json
from pathlib import Path

import numpy as np
import pytest

from timenet.config import settings
from timenet.dataset.axis import RegularAxis
from timenet.engine import run_pipeline
from timenet.errors import TimeFValidationError
from timenet.format import control_writer
from timenet.format.control_audit import audit_control_database
from timenet.format.duckdb import connect_control
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.types import AnswerTask, InputModality, TimeSeriesSpec, read_value
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.bash_lab.sensorcaps import SensorCapsConnector, records as module
from timenet_connectors.datasets.bash_lab.sensorcaps.records import SensorCapsMetadata
from timenet_connectors.metadata import RecordMetadata


_FIXTURES = Path(__file__).parent / "fixtures"


# Independent of the connector inventory and parser: fixtures have distinct values and text
# for every partition. Each partition adds 100 to the previous partition's sensor values.
_PARTITIONS = ("hhar_v2", "motion_v2", "pamap2_v2", "shoaib_v2", "uci_v2")
# The fixture's channels, as its values were generated: each entry's index is its value formula's.
_CHANNELS = (
    ("accelerometer", "x", 1.1),
    ("accelerometer", "y", 2.2),
    ("accelerometer", "z", 3.3),
    ("gyroscope", "x", 0.123456789012345),
    ("gyroscope", "y", -0.2),
    ("gyroscope", "z", 0.3),
)
# Every partition, PAMAP2 included, keeps the source's labels. TimeF orders signals by name.
_SIGNAL_NAMES = (
    "accelerometer_x",
    "accelerometer_y",
    "accelerometer_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
)
# Literal pins, so moving to another repo or commit is a deliberate test edit.
_DATASET_ANNOTATIONS = {
    "source_repository": "BASH-Lab/SensorCaps",
    "source_revision": "4242e07feaacb0ee3aff86182a62914fa7abacef",
}

# The 12 derived features, in source order, as the user message prints them: NumPy text that can span
# lines and hold complex numbers, ellipses, blank lines and trailing spaces.
_FEATURE_TEXT = {
    "Gyroscope moving average": "[[ 0.1  0.2  0.3]\n [ 0.4  0.5  0.6]]",
    "Accelerometer moving average": "[[9.1 9.2 9.3]]",
    "Gyroscope FFT": "[[1.+2.j 3.-4.j]\n ...\n\n [5.+6.j 7.-8.j]]",
    "Accelerometer FFT": "[[ 2.07+0.j -1.08+5.55j]]",
    "Gyroscope minimum": "[-1. -2. -3.]",
    "Accelerometer minimum": "[-6.5 -6.4\n -6.3]",
    "Gyroscope maximum": "[1. 2. 3.]",
    "Accelerometer maximum": "[7.5 7.6 7.7]",
    "Gyroscope median": "[0. 0.1 0.2]",
    "Accelerometer median": "[1.09 1.11 1.12]",
    "Gyroscope variance": "[1.2e-05 1.6e-05]",
    "Accelerometer variance": "[0.1 0.2 0.3]",
}
_FEATURES = "".join(f"{heading}: {text}\n" for heading, text in _FEATURE_TEXT.items()).replace(
    "Accelerometer variance: [0.1 0.2 0.3]\n", "Accelerometer variance: [0.1 0.2 0.3]  \n"
)


def _feature_key(heading):
    return heading.lower().replace(" ", "_")


def _channel_index(name):
    return next(index for index, (sensor, axis, _) in enumerate(_CHANNELS) if f"{sensor}_{axis}" == name)


def _expected_values(partition_index, row_index, channel_index):
    steps = np.arange(60, dtype=np.float64)
    return (
        _CHANNELS[channel_index][2]
        + row_index * 10
        + steps * (channel_index + 1) / 10
        + steps**2 / 1000
        + partition_index * 100
    )


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
        content = "\n".join(lines) + "\n" + _FEATURES
        if partition == "pamap2_v2":
            # As in the release, PAMAP2's location follows the derived features.
            content += "Location: ankle"
            row["messages"][0]["content"] += " Location is supplied."
        row["messages"][1]["content"] = content
    return rows


@pytest.fixture(autouse=True)
def _fixture_inventory(monkeypatch):
    """Pin the inventory to the fixtures: two rows per partition."""
    monkeypatch.setattr(SensorCapsConnector, "SOURCE_ROWS", dict.fromkeys(_PARTITIONS, 2))


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _refs(tmp_path):
    refs = []
    for partition in _PARTITIONS:
        path = tmp_path / f"{partition}.jsonl"
        _write_rows(path, _rows(partition))
        refs.append(HubFileRef(path, partition))
    return refs


def _annotations(owner):
    return {annotation.key: read_value(annotation.value) for annotation in owner.annotations}


def _signal(record, name):
    return next(signal for signal in record.signals if signal.name == name)


def _write_row(tmp_path, row):
    path = tmp_path / "row.jsonl"
    _write_rows(path, [row])
    return HubFileRef(path, "hhar_v2")


def _with_user_message(row, edit):
    row["messages"][1]["content"] = edit(row["messages"][1]["content"])
    return row


def _assert_records_and_tasks(records, tasks):
    assert len(records) == len(tasks) == 10
    records_by_row = {}
    for record in records:
        provenance = RecordMetadata.from_json(record.metadata, [SensorCapsMetadata]).extension(SensorCapsMetadata)
        records_by_row[provenance.partition, provenance.row] = record
    tasks_by_record = {task.inputs[0].id: task for task in tasks}
    for partition_index, partition in enumerate(_PARTITIONS):
        for row_index, row in enumerate(_rows(partition)):
            record = records_by_row[partition, row_index]
            task = tasks_by_record[record.id]
            assert record.metadata == {"sensorcaps": {"partition": partition, "row": row_index}}
            assert record.start_time.timestamp is None
            assert len(record.sources) == 1
            source = record.sources[0]
            assert source.name == "IMU"
            assert source.sources == ()
            assert len(source.signals) == len(record.signals) == 6
            # In the user message's order: the summary, each feature's text without its heading or
            # surrounding whitespace, then the location.
            expected_annotations = {"summary": f"synthetic café {partition} movement {row_index}"}
            expected_annotations |= {_feature_key(heading): text for heading, text in _FEATURE_TEXT.items()}
            if partition == "pamap2_v2":
                expected_annotations["location"] = "ankle"
            assert _annotations(record) == expected_annotations
            assert [annotation.key for annotation in record.annotations] == list(expected_annotations)
            signals = {signal.name: signal for signal in source.signals}
            assert list(signals) == list(_SIGNAL_NAMES)
            for name, signal in signals.items():
                sensor = name.split("_")[0]
                assert signal.spec == TimeSeriesSpec(
                    spec_type=sensor, name=sensor.capitalize(), unit_value=None, dtype="float64"
                )
                # 60 readings at the 10 Hz the paper and the dataset card state.
                assert signal.time_axis == RegularAxis.from_rate_hz(10)
                assert signal.n_values == 60
                actual = signal.to_numpy()
                assert actual.dtype == np.float64
                expected = _expected_values(partition_index, row_index, _channel_index(name))
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
            assert type(task) is AnswerTask
            assert len(task.inputs) == 1 and task.inputs[0] is record
            assert task.targets == (row["messages"][2]["content"],)
            # The instruction GPT-4o-mini received, PAMAP2's naming the location.
            assert task.prompt == row["messages"][0]["content"]
            assert task.split is None
            assert task.metadata == {}
            # The task's context is everything the caption was generated from, in Record order.
            assert len(task.input_annotations) == len(record.annotations)
            for given, attached in zip(task.input_annotations, record.annotations, strict=True):
                assert given is attached or (given.key, given.value) == (attached.key, attached.value)
            assert task.resolved_input_modalities == frozenset({InputModality.TEXT, InputModality.TIME_SERIES})


def test_conversion_follows_the_paper_and_reiterates_tasks(tmp_path):
    dataset = SensorCapsConnector().convert(_refs(tmp_path))
    assert _annotations(dataset) == _DATASET_ANNOTATIONS
    assert dataset.derive_schema().tasks == (AnswerTask,)
    # Verify complete content on both passes, including row-to-record associations.
    for _ in range(2):
        _assert_records_and_tasks(dataset.records, list(dataset.iter_tasks()))


def test_channel_reads_reuse_rows_and_reread_offsets(monkeypatch, tmp_path):
    dataset = SensorCapsConnector().convert(_refs(tmp_path)[:1])
    original = module.parse_row
    calls = []

    def parse(raw, context):
        calls.append(context)
        return original(raw, context)

    monkeypatch.setattr(module, "parse_row", parse)
    first, second = dataset.records
    for signal in first.signals:
        assert len(signal.to_arrow()) == 60
    assert len(calls) == 1  # six channels decode only one source row
    np.testing.assert_allclose(_signal(second, "gyroscope_x").to_numpy(), _expected_values(0, 1, 3))
    assert len(calls) == 2
    np.testing.assert_allclose(_signal(first, "accelerometer_x").to_numpy(), _expected_values(0, 0, 0))
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
    with TimeFReader(open_local_version(version_dir)) as reader:
        assert reader.metadata == connector.metadata()
        assert reader.schema.tasks == (AnswerTask,)
        restored = reader.read()
        _assert_records_and_tasks(restored.records, restored.tasks)
        records = list(reader.iter_records())
        _assert_records_and_tasks(records, list(reader.iter_tasks(inputs=records)))
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


@pytest.mark.parametrize(
    ("edit", "pattern"),
    [
        (lambda user: user.replace("Gyroscope median: ", "Gyroscope mean: "), "unexpected line"),
        (
            lambda user: user.replace("Gyroscope median: [0. 0.1 0.2]\n", ""),
            r"missing derived features: \['Gyroscope median'\]",
        ),
        (lambda user: user + "Gyroscope FFT: [0.]\n", "repeated Gyroscope FFT section"),
        (
            lambda user: user.replace("\nAccelerometer FFT:", "\n\nAccelerometer FFT:").replace(
                "Summary:", "\nSummary:"
            ),
            "unexpected line",
        ),
        (lambda user: user.replace(" [ 0.4", "[ 0.4"), "unexpected line"),
    ],
)
def test_unexpected_user_message_layout_fails_with_source_context(tmp_path, edit, pattern):
    ref = _write_row(tmp_path, _with_user_message(_rows()[0], edit))
    with pytest.raises(TimeFValidationError, match=pattern) as error:
        SensorCapsConnector().convert([ref])
    assert "hhar_v2, row 0" in str(error.value)


def test_a_system_message_that_differs_within_a_file_fails(tmp_path):
    first, second = _rows()
    second["messages"][0]["content"] += " Changed."
    path = tmp_path / "hhar_v2.jsonl"
    _write_rows(path, [first, second])
    with pytest.raises(TimeFValidationError, match="system message differs") as error:
        SensorCapsConnector().convert([HubFileRef(path, "hhar_v2")])
    assert "hhar_v2, row 1" in str(error.value)


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
        np.testing.assert_array_equal(signal.to_numpy(), np.full(60, expected, dtype=np.float64))


def test_pinned_inventory_mismatch_fails(monkeypatch, tmp_path):
    monkeypatch.setitem(SensorCapsConnector.SOURCE_ROWS, "hhar_v2", 3)
    with pytest.raises(TimeFValidationError, match=r"hhar_v2\): expected 3 rows, found 2"):
        SensorCapsConnector().convert(_refs(tmp_path)[:1])


@pytest.mark.parametrize("n_readings", [1, 59, 61])
def test_unexpected_sample_count_fails(tmp_path, n_readings):
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


def test_values_keep_the_six_decimals_the_generator_read(monkeypatch, tmp_path):
    # PAMAP2's -61.793769 becomes -61.79376984 as float32.
    monkeypatch.setitem(SensorCapsConnector.SOURCE_ROWS, "hhar_v2", 1)
    row = _rows()[0]
    lines = row["messages"][1]["content"].splitlines()
    for index, line in enumerate(lines):
        heading, _, _ = line.partition(":")
        if heading in {"Accelerometer", "Gyroscope"}:
            lines[index] = f"{heading}: " + "[" + ", ".join(["[-0.019372, 0.01128, -61.793769]"] * 60) + "]"
    row["messages"][1]["content"] = "\n".join(lines)
    dataset = SensorCapsConnector().convert([_write_row(tmp_path, row)])
    z = _signal(dataset.records[0], "gyroscope_z").to_numpy()
    assert {f"{value:.6f}" for value in z} == {"-61.793769"}
