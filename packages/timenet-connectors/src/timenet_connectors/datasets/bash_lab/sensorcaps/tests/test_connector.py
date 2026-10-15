import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import yaml

from timenet.config import settings
from timenet.dataset.axis import OrdinalAxis
from timenet.engine import run_pipeline, store_dataset
from timenet.errors import TimeFValidationError
from timenet.format import control_writer
from timenet.format.control_audit import audit_control_database
from timenet.format.duckdb import connect_control
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import AnswerTask, License
from timenet_connectors import discovery
from timenet_connectors.datasets.bash_lab.sensorcaps import SensorCapsConnector, connector as module
from timenet_connectors.datasets.bash_lab.sensorcaps.connector import (
    HF_REPO,
    SOURCE_REVISION,
    SOURCE_ROWS,
    SensorCapsFileRef,
    partition_rows,
)


_FIXTURES = Path(__file__).parent / "fixtures"


def _refs(tmp_path, version="v2"):
    if version == "v2":
        rows = [json.loads(line) for line in (_FIXTURES / "v2.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in rows:
            row["messages"][1]["content"] += "\nLocation: ankle"
        (tmp_path / "pamap2.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    return [
        SensorCapsFileRef(
            tmp_path / "pamap2.jsonl" if partition.startswith("pamap2_") else _FIXTURES / f"{version}.jsonl", partition
        )
        for partition in partition_rows(version)
    ]


def _configured_connector(tmp_path, version):
    card = yaml.safe_load(SensorCapsConnector._card_path().read_text())
    card["build_options"] = {"caption_version": version}
    card["dataset_version"] = "1.0.1" if version == "v1" else "1.0.0"
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(card))

    class ConfiguredConnector(SensorCapsConnector):
        CARD = path

    return ConfiguredConnector()


def _rows(file):
    return [json.loads(line) for line in file.path.read_text(encoding="utf-8").splitlines()]


def _annotations(owner):
    return {annotation.key: annotation.value for annotation in owner.annotations}


def _write_row(tmp_path, row):
    path = tmp_path / "row.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return SensorCapsFileRef(path, "hhar_v2")


def test_discovery_and_card():
    assert discovery.resolve("bash-lab/sensorcaps") is SensorCapsConnector
    assert "bash-lab/sensorcaps" in discovery.available()
    card = SensorCapsConnector().metadata()
    assert card.dataset_id == "bash-lab/sensorcaps"
    assert card.license is License.OTHER
    assert card.license_url and "cl-eco-extr" in card.license_url
    assert SOURCE_REVISION in card.description
    assert card.build_options == {"caption_version": "v2"}
    assert sum(SOURCE_ROWS.values()) == sum(partition_rows("v2").values()) == 35960
    assert sum(partition_rows("v1").values()) == 26288


def test_missing_download_dependency(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match="without --no-isolation"):
        SensorCapsConnector().download(tmp_path)


@pytest.mark.parametrize(("version", "expected"), [("v1", 26288), ("v2", 35960)])
def test_card_selects_caption_version(monkeypatch, tmp_path, version, expected):
    import huggingface_hub  # noqa: PLC0415

    seen = []

    def download(repo_id, **kwargs):
        seen.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    connector = _configured_connector(tmp_path, version)
    downloaded = connector.download(tmp_path)
    assert sum(ref.expected_rows or 0 for ref in downloaded) == expected
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
    assert all(ref.partition.endswith(version) for ref in downloaded)
    assert any(ref.partition.startswith("pamap2_") for ref in downloaded) == (version == "v2")
    dataset = connector.convert(_refs(tmp_path, version))
    assert len(dataset.records) == len(downloaded) * 2
    assert _annotations(dataset)["caption_version"] == version
    assert all(_annotations(record)["caption_version"] == version for record in dataset.records)
    with pytest.raises(TimeFValidationError, match="selected caption version"):
        connector.convert(_refs(tmp_path, "v1" if version == "v2" else "v2"))


@pytest.mark.parametrize("options", [{"caption_version": "v3"}, {"caption_versions": "v1"}])
def test_invalid_build_options_fail_before_download(tmp_path, options):
    connector = _configured_connector(tmp_path, "v2")
    card = yaml.safe_load(connector._card_path().read_text())
    card["build_options"] = options
    connector._card_path().write_text(yaml.safe_dump(card))
    with pytest.raises(TimeFValidationError, match="invalid SensorCaps build_options"):
        type(connector)()


def test_conversion_builds_channels_provenance_and_reiterable_tasks(tmp_path):
    dataset = SensorCapsConnector().convert(_refs(tmp_path))
    schema = dataset.derive_schema()
    assert len(dataset.records) == 10
    assert schema.tasks == (AnswerTask,)
    assert {spec.spec_type for spec in schema.time_series_specs} == {"accelerometer", "gyroscope"}
    assert _annotations(dataset) == {
        "source_repository": HF_REPO,
        "source_revision": SOURCE_REVISION,
        "declared_sampling_rate_hz": 10,
        "caption_version": "v2",
    }
    tasks = list(dataset.iter_tasks())
    assert [task.id for task in dataset.iter_tasks()] == [task.id for task in tasks]
    assert len({record.id for record in dataset.records}) == 10
    for record in dataset.records:
        assert len(record.sources) == 1
        assert [signal.name for signal in record.signals] == [
            f"{sensor}_{axis}" for sensor in ("accelerometer", "gyroscope") for axis in ("x", "y", "z")
        ]
        metadata = _annotations(record)
        assert metadata["source_file"] == f"data/{metadata['source_partition']}-00000-of-00001.jsonl"
        assert ("sensor_location" in metadata) == (metadata["source_partition"] == "pamap2_v2")


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
        assert len(signal.to_arrow()) == 2
    assert len(calls) == 1  # six channels decode only one source row
    np.testing.assert_array_equal(second.signals[0].to_numpy(), np.asarray([1.1, -4.4, 7.7], dtype=np.float32))
    assert len(calls) == 2
    np.testing.assert_array_equal(first.signals[3].to_numpy(), np.asarray([0.123456789012345, 0.4], dtype=np.float32))
    assert len(calls) == 3  # revisiting evicted rows reads by byte offset, including UTF-8


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_real_storage_round_trip_preserves_semantics(monkeypatch, tmp_path, version):
    # Cross real task-batch boundaries without needing a 10,001-row fixture corpus.
    monkeypatch.setattr(control_writer, "_TASK_BATCH_SIZE", 3)
    refs = _refs(tmp_path, version)
    dataset = _configured_connector(tmp_path, version).convert(refs)
    version_dir = store_dataset(dataset, tmp_path)
    with connect_control(version_dir / "control.duckdb", read_only=True) as connection:
        audit_control_database(connection)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        records = list(reader.iter_records())
        tasks = list(reader.iter_tasks(records))
        assert len(records) == len(tasks) == len(refs) * 2
        assert reader.metadata.build_options == {"caption_version": version}
        for index, (record, task) in enumerate(zip(records, tasks, strict=True)):
            source_row = _rows(refs[index // 2])[index % 2]
            assert task.inputs[0] is record
            assert task.targets == (source_row["messages"][2]["content"],)
            assert task.metadata == {
                "system_message": source_row["messages"][0]["content"],
                "user_message": source_row["messages"][1]["content"],
            }
            assert task.prompt is None and task.split is None and task.input_annotations == ()
            assert _annotations(record) == _annotations(dataset.records[index])
            for restored_signal, original_signal in zip(record.signals, dataset.records[index].signals, strict=True):
                np.testing.assert_array_equal(restored_signal.to_numpy(), original_signal.to_numpy())
                assert restored_signal.to_numpy().dtype == np.float32
                assert restored_signal.spec.unit_value is None
                assert isinstance(restored_signal.time_axis, OrdinalAxis)
        reader.verify()


def test_engine_keeps_sources_until_write_then_cleans_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("TIMENET_CACHE", str(tmp_path / "cache"))
    cache = settings().cache_dir / "bash-lab/sensorcaps"

    def download(self, cache_dir):
        assert cache_dir == cache
        refs = []
        for source in _refs(tmp_path):
            target = cache_dir / f"{source.partition}.jsonl"
            shutil.copyfile(source.path, target)
            refs.append(SensorCapsFileRef(target, source.partition))
        return refs

    monkeypatch.setattr(SensorCapsConnector, "download", download)
    version_dir = run_pipeline(SensorCapsConnector(), tmp_path / "registry")
    assert not cache.exists()
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
        assert len(restored.tasks) == 10
        np.testing.assert_array_equal(
            restored.records[0].signals[0].to_numpy(), np.asarray([1.1, -4.4], dtype=np.float32)
        )
        user_message = restored.tasks[0].metadata["user_message"]
        assert isinstance(user_message, str) and "Gyroscope:" in user_message


@pytest.mark.parametrize(
    ("sensor", "pattern"),
    [
        ("[]", "nonempty"),
        ("[[1,2,3], [4,5]]", "three values"),
        ('[[1,2,"3"]]', "finite numbers"),
        ("[[NaN,2,3]]", "finite numbers"),
        ("[[1,2,3]]", "lengths differ"),
    ],
)
def test_invalid_sensor_data_has_source_context(tmp_path, sensor, pattern):
    row = _rows(_refs(tmp_path)[0])[0]
    user = row["messages"][1]["content"]
    lines = [f"Gyroscope: {sensor}" if line.startswith("Gyroscope:") else line for line in user.splitlines()]
    row["messages"][1]["content"] = "\n".join(lines)
    ref = _write_row(tmp_path, row)
    with pytest.raises(TimeFValidationError, match=pattern) as error:
        SensorCapsConnector().convert([ref])
    assert str(ref.path) in str(error.value)
    assert "hhar_v2, row 0" in str(error.value)


@pytest.mark.parametrize("case", ["json", "roles", "caption", "missing"])
def test_invalid_messages_and_sections(tmp_path, case):
    row = _rows(_refs(tmp_path)[0])[0]
    if case == "roles":
        row["messages"][1]["role"] = "assistant"
    elif case == "caption":
        row["messages"][2]["content"] = "  "
    elif case == "missing":
        row["messages"][1]["content"] = "Summary: movement"
    ref = _write_row(tmp_path, row)
    if case == "json":
        ref.path.write_text("{not json}\n")
    with pytest.raises(TimeFValidationError, match="hhar_v2, row 0"):
        SensorCapsConnector().convert([ref])


def test_empty_source_is_rejected(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.touch()
    with pytest.raises(TimeFValidationError, match="empty source partition"):
        SensorCapsConnector().convert([SensorCapsFileRef(empty, "hhar_v2")])


def test_pinned_downloads_enforce_partition_counts(tmp_path):
    fixture = _refs(tmp_path)[0]
    with pytest.raises(TimeFValidationError, match=r"hhar_v2.*expected 9166 rows, found 2"):
        SensorCapsConnector().convert([SensorCapsFileRef(fixture.path, fixture.partition, SOURCE_ROWS["hhar"])])
