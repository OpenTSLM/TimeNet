"""Offline contract tests for the UCI-HAR connector."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
import zipfile

import pytest

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.engine import store_dataset
from timenet.errors import TimeFFormatError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import ClassificationTask
from timenet_connectors.datasets.uci.har.connector import (
    ACTIVITY_VOCABULARY_ID,
    ARCHIVE_FILENAME,
    UciHarConnector,
    UciHarSource,
)


_SIGNALS = (
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
)


def _values(row: int, channel: int, split_offset: int = 0) -> str:
    """Return one 128-value source row with a float64-sensitive leading value."""
    start = 1.0000000000000002 + split_offset + row * 10.0 + channel / 100.0
    return " ".join(repr(start + index / 1000.0) for index in range(128)) + "\n"


def _write_split(root: Path, split: str, subjects: tuple[int, ...], labels: tuple[int, ...]) -> None:
    """Write a minimal official-layout split."""
    split_dir = root / split
    signals_dir = split_dir / "Inertial Signals"
    signals_dir.mkdir(parents=True)
    (split_dir / f"subject_{split}.txt").write_text("".join(f"{value}\n" for value in subjects))
    (split_dir / f"y_{split}.txt").write_text("".join(f"{value}\n" for value in labels))
    for channel, signal in enumerate(_SIGNALS):
        split_offset = 0 if split == "train" else 100
        (signals_dir / f"{signal}_{split}.txt").write_text(
            "".join(_values(row, channel, split_offset) for row in range(len(subjects)))
        )


@pytest.fixture
def source(tmp_path: Path) -> UciHarSource:
    """Build a tiny source tree with both official splits."""
    root = tmp_path / "UCI HAR Dataset"
    root.mkdir()
    (root / "activity_labels.txt").write_text(
        "1 WALKING\n2 WALKING_UPSTAIRS\n3 WALKING_DOWNSTAIRS\n4 SITTING\n5 STANDING\n6 LAYING\n"
    )
    _write_split(root, "train", (1, 2), (1, 6))
    _write_split(root, "test", (3,), (2,))
    return UciHarSource(dataset_root=root)


def test_is_a_connector_and_has_the_official_card() -> None:
    """The discoverable connector identifies the UCI-HAR release."""
    connector = UciHarConnector()
    assert isinstance(connector, BaseConnector)
    assert connector.metadata().dataset_id == "uci/har"
    assert str(connector.metadata().license) == "CC-BY-4.0"


def test_convert_preserves_all_nine_channels_and_source_metadata(source: UciHarSource) -> None:
    """Rows become one 128-step record with source-native split, subject, and row facts."""
    dataset = UciHarConnector().convert([source])
    assert isinstance(dataset, TimeFDataset)
    assert [record.record_id for record in dataset.records] == ["uci-har-train-1", "uci-har-train-2", "uci-har-test-1"]
    record = dataset.records[0]
    assert record.subject_ids == ("1",)
    assert len(record.time_series) == 9
    assert [series.signal for series in record.time_series] == list(_SIGNALS)
    assert all(series.n_values == 128 for series in record.time_series)
    assert all(series.time_axis == RegularAxis.from_rate_hz(50) for series in record.time_series)
    assert all(str(series.spec.dtype) == "float64" for series in record.time_series)
    assert [series.source_id for series in record.time_series] == [
        f"train/Inertial Signals/{signal}_train.txt" for signal in _SIGNALS
    ]
    assert {annotation.key: annotation.value for annotation in record.annotations} == {
        "activity_id": 1,
        "split": "train",
        "source_row": 1,
        "subject_id": 1,
    }


def test_labels_are_classification_tasks_using_the_registered_exact_vocabulary(source: UciHarSource) -> None:
    """Official activity labels stay numeric in provenance and exact in task targets."""
    dataset = UciHarConnector().convert([source])
    vocabularies = {annotation.id: annotation for annotation in dataset.registered_annotations}
    assert vocabularies[ACTIVITY_VOCABULARY_ID].value == [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ]
    all_tasks = list(dataset.iter_tasks())
    tasks = [task for task in all_tasks if isinstance(task, ClassificationTask)]
    assert len(tasks) == len(all_tasks)
    assert [task.target for task in tasks] == ["WALKING", "LAYING", "WALKING_UPSTAIRS"]
    assert all(task.target_schema == ACTIVITY_VOCABULARY_ID for task in tasks)
    assert [task.record_ids for task in tasks] == [(record.record_id,) for record in dataset.records]


def test_lazy_values_keep_float64_precision_and_are_repeatable(source: UciHarSource) -> None:
    """Loaders seek a source line on each read and do not narrow values to float32."""
    series = UciHarConnector().convert([source]).records[0].time_series[0]
    first = series.to_arrow()
    second = series.to_arrow()
    assert str(first.type) == "double"
    assert first.to_pylist() == second.to_pylist()
    assert first[0].as_py() == 1.0000000000000002  # noqa: RUF069 (exact source fidelity)


def test_roundtrip_preserves_every_fixture_value_and_provenance(source: UciHarSource, tmp_path: Path) -> None:
    """A stored TimeF release keeps all source values, task labels, and source identifiers exactly."""
    dataset = UciHarConnector().convert([source])
    dataset.derive_schema()
    version_dir = store_dataset(dataset, tmp_path / "out")
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
        before_by_id = {record.record_id: record for record in dataset.records}
        after_by_id = {record.record_id: record for record in restored.records}
        assert after_by_id.keys() == before_by_id.keys()
        assert {task.record_ids: task.target for task in restored.tasks} == {
            task.record_ids: task.target for task in dataset.tasks
        }
        for record_id, before in before_by_id.items():
            after = after_by_id[record_id]
            assert before.subject_ids == after.subject_ids
            assert before.annotations == after.annotations
            for expected, actual in zip(before.time_series, after.time_series, strict=True):
                assert expected.signal == actual.signal
                assert expected.source_id == actual.source_id
                assert expected.to_arrow().to_pylist() == actual.to_arrow().to_pylist()


def test_lazy_reads_are_independent_for_reordered_records(source: UciHarSource) -> None:
    """Offsets locate the same exact source row on repeated non-sequential reads."""
    records = UciHarConnector().convert([source]).records
    requested = (2, 0, 2, 1)
    actual = [records[index].time_series[8].to_arrow().to_pylist() for index in requested]
    expected = []
    for index in requested:
        split = "test" if index == 2 else "train"
        row = 0 if index in {0, 2} else 1
        path = source.dataset_root / split / "Inertial Signals" / f"body_gyro_z_{split}.txt"
        expected.append([float(value) for value in path.read_text().splitlines()[row].split()])
    assert actual == expected


@pytest.mark.parametrize(
    ("relative", "replacement", "match"),
    [
        ("train/y_train.txt", "7\n6\n", "unknown activity label"),
        ("train/subject_train.txt", "31\n2\n", "subject"),
        ("train/Inertial Signals/body_gyro_z_train.txt", "1 2\n", "128"),
        ("train/Inertial Signals/body_acc_x_train.txt", _values(0, 3), "disagree"),
        ("train/Inertial Signals/body_gyro_y_train.txt", "nan " * 128 + "\n" + _values(1, 7), "non-finite"),
        ("activity_labels.txt", "1 WALKING\n1 WALKING\n", "declared twice"),
    ],
)
def test_convert_rejects_malformed_official_rows(
    source: UciHarSource, relative: str, replacement: str, match: str
) -> None:
    """Corrupt labels, subjects, and signal shapes fail before a dataset is emitted."""
    path = source.dataset_root / relative
    path.write_text(replacement)
    with pytest.raises(TimeFFormatError, match=match):
        UciHarConnector().convert([source])


def test_download_extracts_only_required_members_and_validates_cached_archive(tmp_path: Path, monkeypatch) -> None:
    """The nested release stays cached and unrelated archive paths never reach the dataset tree."""
    outer = tmp_path / ARCHIVE_FILENAME
    inner_bytes = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner_bytes, "w") as inner:
        inner.writestr("UCI HAR Dataset/activity_labels.txt", "1 WALKING\n2 WALKING_UPSTAIRS\n3 WALKING_DOWNSTAIRS\n")
        for split in ("train", "test"):
            inner.writestr(f"UCI HAR Dataset/{split}/subject_{split}.txt", "1\n")
            inner.writestr(f"UCI HAR Dataset/{split}/y_{split}.txt", "1\n")
            for signal in _SIGNALS:
                inner.writestr(f"UCI HAR Dataset/{split}/Inertial Signals/{signal}_{split}.txt", _values(0, 0))
        inner.writestr("UCI HAR Dataset/../../escaped.txt", "must not extract")
    with zipfile.ZipFile(outer, "w") as archive:
        archive.writestr("UCI HAR Dataset.zip", inner_bytes.read_bytes())
        archive.writestr("UCI HAR Dataset.names", "metadata")
    original_archive = outer.read_bytes()

    monkeypatch.setattr(
        "timenet_connectors.datasets.uci.har.connector.ARCHIVE_SHA256",
        hashlib.sha256(outer.read_bytes()).hexdigest(),
    )

    async def no_network(*args, **kwargs):
        raise AssertionError("the validated cached archive must not download")

    monkeypatch.setattr("timenet_connectors.datasets.uci.har.connector.download_files", no_network)
    refs = asyncio.run(UciHarConnector().download_async(tmp_path))
    assert refs[0].dataset_root == tmp_path / "UCI HAR Dataset"
    assert outer.exists()
    assert outer.read_bytes() == original_archive
    assert not (tmp_path / "escaped.txt").exists()

    source_file = tmp_path / "UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt"
    original_signal = source_file.read_bytes()
    source_file.write_bytes(b"0" * len(original_signal))
    asyncio.run(UciHarConnector().download_async(tmp_path))
    assert source_file.read_bytes() == original_signal
    source_file.write_bytes(b"partial")
    asyncio.run(UciHarConnector().download_async(tmp_path))
    assert source_file.read_bytes() == original_signal

    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside bytes")
    source_file.unlink()
    source_file.symlink_to(outside)
    with pytest.raises(TimeFFormatError, match="symlink"):
        asyncio.run(UciHarConnector().download_async(tmp_path))
    assert outside.read_bytes() == b"outside bytes"


def test_download_keeps_a_corrupt_outer_archive_for_diagnosis(tmp_path: Path, monkeypatch) -> None:
    """A bad caller cache is reported without deletion or network activity."""
    outer = tmp_path / ARCHIVE_FILENAME
    outer.write_bytes(b"not the official archive")

    async def no_network(*args, **kwargs):
        raise AssertionError("a corrupt cache must not be silently replaced")

    monkeypatch.setattr("timenet_connectors.datasets.uci.har.connector.download_files", no_network)
    with pytest.raises(TimeFFormatError, match="SHA-256 mismatch"):
        asyncio.run(UciHarConnector().download_async(tmp_path))
    assert outer.read_bytes() == b"not the official archive"


def test_download_rejects_an_outer_archive_without_the_fixed_nested_member(tmp_path: Path, monkeypatch) -> None:
    """The official outer layout must include the named nested release archive."""
    outer = tmp_path / ARCHIVE_FILENAME
    with zipfile.ZipFile(outer, "w") as archive:
        archive.writestr("UCI HAR Dataset.names", "metadata")
    monkeypatch.setattr(
        "timenet_connectors.datasets.uci.har.connector.ARCHIVE_SHA256",
        hashlib.sha256(outer.read_bytes()).hexdigest(),
    )
    with pytest.raises(TimeFFormatError, match="missing nested member"):
        asyncio.run(UciHarConnector().download_async(tmp_path))


def test_convert_rejects_subjects_shared_between_frozen_splits(source: UciHarSource) -> None:
    """The source's train/test subject partition is part of its frozen split contract."""
    (source.dataset_root / "test/subject_test.txt").write_text("1\n")
    with pytest.raises(TimeFFormatError, match="overlap"):
        UciHarConnector().convert([source])
