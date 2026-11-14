"""Offline contract tests for PAMAP2."""

from __future__ import annotations

import math
from pathlib import Path
import zipfile

import pytest

from timenet.engine import store_dataset
from timenet.errors import TimeFFormatError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import ClassificationTask
import timenet_connectors.datasets.uci.pamap2.connector as pamap2_connector
from timenet_connectors.datasets.uci.pamap2.connector import ROOT, Pamap2Connector, Pamap2Source, _extract_members


def _row(timestamp: str, label: int, first: str, offset: int) -> str:
    """Return one valid native row with a distinct value in every source column."""
    return " ".join((timestamp, str(label), first, *(str(offset + index) for index in range(51))))


def test_protocol_file_preserves_named_float64_streams_and_activity_runs(tmp_path: Path) -> None:
    """Native time, NaN, all 52 measurements, and label zero survive conversion."""
    root = tmp_path / ROOT / "Protocol"
    root.mkdir(parents=True)
    (root / "subject101.dat").write_text(_row("0.00", 0, "80", 0) + "\n" + _row("0.01", 4, "nan", 100) + "\n")
    dataset = Pamap2Connector().convert([Pamap2Source(dataset_root=root.parent)])
    record = dataset.records[0]
    assert len(record.time_series) == 52
    assert [series.signal for series in record.time_series[:5]] == [
        "heart_rate",
        "hand_temperature_celsius",
        "hand_acceleration_16g_x",
        "hand_acceleration_16g_y",
        "hand_acceleration_16g_z",
    ]
    assert record.time_series[-1].signal == "ankle_orientation_invalid_q4"
    assert record.time_series[0].to_arrow().to_pylist()[0] == pytest.approx(80.0)
    assert math.isnan(record.time_series[0].to_arrow().to_pylist()[1])
    assert record.time_series[-1].to_arrow().to_pylist() == [50.0, 150.0]
    tasks = [task for task in dataset.iter_tasks() if isinstance(task, ClassificationTask)]
    assert [task.target for task in tasks] == ["other_transient", "walking"]
    dataset.derive_schema()
    version = store_dataset(dataset, tmp_path / "out")
    with TimeFReader(DatasetVersion.open_local(version)) as reader:
        restored = reader.read().records[0]
    assert restored.time_series[-1].to_arrow().to_pylist() == [50.0, 150.0]
    assert math.isnan(restored.time_series[0].to_arrow().to_pylist()[1])
    assert restored.time_series[0].source_id == "Protocol/subject101.dat"


def test_nested_archive_cache_is_verified_before_reuse(tmp_path: Path) -> None:
    """The nested ZIP is extracted once and a changed cache member is rejected."""
    archive = tmp_path / "outer.zip"
    nested = tmp_path / f"{ROOT}.zip"
    with zipfile.ZipFile(nested, "w") as source:
        for subject in range(101, 110):
            source.writestr(f"{ROOT}/Protocol/subject{subject}.dat", _row("0.00", 0, "80", subject) + "\n")
    with zipfile.ZipFile(archive, "w") as source:
        source.write(nested, nested.name)
        source.writestr("readme.pdf", b"source documentation")
    root = _extract_members(archive, tmp_path / "cache")
    assert _extract_members(archive, tmp_path / "cache") == root
    (root / "Protocol" / "subject101.dat").write_text("changed\n")
    with pytest.raises(TimeFFormatError, match="cached extraction has changed"):
        _extract_members(archive, tmp_path / "cache")


def test_one_record_parse_cache_is_shared_across_channels_and_evicted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All streams and timestamps share one parse, while a second recording evicts the first."""
    root = tmp_path / ROOT / "Protocol"
    root.mkdir(parents=True)
    for subject, offset in ((101, 0), (102, 1000)):
        (root / f"subject{subject}.dat").write_text(
            _row("0.00", 0, str(offset), offset) + "\n" + _row("0.01", 4, str(offset + 1), offset + 100) + "\n"
        )
    dataset = Pamap2Connector().convert([Pamap2Source(dataset_root=root.parent)])
    pamap2_connector._cached_record.cache_clear()
    source_row = pamap2_connector._row
    calls = 0

    def count_rows(path: Path, row: int, line: bytes) -> tuple[int, int, tuple[float, ...]]:
        nonlocal calls
        calls += 1
        return source_row(path, row, line)

    monkeypatch.setattr(pamap2_connector, "_row", count_rows)
    first, second = dataset.records
    for series in first.time_series:
        series.to_arrow()
    first.time_series[0].time_offsets_us()
    assert calls == 2
    second.time_series[0].to_arrow()
    assert calls == 4
    first.time_series[0].to_arrow()
    assert calls == 6
