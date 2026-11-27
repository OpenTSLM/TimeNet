"""Offline contract tests for Shoaib CSV parsing."""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from timenet.engine import store_dataset
from timenet.errors import TimeFFormatError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet_connectors.datasets.utwente.shoaib import connector
from timenet_connectors.datasets.utwente.shoaib.connector import ShoaibConnector, ShoaibParticipant


def test_convert_retains_five_timestamp_tokens_sixty_channels_and_activity(tmp_path: Path) -> None:
    """The native 70-field layout remains reconstructible without timestamp coercion."""
    path = tmp_path / "Participant_1.csv"
    headers = ["position"] + [""] * 69
    fields = ["time_stamp", "Ax", "Ay", "Az", "Lx", "Ly", "Lz", "Gx", "Gy", "Gz", "Mx", "My", "Mz", ""] * 5
    row = []
    for position in range(5):
        row.extend(["1.39E+12"] + [str(position + value / 10) for value in range(1, 13)] + [""])
    row[-1] = "walking"
    path.write_text(",".join(headers) + "\n" + ",".join(fields) + "\n" + ",".join(row) + "\n")
    dataset = ShoaibConnector().convert([ShoaibParticipant(path=path, participant="1")])
    record = dataset.records[0]
    assert len(record.time_series) == 66
    assert record.time_series[0].to_arrow().to_pylist() == ["1.39E+12"]
    assert record.time_series[1].to_arrow().to_pylist() == [0.1]
    assert record.time_series[-1].to_arrow().to_pylist() == ["walking"]


def test_fixture_persisted_roundtrip_keeps_timestamp_and_activity(tmp_path: Path) -> None:
    """The fixture's native timestamp token and activity stream survive TimeF storage."""
    path = tmp_path / "Participant_1.csv"
    headers = ["position"] + [""] * 69
    fields = ["time_stamp", "Ax", "Ay", "Az", "Lx", "Ly", "Lz", "Gx", "Gy", "Gz", "Mx", "My", "Mz", ""] * 5
    row = ["1.39E+12", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1", "2", "3", ""] * 5
    row[-1] = "walking"
    path.write_text("\n".join((",".join(headers), ",".join(fields), ",".join(row), "")))
    dataset = ShoaibConnector().convert([ShoaibParticipant(path=path, participant="1")])
    dataset.derive_schema()
    version = store_dataset(dataset, tmp_path / "out")
    with TimeFReader(DatasetVersion.open_local(version)) as reader:
        restored = reader.read().records[0]
    assert restored.time_series[0].to_arrow().to_pylist() == ["1.39E+12"]
    assert restored.time_series[-1].to_arrow().to_pylist() == ["walking"]


@pytest.fixture
def archive_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    archive = tmp_path / "shoaib.rar"
    archive.write_bytes(b"pinned fixture archive")
    monkeypatch.setattr(connector, "SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    names = [f"DataSet/Participant_{i}.csv" for i in range(1, 11)] + ["DataSet/Readme.txt"]
    entries = [
        SimpleNamespace(
            pathname=name,
            isreg=True,
            islnk=False,
            issym=False,
            isdir=False,
            get_blocks=lambda: iter((b"original header\noriginal body\n",)),
        )
        for name in names
    ]

    @contextmanager
    def file_reader(_path):
        yield entries

    monkeypatch.setitem(sys.modules, "libarchive", SimpleNamespace(file_reader=file_reader))
    return tmp_path, entries


def test_download_rechecks_every_cached_byte(archive_cache) -> None:
    cache, _entries = archive_cache
    first = ShoaibConnector().download(cache)
    second = ShoaibConnector().download(cache)
    assert len(first) == len(second) == 10
    assert (cache / "DataSet" / "Readme.txt").is_file()
    assert not list(cache.glob(".shoaib-*"))


@pytest.mark.parametrize("damage", ["header", "body", "readme", "missing", "extra", "file_link", "root_link"])
def test_download_rejects_untrusted_extracted_cache(archive_cache, damage: str) -> None:
    cache, _entries = archive_cache
    ShoaibConnector().download(cache)
    root = cache / "DataSet"
    target = root / "Participant_1.csv"
    if damage in {"header", "body"}:
        target.write_bytes(
            b"changed header\noriginal body\n" if damage == "header" else b"original header\nchanged body\n"
        )
    elif damage == "readme":
        (root / "Readme.txt").write_text("changed provenance")
    elif damage == "missing":
        target.unlink()
    elif damage == "extra":
        (root / "Participant_11.csv").write_text("unexpected")
    elif damage == "file_link":
        target.unlink()
        target.symlink_to(root / "Participant_2.csv")
    else:
        root.rename(cache / "old")
        root.symlink_to(cache / "old", target_is_directory=True)
    with pytest.raises(TimeFFormatError, match="cache"):
        ShoaibConnector().download(cache)
    assert not list(cache.glob(".shoaib-*"))


@pytest.mark.parametrize("damage", ["symlink", "hardlink", "device", "duplicate", "missing", "traversal"])
def test_download_rejects_invalid_archive_members(archive_cache, damage: str) -> None:
    cache, entries = archive_cache
    if damage == "symlink":
        entries[0].issym = True
    elif damage == "hardlink":
        entries[0].islnk = True
    elif damage == "device":
        entries[0].isreg = False
    elif damage == "duplicate":
        entries.append(entries[0])
    elif damage == "missing":
        entries.pop()
    else:
        entries[0].pathname = "DataSet/../../escape.csv"
    with pytest.raises(TimeFFormatError, match="archive"):
        ShoaibConnector().download(cache)
    assert not (cache / "DataSet").exists()
    assert not list(cache.glob(".shoaib-*"))
