from collections import Counter

import pytest

from timenet.client import TimeNet
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.format.checksums import file_checksum
from timenet.manifest import LockedDependency
from timenet.registry import LocalRegistry
from timenet.testing import make_dataset
from timenet.types import DatasetMetadata, DatasetRef, License, Version
from timenet.writer import TimeFWriter


def write_layer(root, name, parents=(), *, dependencies=None):
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id=name,
            dataset_version=Version(1, 0, 0),
            name=name,
            description="Test layer",
            license=License.MIT,
            parents=tuple(DatasetRef(dataset_id=parent, version=Version(1, 0, 0)) for parent in parents),
        )
    )
    registry = LocalRegistry(root)
    lock = {}
    for parent in parents:
        manifest = registry.get_manifest(parent, "1.0.0")
        for entry in manifest.dependencies:
            lock[entry.dataset] = entry
        direct = LockedDependency(
            dataset_id=parent,
            version=Version(1, 0, 0),
            manifest_checksum=file_checksum(root / parent / "1.0.0/manifest.json"),
        )
        lock[direct.dataset] = direct
    dataset.set_dependencies(tuple(lock.values()) if dependencies is None else dependencies)
    for record in make_dataset().records:
        dataset.add_record(record=record)
    dataset.derive_schema()
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    return registry.get_manifest(name, "1.0.0")


@pytest.mark.parametrize("operation", ["load", "download"])
@pytest.mark.parametrize("transitive", [False, True])
def test_wrong_dependency_checksum_is_rejected(tmp_path, operation, transitive):
    write_layer(tmp_path, "test/base")
    middle = write_layer(tmp_path, "test/middle", ("test/base",))
    parents = ("test/middle",) if transitive else ("test/base",)
    lock = [LockedDependency(dataset_id="test/base", version=Version(1, 0, 0), manifest_checksum="sha256:" + "f" * 64)]
    if transitive:
        lock.append(
            LockedDependency(
                dataset_id=middle.dataset_id,
                version=Version(1, 0, 0),
                manifest_checksum=file_checksum(tmp_path / "test/middle/1.0.0/manifest.json"),
            )
        )
    write_layer(tmp_path, "test/child", parents, dependencies=lock)
    client = TimeNet(tmp_path, storage_path=tmp_path / "download")
    with pytest.raises(TimeFFormatError, match="checksum"):
        getattr(client, operation)("test/child")
    assert not (tmp_path / "download/test/child/1.0.0/manifest.json").exists()


def test_incomplete_transitive_lock_is_rejected(tmp_path):
    write_layer(tmp_path, "test/base")
    write_layer(tmp_path, "test/middle", ("test/base",))
    direct = LockedDependency(
        dataset_id="test/middle",
        version=Version(1, 0, 0),
        manifest_checksum=file_checksum(tmp_path / "test/middle/1.0.0/manifest.json"),
    )
    write_layer(tmp_path, "test/child", ("test/middle",), dependencies=(direct,))
    with pytest.raises(TimeFFormatError, match="complete parent closure"):
        LocalRegistry(tmp_path).open_reader("test/child")


def test_diamond_opens_each_version_once(tmp_path):
    write_layer(tmp_path, "test/base")
    for name in ("test/left", "test/right"):
        write_layer(tmp_path, name, ("test/base",))
    write_layer(tmp_path, "test/child", ("test/left", "test/right"))

    class Registry(LocalRegistry):
        def __init__(self, root):
            super().__init__(root)
            self.opens = Counter()

        def open_version(self, dataset_id, version=None):
            self.opens[dataset_id] += 1
            return super().open_version(dataset_id, version)

    registry = Registry(tmp_path)
    with registry.open_reader("test/child"):
        assert registry.opens == dict.fromkeys(("test/base", "test/left", "test/right", "test/child"), 1)
