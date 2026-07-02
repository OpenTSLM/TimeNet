import dataclasses

import pytest

from timenet.client import TimeNet
from timenet.dataset import TimeFDataset
from timenet.errors import DatasetNotFoundError
from timenet.manifest import Manifest
from timenet.testing import assert_datasets_equal, make_dataset
from timenet.types import Domain, Version
from timenet.writer import TimeFWriter


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("TIMENET_HOME", "TIMENET_STORAGE", "TIMENET_CACHE", "TIMENET_REGISTRY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def registry_root(tmp_path):
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "reg", dataset) as writer:
        writer.write()
    return tmp_path / "reg"


def test_no_args_uses_local_home_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "home" / "registry", dataset) as writer:  # the default local registry
        writer.write()
    client = TimeNet()  # no registry, no storage
    assert {m.dataset_id for m in client.list()} == {"hello_world"}
    assert client.download("hello_world") == tmp_path / "home" / "storage" / "hello_world" / "1.0.0"


def test_list(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    assert {m.dataset_id for m in client.list()} == {"hello_world"}


def test_get_returns_manifest(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    assert isinstance(client.get("hello_world"), Manifest)


def test_search(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    assert {m.dataset_id for m in client.search(domain=Domain.GENERAL)} == {"hello_world"}
    assert client.search(domain=Domain.CARDIOLOGY) == []


def test_download_copies_into_storage(registry_root, tmp_path):
    storage = tmp_path / "store"
    client = TimeNet(registry_root, storage_path=storage)
    version_dir = client.download("hello_world")
    assert version_dir == storage / "hello_world" / "1.0.0"
    assert (version_dir / "manifest.json").exists()
    assert list(version_dir.glob("time_series/shard-*.parquet"))


def test_download_is_idempotent(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    first = client.download("hello_world")
    second = client.download("hello_world")
    assert first == second


def test_force_redownload_replaces_atomically(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    first = client.download("hello_world")
    (first / "stale.txt").write_text("left over")  # a marker that a clean replace should remove

    again = client.download("hello_world", force=True)
    assert again == first
    assert (again / "manifest.json").exists()
    assert not (again / "stale.txt").exists()  # replaced wholesale, not written in place
    assert not list(again.parent.glob("*.tmp-*"))  # staging dir cleaned up


def test_download_sweeps_stale_staging(registry_root, tmp_path):
    storage = tmp_path / "store"
    client = TimeNet(registry_root, storage_path=storage)
    # A hard-killed download skips the cleanup finally, leaking a <version>.tmp-* dir.
    stale = storage / "hello_world" / "1.0.0.tmp-deadbeef"
    stale.mkdir(parents=True)
    (stale / "junk.parquet").write_text("partial")

    client.download("hello_world")
    assert not list((storage / "hello_world").glob("*.tmp-*"))  # swept before staging a fresh copy


def test_load_round_trips(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    restored = client.load("hello_world")
    assert isinstance(restored, TimeFDataset)
    assert_datasets_equal(make_dataset(), restored)


def test_registry_env_var(registry_root, tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_REGISTRY", str(registry_root))
    client = TimeNet(storage_path=tmp_path / "store")
    assert {m.dataset_id for m in client.list()} == {"hello_world"}


@pytest.fixture
def versioned_registry(tmp_path):
    """A registry holding hello_world at both 1.0.0 and 1.1.0."""
    root = tmp_path / "vreg"
    for version in (Version(1, 0, 0), Version(1, 1, 0)):
        dataset = make_dataset()
        dataset._metadata = dataclasses.replace(dataset.metadata, dataset_version=version)
        dataset.derive_schema()
        with TimeFWriter(root, dataset) as writer:
            writer.write()
    return root


def test_version_ref_pins_and_defaults_to_latest(versioned_registry, tmp_path):
    client = TimeNet(versioned_registry, storage_path=tmp_path / "store")
    assert str(client.get("hello_world").metadata.dataset_version) == "1.1.0"  # no ref -> latest
    assert str(client.get("hello_world@latest").metadata.dataset_version) == "1.1.0"
    assert str(client.get("hello_world@1.0.0").metadata.dataset_version) == "1.0.0"  # pinned
    assert str(client.get("hello_world", "1.0.0").metadata.dataset_version) == "1.0.0"  # explicit arg


def test_version_ref_download_pins(versioned_registry, tmp_path):
    client = TimeNet(versioned_registry, storage_path=tmp_path / "store")
    assert client.download("hello_world@1.0.0") == tmp_path / "store" / "hello_world" / "1.0.0"


def test_version_ref_missing_pin_raises(versioned_registry, tmp_path):
    client = TimeNet(versioned_registry, storage_path=tmp_path / "store")
    with pytest.raises(DatasetNotFoundError):
        client.get("hello_world@9.9.9")


def test_version_given_twice_raises(versioned_registry, tmp_path):
    client = TimeNet(versioned_registry, storage_path=tmp_path / "store")
    with pytest.raises(ValueError, match="twice"):
        client.get("hello_world@1.0.0", "1.1.0")
