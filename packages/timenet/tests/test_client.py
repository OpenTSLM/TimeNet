import dataclasses

import numpy as np
import pytest

from timenet.client import TimeNet
from timenet.control_plane import TimeFReader, TimeFWriter
from timenet.errors import TimeFFormatError, TimeNetAccessError, TimeNetDatasetNotFoundError
from timenet.manifest import Manifest
from timenet.manifest.files import FilePart
from timenet.registry import TIMENET_REGISTRY_URL, RemoteRegistry
from timenet.testing import make_dataset
from timenet.types import Access, DatasetMetadata, License, Version


DATASET_ID = "test/bedside"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("TIMENET_HOME", "TIMENET_STORAGE", "TIMENET_CACHE", "TIMENET_REGISTRY"):
        monkeypatch.delenv(var, raising=False)


def _write(root, dataset=None):
    dataset = dataset if dataset is not None else make_dataset(n_records=2, n_values=64)
    with TimeFWriter(root, dataset.metadata) as writer:
        writer.write(dataset)
    return dataset


@pytest.fixture
def registry_root(tmp_path):
    _write(tmp_path / "reg")
    return tmp_path / "reg"


def _assert_matches_source(reader: TimeFReader, source=None) -> None:
    """Every record, task and signal of the written dataset came back out of the control plane."""
    source = source if source is not None else make_dataset(n_records=2, n_values=64)
    assert reader.record_ids() == sorted(record.id for record in source.records)
    assert reader.task_ids() == sorted(task.id for task in source.tasks)
    for record in source.records:
        view = reader.record(record.id)
        assert [signal.signal_id for signal in view.signals()] == [signal.id for signal in record.signals()]
        for signal in record.signals():
            np.testing.assert_array_equal(reader.values(signal.id), signal.values)


def test_no_args_defaults_to_the_hosted_registry(monkeypatch):
    monkeypatch.delenv("TIMENET_REGISTRY", raising=False)
    client = TimeNet()  # no registry, no storage
    assert isinstance(client._registry, RemoteRegistry)
    assert client._registry._base_url == TIMENET_REGISTRY_URL


def test_timenet_registry_env_selects_a_local_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TIMENET_REGISTRY", str(tmp_path / "home" / "registry"))
    _write(tmp_path / "home" / "registry")
    client = TimeNet()  # registry comes from $TIMENET_REGISTRY
    assert {m.dataset_id for m in client.list()} == {DATASET_ID}
    assert client.download(DATASET_ID) == tmp_path / "home" / "storage" / DATASET_ID / "1.0.0"


def test_list(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    assert {m.dataset_id for m in client.list()} == {DATASET_ID}


def test_get_returns_manifest(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    assert isinstance(client.get(DATASET_ID), Manifest)


def test_search(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    domain = make_dataset(n_records=1, n_values=8).metadata.domains[0]
    assert {m.dataset_id for m in client.search(domain=domain)} == {DATASET_ID}
    assert client.search(license=License.GPL_3_0) == []


def test_download_copies_into_storage(registry_root, tmp_path):
    storage = tmp_path / "store"
    client = TimeNet(registry_root, storage_path=storage)
    version_dir = client.download(DATASET_ID)
    assert version_dir == storage / DATASET_ID / "1.0.0"
    assert (version_dir / "manifest.json").exists()
    assert (version_dir / "control.duckdb").exists()
    assert list(version_dir.glob("time_series/part-*.parquet"))


def test_download_reports_progress(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    reported: list[int] = []
    client.download(DATASET_ID, progress_cb=reported.append)
    expected = sum(part.size for part in client.get(DATASET_ID).files.all_files())
    assert reported  # the callback fired
    assert sum(reported) == expected  # summed to the version's total byte size


def test_download_rejects_path_traversal(registry_root, tmp_path):
    # a corrupt manifest relpath must not let a download write outside the target directory
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    manifest = client.get(DATASET_ID)
    bad = dataclasses.replace(
        manifest,
        files=dataclasses.replace(manifest.files, time_series=(FilePart("../../escape.txt", "sha256:0", 0),)),
    )
    with pytest.raises(TimeFFormatError, match="escapes"):
        client._registry.download_version(DATASET_ID, "1.0.0", tmp_path / "target", manifest=bad)


def test_download_is_idempotent(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    first = client.download(DATASET_ID)
    second = client.download(DATASET_ID)
    assert first == second


def test_force_redownload_replaces_atomically(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    first = client.download(DATASET_ID)
    (first / "stale.txt").write_text("left over")  # a marker that a clean replace should remove

    again = client.download(DATASET_ID, force=True)
    assert again == first
    assert (again / "manifest.json").exists()
    assert not (again / "stale.txt").exists()  # replaced wholesale, not written in place
    assert not list(again.parent.glob("*.tmp-*"))  # staging dir cleaned up


def test_download_leaves_sibling_staging_dirs_untouched(registry_root, tmp_path):
    storage = tmp_path / "store"
    client = TimeNet(registry_root, storage_path=storage)
    # A concurrent download of the same version has a live <version>.tmp-* dir. download() must not
    # delete a sibling staging dir: sweeping siblings would corrupt that other download mid-write.
    sibling = storage / DATASET_ID / "1.0.0.tmp-deadbeef"
    sibling.mkdir(parents=True)
    (sibling / "inflight.parquet").write_text("partial")

    version_dir = client.download(DATASET_ID)
    assert sibling.exists()  # left alone, not swept
    assert (version_dir / "manifest.json").exists()  # the download still completed


def test_open_round_trips(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    with client.open(DATASET_ID) as reader:
        assert isinstance(reader, TimeFReader)
        _assert_matches_source(reader)


def test_open_reads_in_place_without_downloading(registry_root, tmp_path):
    # open() reads through the registry's open_version handle, not download-then-read, so it must not
    # write a copy into local storage the way download() does.
    storage = tmp_path / "store"
    client = TimeNet(registry_root, storage_path=storage)
    with client.open(DATASET_ID):
        pass
    assert not (storage / DATASET_ID).exists()  # nothing fetched into the download cache


def test_registry_env_var(registry_root, tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_REGISTRY", str(registry_root))
    client = TimeNet(storage_path=tmp_path / "store")
    assert {m.dataset_id for m in client.list()} == {DATASET_ID}


@pytest.fixture
def versioned_registry(tmp_path):
    """A registry holding test/bedside at both 1.0.0 and 1.1.0."""
    root = tmp_path / "vreg"
    for version in (Version(1, 0, 0), Version(1, 1, 0)):
        dataset = make_dataset(n_records=1, n_values=64)
        dataset.metadata = dataclasses.replace(dataset.metadata, dataset_version=version)
        _write(root, dataset)
    return root


def test_version_ref_pins_and_defaults_to_latest(versioned_registry, tmp_path):
    client = TimeNet(versioned_registry, storage_path=tmp_path / "store")
    assert str(client.get(DATASET_ID).metadata.dataset_version) == "1.1.0"  # no ref -> latest
    assert str(client.get(f"{DATASET_ID}@latest").metadata.dataset_version) == "1.1.0"
    assert str(client.get(f"{DATASET_ID}@1.0.0").metadata.dataset_version) == "1.0.0"  # pinned
    assert str(client.get(DATASET_ID, "1.0.0").metadata.dataset_version) == "1.0.0"  # explicit arg


def test_version_ref_download_pins(versioned_registry, tmp_path):
    client = TimeNet(versioned_registry, storage_path=tmp_path / "store")
    assert client.download(f"{DATASET_ID}@1.0.0") == tmp_path / "store" / DATASET_ID / "1.0.0"


def test_version_ref_missing_pin_raises(versioned_registry, tmp_path):
    client = TimeNet(versioned_registry, storage_path=tmp_path / "store")
    with pytest.raises(TimeNetDatasetNotFoundError):
        client.get(f"{DATASET_ID}@9.9.9")


def test_version_given_twice_raises(versioned_registry, tmp_path):
    client = TimeNet(versioned_registry, storage_path=tmp_path / "store")
    with pytest.raises(ValueError, match="twice"):
        client.get(f"{DATASET_ID}@1.0.0", "1.1.0")


def test_open_pins_the_requested_version(versioned_registry, tmp_path):
    client = TimeNet(versioned_registry, storage_path=tmp_path / "store")
    with client.open(f"{DATASET_ID}@1.0.0") as reader:
        _assert_matches_source(reader, make_dataset(n_records=1, n_values=64))


def test_open_rejects_a_credentialed_dataset_from_a_hosted_registry(tmp_path):
    client = TimeNet(tmp_path / "registry", storage_path=tmp_path / "store")
    meta = DatasetMetadata(
        dataset_id="org/gated",
        dataset_version=Version(1, 0, 0),
        name="Gated",
        description="A credentialed dataset.",
        license=License.CC_BY_4_0,
        access=Access.CREDENTIALED,
        access_url="https://physionet.example/dua",
    )

    class _Hosted:  # not a LocalRegistry, so the build-your-own gate applies
        def get_manifest(self, dataset_id, version=None):
            return type("_M", (), {"metadata": meta})()

    client._registry = _Hosted()  # ty: ignore[invalid-assignment]
    with pytest.raises(TimeNetAccessError, match="build it locally"):
        client.open("org/gated")
    with pytest.raises(TimeNetAccessError, match="Get access at"):
        client.download("org/gated")
