import dataclasses
from pathlib import Path

import pytest

from timenet import client as client_module
from timenet.client import TimeNet
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError, TimeNetDatasetNotFoundError
from timenet.manifest import Manifest
from timenet.manifest.files import FilePart
from timenet.registry import TIMENET_REGISTRY_URL, RemoteRegistry
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


def test_no_args_defaults_to_the_hosted_registry(monkeypatch):
    monkeypatch.delenv("TIMENET_REGISTRY", raising=False)
    client = TimeNet()  # no registry, no storage
    assert isinstance(client._registry, RemoteRegistry)
    assert client._registry._base_url == TIMENET_REGISTRY_URL


def test_timenet_registry_env_selects_a_local_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TIMENET_REGISTRY", str(tmp_path / "home" / "registry"))
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "home" / "registry", dataset) as writer:
        writer.write()
    client = TimeNet()  # registry comes from $TIMENET_REGISTRY
    assert {m.dataset_id for m in client.list()} == {"timenet/hello-world"}
    assert client.download("timenet/hello-world") == tmp_path / "home" / "storage" / "timenet/hello-world" / "1.0.0"


def test_list(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    assert {m.dataset_id for m in client.list()} == {"timenet/hello-world"}


def test_get_returns_manifest(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    assert isinstance(client.get("timenet/hello-world"), Manifest)


def test_search(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    assert {m.dataset_id for m in client.search(domain=Domain.GENERAL)} == {"timenet/hello-world"}
    assert client.search(domain=Domain.CARDIOLOGY) == []


def test_download_copies_into_storage(registry_root, tmp_path):
    storage = tmp_path / "store"
    client = TimeNet(registry_root, storage_path=storage)
    version_dir = client.download("timenet/hello-world")
    assert version_dir == storage / "timenet/hello-world" / "1.0.0"
    assert (version_dir / "manifest.json").exists()
    assert list(version_dir.glob("time_series/part-*.parquet"))


def test_download_reports_progress(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    reported: list[int] = []
    client.download("timenet/hello-world", progress_cb=reported.append)
    expected = sum(part.size for part in client.get("timenet/hello-world").files.all_files())
    assert reported  # the callback fired
    assert sum(reported) == expected  # summed to the version's total byte size


def test_download_rejects_path_traversal(registry_root, tmp_path):
    # a corrupt manifest relpath must not let a download write outside the target directory
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    manifest = client.get("timenet/hello-world")
    bad = dataclasses.replace(
        manifest,
        files=dataclasses.replace(manifest.files, samples=(FilePart("../../escape.txt", "sha256:0", 0),)),
    )
    with pytest.raises(TimeFFormatError, match="escapes"):
        client._registry.download_version("timenet/hello-world", "1.0.0", tmp_path / "target", manifest=bad)


def test_download_is_idempotent(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    first = client.download("timenet/hello-world")
    second = client.download("timenet/hello-world")
    assert first == second


def test_force_redownload_replaces_atomically(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    first = client.download("timenet/hello-world")
    (first / "stale.txt").write_text("left over")  # a marker that a clean replace should remove

    again = client.download("timenet/hello-world", force=True)
    assert again == first
    assert (again / "manifest.json").exists()
    assert not (again / "stale.txt").exists()  # replaced wholesale, not written in place
    assert not list(again.parent.glob("*.tmp-*"))  # staging dir cleaned up


def test_download_leaves_sibling_staging_dirs_untouched(registry_root, tmp_path):
    storage = tmp_path / "store"
    client = TimeNet(registry_root, storage_path=storage)
    # A concurrent download of the same version has a live <version>.tmp-* dir. download() must not
    # delete a sibling staging dir: sweeping siblings would corrupt that other download mid-write.
    sibling = storage / "timenet/hello-world" / "1.0.0.tmp-deadbeef"
    sibling.mkdir(parents=True)
    (sibling / "inflight.parquet").write_text("partial")

    version_dir = client.download("timenet/hello-world")
    assert sibling.exists()  # left alone, not swept
    assert (version_dir / "manifest.json").exists()  # the download still completed


def test_load_round_trips(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    restored = client.load("timenet/hello-world")
    assert isinstance(restored, TimeFDataset)
    assert_datasets_equal(make_dataset(), restored)


def test_load_reads_in_place_without_downloading(registry_root, tmp_path):
    # load() reads through the registry's open_version handle, not download-then-read, so it must not
    # write a copy into local storage the way download() does.
    storage = tmp_path / "store"
    client = TimeNet(registry_root, storage_path=storage)
    client.load("timenet/hello-world")
    assert not (storage / "timenet/hello-world").exists()  # nothing fetched into the download cache


def test_load_torch(registry_root, tmp_path):
    torch = pytest.importorskip("torch")
    from torch.utils.data import Dataset  # noqa: PLC0415

    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    ds = client.load_torch("timenet/hello-world")
    assert isinstance(ds, Dataset)
    assert len(ds) == len(make_dataset().samples)
    assert isinstance(ds[0]["series"][0], torch.Tensor)


def test_registry_env_var(registry_root, tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_REGISTRY", str(registry_root))
    client = TimeNet(storage_path=tmp_path / "store")
    assert {m.dataset_id for m in client.list()} == {"timenet/hello-world"}


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
    assert str(client.get("timenet/hello-world").metadata.dataset_version) == "1.1.0"  # no ref -> latest
    assert str(client.get("timenet/hello-world@latest").metadata.dataset_version) == "1.1.0"
    assert str(client.get("timenet/hello-world@1.0.0").metadata.dataset_version) == "1.0.0"  # pinned
    assert str(client.get("timenet/hello-world", "1.0.0").metadata.dataset_version) == "1.0.0"  # explicit arg


def test_version_ref_download_pins(versioned_registry, tmp_path):
    client = TimeNet(versioned_registry, storage_path=tmp_path / "store")
    assert client.download("timenet/hello-world@1.0.0") == tmp_path / "store" / "timenet/hello-world" / "1.0.0"


def test_version_ref_missing_pin_raises(versioned_registry, tmp_path):
    client = TimeNet(versioned_registry, storage_path=tmp_path / "store")
    with pytest.raises(TimeNetDatasetNotFoundError):
        client.get("timenet/hello-world@9.9.9")


def test_version_given_twice_raises(versioned_registry, tmp_path):
    client = TimeNet(versioned_registry, storage_path=tmp_path / "store")
    with pytest.raises(ValueError, match="twice"):
        client.get("timenet/hello-world@1.0.0", "1.1.0")


def test_load_names_the_buildable_version_when_a_pin_cannot_be_built(registry_root, tmp_path, monkeypatch):
    # The builder declares only one version, so a pin it cannot satisfy must say which version it
    # builds, and must fail before the expensive build rather than after it.
    class _Builder:
        built = False

        def knows(self, dataset_id):
            return True

        def declared_version(self, dataset_id):
            return "1.0.0"

        def build(self, dataset_id, root, *, force=False):
            self.built = True
            return Path(root) / dataset_id / "1.0.0"

    builder = _Builder()
    monkeypatch.setattr(client_module, "find_builder", lambda dataset_id: builder)
    client = TimeNet(registry_root, storage_path=tmp_path / "store")

    with pytest.raises(TimeNetDatasetNotFoundError, match=r"builds version 1\.0\.0, not the requested 9\.9\.9"):
        client.load("timenet/hello-world@9.9.9")
    assert not builder.built  # the pin was rejected before the build ran


@pytest.mark.parametrize("pin", [None, "latest", ""])
def test_load_accepts_the_latest_sentinels_after_a_build(tmp_path, monkeypatch, pin):
    # Every registry reads "latest" and "" as the latest version, so they are not pins the build
    # can miss: the version the builder just committed satisfies them.
    class _Builder:
        def knows(self, dataset_id):
            return True

        def declared_version(self, dataset_id):
            return "1.0.0"

        def build(self, dataset_id, root, *, force=False):
            dataset = make_dataset()
            dataset.derive_schema()
            with TimeFWriter(Path(root), dataset) as writer:
                writer.write()
            return Path(root) / dataset_id / "1.0.0"

    empty_root = tmp_path / "reg"
    empty_root.mkdir()
    monkeypatch.setattr(client_module, "find_builder", lambda dataset_id: _Builder())
    client = TimeNet(empty_root, storage_path=tmp_path / "store")

    assert str(client.load("timenet/hello-world", pin).metadata.dataset_version) == "1.0.0"


def test_load_does_not_build_when_auto_build_is_false(registry_root, tmp_path, monkeypatch):
    # A caller can opt out of the build-on-miss, so a notebook load fails fast instead of starting a
    # multi-GB download and a build.
    class _Builder:
        built = False

        def knows(self, dataset_id):
            return True

        def declared_version(self, dataset_id):
            return "1.0.0"

        def build(self, dataset_id, root, *, force=False):
            self.built = True
            return Path(root) / dataset_id / "1.0.0"

    builder = _Builder()
    monkeypatch.setattr(client_module, "find_builder", lambda dataset_id: builder)
    client = TimeNet(registry_root, storage_path=tmp_path / "store")

    with pytest.raises(TimeNetDatasetNotFoundError):
        client.load("timenet/hello-world@9.9.9", auto_build=False)
    assert not builder.built  # auto_build=False skips the connector entirely
