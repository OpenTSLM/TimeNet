from pathlib import Path
import shutil

from pydantic import ValidationError
import pytest

from timenet import builders as builders_module, client as client_module
from timenet.client import TimeNet
from timenet.connectors import BuildContext
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError, TimeNetAccessError, TimeNetDatasetNotFoundError
from timenet.manifest import Manifest
from timenet.manifest.files import FilePart
from timenet.reader import DatasetHandle, TimeFReader
from timenet.registry import TIMENET_REGISTRY_URL, LocalRegistry, RemoteRegistry, open_local_version
from timenet.testing import assert_datasets_equal, make_dataset
from timenet.types import Access, DatasetMetadata, Domain, License, Version
from timenet.types.object_kind import ObjectKind
from timenet.types.reference_keys import InputAlias
from timenet.types.references import DatasetRef
from timenet.writer import TimeFWriter


def test_local_child_fetches_complete_parents_before_returning_a_handle(tmp_path):
    parent = make_dataset()
    origin = LocalRegistry(tmp_path / "origin")
    origin.store(parent)
    child = TimeFDataset(metadata=parent.metadata.model_copy(update={"dataset_id": "test/child"}))
    with TimeNet(origin).get_handle(parent.metadata.dataset_id) as handle:
        child.import_from(handle, alias=InputAlias("parent"), records=[parent.records[0].id])
    origin.store(child)
    cache = tmp_path / "cache"
    shutil.copytree(origin.root / "test/child/1.0.0", cache / "test/child/1.0.0")
    client = TimeNet(cache)
    with pytest.raises(TimeNetDatasetNotFoundError, match=parent.metadata.dataset_id):
        client.get_handle("test/child", auto_build=False)
    with client.get_handle("test/child", fetch_parents=str(origin.root)) as handle:
        assert (cache / parent.metadata.dataset_id / "1.0.0/control.duckdb").is_file()
        shutil.rmtree(origin.root)
        assert handle.read().records[0].signals[0].to_arrow().equals(parent.records[0].signals[0].to_arrow())
    with client.load("test/child") as handle:
        assert handle.reader._control is None
        assert handle.read().records


def test_resampled_child_reads_and_downloads_without_its_build_input(tmp_path):
    parent = make_dataset()
    origin = LocalRegistry(tmp_path / "origin")
    origin.store(parent)
    reference = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
    metadata = parent.metadata.model_copy(
        update={"dataset_id": "test/resampled", "inputs": {InputAlias("parent"): reference}}
    )
    with BuildContext.open(metadata, origin) as context:
        original = (
            context.inputs[InputAlias("parent")].get_handle(ObjectKind.SIGNAL, parent.records[0].signals[0].id).read()
        )
        signal = Signal.from_values(
            original.to_numpy()[::2], spec=original.spec, name=original.name, time_axis=RegularAxis.from_rate_hz(8)
        )
        child = TimeFDataset(metadata=metadata)
        child.add_record(record=Record(record_id="resampled", sources=(Source(name="Monitor", signals=(signal,)),)))
        child.import_from(context.inputs[InputAlias("parent")])
        context.check_imports(child)
        origin.store(child)
    manifest = origin.get_manifest(metadata.dataset_id)
    assert manifest.input_locks[InputAlias("parent")].source == reference
    assert manifest.payload_sources == {}
    shutil.rmtree(origin.root / reference.dataset_id)

    # Reading the compiled result needs no original samples, even when fetching/building is enabled.
    with TimeNet(origin).load(metadata.dataset_id, fetch_parents=str(origin.root), build_parents=True) as handle:
        assert handle.read().records[0].signals[0].to_arrow().equals(signal.to_arrow())
    # Running the connector again still requires its declared input.
    with pytest.raises(TimeNetDatasetNotFoundError, match=reference.dataset_id):
        BuildContext.open(metadata, origin)

    cache = tmp_path / "download"
    directory = TimeNet(origin, storage_path=cache).download(metadata.dataset_id)
    assert not (cache / reference.dataset_id).exists()
    shutil.rmtree(origin.root)
    version = open_local_version(directory)
    assert version.manifest.input_locks == manifest.input_locks
    with TimeFReader(version) as reader:
        reader.verify()
        assert reader.get_handle().read().records[0].signals[0].to_arrow().equals(signal.to_arrow())


@pytest.mark.parametrize("declared_version", ["1.0.0", "2.0.0"])
def test_local_child_builds_only_the_exact_missing_physical_source(tmp_path, monkeypatch, declared_version):
    parent = make_dataset()
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(parent)
    child = TimeFDataset(metadata=parent.metadata.model_copy(update={"dataset_id": "test/child"}))
    with TimeNet(registry).get_handle(parent.metadata.dataset_id) as handle:
        child.import_from(handle, alias=InputAlias("parent"), records=[parent.records[0].id])
    registry.store(child)
    shutil.rmtree(registry.root / parent.metadata.dataset_id)
    calls = []

    class Builder:
        def declared_version(self, dataset_id):
            return declared_version

        def build(self, dataset_id, root, *, force=False, build_parents=False, fetch_parents=None):
            calls.append((dataset_id, force, build_parents, fetch_parents))
            LocalRegistry(root).store(make_dataset(), force=force)
            return root / dataset_id / declared_version

    monkeypatch.setattr(builders_module, "find_builder", lambda dataset_id: Builder())
    if declared_version != "1.0.0":
        with pytest.raises(TimeFValidationError, match="exact input"):
            TimeNet(registry).get_handle("test/child", auto_build=False, build_parents=True)
        assert calls == []
        return
    with TimeNet(registry).get_handle("test/child", auto_build=False, build_parents=True) as handle:
        assert handle.read().records[0].signals[0].to_arrow().equals(parent.records[0].signals[0].to_arrow())
    assert calls == [(parent.metadata.dataset_id, False, True, None)]


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


def test_manifest_file_rejects_path_traversal():
    with pytest.raises(ValidationError, match="stay within the dataset root"):
        FilePart(path="../../escape.txt", checksum="sha256:" + "0" * 64, size=0)


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
    with client.load("timenet/hello-world") as handle:
        assert isinstance(handle, DatasetHandle)
        assert handle.reader._control is None
        restored = handle.read()
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
    assert len(ds) == len(make_dataset().records)
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
        dataset._metadata = dataset.metadata.model_copy(update={"dataset_version": version})
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

        def build(self, dataset_id, root, *, force=False, build_parents=False, fetch_parents=None):
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

        def build(self, dataset_id, root, *, force=False, build_parents=False, fetch_parents=None):
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

        def build(self, dataset_id, root, *, force=False, build_parents=False, fetch_parents=None):
            self.built = True
            return Path(root) / dataset_id / "1.0.0"

    builder = _Builder()
    monkeypatch.setattr(client_module, "find_builder", lambda dataset_id: builder)
    client = TimeNet(registry_root, storage_path=tmp_path / "store")

    with pytest.raises(TimeNetDatasetNotFoundError):
        client.load("timenet/hello-world@9.9.9", auto_build=False)
    assert not builder.built  # auto_build=False skips the connector entirely


def test_load_rejects_a_credentialed_dataset_from_a_hosted_registry(tmp_path):
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
        client.load("org/gated")
    with pytest.raises(TimeNetAccessError, match="Get access at"):
        client.download("org/gated")
