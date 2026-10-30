from pathlib import Path
import shutil
from types import SimpleNamespace

import httpx
from pydantic import ValidationError
import pytest

from timenet import builders as builders_module, client as client_module
from timenet.client import TimeNet
from timenet.connectors import BaseConnector
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.engine import open_parents, run_pipeline
from timenet.errors import (
    TimeFFormatError,
    TimeFValidationError,
    TimeNetAccessError,
    TimeNetDatasetNotFoundError,
    TimeNetFileError,
    TimeNetMissingParentError,
)
from timenet.format.stored_values import StoredText
from timenet.manifest import Manifest
from timenet.manifest.files import FilePart
from timenet.reader import Loading, LoadPlan, TimeFReader
from timenet.registry import TIMENET_REGISTRY_URL, LocalRegistry, RemoteRegistry, open_local_version
from timenet.testing import assert_datasets_equal, make_dataset
from timenet.types import Access, DatasetMetadata, Domain, License, Version
from timenet.types.access import AccessRequirement
from timenet.types.parent_dataset_card import ParentDatasetCard
from timenet.types.reference_keys import ParentAlias, SourceKey
from timenet.types.references import DatasetRef
from timenet.writer import TimeFWriter


def _child_of_the_demo(root: Path) -> tuple[LocalRegistry, DatasetRef]:
    """Store the demo dataset and ``test/child``, which reads one of its Records, in a local registry.

    Returns:
        The registry, and the demo release, which is the payload source of the child.
    """
    parent = make_dataset()
    registry = LocalRegistry(root)
    registry.store(parent)
    reference = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
    child = TimeFDataset(
        metadata=parent.metadata.model_copy(
            update={"dataset_id": "test/child", "parents": {ParentAlias("parent"): reference}}
        )
    )
    with TimeNet(registry).open(parent.metadata.dataset_id) as reader:
        child.import_from(reader, alias=ParentAlias("parent"), records=[parent.records[0].id])
    registry.store(child)
    return registry, reference


class _Builder:
    """A builder backend whose cards declare ``VERSION``.

    A build stores the demo dataset under the release it builds, in ``BACKEND``. It hands its pins to
    the writer, which publishes the release only if it reproduces the pinned files.
    """

    VERSION = "1.0.0"
    BACKEND = "parquet"

    def __init__(self, calls):
        self.calls = calls

    def knows(self, dataset_id):
        return True

    def declared_version(self, dataset_id):
        return self.VERSION

    def card(self, dataset_id):
        version = DatasetRef.parse(f"{dataset_id}@{self.VERSION}").version
        return make_dataset().metadata.model_copy(update={"dataset_id": dataset_id, "dataset_version": version})

    def build(self, dataset_id, root, *, force=False, build_parents=False, fetch_from=None, pinned_by=None):  # noqa: PLR0913
        self.calls.append((dataset_id, force, build_parents, fetch_from, pinned_by))
        dataset = make_dataset()
        dataset._metadata = self.card(dataset_id)
        LocalRegistry(root).store(dataset, force=force, values_backend=self.BACKEND, pinned_by=pinned_by)
        return root / dataset_id / self.VERSION


def test_a_local_child_copies_its_missing_payload_source_from_fetch_from(tmp_path):
    origin, reference = _child_of_the_demo(tmp_path / "origin")
    cache = tmp_path / "cache"
    shutil.copytree(origin.root / "test/child/1.0.0", cache / "test/child/1.0.0")
    client = TimeNet(cache)
    with pytest.raises(TimeNetMissingParentError) as caught:
        client.open("test/child", auto_build=False)
    assert (caught.value.missing, caught.value.needed_by) == (reference, DatasetRef.parse("test/child@1.0.0"))
    assert str(caught.value).endswith(
        ". Copy it with fetch_from=<registry> or build it from its connector with auto_build=True."
    )
    with client.open("test/child", fetch_from=str(origin.root)) as reader:
        assert (cache / reference.dataset_id / "1.0.0/control.duckdb").is_file()
        shutil.rmtree(origin.root)
        assert reader.read().records[0].signals[0].to_arrow().equals(make_dataset().records[0].signals[0].to_arrow())
    with client.open("test/child") as reader:
        assert reader._control is None
        assert reader.read().records


@pytest.mark.parametrize("in_fetch_from", [True, False])
def test_a_missing_version_is_copied_from_fetch_from_before_it_is_built(tmp_path, monkeypatch, in_fetch_from):
    origin = LocalRegistry(tmp_path / "origin")
    if in_fetch_from:
        _child_of_the_demo(origin.root)
    calls = []
    monkeypatch.setattr(client_module, "find_builder", lambda dataset_id: _Builder(calls))

    with TimeNet(tmp_path / "local").open("test/child", fetch_from=origin) as reader:
        assert reader.read().records

    if in_fetch_from:
        assert calls == []
        assert (tmp_path / "local/test/child/1.0.0/manifest.json").is_file()
        return
    # The build gets the same registry to copy its own missing parents from.
    assert calls == [("test/child", False, True, origin, None)]


class _Resampled(BaseConnector[str]):
    """Writes the first demo Signal at half its rate, from its declared parent, into a Record of its own."""

    def metadata(self):
        reference = DatasetRef(
            dataset_id=make_dataset().metadata.dataset_id, version=make_dataset().metadata.dataset_version
        )
        return make_dataset().metadata.model_copy(
            update={"dataset_id": "test/resampled", "parents": {ParentAlias("parent"): reference}}
        )

    def download(self, cache_dir):
        return []

    def compose(self, raw_refs, context):
        original = context.parents[ParentAlias("parent")].get(Signal, make_dataset().records[0].signals[0].id)
        signal = Signal.from_values(
            original.to_numpy()[::2], spec=original.spec, name=original.name, time_axis=RegularAxis.from_rate_hz(8)
        )
        child = TimeFDataset(metadata=self.metadata())
        child.add_record(record=Record(sources=(Source(name="Monitor", signals=(signal,)),)))
        return child


def test_resampled_child_reads_and_downloads_without_its_build_only_parent(tmp_path):
    origin = LocalRegistry(tmp_path / "origin")
    origin.store(make_dataset())
    metadata = _Resampled().metadata()
    reference = metadata.parents[ParentAlias("parent")]
    run_pipeline(_Resampled(), origin.root, cache_dir=tmp_path / "build-cache")
    manifest = origin.get_manifest(metadata.dataset_id)
    assert manifest.parent_cards[ParentAlias("parent")].source == reference
    assert manifest.payload_sources == {}
    expected = make_dataset().records[0].signals[0].to_arrow()[::2]
    shutil.rmtree(origin.root / reference.dataset_id)

    # Reading the compiled result needs no original samples, even when fetching is enabled.
    loaded = TimeNet(origin).load(metadata.dataset_id, fetch_from=str(origin.root))
    assert loaded.records[0].signals[0].to_arrow().equals(expected)
    # Running the connector again still requires its declared parent.
    with pytest.raises(TimeNetDatasetNotFoundError, match=reference.dataset_id):
        open_parents(metadata, origin)

    cache = tmp_path / "download"
    directory = TimeNet(origin, storage_path=cache).download(metadata.dataset_id)
    assert not (cache / reference.dataset_id).exists()
    shutil.rmtree(origin.root)
    version = open_local_version(directory)
    assert version.manifest.parent_cards == manifest.parent_cards
    with TimeFReader(version) as reader:
        reader.verify()
        assert reader.read().records[0].signals[0].to_arrow().equals(expected)


@pytest.mark.parametrize("case", ["exact", "differs", "other-version"])
def test_a_missing_payload_source_builds_with_its_own_connector_against_the_pins(tmp_path, monkeypatch, case):
    registry, reference = _child_of_the_demo(tmp_path / "registry")
    pins = registry.get_manifest("test/child").payload_sources[SourceKey(str(reference))]
    shutil.rmtree(registry.root / reference.dataset_id)
    calls = []
    monkeypatch.setattr(_Builder, "VERSION", "2.0.0" if case == "other-version" else "1.0.0")
    monkeypatch.setattr(_Builder, "BACKEND", "zarr" if case == "differs" else "parquet")
    monkeypatch.setattr(builders_module, "find_builder", lambda dataset_id: _Builder(calls))

    with pytest.raises(TimeNetMissingParentError, match=r"auto_build=True\.$"):
        TimeNet(registry).open("test/child", auto_build=False)
    assert calls == []
    if case == "other-version":
        # No installed connector declares the exact version of the source that the child pins.
        with pytest.raises(TimeNetMissingParentError, match=r"\. Copy it with fetch_from=<registry>\.$"):
            TimeNet(registry).open("test/child")
        assert calls == []
        return
    if case == "differs":
        pytest.importorskip("zarr")
        # A Zarr build writes other files than the Parquet files that the child pins.
        with pytest.raises(TimeFFormatError, match=r"Nothing was published"):
            TimeNet(registry).open("test/child")
        assert not registry.exists(reference.dataset_id, str(reference.version))
        fetch_from = None
    else:
        # A registry to copy from that lacks the source: the build gets it for the parents of the source.
        fetch_from = LocalRegistry(tmp_path / "empty")
        with TimeNet(registry).open("test/child", fetch_from=fetch_from) as reader:
            signal = reader.read().records[0].signals[0]
            assert signal.to_arrow().equals(make_dataset().records[0].signals[0].to_arrow())
    # The source builds with its own connector, which prepares its parents in the same order, against
    # the pins of the child that reads it.
    assert calls == [(reference.dataset_id, False, True, fetch_from, {DatasetRef.parse("test/child@1.0.0"): pins})]


@pytest.mark.parametrize("damaged", ["requested", "payload-source"])
def test_a_damaged_release_is_repaired_from_fetch_from_and_never_built_over(tmp_path, monkeypatch, damaged):
    registry, reference = _child_of_the_demo(tmp_path / "registry")
    origin = tmp_path / "origin"
    shutil.copytree(registry.root, origin)
    release = DatasetRef.parse("test/child@1.0.0") if damaged == "requested" else reference
    control = registry.root / release.dataset_id / str(release.version) / "control.duckdb"
    size = control.stat().st_size
    control.write_bytes(b"cut short")
    calls = []
    monkeypatch.setattr(builders_module, "find_builder", lambda dataset_id: _Builder(calls))
    monkeypatch.setattr(client_module, "find_builder", lambda dataset_id: _Builder(calls))

    with pytest.raises(TimeNetFileError) as caught:
        TimeNet(registry).open("test/child")
    assert str(caught.value) == (
        f"{release}: file 'control.duckdb' has 9 bytes, expected {size:,} bytes. Repair it with fetch_from=<registry>."
    )
    with TimeNet(registry).open("test/child", fetch_from=str(origin)) as reader:
        assert reader.read().records
    assert control.stat().st_size == size
    assert calls == []


def test_a_corrupted_copy_from_fetch_from_names_the_transfer_without_the_repair_hint(tmp_path):
    origin, _ = _child_of_the_demo(tmp_path / "origin")
    control = origin.root / "test/child/1.0.0/control.duckdb"
    data = control.read_bytes()
    control.write_bytes(bytes([data[0] ^ 1]) + data[1:])

    with pytest.raises(TimeNetFileError) as caught:
        TimeNet(tmp_path / "local").open("test/child", auto_build=False, fetch_from=origin)

    # The registry to copy from served the file, so a hint to copy from a registry would not help.
    assert caught.value.owner == f"download of test/child@1.0.0 from {origin}"
    assert caught.value.hint is None


def test_a_remote_client_takes_no_fetch_from_and_never_builds(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(client_module, "find_builder", lambda dataset_id: _Builder(calls))
    remote = RemoteRegistry(
        "http://api.local", transport=httpx.MockTransport(lambda request: httpx.Response(404)), cache_dir=tmp_path
    )

    with pytest.raises(TimeFValidationError, match="fetch_from copies releases into a local registry"):
        TimeNet(remote).open("test/child", fetch_from=str(tmp_path))
    with pytest.raises(TimeNetDatasetNotFoundError):
        TimeNet(remote).open("test/child")
    assert calls == []


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


def test_open_returns_a_lazy_reader(registry_root, tmp_path):
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    with client.open("timenet/hello-world") as reader:
        assert isinstance(reader, TimeFReader)
        assert reader._control is None  # the catalog opens on the first read
        dataset = reader.read()
        signals = [signal for record in dataset.records for signal in record.signals]
        assert signals
        assert not any(signal._values.is_loaded for signal in signals)
        assert_datasets_equal(make_dataset(), dataset)


def test_load_round_trips_with_every_value_loaded_and_its_reader_closed(tmp_path, monkeypatch):
    expected = make_dataset()
    # Over the inline limit, so the prompt is packed in a payload file.
    expected.tasks[0].prompt = "A detailed clinical report. " * 300
    spec = expected.records[0].signals[0].spec
    irregular = Signal.from_irregular([1.0, 2.0, 3.0], time_offsets_us=[0, 2, 5], spec=spec, name="II")
    expected.add_record(record=Record(sources=(Source(name="env", signals=(irregular,)),)))
    expected.derive_schema()
    with TimeFWriter(tmp_path / "reg", expected) as writer:
        writer.write()
    closed: list[TimeFReader] = []
    close = TimeFReader.close

    def counting_close(reader):
        closed.append(reader)
        close(reader)

    monkeypatch.setattr(TimeFReader, "close", counting_close)
    restored = TimeNet(tmp_path / "reg", storage_path=tmp_path / "store").load("timenet/hello-world")
    assert isinstance(restored, TimeFDataset)
    assert len(closed) == 1
    signals = [signal for record in restored.records for signal in record.signals]
    assert signals
    assert all(signal._values.is_loaded for signal in signals)
    (offsets,) = [signal._time_offsets for signal in signals if signal.id == irregular.id]
    assert offsets is not None
    assert offsets.is_loaded
    prompt = next(task.prompt for task in restored.tasks if task.id == expected.tasks[0].id)
    assert isinstance(prompt, StoredText)
    assert prompt.is_loaded
    # Every value is in memory, so the dataset reads without its release.
    shutil.rmtree(tmp_path / "reg")
    assert_datasets_equal(expected, restored)


def test_load_reads_in_place_without_downloading(registry_root, tmp_path):
    # load() reads through the registry's open_version handle, not download-then-read, so it must not
    # write a copy into local storage the way download() does.
    storage = tmp_path / "store"
    client = TimeNet(registry_root, storage_path=storage)
    client.load("timenet/hello-world")
    assert not (storage / "timenet/hello-world").exists()  # nothing fetched into the download cache


def test_open_torch(registry_root, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from torch.utils.data import Dataset  # noqa: PLC0415

    def read_everything(*args, **kwargs):
        raise AssertionError("open_torch reads one batch at a time, not the whole dataset")

    closed: list[TimeFReader] = []
    close = TimeFReader.close

    def counting_close(reader):
        closed.append(reader)
        close(reader)

    monkeypatch.setattr(TimeFReader, "read", read_everything)
    monkeypatch.setattr(TimeFReader, "close", counting_close)
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    with client.open_torch("timenet/hello-world") as ds:
        assert isinstance(ds, Dataset)
        assert len(ds) == len(make_dataset().records)
        assert isinstance(ds[0]["series"][0], torch.Tensor)
        assert closed == []
    # The view closes the reader it holds at the end of the block.
    assert len(closed) == 1


@pytest.mark.parametrize("arrays_loaded", [True, False], ids=["default-plan", "text-only-plan"])
def test_open_torch_passes_its_load_plan_to_the_view(registry_root, tmp_path, arrays_loaded):
    pytest.importorskip("torch")
    client = TimeNet(registry_root, storage_path=tmp_path / "store")
    view = (
        client.open_torch("timenet/hello-world")
        if arrays_loaded
        else client.open_torch("timenet/hello-world", load=LoadPlan(text=Loading.EAGER))
    )
    with view:
        # The item's tasks reference the item's Record, which the batch read built with the plan.
        record = view[0]["tasks"][0].inputs[0]
        assert record.id == make_dataset().records[0].id
        assert {signal._values.is_loaded for signal in record.signals} == {arrays_loaded}


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


def test_open_and_load_read_the_version_asked_for(versioned_registry, tmp_path):
    client = TimeNet(versioned_registry, storage_path=tmp_path / "store")
    with client.open("timenet/hello-world", version="1.0.0") as reader:
        assert str(reader.metadata.dataset_version) == "1.0.0"
    assert str(client.load("timenet/hello-world", version="1.0.0").metadata.dataset_version) == "1.0.0"


def test_open_torch_reads_the_version_asked_for(versioned_registry, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    opened: list[TimeFReader] = []
    original_open = TimeNet.open

    def recording_open(client, *args, **kwargs):
        opened.append(original_open(client, *args, **kwargs))
        return opened[-1]

    monkeypatch.setattr(TimeNet, "open", recording_open)
    TimeNet(versioned_registry, storage_path=tmp_path / "store").open_torch("timenet/hello-world", version="1.0.0")
    (reader,) = opened
    assert str(reader.metadata.dataset_version) == "1.0.0"


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

        def build(self, dataset_id, root, *, force=False, build_parents=False, fetch_from=None):
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

        def build(self, dataset_id, root, *, force=False, build_parents=False, fetch_from=None):
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


@pytest.mark.parametrize("entry_point", ["open", "load", "open_torch"])
def test_load_does_not_build_when_auto_build_is_false(registry_root, tmp_path, monkeypatch, entry_point):
    # A caller can opt out of the build-on-miss, so a notebook load fails fast instead of starting a
    # multi-GB download and a build. The builder declares the requested version, so only
    # auto_build=False keeps it from running.
    class _Builder:
        built = False

        def knows(self, dataset_id):
            return True

        def declared_version(self, dataset_id):
            return "9.9.9"

        def build(self, dataset_id, root, *, force=False, build_parents=False, fetch_from=None):
            self.built = True
            return Path(root) / dataset_id / "1.0.0"

    if entry_point == "open_torch":
        pytest.importorskip("torch")
    builder = _Builder()
    monkeypatch.setattr(client_module, "find_builder", lambda dataset_id: builder)
    client = TimeNet(registry_root, storage_path=tmp_path / "store")

    # The hint names the keywords that could obtain the release, in a sentence of its own.
    hint = r"\. Copy it with fetch_from=<registry> or build it from its connector with auto_build=True\.$"
    with pytest.raises(TimeNetDatasetNotFoundError, match=hint):
        getattr(client, entry_point)("timenet/hello-world@9.9.9", auto_build=False)
    assert not builder.built  # auto_build=False skips the connector entirely


def _card(dataset_id, access, access_url=None):
    return DatasetMetadata(
        dataset_id=dataset_id,
        dataset_version=Version(1, 0, 0),
        name="Gated",
        description="A credentialed dataset.",
        license=License.CC_BY_4_0,
        access=access,
        access_url=access_url,
    )


def _hosted(client, metadata, lineage=()):
    """Serve one manifest from a registry that is not local, so the build-your-own gate applies."""

    class _Hosted:
        def get_manifest(self, dataset_id, version=None):
            return SimpleNamespace(metadata=metadata, lineage=lineage)

    client._registry = _Hosted()


def test_load_rejects_a_credentialed_dataset_from_a_hosted_registry(tmp_path):
    client = TimeNet(tmp_path / "registry", storage_path=tmp_path / "store")
    _hosted(client, _card("org/gated", Access.CREDENTIALED, "https://physionet.example/dua"))
    message = (
        "org/gated@1.0.0 is credentialed. Get access at https://physionet.example/dua for org/gated@1.0.0. "
        "TimeNet does not host its data, so build it locally with `timenet-build build org/gated`."
    )
    for entry_point in (client.open, client.load, client.download):
        with pytest.raises(TimeNetAccessError) as caught:
            entry_point("org/gated")
        assert str(caught.value) == message


def test_the_hint_for_a_hosted_child_names_the_access_page_of_each_ancestor(tmp_path):
    client = TimeNet(tmp_path / "registry", storage_path=tmp_path / "store")
    # The card of the child stays open and has no access page. Its parents set the access.
    parents = (
        _card("org/a", Access.CREDENTIALED, "https://a.example"),
        _card("org/b", Access.RESTRICTED, "https://b.example"),
    )
    _hosted(client, _card("org/child", Access.OPEN), tuple(ParentDatasetCard(metadata=card) for card in parents))

    with pytest.raises(TimeNetAccessError) as caught:
        client.open("org/child")

    assert caught.value.requirements == (
        AccessRequirement(DatasetRef.parse("org/a@1.0.0"), Access.CREDENTIALED, "https://a.example"),
        AccessRequirement(DatasetRef.parse("org/b@1.0.0"), Access.RESTRICTED, "https://b.example"),
    )
    assert str(caught.value) == (
        "org/child@1.0.0 is restricted. "
        "Get access at https://a.example for org/a@1.0.0 and at https://b.example for org/b@1.0.0. "
        "TimeNet does not host its data, so build it locally with `timenet-build build org/child`."
    )
