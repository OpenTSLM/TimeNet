import shutil
from types import SimpleNamespace

from _demo_child import _child_of_the_demo
from _fake_registry import build_service_fake
import httpx
from pydantic import ValidationError
import pytest

from timenet import builders as builders_module
from timenet.client import TimeNet, default_sources
from timenet.connectors import BaseConnector
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.engine import open_parents, run_pipeline
from timenet.errors import (
    TimeFValidationError,
    TimeNetAccessError,
    TimeNetDatasetNotFoundError,
    TimeNetMissingParentError,
    TimeNetRegistryError,
    TimeNetVersionUnavailableError,
)
from timenet.format.stored_values import StoredText
from timenet.manifest.files import FilePart
from timenet.reader import Loading, LoadPlan, TimeFReader
from timenet.registry import TIMENET_REGISTRY_URL, LocalRegistry, RemoteRegistry, open_local_version
from timenet.sources import InstalledConnectors
from timenet.testing import assert_datasets_equal, install_backends, make_dataset
from timenet.types import Access, DatasetMetadata, Domain, License, Version
from timenet.types.access import AccessRequirement
from timenet.types.parent_dataset_card import ParentDatasetCard
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef
from timenet.writer import TimeFWriter


class _Builder:
    """A builder backend whose cards declare ``VERSION``.

    A build stores the demo dataset under the release it builds, in ``BACKEND``. It hands its pins to
    the writer, which publishes the release only if it reproduces the pinned files.
    """

    VERSION = "1.0.0"
    BACKEND = "parquet"

    def __init__(self, calls):
        self.calls = calls

    def dataset_ids(self):
        return ("test/child", "timenet/hello-world")

    def card(self, dataset_id):
        version = DatasetRef.parse(f"{dataset_id}@{self.VERSION}").version
        return make_dataset().metadata.model_copy(update={"dataset_id": dataset_id, "dataset_version": version})

    def build(self, release, store, *, force, pinned_by, options):
        self.calls.append((release, force, dict(pinned_by)))
        dataset = make_dataset()
        dataset._metadata = self.card(release.dataset_id)
        store.put_dataset(dataset, force=force, values_backend=self.BACKEND, pinned_by=pinned_by or None)


def test_a_child_reads_its_missing_payload_source_from_a_later_registry(tmp_path):
    origin, reference = _child_of_the_demo(tmp_path / "origin")
    cache = tmp_path / "cache"
    shutil.copytree(origin.root / "test/child/1.0.0", cache / "test/child/1.0.0")
    store = LocalRegistry(cache)
    with pytest.raises(TimeNetMissingParentError) as caught:
        TimeNet(sources=[store]).open("test/child")
    assert (caught.value.missing, caught.value.needed_by) == (reference, DatasetRef.parse("test/child@1.0.0"))
    assert str(caught.value).endswith(
        ". Add a registry that holds it to the sources, or add InstalledConnectors() to build it."
    )
    with TimeNet(sources=[store, str(origin.root), InstalledConnectors()]).open("test/child") as reader:
        assert reader.read().records[0].signals[0].to_arrow().equals(make_dataset().records[0].signals[0].to_arrow())
    # The read opens the payload source in origin, where it is, and copies nothing.
    assert not (cache / reference.dataset_id).exists()


@pytest.mark.parametrize("in_origin", [True, False])
def test_a_missing_version_is_read_from_a_registry_before_it_is_built(tmp_path, monkeypatch, in_origin):
    origin = LocalRegistry(tmp_path / "origin")
    if in_origin:
        _child_of_the_demo(origin.root)
    calls = []
    install_backends(monkeypatch, _Builder(calls))
    store = LocalRegistry(tmp_path / "local")

    with TimeNet(sources=[store, origin, InstalledConnectors()]).open("test/child") as reader:
        assert reader.read().records

    if in_origin:
        assert calls == []
        assert not (tmp_path / "local").exists()
        return
    assert calls == [(DatasetRef.parse("test/child@1.0.0"), False, {})]


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


def test_resampled_child_reads_and_downloads_without_its_build_only_parent(tmp_path, monkeypatch):
    origin = LocalRegistry(tmp_path / "origin")
    origin.put_dataset(make_dataset())
    metadata = _Resampled().metadata()
    reference = metadata.parents[ParentAlias("parent")]
    run_pipeline(_Resampled(), origin.root, cache_dir=tmp_path / "build-cache")
    manifest = origin.get_manifest(metadata.dataset_id)
    assert manifest.parent_cards[ParentAlias("parent")].source == reference
    assert manifest.payload_sources == {}
    expected = make_dataset().records[0].signals[0].to_arrow()[::2]
    shutil.rmtree(origin.root / reference.dataset_id)

    # Reading the compiled result needs no original samples.
    loaded = TimeNet(sources=[origin]).load(metadata.dataset_id)
    assert loaded.records[0].signals[0].to_arrow().equals(expected)
    # Running the connector again still requires its declared parent.
    with pytest.raises(TimeNetDatasetNotFoundError, match=reference.dataset_id):
        open_parents(metadata, origin)

    cache = tmp_path / "download"
    monkeypatch.setenv("TIMENET_STORAGE", str(cache))
    directory = TimeNet(sources=[origin]).download(metadata.dataset_id)
    assert not (cache / reference.dataset_id).exists()
    shutil.rmtree(origin.root)
    version = open_local_version(directory)
    assert version.manifest.parent_cards == manifest.parent_cards
    with TimeFReader(version) as reader:
        reader.verify()
        assert reader.read().records[0].signals[0].to_arrow().equals(expected)


def test_a_list_of_one_remote_registry_never_builds(tmp_path, monkeypatch):
    calls = []
    install_backends(monkeypatch, _Builder(calls))
    # The remote registry holds the child, but not its payload source, which the installed backend builds.
    origin, reference = _child_of_the_demo(tmp_path / "origin")
    shutil.rmtree(origin.root / reference.dataset_id)
    transport, _ = build_service_fake(origin.root)
    remote = RemoteRegistry("http://api.local", transport=transport, cache_dir=tmp_path / "cache")
    hint = "Add a registry that holds it to the sources, or add InstalledConnectors() to build it."

    with pytest.raises(TimeNetMissingParentError) as missing:
        TimeNet(sources=[remote]).open("test/child")
    assert str(missing.value) == (
        f"payload source {reference} of test/child@1.0.0 is not in registry http://api.local. {hint}"
    )
    with pytest.raises(TimeNetDatasetNotFoundError) as caught:
        TimeNet(sources=[remote]).open(reference.dataset_id)
    assert str(caught.value) == f"{reference.dataset_id} is in none of the sources: registry http://api.local. {hint}"
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


@pytest.fixture
def storage(tmp_path, monkeypatch):
    """The directory that ``download()`` copies into."""
    monkeypatch.setenv("TIMENET_STORAGE", str(tmp_path / "storage"))
    return tmp_path / "storage"


def _entry(source):
    """Describe one source of a list: a local registry by its root, another source by its text."""
    if isinstance(source, LocalRegistry):
        return ("local", source.root)
    if isinstance(source, InstalledConnectors):
        return "installed connectors"
    return str(source)


@pytest.mark.parametrize("registry", [None, "https://registry.example.test", "local"])
def test_the_default_list_follows_timenet_registry(tmp_path, monkeypatch, registry):
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    store = ("local", tmp_path / "home" / "registry")
    if registry == "local":
        monkeypatch.setenv("TIMENET_REGISTRY", str(tmp_path / "elsewhere"))
        expected = [("local", tmp_path / "elsewhere"), "installed connectors"]
    elif registry is not None:
        monkeypatch.setenv("TIMENET_REGISTRY", registry)
        expected = [store, f"registry {registry}", "installed connectors"]
    else:
        expected = [store, f"registry {TIMENET_REGISTRY_URL}", "installed connectors"]
    client = TimeNet()

    assert [_entry(source) for source in client._sources] == expected
    assert client._store is client._sources[0]
    # The argument comes before $TIMENET_REGISTRY, and adds a remote to a local $TIMENET_REGISTRY.
    with_remote = [_entry(source) for source in default_sources(remote_registry="https://other.example.test")]
    assert with_remote == [expected[0], "registry https://other.example.test", "installed connectors"]


def test_timenet_registry_env_selects_a_local_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TIMENET_REGISTRY", str(tmp_path / "home" / "registry"))
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "home" / "registry", dataset) as writer:
        writer.write()
    client = TimeNet()  # the store comes from $TIMENET_REGISTRY
    assert [_entry(source) for source in client._sources] == [
        ("local", tmp_path / "home" / "registry"),
        "installed connectors",
    ]
    assert {m.dataset_id for m in client.list()} == {"timenet/hello-world"}
    assert client.download("timenet/hello-world") == tmp_path / "home" / "storage" / "timenet/hello-world" / "1.0.0"


def test_a_client_takes_an_ordered_list_of_sources(tmp_path):
    store = LocalRegistry(tmp_path / "store")
    client = TimeNet(sources=[store, str(tmp_path / "a"), tmp_path / "b", InstalledConnectors()])

    first, second, third, fourth = client._sources
    assert first is store
    for source, root in ((second, tmp_path / "a"), (third, tmp_path / "b")):
        assert isinstance(source, LocalRegistry)
        assert source.root == root
    assert fourth == InstalledConnectors()


@pytest.mark.parametrize("writable", ["class", "path"])
def test_builds_land_in_the_first_local_registry_that_is_not_read_only(tmp_path, monkeypatch, writable):
    install_backends(monkeypatch, _Builder([]))
    roots = [tmp_path / name for name in ("read-only", "first", "later")]
    first = LocalRegistry(roots[1]) if writable == "class" else str(roots[1])
    sources = [LocalRegistry(roots[0], read_only=True), first, LocalRegistry(roots[2]), InstalledConnectors()]

    TimeNet(sources=sources).load("timenet/hello-world")

    built = [root.name for root in roots if LocalRegistry(root).exists("timenet/hello-world", "1.0.0")]
    assert built == ["first"]


def test_a_client_rejects_one_source_given_as_a_string(tmp_path):
    for value in (str(tmp_path), tmp_path):
        with pytest.raises(TimeFValidationError) as caught:
            TimeNet(value)
        assert str(caught.value) == (
            f"sources takes a list of sources. Pass sources=[{str(tmp_path)!r}], or "
            f"sources=[{str(tmp_path)!r}, InstalledConnectors()] "
            "to build what the path lacks"
        )


def test_a_client_rejects_one_registry_given_without_a_list(tmp_path):
    remote = RemoteRegistry("http://api.local", transport=httpx.MockTransport(lambda request: httpx.Response(404)))
    for registry in (LocalRegistry(tmp_path), remote):
        with pytest.raises(TimeFValidationError, match=r"^sources takes a list of sources\. Pass sources=\["):
            TimeNet(registry)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "sources[0] is blank; give a registry URL or a local path"),
        (
            "ftp://x",
            "sources[0] 'ftp://x' has an unsupported registry scheme. "
            "Use an http(s)://, s3://, timenet:// or file:// URL, or a local path.",
        ),
    ],
    ids=["blank", "unsupported-scheme"],
)
def test_a_client_rejects_a_string_that_is_not_a_registry(value, message):
    with pytest.raises(TimeNetRegistryError) as caught:
        TimeNet(sources=[value])
    assert str(caught.value) == message


def test_a_client_rejects_an_empty_list():
    with pytest.raises(TimeFValidationError, match=r"^sources is empty\."):
        TimeNet(sources=[])


def test_a_client_rejects_an_item_that_is_not_a_source(tmp_path):
    with pytest.raises(TimeFValidationError, match=r"^sources\[1\] is not a source, a registry URL or a path: 7$"):
        TimeNet(sources=[str(tmp_path), 7])  # ty: ignore[invalid-argument-type]


def test_installed_connectors_need_a_local_registry_that_is_not_read_only(tmp_path):
    with pytest.raises(TimeFValidationError) as caught:
        TimeNet(sources=[LocalRegistry(tmp_path, read_only=True), InstalledConnectors()])
    assert str(caught.value) == (
        "InstalledConnectors() needs a writable local registry to build into. "
        "Add a LocalRegistry without read_only=True to the sources"
    )


def test_a_list_without_installed_connectors_never_builds(registry_root, monkeypatch):
    calls = []
    monkeypatch.setattr(_Builder, "VERSION", "2.0.0")
    install_backends(monkeypatch, _Builder(calls))
    client = TimeNet(sources=[LocalRegistry(registry_root)])
    assert str(client.load("timenet/hello-world").metadata.dataset_version) == "1.0.0"
    assert calls == []


def test_get_list_and_search_ask_the_first_registry_that_is_not_the_store(registry_root, tmp_path):
    store = LocalRegistry(tmp_path / "store")
    other = make_dataset()
    other._metadata = other.metadata.model_copy(update={"dataset_id": "test/other"})
    store.put_dataset(other)
    client = TimeNet(sources=[store, str(registry_root), InstalledConnectors()])

    assert [metadata.dataset_id for metadata in client.list()] == ["timenet/hello-world"]
    assert [metadata.dataset_id for metadata in client.search(query="hello")] == ["timenet/hello-world"]
    assert client.get("timenet/hello-world").dataset_id == "timenet/hello-world"
    with pytest.raises(TimeNetDatasetNotFoundError):
        client.get("test/other")
    # Without such a registry, they ask the store.
    alone = TimeNet(sources=[store, InstalledConnectors()])
    assert [metadata.dataset_id for metadata in alone.list()] == ["test/other"]


def test_list(registry_root):
    client = TimeNet(sources=[str(registry_root)])
    assert {m.dataset_id for m in client.list()} == {"timenet/hello-world"}


def test_search(registry_root):
    client = TimeNet(sources=[str(registry_root)])
    assert {m.dataset_id for m in client.search(domain=Domain.GENERAL)} == {"timenet/hello-world"}
    assert client.search(domain=Domain.CARDIOLOGY) == []


def test_download_copies_into_storage(registry_root, storage):
    client = TimeNet(sources=[str(registry_root)])
    version_dir = client.download("timenet/hello-world")
    assert version_dir == storage / "timenet/hello-world" / "1.0.0"
    assert (version_dir / "manifest.json").exists()
    assert list(version_dir.glob("time_series/part-*.parquet"))


def test_download_reports_progress(registry_root, storage):
    client = TimeNet(sources=[str(registry_root)])
    reported: list[int] = []
    client.download("timenet/hello-world", progress_cb=reported.append)
    expected = sum(part.size for part in client.get("timenet/hello-world").files.all_files())
    assert reported  # the callback fired
    assert sum(reported) == expected  # summed to the version's total byte size


def test_manifest_file_rejects_path_traversal():
    with pytest.raises(ValidationError, match="stay within the dataset root"):
        FilePart(path="../../escape.txt", checksum="sha256:" + "0" * 64, size=0)


def test_download_is_idempotent(registry_root, storage):
    client = TimeNet(sources=[str(registry_root)])
    first = client.download("timenet/hello-world")
    second = client.download("timenet/hello-world")
    assert first == second


def test_force_redownload_replaces_atomically(registry_root, storage):
    client = TimeNet(sources=[str(registry_root)])
    first = client.download("timenet/hello-world")
    (first / "stale.txt").write_text("left over")  # a marker that a clean replace should remove

    again = client.download("timenet/hello-world", force=True)
    assert again == first
    assert (again / "manifest.json").exists()
    assert not (again / "stale.txt").exists()  # replaced wholesale, not written in place
    assert not list(again.parent.glob("*.tmp-*"))  # staging dir cleaned up


def test_download_leaves_sibling_staging_dirs_untouched(registry_root, storage):
    client = TimeNet(sources=[str(registry_root)])
    # A concurrent download of the same version has a live <version>.tmp-* dir. download() must not
    # delete a sibling staging dir: sweeping siblings would corrupt that other download mid-write.
    sibling = storage / "timenet/hello-world" / "1.0.0.tmp-deadbeef"
    sibling.mkdir(parents=True)
    (sibling / "inflight.parquet").write_text("partial")

    version_dir = client.download("timenet/hello-world")
    assert sibling.exists()  # left alone, not swept
    assert (version_dir / "manifest.json").exists()  # the download still completed


def test_open_returns_a_lazy_reader(registry_root):
    client = TimeNet(sources=[str(registry_root)])
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
    restored = TimeNet(sources=[str(tmp_path / "reg")]).load("timenet/hello-world")
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


def test_load_reads_in_place_without_downloading(registry_root, tmp_path, storage):
    # load() reads the release where it is, not download-then-read, so it must not write a copy into
    # the store or into local storage the way download() does.
    store = tmp_path / "store"
    client = TimeNet(sources=[LocalRegistry(store), str(registry_root), InstalledConnectors()])
    client.load("timenet/hello-world")
    assert not store.exists()  # nothing copied into the store
    assert not (storage / "timenet/hello-world").exists()  # nothing fetched into the download cache


def test_open_torch(registry_root, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from torch.utils.data import Dataset  # ruff: ignore[import-outside-top-level]

    def read_everything(*args, **kwargs):
        raise AssertionError("open_torch reads one batch at a time, not the whole dataset")

    closed: list[TimeFReader] = []
    close = TimeFReader.close

    def counting_close(reader):
        closed.append(reader)
        close(reader)

    monkeypatch.setattr(TimeFReader, "read", read_everything)
    monkeypatch.setattr(TimeFReader, "close", counting_close)
    client = TimeNet(sources=[str(registry_root)])
    with client.open_torch("timenet/hello-world") as ds:
        assert isinstance(ds, Dataset)
        assert len(ds) == len(make_dataset().records)
        assert isinstance(ds[0]["series"][0], torch.Tensor)
        assert closed == []
    # The view closes the reader it holds at the end of the block.
    assert len(closed) == 1


@pytest.mark.parametrize("arrays_loaded", [True, False], ids=["default-plan", "text-only-plan"])
def test_open_torch_passes_its_load_plan_to_the_view(registry_root, arrays_loaded):
    pytest.importorskip("torch")
    client = TimeNet(sources=[str(registry_root)])
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


def test_registry_env_var(registry_root, monkeypatch):
    monkeypatch.setenv("TIMENET_REGISTRY", str(registry_root))
    client = TimeNet()
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


def test_version_ref_pins_and_defaults_to_latest(versioned_registry):
    client = TimeNet(sources=[str(versioned_registry)])
    assert str(client.get("timenet/hello-world").metadata.dataset_version) == "1.1.0"  # no ref -> latest
    assert str(client.get("timenet/hello-world@latest").metadata.dataset_version) == "1.1.0"
    assert str(client.get("timenet/hello-world@1.0.0").metadata.dataset_version) == "1.0.0"  # pinned
    assert str(client.get("timenet/hello-world", "1.0.0").metadata.dataset_version) == "1.0.0"  # explicit arg


def test_open_and_load_read_the_version_asked_for(versioned_registry):
    client = TimeNet(sources=[str(versioned_registry)])
    with client.open("timenet/hello-world", version="1.0.0") as reader:
        assert str(reader.metadata.dataset_version) == "1.0.0"
    assert str(client.load("timenet/hello-world", version="1.0.0").metadata.dataset_version) == "1.0.0"


def test_open_torch_reads_the_version_asked_for(versioned_registry, monkeypatch):
    pytest.importorskip("torch")
    opened: list[TimeFReader] = []
    original_open = TimeNet.open

    def recording_open(client, *args, **kwargs):
        opened.append(original_open(client, *args, **kwargs))
        return opened[-1]

    monkeypatch.setattr(TimeNet, "open", recording_open)
    TimeNet(sources=[str(versioned_registry)]).open_torch("timenet/hello-world", version="1.0.0")
    (reader,) = opened
    assert str(reader.metadata.dataset_version) == "1.0.0"


def test_version_ref_download_pins(versioned_registry, storage):
    client = TimeNet(sources=[str(versioned_registry)])
    assert client.download("timenet/hello-world@1.0.0") == storage / "timenet/hello-world" / "1.0.0"


def test_version_ref_missing_pin_raises(versioned_registry):
    client = TimeNet(sources=[str(versioned_registry)])
    with pytest.raises(TimeNetDatasetNotFoundError):
        client.get("timenet/hello-world@9.9.9")


def test_version_given_twice_raises(versioned_registry):
    client = TimeNet(sources=[str(versioned_registry)])
    with pytest.raises(ValueError, match="twice"):
        client.get("timenet/hello-world@1.0.0", "1.1.0")


def test_load_names_the_buildable_version_when_a_pin_cannot_be_built(registry_root, monkeypatch):
    # The builder declares only one version, so a pin it cannot satisfy must say which version it
    # builds, and must fail before the expensive build rather than after it.
    calls = []
    install_backends(monkeypatch, _Builder(calls))
    client = TimeNet(sources=[LocalRegistry(registry_root), InstalledConnectors()])

    with pytest.raises(TimeNetVersionUnavailableError, match=r"builds version 1\.0\.0, not the requested 9\.9\.9"):
        client.load("timenet/hello-world@9.9.9")
    assert calls == []  # the pin was rejected before the build ran


@pytest.mark.parametrize("connectors", [True, False])
def test_a_release_that_no_source_provides_names_how_to_get_it(tmp_path, monkeypatch, connectors):
    monkeypatch.setattr(builders_module, "entry_points", lambda group: [])
    sources = [
        LocalRegistry(tmp_path / "reg"),
        LocalRegistry(tmp_path / "empty"),
        *([InstalledConnectors()] if connectors else []),
    ]

    with pytest.raises(TimeNetDatasetNotFoundError) as caught:
        TimeNet(sources=sources).open("x/y")

    message = f"x/y is in none of the sources: {', '.join(str(entry) for entry in sources)}."
    # The hint names InstalledConnectors() only if the list does not have it.
    hint = " Add a registry that holds it to the sources"
    hint += "." if connectors else ", or add InstalledConnectors() to build it."
    assert str(caught.value) == message + hint


def test_load_accepts_the_latest_sentinel_after_a_build(tmp_path, monkeypatch):
    # "latest" is not a pin the build can miss: the version the builder just committed satisfies it.
    empty_root = tmp_path / "reg"
    empty_root.mkdir()
    install_backends(monkeypatch, _Builder([]))
    client = TimeNet(sources=[LocalRegistry(empty_root), InstalledConnectors()])

    assert str(client.load("timenet/hello-world", "latest").metadata.dataset_version) == "1.0.0"


@pytest.mark.parametrize("entry_point", ["open", "load", "open_torch"])
def test_a_list_without_installed_connectors_does_not_build_a_missing_version(registry_root, monkeypatch, entry_point):
    # A list without InstalledConnectors() does not build on a miss, so a notebook load fails fast
    # instead of starting a multi-GB download and a build. The builder declares the requested
    # version, so only the list keeps it from running.
    if entry_point == "open_torch":
        pytest.importorskip("torch")
    calls = []
    monkeypatch.setattr(_Builder, "VERSION", "9.9.9")
    install_backends(monkeypatch, _Builder(calls))
    client = TimeNet(sources=[LocalRegistry(registry_root)])

    # The hint names the sources that could provide the release, in a sentence of its own.
    hint = r"\. Add a registry that holds it to the sources, or add InstalledConnectors\(\) to build it\.$"
    with pytest.raises(TimeNetDatasetNotFoundError, match=hint):
        getattr(client, entry_point)("timenet/hello-world@9.9.9")
    assert calls == []  # the list skips the connector entirely


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


def _hosted(tmp_path, metadata, lineage=()):
    """Return a registry that is not local and serves one manifest, so the build-your-own gate applies."""

    class _Hosted(RemoteRegistry):
        def get_manifest(self, dataset_id, version=None):
            return SimpleNamespace(metadata=metadata, lineage=lineage)

    return _Hosted(
        "http://api.local",
        transport=httpx.MockTransport(lambda request: httpx.Response(404)),
        cache_dir=tmp_path / "cache",
    )


def test_load_rejects_a_credentialed_dataset_from_a_hosted_registry(tmp_path):
    client = TimeNet(
        sources=[_hosted(tmp_path, _card("org/gated", Access.CREDENTIALED, "https://physionet.example/dua"))]
    )
    message = (
        "org/gated@1.0.0 is credentialed. Get access at https://physionet.example/dua for org/gated@1.0.0. "
        "No source serves its files. Add InstalledConnectors() to the sources, or build it locally with "
        "`timenet-build build org/gated`."
    )
    for entry_point in (client.open, client.load):
        with pytest.raises(TimeNetAccessError) as caught:
            entry_point("org/gated")
        assert str(caught.value) == message


def test_the_access_hint_says_that_no_installed_connector_builds_the_release(tmp_path):
    card = _card("org/gated", Access.CREDENTIALED, "https://physionet.example/dua")
    client = TimeNet(sources=[LocalRegistry(tmp_path / "store"), _hosted(tmp_path, card), InstalledConnectors()])

    with pytest.raises(TimeNetAccessError) as caught:
        client.open("org/gated")

    assert str(caught.value) == (
        "org/gated@1.0.0 is credentialed. Get access at https://physionet.example/dua for org/gated@1.0.0. "
        "No source serves its files, and no installed connector builds this release. "
        "Install its connector package and build it locally with `timenet-build build org/gated`."
    )


def test_the_hint_for_a_hosted_child_names_the_access_page_of_each_ancestor(tmp_path):
    # The card of the child stays open and has no access page. Its parents set the access.
    parents = (
        _card("org/a", Access.CREDENTIALED, "https://a.example"),
        _card("org/b", Access.RESTRICTED, "https://b.example"),
    )
    lineage = tuple(ParentDatasetCard(metadata=card) for card in parents)
    client = TimeNet(sources=[_hosted(tmp_path, _card("org/child", Access.OPEN), lineage)])

    with pytest.raises(TimeNetAccessError) as caught:
        client.open("org/child")

    assert caught.value.requirements == (
        AccessRequirement(DatasetRef.parse("org/a@1.0.0"), Access.CREDENTIALED, "https://a.example"),
        AccessRequirement(DatasetRef.parse("org/b@1.0.0"), Access.RESTRICTED, "https://b.example"),
    )
    assert str(caught.value) == (
        "org/child@1.0.0 is restricted. "
        "Get access at https://a.example for org/a@1.0.0 and at https://b.example for org/b@1.0.0. "
        "No source serves its files. Add InstalledConnectors() to the sources, or build it locally with "
        "`timenet-build build org/child`."
    )
