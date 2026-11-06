from pathlib import Path
import re
import shutil
from types import SimpleNamespace

from _demo_child import _child_of_the_demo
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
    TimeNetFileError,
    TimeNetMissingParentError,
    TimeNetRegistryError,
    TimeNetVersionConflictError,
    TimeNetVersionUnavailableError,
)
from timenet.format.stored_values import StoredText
from timenet.manifest import Manifest
from timenet.manifest.files import FilePart
from timenet.progress import BytesMoved, ProgressEvent, StepFinished, StepStarted
from timenet.reader import Loading, LoadPlan, TimeFReader
from timenet.registry import TIMENET_REGISTRY_URL, LocalRegistry, RemoteRegistry, open_local_version
from timenet.sources import InstalledConnectors
from timenet.testing import assert_datasets_equal, install_backends, make_dataset
from timenet.types import Access, Annotation, DatasetMetadata, Domain, License, Version
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


def test_a_child_reads_its_missing_payload_source_from_a_later_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(builders_module, "entry_points", lambda group: [])
    origin, reference = _child_of_the_demo(tmp_path / "origin")
    cache = tmp_path / "cache"
    shutil.copytree(origin.root / "test/child/1.0.0", cache / "test/child/1.0.0")
    store = LocalRegistry(cache, store=True)
    with pytest.raises(TimeNetMissingParentError) as caught:
        TimeNet(sources=[store]).open("test/child")
    assert (caught.value.missing, caught.value.needed_by) == (reference, DatasetRef.parse("test/child@1.0.0"))
    assert str(caught.value).endswith(
        ". Add a registry that holds it to the sources, or add InstalledConnectors() to build it."
    )
    # No installed connector builds the payload source, so with InstalledConnectors() the hint names only a registry.
    with pytest.raises(TimeNetMissingParentError, match=r"\. Add a registry that holds it to the sources\.$"):
        TimeNet(sources=[store, InstalledConnectors()]).open("test/child")
    with TimeNet(sources=[store, str(origin.root), InstalledConnectors()]).open("test/child") as reader:
        assert reader.read().records[0].signals[0].to_arrow().equals(make_dataset().records[0].signals[0].to_arrow())
    # The read opens the payload source in origin, where it is, and copies nothing.
    assert not (cache / reference.dataset_id).exists()


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

    store = LocalRegistry(tmp_path / "store", store=True)
    directory = TimeNet(sources=[store, origin]).download(metadata.dataset_id)
    assert not (store.root / reference.dataset_id).exists()
    shutil.rmtree(origin.root)
    version = open_local_version(directory)
    assert version.manifest.parent_cards == manifest.parent_cards
    with TimeFReader(version) as reader:
        reader.verify()
        assert reader.read().records[0].signals[0].to_arrow().equals(expected)


def test_a_damaged_release_outside_the_store_names_no_download(tmp_path):
    origin, _ = _child_of_the_demo(tmp_path / "origin")
    mirror = LocalRegistry(tmp_path / "mirror")
    shutil.copytree(origin.root, mirror.root)
    (mirror.root / "test/child/1.0.0/control.duckdb").write_bytes(b"cut short")
    store = LocalRegistry(tmp_path / "store", store=True)

    # download() replaces only a copy in the store, so it cannot repair the mirror.
    with pytest.raises(TimeNetFileError) as caught:
        TimeNet(sources=[store, mirror, origin]).open("test/child")
    assert (caught.value.release, caught.value.hint) == (DatasetRef.parse("test/child@1.0.0"), None)


def test_a_damaged_build_in_the_store_names_the_build_that_replaces_it(tmp_path, monkeypatch):
    calls = []
    install_backends(monkeypatch, _Builder(calls))
    store = LocalRegistry(tmp_path / "store", store=True)
    # A registry in the list that lacks the release, as the hosted registry lacks a local build.
    empty = RemoteRegistry(
        "http://api.local",
        transport=httpx.MockTransport(lambda request: httpx.Response(404)),
        store=tmp_path / "cache",
    )
    client = TimeNet(sources=[store, empty, InstalledConnectors()])
    client.load("timenet/hello-world@1.0.0")  # the build puts the release into the store
    (store.root / "timenet/hello-world/1.0.0/control.duckdb").write_bytes(b"cut short")

    # No registry other than the store provides the release, so download() cannot replace it.
    hint = f"Build it again with `timenet-build build timenet/hello-world --out {store.root} --force`."
    with pytest.raises(TimeNetFileError) as caught:
        client.open("timenet/hello-world@1.0.0")
    assert caught.value.hint == hint
    assert len(calls) == 1  # nothing builds over the damaged copy


@pytest.fixture
def registry_root(tmp_path):
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "reg", dataset) as writer:
        writer.write()
    return tmp_path / "reg"


def _entry(source):
    """Describe one source of a list: a store by its root, another source by its text."""
    if isinstance(source, LocalRegistry) and source.is_store:
        return ("store", source.root)
    if isinstance(source, InstalledConnectors):
        return "installed connectors"
    return str(source)


def test_the_default_list_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(tmp_path / "store"))
    monkeypatch.setenv("TIMENET_REMOTE_REGISTRY", str(tmp_path / "remote"))
    monkeypatch.setenv("TIMENET_AUTO_BUILD", "false")
    store, remote = default_sources()
    assert isinstance(store, LocalRegistry) and isinstance(remote, LocalRegistry)
    assert (type(store), store.root, store.is_store) == (LocalRegistry, tmp_path / "store", True)
    assert (type(remote), remote.root, remote.is_store) == (LocalRegistry, tmp_path / "remote", False)
    # TimeNet() uses the same list.
    client = TimeNet()
    assert [_entry(source) for source in client._sources] == [
        ("store", tmp_path / "store"),
        f"local registry {tmp_path / 'remote'}",
    ]
    assert client._store is client._sources[0]


@pytest.mark.usefixtures("home")
def test_the_default_list_without_variables(monkeypatch):
    for name in ("LOCAL_REGISTRY", "REMOTE_REGISTRY", "AUTO_BUILD", "CACHE"):
        monkeypatch.delenv(f"TIMENET_{name}", raising=False)
    store, remote, connectors = default_sources()
    assert isinstance(store, LocalRegistry) and isinstance(remote, RemoteRegistry)
    assert (type(store), store.root, store.is_store) == (LocalRegistry, Path("~/.timenet/registry").expanduser(), True)
    assert (str(remote), remote.local_root) == (f"registry {TIMENET_REGISTRY_URL}", store.root)
    assert type(connectors) is InstalledConnectors


def test_default_sources_arguments_override_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(tmp_path / "env-store"))
    monkeypatch.setenv("TIMENET_REMOTE_REGISTRY", str(tmp_path / "env-remote"))
    monkeypatch.setenv("TIMENET_AUTO_BUILD", "false")
    sources = default_sources(local_registry=tmp_path / "store", remote_registry=tmp_path / "remote", auto_build=True)
    assert [_entry(source) for source in sources] == [
        ("store", tmp_path / "store"),
        f"local registry {tmp_path / 'remote'}",
        "installed connectors",
    ]


def test_default_sources_names_the_variable_of_a_registry_that_cannot_open(monkeypatch):
    monkeypatch.setenv("TIMENET_REMOTE_REGISTRY", "gs://bucket/registry")
    with pytest.raises(
        TimeNetRegistryError,
        match=r"^TIMENET_REMOTE_REGISTRY 'gs://bucket/registry' has an unsupported registry scheme",
    ):
        default_sources()


def test_a_client_takes_an_ordered_list_of_sources(tmp_path):
    store = LocalRegistry(tmp_path / "store", store=True)
    client = TimeNet(sources=[store, str(tmp_path / "a"), tmp_path / "b", InstalledConnectors()])

    first, second, third, fourth = client._sources
    assert first is store
    for source, root in ((second, tmp_path / "a"), (third, tmp_path / "b")):
        assert isinstance(source, LocalRegistry)
        assert (source.root, source.is_store) == (root, False)
    assert fourth == InstalledConnectors()
    assert client._store is store


def test_a_client_rejects_one_source_given_as_a_string(tmp_path):
    for value in (str(tmp_path), tmp_path):
        with pytest.raises(TimeFValidationError) as caught:
            TimeNet(value)
        assert str(caught.value) == (
            f"sources takes a list of sources. Pass sources=[{str(tmp_path)!r}], or "
            f"sources=[LocalRegistry({str(tmp_path)!r}, store=True), InstalledConnectors()] "
            "to build what the path lacks"
        )


def test_a_client_rejects_one_registry_given_without_a_list(tmp_path):
    with pytest.raises(TimeFValidationError, match=r"^sources takes a list of sources\. Pass sources=\["):
        TimeNet(LocalRegistry(tmp_path))  # ty: ignore[invalid-argument-type]


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


def test_a_client_accepts_at_most_one_store(tmp_path):
    stores = [LocalRegistry(tmp_path / name, store=True) for name in ("a", "b")]
    with pytest.raises(TimeFValidationError, match=r"^sources has 2 stores: "):
        TimeNet(sources=[*stores, InstalledConnectors()])


def test_a_client_rejects_a_store_that_overlaps_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_CACHE", str(tmp_path / "cache"))
    given = LocalRegistry(tmp_path / "cache" / "store", store=True)
    with pytest.raises(TimeFValidationError, match=r"^The store \(.*\) and TIMENET_CACHE \(.*\) overlap\."):
        TimeNet(sources=[given, InstalledConnectors()])


def test_a_list_with_installed_connectors_needs_a_store(tmp_path):
    with pytest.raises(TimeFValidationError, match=r"LocalRegistry\(<path>, store=True\)"):
        TimeNet(sources=[LocalRegistry(tmp_path), InstalledConnectors()])


def test_get_list_and_search_ask_the_first_registry_that_is_not_the_store(registry_root, tmp_path):
    store = LocalRegistry(tmp_path / "store", store=True)
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


def test_get_returns_manifest(registry_root):
    client = TimeNet(sources=[str(registry_root)])
    assert isinstance(client.get("timenet/hello-world"), Manifest)


def test_search(registry_root):
    client = TimeNet(sources=[str(registry_root)])
    assert {m.dataset_id for m in client.search(domain=Domain.GENERAL)} == {"timenet/hello-world"}
    assert client.search(domain=Domain.CARDIOLOGY) == []


def test_download_copies_the_release_and_its_payload_sources_into_the_store(tmp_path):
    origin, _ = _child_of_the_demo(tmp_path / "origin")
    store = LocalRegistry(tmp_path / "store", store=True)
    path = TimeNet(sources=[store, origin.root]).download("test/child")
    assert path == tmp_path / "store/test/child/1.0.0"
    assert store.exists("timenet/hello-world", "1.0.0")


def test_download_without_a_store_raises(registry_root):
    with pytest.raises(TimeFValidationError) as caught:
        TimeNet(sources=[str(registry_root)]).download("timenet/hello-world")
    assert str(caught.value) == (
        "download() copies into the store; add LocalRegistry(<path>, store=True) to the sources"
    )


def test_download_of_a_release_that_only_a_connector_provides_builds_nothing(tmp_path, monkeypatch):
    calls = []
    install_backends(monkeypatch, _Builder(calls))
    store = LocalRegistry(tmp_path / "store", store=True)

    with pytest.raises(TimeNetDatasetNotFoundError) as caught:
        TimeNet(sources=[store, InstalledConnectors()]).download("timenet/hello-world")

    assert str(caught.value) == (
        "timenet/hello-world@1.0.0 is not built yet: only an installed connector provides it. "
        "open() or load() builds it into the store."
    )
    assert calls == []
    assert not store.root.exists()


def test_download_of_latest_fails_when_a_connector_declares_a_newer_version(registry_root, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(_Builder, "VERSION", "2.0.0")
    install_backends(monkeypatch, _Builder(calls))
    store = LocalRegistry(tmp_path / "store", store=True)
    client = TimeNet(sources=[store, str(registry_root), InstalledConnectors()])

    # "Latest" is the version of the connector, which no registry holds.
    with pytest.raises(TimeNetDatasetNotFoundError, match=r"^timenet/hello-world@2\.0\.0 is not built yet: "):
        client.download("timenet/hello-world")
    assert calls == []
    # The registry still provides its exact version.
    assert client.download("timenet/hello-world@1.0.0") == store.root / "timenet/hello-world/1.0.0"


def test_download_of_a_release_that_no_source_provides_names_each_source(tmp_path, monkeypatch):
    monkeypatch.setattr(builders_module, "entry_points", lambda group: [])
    sources = [LocalRegistry(tmp_path / "store", store=True), LocalRegistry(tmp_path / "empty"), InstalledConnectors()]

    with pytest.raises(TimeNetDatasetNotFoundError) as caught:
        TimeNet(sources=sources).download("x/y")

    # download() never builds, so the hint names only a registry.
    names = ", ".join(str(source) for source in sources)
    assert str(caught.value) == f"x/y is in none of the sources: {names}. Add a registry that holds it to the sources."


def test_download_of_a_version_that_the_connector_does_not_build_names_the_built_version(tmp_path, monkeypatch):
    calls = []
    install_backends(monkeypatch, _Builder(calls))
    client = TimeNet(sources=[LocalRegistry(tmp_path / "store", store=True), InstalledConnectors()])

    with pytest.raises(TimeNetVersionUnavailableError, match=r"builds version 1\.0\.0, not the requested 9\.9\.9"):
        client.download("timenet/hello-world@9.9.9")
    assert calls == []


def test_download_copies_each_release_from_the_first_registry_that_provides_it(tmp_path):
    origin, reference = _child_of_the_demo(tmp_path / "origin")
    # The first registry holds only the child, and the second only the payload source that it reads.
    shutil.copytree(origin.root / "test/child", tmp_path / "first/test/child")
    shutil.copytree(origin.root / reference.dataset_id, tmp_path / "second" / reference.dataset_id)
    store = LocalRegistry(tmp_path / "store", store=True)

    path = TimeNet(sources=[store, tmp_path / "first", tmp_path / "second"]).download("test/child")

    assert path == store.root / "test/child/1.0.0"
    with TimeNet(sources=[store]).open("test/child") as reader:
        reader.verify()


@pytest.mark.parametrize("case", ["force", "damaged"])
def test_download_refuses_a_store_copy_with_other_files(registry_root, tmp_path, case):
    store = LocalRegistry(tmp_path / "store", store=True)
    other = make_dataset()
    other.records[0].annotate(Annotation(key="build", value="another"))
    store.put_dataset(other)
    directory = store.root / "timenet/hello-world/1.0.0"
    client = TimeNet(sources=[store, str(registry_root)])
    if case == "force":
        # A complete copy in the store returns at once, so only force=True compares its files.
        assert client.download("timenet/hello-world") == directory
    else:
        (directory / "control.duckdb").write_bytes(b"cut short")
    before = (directory / "manifest.json").read_text()

    with pytest.raises(TimeNetVersionConflictError, match="new version"):
        client.download("timenet/hello-world", force=case == "force")

    assert (directory / "manifest.json").read_text() == before


@pytest.mark.parametrize("case", ["force", "damaged"])
def test_download_names_the_store_when_no_other_registry_provides_its_copy(tmp_path, monkeypatch, case):
    calls = []
    install_backends(monkeypatch, _Builder(calls))
    store = LocalRegistry(tmp_path / "store", store=True)
    store.put_dataset(make_dataset())
    client = TimeNet(sources=[store, InstalledConnectors()])

    # The store holds the release, so the error is about its copy, not about a build.
    if case == "force":
        message = r"^No registry other than the store provides timenet/hello-world@1\.0\.0, so download\(force=True\)"
        with pytest.raises(TimeNetDatasetNotFoundError, match=message):
            client.download("timenet/hello-world", force=True)
    else:
        (store.root / "timenet/hello-world/1.0.0/control.duckdb").write_bytes(b"cut short")
        # The installed connector builds the release, so the hint names the build that replaces it.
        hint = f"Build it again with `timenet-build build timenet/hello-world --out {store.root} --force`."
        with pytest.raises(TimeNetFileError) as caught:
            client.download("timenet/hello-world")
        assert caught.value.hint == hint
    assert calls == []


@pytest.mark.parametrize("case", ["force", "damaged"])
def test_download_counts_a_registry_over_the_directory_of_the_store_as_the_store(tmp_path, monkeypatch, case):
    monkeypatch.setattr(builders_module, "entry_points", lambda group: [])
    root = tmp_path / "reg"
    LocalRegistry(root).put_dataset(make_dataset())
    directory = root / "timenet/hello-world/1.0.0"
    if case == "damaged":
        (directory / "control.duckdb").write_bytes(b"cut short")
    before = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    # As for `timenet download timenet/hello-world --local-registry ./reg --remote-registry ./reg`:
    # the registry after the store reads the directory of the store, so it cannot replace its copy.
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(root))
    client = TimeNet(sources=default_sources(remote_registry=root))

    if case == "force":
        with pytest.raises(TimeNetDatasetNotFoundError) as caught:
            client.download("timenet/hello-world", force=True)
        assert str(caught.value) == (
            "No registry other than the store provides timenet/hello-world@1.0.0, so download(force=True) "
            "cannot copy it again."
        )
    else:
        with pytest.raises(TimeNetFileError) as caught:
            client.download("timenet/hello-world")
        assert caught.value.hint == "No registry other than the store provides it, so download() cannot replace it."
    assert {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()} == before


def test_download_counts_a_case_variant_of_the_store_directory_as_the_store(tmp_path):
    (tmp_path / "Probe").mkdir()
    if not (tmp_path / "probe").exists():
        pytest.skip("the temporary folder tells upper and lower case apart")
    store = LocalRegistry(tmp_path / "store", store=True)
    store.put_dataset(make_dataset())
    client = TimeNet(sources=[store, LocalRegistry(tmp_path / "STORE")])

    # The second registry reads the directory of the store, so it cannot copy the release again.
    with pytest.raises(TimeNetDatasetNotFoundError, match=r"^No registry other than the store provides "):
        client.download("timenet/hello-world", force=True)


@pytest.mark.parametrize("case", ["missing", "damaged"])
def test_download_completes_the_payload_sources_of_a_complete_store_copy(tmp_path, case):
    origin, reference = _child_of_the_demo(tmp_path / "origin")
    store = LocalRegistry(tmp_path / "store", store=True)
    client = TimeNet(sources=[store, origin])
    client.download("test/child")
    source = store.root / reference.dataset_id / str(reference.version)
    if case == "missing":
        shutil.rmtree(source)
    else:
        (source / "control.duckdb").write_bytes(b"cut short")

    # The child is complete in the store, but the payload source that it reads is not.
    assert client.download("test/child") == store.root / "test/child/1.0.0"
    with TimeFReader(store.open_release(reference)) as reader:
        reader.verify()


def test_download_repairs_a_damaged_store_copy(tmp_path):
    origin, reference = _child_of_the_demo(tmp_path / "origin")
    store = LocalRegistry(tmp_path / "store", store=True)
    client = TimeNet(sources=[store, origin.root])
    client.download("test/child")
    (store.root / reference.dataset_id / str(reference.version) / "control.duckdb").write_bytes(b"cut short")

    # The read stops at the damaged payload source, and names the download that repairs it.
    hint = f"Copy an intact release into the store with download('{reference}')."
    with pytest.raises(TimeNetFileError, match=re.escape(hint) + "$"):
        client.open("test/child")
    client.download(str(reference))

    with client.open("test/child") as reader:
        reader.verify()


def test_download_returns_a_complete_store_copy_at_once(registry_root, tmp_path):
    client = TimeNet(sources=[LocalRegistry(tmp_path / "store", store=True), str(registry_root)])
    first = client.download("timenet/hello-world")
    # The store copy is complete, so the next download needs no registry.
    shutil.rmtree(registry_root)
    events: list[ProgressEvent] = []
    assert client.download("timenet/hello-world", progress=events.append) == first
    assert events == []  # nothing is copied again


def test_download_with_force_copies_again(registry_root, tmp_path):
    client = TimeNet(sources=[LocalRegistry(tmp_path / "store", store=True), str(registry_root)])
    first = client.download("timenet/hello-world")
    (first / "stale.txt").write_text("left over")  # a marker that a clean replace removes

    again = client.download("timenet/hello-world", force=True)
    assert again == first
    assert (again / "manifest.json").exists()
    assert not (again / "stale.txt").exists()  # replaced as a whole, not written in place
    assert not list(again.parent.glob("*.tmp-*"))  # the staging directory is gone


def test_download_reports_its_steps_and_bytes(tmp_path):
    origin, reference = _child_of_the_demo(tmp_path / "origin")
    child = DatasetRef.parse("test/child@1.0.0")
    events: list[ProgressEvent] = []
    TimeNet(sources=[LocalRegistry(tmp_path / "store", store=True), origin.root]).download(
        "test/child", progress=events.append
    )

    assert [event for event in events if not isinstance(event, BytesMoved)] == [
        StepStarted(reference, "copy"),
        StepFinished(reference, "copy"),
        StepStarted(child, "copy"),
        StepFinished(child, "copy"),
    ]
    for release in (reference, child):
        done = [event.done for event in events if isinstance(event, BytesMoved) and event.url == str(release)]
        manifest = origin.get_manifest(release.dataset_id, str(release.version))
        assert done[-1] == sum(part.size for part in manifest.files.all_files())


def test_manifest_file_rejects_path_traversal():
    with pytest.raises(ValidationError, match="stay within the dataset root"):
        FilePart(path="../../escape.txt", checksum="sha256:" + "0" * 64, size=0)


def test_download_leaves_sibling_staging_dirs_untouched(registry_root, tmp_path):
    store = LocalRegistry(tmp_path / "store", store=True)
    client = TimeNet(sources=[store, str(registry_root)])
    # A concurrent download of the same version has a live <version>.tmp-* dir. download() must not
    # delete a sibling staging dir: sweeping siblings would corrupt that other download mid-write.
    sibling = store.root / "timenet/hello-world" / "1.0.0.tmp-deadbeef"
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


def test_load_reads_in_place_without_downloading(registry_root, tmp_path, monkeypatch):
    # load() reads the release where it is, not download-then-read, so it must not write a copy into
    # the store the way download() does.
    calls = []
    install_backends(monkeypatch, _Builder(calls))
    store = tmp_path / "store"
    client = TimeNet(sources=[LocalRegistry(store, store=True), str(registry_root), InstalledConnectors()])
    client.load("timenet/hello-world")
    assert calls == []
    assert not store.exists()  # nothing copied into the store


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


def test_version_ref_download_pins(versioned_registry, tmp_path):
    client = TimeNet(sources=[LocalRegistry(tmp_path / "store", store=True), str(versioned_registry)])
    assert client.download("timenet/hello-world@1.0.0") == tmp_path / "store/timenet/hello-world/1.0.0"


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
    client = TimeNet(sources=[LocalRegistry(registry_root, store=True), InstalledConnectors()])

    with pytest.raises(TimeNetVersionUnavailableError, match=r"builds version 1\.0\.0, not the requested 9\.9\.9"):
        client.load("timenet/hello-world@9.9.9")
    assert calls == []  # the pin was rejected before the build ran


@pytest.mark.parametrize("connectors", [True, False])
def test_a_release_that_no_source_provides_names_how_to_get_it(tmp_path, monkeypatch, connectors):
    monkeypatch.setattr(builders_module, "entry_points", lambda group: [])
    sources = [
        LocalRegistry(tmp_path / "reg", store=True),
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


@pytest.mark.parametrize("pin", [None, "latest", ""])
def test_load_accepts_the_latest_sentinels_after_a_build(tmp_path, monkeypatch, pin):
    # Every registry reads "latest" and "" as the latest version, so they are not pins the build
    # can miss: the version the builder just committed satisfies them.
    empty_root = tmp_path / "reg"
    empty_root.mkdir()
    install_backends(monkeypatch, _Builder([]))
    client = TimeNet(sources=[LocalRegistry(empty_root, store=True), InstalledConnectors()])

    assert str(client.load("timenet/hello-world", pin).metadata.dataset_version) == "1.0.0"


def test_a_list_without_installed_connectors_does_not_build_a_missing_version(registry_root, monkeypatch):
    # A list without InstalledConnectors() does not build on a miss, so a notebook load fails fast
    # instead of starting a multi-GB download and a build. The builder declares the requested
    # version, so only the list keeps it from running.
    calls = []
    monkeypatch.setattr(_Builder, "VERSION", "9.9.9")
    install_backends(monkeypatch, _Builder(calls))
    client = TimeNet(sources=[LocalRegistry(registry_root, store=True)])

    # The hint names the sources that could provide the release, in a sentence of its own.
    hint = r"\. Add a registry that holds it to the sources, or add InstalledConnectors\(\) to build it\.$"
    with pytest.raises(TimeNetDatasetNotFoundError, match=hint):
        client.load("timenet/hello-world@9.9.9")
    assert calls == []  # the list skips the connector entirely


def test_latest_builds_a_newer_version_that_the_connector_declares(registry_root, monkeypatch):
    calls = []
    monkeypatch.setattr(_Builder, "VERSION", "2.0.0")
    install_backends(monkeypatch, _Builder(calls))
    client = TimeNet(sources=[LocalRegistry(registry_root, store=True), InstalledConnectors()])
    with client.open("timenet/hello-world") as reader:
        assert str(reader.metadata.dataset_version) == "2.0.0"
    assert calls == [(DatasetRef.parse("timenet/hello-world@2.0.0"), False, {})]


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
        store=tmp_path / "cache",
    )


def test_load_rejects_a_credentialed_dataset_from_a_hosted_registry(tmp_path):
    hosted = _hosted(tmp_path, _card("org/gated", Access.CREDENTIALED, "https://physionet.example/dua"))
    store = LocalRegistry(tmp_path / "store", store=True)
    client = TimeNet(sources=[store, hosted])
    message = (
        "org/gated@1.0.0 is credentialed. Get access at https://physionet.example/dua for org/gated@1.0.0. "
        "No source serves its files. Add InstalledConnectors() to the sources, or build it locally with "
        "`timenet-build build org/gated`."
    )
    with pytest.raises(TimeNetAccessError) as caught:
        client.load("org/gated")
    assert str(caught.value) == message
    assert not store.root.exists()


def test_the_access_hint_says_that_no_installed_connector_builds_the_release(tmp_path, monkeypatch):
    monkeypatch.setattr(builders_module, "entry_points", lambda group: [])
    card = _card("org/gated", Access.CREDENTIALED, "https://physionet.example/dua")
    client = TimeNet(
        sources=[LocalRegistry(tmp_path / "store", store=True), _hosted(tmp_path, card), InstalledConnectors()]
    )

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
