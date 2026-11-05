import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest

import timenet
from timenet import builders
from timenet.builders import BuildOptions
from timenet.errors import (
    TimeFValidationError,
    TimeNetAccessError,
    TimeNetDatasetNotFoundError,
    TimeNetMissingParentError,
    TimeNetRegistryError,
)
from timenet.registry import LocalRegistry, RemoteRegistry
from timenet.sources import NotFound
from timenet.testing import install_backends, make_dataset
from timenet.types import Access
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef


PARENT = DatasetRef.parse("test/parent@1.0.0")
CHILD = DatasetRef.parse("test/child@1.0.0")
HELLO = DatasetRef.parse("timenet/hello-world@1.0.0")


def _card(reference, **update):
    """Return the card of the demo dataset under another release."""
    return make_dataset().metadata.model_copy(
        update={"dataset_id": reference.dataset_id, "dataset_version": reference.version, **update}
    )


def _demo(card):
    """Return the demo dataset under ``card``."""
    dataset = make_dataset()
    dataset._metadata = card
    return dataset


class _Backend:
    """A builder backend of another package. It builds the demo dataset under each card that it holds."""

    def __init__(self, cards, calls):
        self.cards = {card.dataset_id: card for card in cards}
        self.calls = calls

    def dataset_ids(self):
        return list(self.cards)

    def card(self, dataset_id):
        return self.cards.get(dataset_id)

    def build(self, release, store, *, force, pinned_by, options):
        self.calls.append((release, options))
        store.put_dataset(_demo(self.cards[release.dataset_id]), force=force, pinned_by=pinned_by or None)


@pytest.fixture
def cards(monkeypatch):
    """Install a card for CHILD and a card for PARENT. CHILD declares PARENT.

    Returns:
        The list of the calls of the backend: ``(release, options)`` for each build.
    """
    calls = []
    backend = _Backend([_card(CHILD, parents={ParentAlias("parent"): PARENT}), _card(PARENT)], calls)
    install_backends(monkeypatch, backend)
    return calls


def _store_parent(registry):
    """Put a PARENT release into a registry."""
    registry.put_dataset(_demo(_card(PARENT)))


def test_build_with_store_and_parents_from_takes_the_store_and_the_sources_only_from_them(tmp_path, monkeypatch, cards):
    env_store, origin = tmp_path / "env-store", LocalRegistry(tmp_path / "origin")
    _store_parent(origin)
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(env_store))
    monkeypatch.setenv("TIMENET_REMOTE_REGISTRY", str(origin.root))
    store = tmp_path / "store"
    # The remote registry of the environment holds the parent, but parents_from is empty.
    with pytest.raises(TimeNetMissingParentError) as caught:
        timenet.build(CHILD.dataset_id, store=store, parents_from=[])
    assert (
        caught.value.hint == "Add a registry that holds it to parents_from, or add InstalledConnectors() to build it."
    )
    assert not env_store.exists()
    assert cards == []


def test_build_without_parents_from_copies_a_parent_from_the_remote_registry(tmp_path, monkeypatch, cards):
    origin = LocalRegistry(tmp_path / "origin")
    _store_parent(origin)
    monkeypatch.setenv("TIMENET_REMOTE_REGISTRY", str(origin.root))
    plan = timenet.build(CHILD.dataset_id, store=tmp_path / "store")
    # default_sources() opens its own LocalRegistry on origin, so compare the entry by its root.
    first = plan.steps[0]
    assert isinstance(first.entry, LocalRegistry)
    assert (first.release, first.action, first.entry.root) == (PARENT, "copy", origin.root)
    assert LocalRegistry(tmp_path / "store").exists(CHILD.dataset_id, "1.0.0")
    assert [release for release, _ in cards] == [CHILD]


def test_build_lands_in_the_store_that_the_environment_names(tmp_path, monkeypatch, cards):
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(tmp_path / "env-store"))
    plan = timenet.build(PARENT.dataset_id)
    assert isinstance(plan, timenet.Plan)
    assert LocalRegistry(tmp_path / "env-store").exists(PARENT.dataset_id, "1.0.0")


def test_auto_build_false_leaves_the_connectors_out_of_the_default_parents(tmp_path, monkeypatch, cards):
    # The remote registry of the environment is empty, so only the installed connector builds the parent.
    store = tmp_path / "store"
    monkeypatch.setenv("TIMENET_AUTO_BUILD", "false")
    with pytest.raises(TimeNetMissingParentError):
        timenet.build(CHILD.dataset_id, store=store)
    assert cards == []

    monkeypatch.setenv("TIMENET_AUTO_BUILD", "true")
    timenet.build(CHILD.dataset_id, store=store)
    assert [release for release, _ in cards] == [PARENT, CHILD]


def test_plan_build_runs_nothing_and_returns_the_plan_that_build_runs(tmp_path, cards):
    origin, store = LocalRegistry(tmp_path / "origin"), tmp_path / "store"
    _store_parent(origin)

    plan = timenet.plan_build(CHILD.dataset_id, store=store, parents_from=[origin])
    assert cards == []
    assert not store.exists()
    assert plan.steps == (
        timenet.Step(PARENT, "copy", origin),
        timenet.Step(CHILD, "build", plan.steps[1].entry, depends_on=(PARENT,)),
    )

    assert timenet.build(CHILD.dataset_id, store=store, parents_from=[origin]) == plan


def test_open_access_only_refuses_a_gated_parent_before_any_step(tmp_path, cards):
    origin, store = LocalRegistry(tmp_path / "origin"), tmp_path / "store"
    gated = _card(PARENT, access=Access.CREDENTIALED, access_url="https://physionet.example/dua")
    origin.put_dataset(_demo(gated))

    with pytest.raises(TimeNetAccessError, match="credentialed"):
        timenet.build(CHILD.dataset_id, store=store, parents_from=[origin], open_access_only=True)
    assert cards == []
    assert not store.exists()


def test_timenet_exports_the_client_and_its_sources(tmp_path, cards):
    client = timenet.TimeNet(
        sources=[timenet.LocalRegistry(tmp_path / "store", store=True), timenet.InstalledConnectors()]
    )
    client.load(PARENT.dataset_id)
    assert [release for release, _ in cards] == [PARENT]


def test_build_passes_keep_cache_and_progress_to_the_backend(tmp_path, cards):
    def callback(event):
        pass

    timenet.build(PARENT.dataset_id, store=tmp_path / "store", parents_from=[], keep_cache=True, progress=callback)
    assert cards == [(PARENT, BuildOptions(keep_cache=True, progress=callback))]


def test_build_of_a_dataset_without_a_connector_package_names_one(tmp_path, monkeypatch):
    monkeypatch.setattr(builders, "entry_points", lambda group: [])
    with pytest.raises(TimeNetDatasetNotFoundError) as caught:
        timenet.build("x/y", store=tmp_path / "store")
    assert caught.value.hint == "Install a connector package that builds it, such as timenet-connectors."


def test_build_rejects_a_store_that_is_not_local(tmp_path, cards):
    remote = RemoteRegistry("https://registry.example.test")
    with pytest.raises(TimeFValidationError) as caught:
        timenet.build(CHILD.dataset_id, store=remote)  # ty: ignore[invalid-argument-type]
    assert str(caught.value) == (
        "store takes a local registry, because a build always lands locally. Publish the result with timenet.publish()."
    )
    # A URL goes through local_registry_path(), which names the problem.
    with pytest.raises(TimeNetRegistryError, match="is remote; the local registry needs a local directory"):
        timenet.build(CHILD.dataset_id, store="s3://bucket/registry")
    assert cards == []


def test_build_rejects_a_given_store_that_overlaps_the_cache(tmp_path, monkeypatch, cards):
    store = tmp_path / "cache" / "store"
    monkeypatch.setenv("TIMENET_CACHE", str(tmp_path / "cache"))
    with pytest.raises(TimeFValidationError, match=r"^The store \(.*\) and TIMENET_CACHE \(.*\) overlap\."):
        timenet.build(CHILD.dataset_id, store=store)
    assert cards == []
    assert not store.exists()


def test_build_rejects_parents_from_given_as_a_string(tmp_path, cards):
    with pytest.raises(TimeFValidationError) as caught:
        timenet.build(CHILD.dataset_id, store=tmp_path / "store", parents_from="timenet://")
    assert str(caught.value) == "parents_from takes a list of sources. Pass parents_from=['timenet://']."


def test_parents_from_names_the_item_that_cannot_be_opened(tmp_path, cards):
    with pytest.raises(TimeNetRegistryError) as caught:
        timenet.plan_build(CHILD.dataset_id, store=tmp_path / "store", parents_from=["timenet://", "foo://x"])
    assert str(caught.value) == (
        "parents_from[1] 'foo://x' has an unsupported registry scheme. "
        "Use an http(s)://, s3://, timenet:// or file:// URL, or a local path."
    )


def test_parents_from_registries_cache_in_the_given_store(tmp_path, monkeypatch, cards):
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(tmp_path / "env"))
    asked = []

    def resolve(self, dataset_id, version):
        asked.append((dataset_id, self.local_root))
        return NotFound()

    # No network: the remote registry records where it looks for complete copies, and lacks the parent.
    monkeypatch.setattr(RemoteRegistry, "resolve", resolve)
    with pytest.raises(TimeNetMissingParentError):
        timenet.plan_build(CHILD.dataset_id, store=tmp_path / "store", parents_from=["https://registry.example.test"])
    assert asked == [(PARENT.dataset_id, tmp_path / "store")]


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Name the store with the environment, and return it empty."""
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(tmp_path / "store"))
    return LocalRegistry(tmp_path / "store")


def test_publish_copies_the_latest_release_of_the_store(tmp_path, store):
    store.put_dataset(_demo(_card(DatasetRef.parse("timenet/hello-world@0.9.0"))))
    store.put_dataset(make_dataset())
    dest = tmp_path / "dest"

    assert timenet.publish("timenet/hello-world", dest=dest) == DatasetRef.parse("timenet/hello-world@1.0.0")
    assert LocalRegistry(dest).exists("timenet/hello-world", "1.0.0")
    assert not LocalRegistry(dest).exists("timenet/hello-world", "0.9.0")


def test_publish_takes_a_source(tmp_path, store):
    origin = LocalRegistry(tmp_path / "origin")
    origin.put_dataset(make_dataset())
    dest = LocalRegistry(tmp_path / "dest")
    events = []

    assert (
        timenet.publish("timenet/hello-world@1.0.0", dest=dest, source=str(origin.root), progress=events.append)
        == HELLO
    )
    assert dest.exists(HELLO.dataset_id, "1.0.0")
    assert not store.root.exists()
    assert [type(event).__name__ for event in events if type(event).__name__ != "BytesMoved"] == [
        "StepStarted",
        "StepFinished",
    ]


def test_publish_of_a_release_that_the_source_lacks_names_the_source(tmp_path):
    source = LocalRegistry(tmp_path / "source")
    for wanted in ("timenet/hello-world", "timenet/hello-world@1.0.0"):
        with pytest.raises(TimeNetDatasetNotFoundError) as caught:
            timenet.publish(wanted, dest=tmp_path / "dest", source=source)
        assert type(caught.value) is TimeNetDatasetNotFoundError
        assert str(caught.value) == f"{wanted} is not in local registry {source.root}."
    assert not (tmp_path / "dest").exists()


def test_publish_of_a_listed_release_that_the_source_does_not_serve_raises_the_access_error(tmp_path, monkeypatch):
    # The hint of the error asks the installed connectors, so the test installs none.
    monkeypatch.setattr(builders, "entry_points", lambda group: [])
    card = _card(
        DatasetRef.parse("org/gated@1.0.0"), access=Access.CREDENTIALED, access_url="https://physionet.example/dua"
    )

    class _Hosted(RemoteRegistry):
        def get_manifest(self, dataset_id, version=None):
            return SimpleNamespace(metadata=card, lineage=())

    hosted = _Hosted("http://api.local", transport=httpx.MockTransport(lambda request: httpx.Response(404)))
    with pytest.raises(TimeNetAccessError) as caught:
        timenet.publish("org/gated", dest=tmp_path / "dest", source=hosted)
    assert caught.value.dataset == DatasetRef.parse("org/gated@1.0.0")
    assert not (tmp_path / "dest").exists()


@pytest.mark.parametrize("keyword", ["dest", "source"])
def test_publish_names_a_registry_that_cannot_be_opened(tmp_path, keyword):
    with pytest.raises(TimeNetRegistryError) as caught:
        if keyword == "dest":
            timenet.publish("timenet/hello-world", dest="foo://x")
        else:
            timenet.publish("timenet/hello-world", dest=tmp_path / "dest", source="foo://x")
    assert str(caught.value) == (
        f"{keyword} 'foo://x' has an unsupported registry scheme. "
        "Use an http(s)://, s3://, timenet:// or file:// URL, or a local path."
    )


def test_import_timenet_loads_the_producer_on_first_use():
    code = "\n".join(
        [
            "import sys, timenet",
            "assert {'Plan', 'Step', 'build', 'plan_build', 'publish'} <= set(dir(timenet))",
            "assert 'timenet.client' not in sys.modules",
            "build = timenet.build",
            "assert 'timenet.client' in sys.modules",
            "assert build is sys.modules['timenet.producer'].build",
            "assert timenet.Plan is sys.modules['timenet.coordinator.plan'].Plan",
            "try:",
            "    timenet.nothing",
            "except AttributeError as error:",
            "    assert str(error) == \"module 'timenet' has no attribute 'nothing'\", error",
            "else:",
            "    raise AssertionError('timenet.nothing exists')",
        ]
    )
    subprocess.run([sys.executable, "-c", code], check=True)
