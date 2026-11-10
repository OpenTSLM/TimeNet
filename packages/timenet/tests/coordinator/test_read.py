from pathlib import Path
import re
import shutil

from _demo_child import _child_of_the_demo
from _fake_registry import build_fake, build_service_fake
import httpx
import numpy as np
import pytest

from timenet import builders
from timenet.coordinator.plan import BuildStep, CopyStep, ReadStep
from timenet.coordinator.read import plan_read, read_release
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import (
    TimeFFormatError,
    TimeFValidationError,
    TimeNetAccessError,
    TimeNetDatasetNotFoundError,
    TimeNetFileError,
    TimeNetMissingParentError,
    TimeNetVersionUnavailableError,
)
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, RemoteRegistry, open_local_version
from timenet.registry._paths import release_dir
from timenet.sources import InstalledConnectors, ReleaseSource
from timenet.testing import install_backends, make_dataset
from timenet.types import Access, TimeSeriesSpec, ureg
from timenet.types.access import AccessRequirement
from timenet.types.reference_keys import ParentAlias, SourceKey
from timenet.types.references import DatasetRef


DEMO = DatasetRef.parse("timenet/hello-world@1.0.0")
CHILD = DatasetRef.parse("test/child@1.0.0")
PARENT = DatasetRef.parse("test/parent@1.0.0")
OTHER = DatasetRef.parse("test/other@1.0.0")
_GATED = {"access": Access.CREDENTIALED, "access_url": "https://example.org/dua"}


def _build(release: DatasetRef, **fields) -> BuildStep:
    """Return the build step of ``release``, with the builder that is installed for it."""
    return BuildStep(release, builders.declared(release.dataset_id), **fields)


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
    """A builder backend of another package.

    It builds CHILD as ``_child_of_the_demo()`` does, and the demo dataset under each other card. A
    build hands its pins to the writer, which publishes the release only if it reproduces them.
    """

    def __init__(self, cards, calls):
        self.cards = {card.dataset_id: card for card in cards}
        self.calls = calls

    def dataset_ids(self):
        return list(self.cards)

    def card(self, dataset_id):
        return self.cards.get(dataset_id)

    def build(self, release, store, *, force, pinned_by, options):
        self.calls.append((release, force, dict(pinned_by)))
        if release == CHILD:
            _child_of_the_demo(store.root)
        else:
            store.put_dataset(_demo(self.cards[release.dataset_id]), force=force, pinned_by=pinned_by or None)


def _install_demo_backend(monkeypatch, *cards) -> list:
    """Make one backend the only installed one. It builds the demo release, and a release for each card.

    Returns:
        The list that receives ``(release, force, pinned_by)`` for each build.
    """
    calls = []
    backend = _Backend((make_dataset().metadata, *cards), calls)
    install_backends(monkeypatch, backend)
    return calls


def _child_reading(store, *cards, child_release=CHILD):
    """Store a child, which reads one Record of the demo dataset under each card, without those releases.

    Args:
        store: The registry to store the child in.
        *cards: The cards of the releases that the child reads.
        child_release: The release of the child.

    Returns:
        The pins of the child on each release that it reads, in the order of its manifest.
    """
    parents = {
        ParentAlias(f"parent{index}"): DatasetRef(dataset_id=card.dataset_id, version=card.dataset_version)
        for index, card in enumerate(cards)
    }
    child = TimeFDataset(metadata=_card(child_release, parents=parents))
    for (alias, parent), card in zip(parents.items(), cards, strict=True):
        store.put_dataset(_demo(card))
        with TimeFReader(open_local_version(release_dir(store.root, parent))) as reader:
            child.import_from(reader, alias=alias, records=[make_dataset().records[0].id])
    store.put_dataset(child)
    for parent in parents.values():
        shutil.rmtree(release_dir(store.root, parent))
    manifest = store.get_manifest(child_release.dataset_id)
    return {pinned.release: pinned for pinned in manifest.payload_sources.values()}


def _gated_remote(tmp_path) -> RemoteRegistry:
    """Return a remote registry that lists DEMO as credentialed, so it does not serve its files."""
    origin = LocalRegistry(tmp_path / "origin")
    origin.put_dataset(_demo(make_dataset().metadata.model_copy(update=_GATED)))
    transport, _ = build_fake(release_dir(origin.root, DEMO))
    return RemoteRegistry("http://api.local", transport=transport, cache_dir=tmp_path / "cache")


def _offline(request):
    raise httpx.ConnectError("the network is down", request=request)


def _count_reads(monkeypatch, registry) -> dict[str, int]:
    """Count each manifest that is read from a registry, by the release that it was asked for."""
    reads, get_manifest = {}, registry.get_manifest

    def counted(dataset_id, version=None):
        reads[f"{dataset_id}@{version}"] = reads.get(f"{dataset_id}@{version}", 0) + 1
        return get_manifest(dataset_id, version)

    monkeypatch.setattr(registry, "get_manifest", counted)
    return reads


def test_a_read_reads_in_place_from_the_first_source_that_holds_the_release(tmp_path):
    store, origin = LocalRegistry(tmp_path / "store"), LocalRegistry(tmp_path / "origin")
    origin.put_dataset(make_dataset())
    plan = plan_read(DEMO.dataset_id, None, sources=[store, origin], store=store)
    assert plan.steps == (ReadStep(DEMO, origin),)
    version = read_release(DEMO.dataset_id, None, sources=[store, origin], store=store)
    assert Path(version.root).is_relative_to(origin.root)
    assert not store.root.exists()


def test_a_damaged_copy_fails_the_read_before_any_step_runs(tmp_path, monkeypatch):
    store, _ = _child_of_the_demo(tmp_path / "store")
    # No registry holds DEMO, so the plan builds it before the read opens CHILD.
    shutil.rmtree(release_dir(store.root, DEMO))
    # Origin and the connector also provide CHILD, but the read stops at the damaged copy, which comes first.
    origin = LocalRegistry(tmp_path / "origin")
    shutil.copytree(store.root, origin.root)
    (release_dir(store.root, CHILD) / "control.duckdb").write_bytes(b"cut short")
    calls = _install_demo_backend(monkeypatch, _card(CHILD, parents={ParentAlias("parent"): DEMO}))
    with pytest.raises(TimeNetFileError, match=r"^test/child@1\.0\.0: file 'control\.duckdb' has 9 bytes"):
        read_release(CHILD.dataset_id, None, sources=[store, origin, InstalledConnectors()], store=store)
    assert calls == []


@pytest.mark.parametrize("answer", ["build", "not-found", "not-served"])
def test_a_damaged_copy_in_the_store_fails_a_read_that_no_registry_serves(tmp_path, monkeypatch, answer):
    # The store is not in the list, so no registry answers for its damaged copy.
    store = LocalRegistry(tmp_path / "store")
    store.put_dataset(make_dataset())
    (release_dir(store.root, DEMO) / "control.duckdb").write_bytes(b"cut short")
    _install_demo_backend(monkeypatch)
    sources: list[ReleaseSource]
    if answer == "build":
        sources = [InstalledConnectors()]
    elif answer == "not-found":
        sources = [LocalRegistry(tmp_path / "empty")]
    else:
        # The damage error comes before the access error of the registry that lists DEMO.
        sources = [_gated_remote(tmp_path)]
    with pytest.raises(TimeNetFileError, match=r"^timenet/hello-world@1\.0\.0: file 'control\.duckdb' has 9 bytes"):
        plan_read(DEMO.dataset_id, "1.0.0", sources=sources, store=store)


def test_a_payload_source_resolves_on_its_own(tmp_path):
    store, _ = _child_of_the_demo(tmp_path / "store")
    shutil.rmtree(release_dir(store.root, DEMO))
    origin = LocalRegistry(tmp_path / "origin")
    origin.put_dataset(make_dataset())
    plan = plan_read(CHILD.dataset_id, None, sources=[store, origin], store=store)
    assert plan.steps == (ReadStep(DEMO, origin), ReadStep(CHILD, store, depends_on=(DEMO,)))
    version = read_release(CHILD.dataset_id, None, sources=[store, origin], store=store)
    assert Path(version.root).is_relative_to(store.root)
    assert Path(version.payload_sources[SourceKey(str(DEMO))].root).is_relative_to(origin.root)
    assert not store.exists(DEMO.dataset_id, str(DEMO.version))


def test_a_payload_source_with_other_files_than_the_pins_fails_the_plan(tmp_path):
    store, _ = _child_of_the_demo(tmp_path / "store")
    shutil.rmtree(release_dir(store.root, DEMO))
    origin = LocalRegistry(tmp_path / "origin")
    # The same release with its values in another format: CHILD pins files that it does not have.
    origin.put_dataset(make_dataset(), values_backend="zarr")
    with pytest.raises(TimeFFormatError, match=r"^payload source timenet/hello-world@1\.0\.0 differs"):
        plan_read(CHILD.dataset_id, None, sources=[store, origin], store=store)


def test_a_missing_payload_source_builds_against_the_pins_of_the_release(tmp_path, monkeypatch):
    store, _ = _child_of_the_demo(tmp_path / "store")
    shutil.rmtree(release_dir(store.root, DEMO))
    pins = store.get_manifest(CHILD.dataset_id).payload_sources[SourceKey(str(DEMO))]
    calls = _install_demo_backend(monkeypatch)
    sources = [store, InstalledConnectors()]
    assert plan_read(CHILD.dataset_id, None, sources=sources, store=store).steps == (
        _build(DEMO, pinned_by={CHILD: pins}),
        ReadStep(CHILD, store, depends_on=(DEMO,)),
    )
    version = read_release(CHILD.dataset_id, None, sources=sources, store=store)
    assert calls == [(DEMO, False, {CHILD: pins})]
    assert Path(version.payload_sources[SourceKey(str(DEMO))].root).is_relative_to(store.root)


def test_a_read_builds_a_release_that_only_a_connector_provides(tmp_path, monkeypatch):
    store = LocalRegistry(tmp_path / "store")
    store.put_dataset(make_dataset())
    calls = _install_demo_backend(monkeypatch, _card(CHILD, parents={ParentAlias("parent"): DEMO}))
    sources = [store, InstalledConnectors()]
    # The build of CHILD reads DEMO, which the store holds, so DEMO gets no step.
    plan = plan_read(CHILD.dataset_id, None, sources=sources, store=store)
    assert plan.steps == (_build(CHILD),)
    version = read_release(CHILD.dataset_id, None, sources=sources, store=store)
    assert calls == [(CHILD, False, {})]
    assert Path(version.root).is_relative_to(store.root)
    # A payload source without a step opens from the store.
    assert Path(version.payload_sources[SourceKey(str(DEMO))].root).is_relative_to(store.root)


@pytest.mark.parametrize(("version", "release"), [(None, "x/y"), ("1.0.0", "x/y@1.0.0")])
def test_a_release_that_no_source_provides_names_every_source(tmp_path, monkeypatch, version, release):
    store = LocalRegistry(tmp_path / "store")
    _install_demo_backend(monkeypatch)
    with pytest.raises(TimeNetDatasetNotFoundError) as caught:
        plan_read("x/y", version, sources=[store, InstalledConnectors()], store=store)
    assert type(caught.value) is TimeNetDatasetNotFoundError
    assert str(caught.value) == f"{release} is in none of the sources: {store}, installed connectors."


def test_a_missing_payload_source_names_the_version_that_the_connector_builds(tmp_path, monkeypatch):
    origin, _ = _child_of_the_demo(tmp_path / "origin")
    shutil.rmtree(release_dir(origin.root, DEMO))
    _install_demo_backend(monkeypatch, _card(DatasetRef.parse(f"{DEMO.dataset_id}@2.0.0")))
    with pytest.raises(TimeNetMissingParentError) as caught:
        plan_read(CHILD.dataset_id, None, sources=[origin, InstalledConnectors()], store=None)
    assert str(caught.value) == (
        f"payload source {DEMO} of {CHILD} is not in {origin} or installed connectors ({DEMO.dataset_id} builds 2.0.0)."
    )


def test_pins_that_the_store_copy_of_a_payload_source_breaks_fail_the_read_before_any_step(tmp_path, monkeypatch):
    # Origin holds CHILD, which reads OTHER and DEMO. Only the connectors provide them. The store
    # holds DEMO with its values in another format, so the plan builds nothing that checks the pins
    # of CHILD on DEMO. The plan builds OTHER first.
    origin, store = LocalRegistry(tmp_path / "origin"), LocalRegistry(tmp_path / "store")
    _child_reading(origin, _card(OTHER), make_dataset().metadata)
    store.put_dataset(make_dataset(), values_backend="zarr")
    calls = _install_demo_backend(monkeypatch, _card(OTHER))
    with pytest.raises(TimeFFormatError, match=r"^payload source timenet/hello-world@1\.0\.0 differs"):
        read_release(CHILD.dataset_id, None, sources=[origin, InstalledConnectors()], store=store)
    assert calls == []


def test_a_release_that_a_source_lists_without_serving_raises_the_access_error(tmp_path):
    with pytest.raises(TimeNetAccessError) as caught:
        read_release(DEMO.dataset_id, None, sources=[_gated_remote(tmp_path)], store=None)
    assert caught.value.requirements == (AccessRequirement(DEMO, Access.CREDENTIALED, _GATED["access_url"]),)


def test_an_exact_version_that_the_connector_does_not_declare_raises_version_unavailable(tmp_path, monkeypatch):
    store = LocalRegistry(tmp_path / "store")
    _install_demo_backend(monkeypatch)
    with pytest.raises(TimeNetVersionUnavailableError) as caught:
        read_release(DEMO.dataset_id, "2.0.0", sources=[store, InstalledConnectors()], store=store)
    assert (caught.value.release, caught.value.declared) == (DatasetRef.parse(f"{DEMO.dataset_id}@2.0.0"), DEMO.version)


def test_latest_without_the_connectors_in_the_list_reads_the_older_copy_and_builds_nothing(tmp_path, monkeypatch):
    store = LocalRegistry(tmp_path / "store")
    store.put_dataset(make_dataset())
    # The installed connector builds a newer version, but the list does not hold the connectors.
    calls = _install_demo_backend(monkeypatch, _card(DatasetRef.parse(f"{DEMO.dataset_id}@2.0.0")))
    assert plan_read(DEMO.dataset_id, "latest", sources=[store], store=store).steps == (ReadStep(DEMO, store),)
    version = read_release(DEMO.dataset_id, "latest", sources=[store], store=store)
    assert version.manifest.metadata.dataset_version == DEMO.version
    assert calls == []


def test_a_read_without_a_version_stops_at_the_first_source_that_serves_it(tmp_path, monkeypatch):
    store = LocalRegistry(tmp_path / "store")
    store.put_dataset(make_dataset())
    # A later registry cannot be reached, and the installed connector builds a newer version.
    down = RemoteRegistry("http://down.local", transport=httpx.MockTransport(_offline), cache_dir=tmp_path / "c")
    calls = _install_demo_backend(monkeypatch, _card(DatasetRef.parse(f"{DEMO.dataset_id}@2.0.0")))

    version = read_release(DEMO.dataset_id, None, sources=[store, down, InstalledConnectors()], store=store)

    assert version.manifest.metadata.dataset_version == DEMO.version
    assert calls == []


def test_a_read_that_builds_reads_the_card_of_the_connector_once(tmp_path, monkeypatch):
    reads = []

    class _Counted(_Backend):
        def card(self, dataset_id):
            reads.append(dataset_id)
            return super().card(dataset_id)

    install_backends(monkeypatch, _Counted((make_dataset().metadata,), []))
    plan_read(DEMO.dataset_id, None, sources=[InstalledConnectors()], store=LocalRegistry(tmp_path / "store"))
    assert reads == [DEMO.dataset_id]


def test_a_read_that_finds_every_release_asks_the_connectors_nothing(tmp_path, monkeypatch):
    asked = []

    class _Counted(_Backend):
        def dataset_ids(self):
            asked.append("dataset_ids")
            return super().dataset_ids()

        def card(self, dataset_id):
            asked.append(dataset_id)
            return super().card(dataset_id)

    install_backends(monkeypatch, _Counted((make_dataset().metadata,), []))
    store, _ = _child_of_the_demo(tmp_path / "store")
    read_release(CHILD.dataset_id, None, sources=[store, InstalledConnectors()], store=store)
    # Only the error for a missing payload source names the sources, with the versions that the connectors build.
    assert asked == []


def test_a_read_that_needs_a_build_without_a_store_raises(monkeypatch):
    _install_demo_backend(monkeypatch)
    with pytest.raises(TimeFValidationError, match=r"timenet/hello-world@1\.0\.0 needs a build"):
        read_release(DEMO.dataset_id, None, sources=[InstalledConnectors()], store=None)


@pytest.mark.parametrize("parent", [PARENT, DatasetRef.parse("x/parent@1.0.0")], ids=["build-first", "read-first"])
def test_a_release_that_a_build_copies_is_read_from_the_store_and_appears_once(tmp_path, monkeypatch, parent):
    # CHILD reads the parent and DEMO. Only the connectors build the parent, and its card declares
    # DEMO, which origin holds. So the build of the parent copies DEMO into the store, and the read
    # takes it there. The read plans the payload sources in the order of their names.
    parent_card = _card(parent, parents={ParentAlias("demo"): DEMO})
    store, origin = LocalRegistry(tmp_path / "store"), LocalRegistry(tmp_path / "origin")
    pins = _child_reading(store, parent_card, make_dataset().metadata)
    origin.put_dataset(make_dataset())
    calls = _install_demo_backend(monkeypatch, parent_card)
    sources = [store, origin, InstalledConnectors()]
    reads = _count_reads(monkeypatch, origin)

    plan = plan_read(CHILD.dataset_id, None, sources=sources, store=store)

    # The read and the build of the parent share what the plan read.
    assert reads == {str(DEMO): 1}
    assert plan.steps == (
        CopyStep(DEMO, origin),
        _build(parent, depends_on=(DEMO,), pinned_by={CHILD: pins[parent]}),
        ReadStep(CHILD, store, depends_on=tuple(pins)),
    )
    version = read_release(CHILD.dataset_id, None, sources=sources, store=store)
    assert calls == [(parent, False, {CHILD: pins[parent]})]
    assert Path(version.payload_sources[SourceKey(str(DEMO))].root).is_relative_to(store.root)


def test_a_step_that_two_builds_need_appears_once(tmp_path, monkeypatch):
    # CHILD reads PARENT and OTHER. Only the connectors build them, and both of their cards declare
    # DEMO, which origin holds. So both builds copy DEMO first.
    cards = [_card(reference, parents={ParentAlias("demo"): DEMO}) for reference in (PARENT, OTHER)]
    store, origin = LocalRegistry(tmp_path / "store"), LocalRegistry(tmp_path / "origin")
    pins = _child_reading(store, *cards)
    origin.put_dataset(make_dataset())
    _install_demo_backend(monkeypatch, *cards)

    plan = plan_read(CHILD.dataset_id, None, sources=[store, origin, InstalledConnectors()], store=store)

    assert [(step.release, step.action) for step in plan.steps] == [
        (DEMO, "copy"),
        *((reference, "build") for reference in pins),
        (CHILD, "read"),
    ]


@pytest.mark.parametrize(
    "parent", [PARENT, DatasetRef.parse("a/parent@1.0.0")], ids=["pinned-build-first", "unpinned-build-first"]
)
def test_two_builds_of_one_release_merge_into_one_build_with_the_pins_of_each(tmp_path, monkeypatch, parent):
    # CHILD reads the parent and OTHER, and the card of the parent declares OTHER. So the build of
    # the parent builds OTHER without pins, and the read builds OTHER with the pins of CHILD. The read
    # plans the payload sources in the order of their names.
    cards = [_card(parent, parents={ParentAlias("other"): OTHER}), _card(OTHER)]
    store = LocalRegistry(tmp_path / "store")
    pins = _child_reading(store, *cards)
    calls = _install_demo_backend(monkeypatch, *cards)
    sources = [store, InstalledConnectors()]

    plan = plan_read(CHILD.dataset_id, None, sources=sources, store=store)

    # The one build of OTHER comes before the build of the parent, which needs it.
    assert plan.steps == (
        _build(OTHER, pinned_by={CHILD: pins[OTHER]}),
        _build(parent, depends_on=(OTHER,), pinned_by={CHILD: pins[parent]}),
        ReadStep(CHILD, store, depends_on=tuple(pins)),
    )
    version = read_release(CHILD.dataset_id, None, sources=sources, store=store)
    assert calls == [(OTHER, False, {CHILD: pins[OTHER]}), (parent, False, {CHILD: pins[parent]})]
    assert Path(version.payload_sources[SourceKey(str(OTHER))].root).is_relative_to(store.root)


def test_two_readers_that_pin_one_file_differently_fail_the_read_before_any_step(tmp_path, monkeypatch):
    # CHILD reads PARENT and OTHER. The card of PARENT declares X, which the store holds, and which
    # reads OTHER too. So the read builds OTHER with the pins of CHILD, and the build of PARENT builds
    # OTHER with the pins of X. X pins one file of OTHER with another checksum, so no build matches both.
    x = DatasetRef.parse("test/x@1.0.0")
    cards = [_card(PARENT, parents={ParentAlias("x"): x}), _card(OTHER)]
    store = LocalRegistry(tmp_path / "store")
    _child_reading(store, cards[1], child_release=x)
    _child_reading(store, *cards)
    manifest = store.get_manifest(x.dataset_id)
    ((key, pinned),) = manifest.payload_sources.items()
    changed = pinned.files[0].model_copy(update={"checksum": "sha256:" + "0" * 64})
    pinned = pinned.model_copy(update={"files": (changed, *pinned.files[1:])})
    (release_dir(store.root, x) / "manifest.json").write_text(
        manifest.model_copy(update={"payload_sources": {key: pinned}}).to_json()
    )
    # A new registry, so that no manifest that it read before stays.
    store = LocalRegistry(store.root)
    calls = _install_demo_backend(monkeypatch, *cards)
    message = f"{CHILD} and {x} pin {changed.path!r} of {OTHER} as different files"
    with pytest.raises(TimeFFormatError, match=f"^{re.escape(message)}"):
        read_release(CHILD.dataset_id, None, sources=[store, InstalledConnectors()], store=store)
    assert calls == []


def test_a_remote_read_fetches_each_manifest_once(tmp_path, monkeypatch):
    origin, _ = _child_of_the_demo(tmp_path / "origin")
    transport, _ = build_service_fake(origin.root)
    remote = RemoteRegistry("http://api.local", transport=transport, cache_dir=tmp_path / "cache")
    reads = _count_reads(monkeypatch, remote)

    version = read_release(CHILD.dataset_id, None, sources=[remote], store=None)

    assert version.manifest == origin.get_manifest(CHILD.dataset_id)
    assert version.payload_sources[SourceKey(str(DEMO))].manifest == origin.get_manifest(DEMO.dataset_id)
    # The registry answers "latest" with the manifest, and the plan and the open use that manifest.
    assert reads == {f"{CHILD.dataset_id}@None": 1, str(DEMO): 1}


def test_latest_reads_the_manifest_of_the_version_that_the_answer_found(tmp_path, monkeypatch):
    origin = LocalRegistry(tmp_path / "origin")
    origin.put_dataset(make_dataset())
    manifest = origin.get_manifest(DEMO.dataset_id, str(DEMO.version))
    reads = _count_reads(monkeypatch, origin)
    # A local registry answers "latest" without its manifest, so the plan reads it at the exact version.
    version = read_release(DEMO.dataset_id, None, sources=[origin], store=None)
    assert version.manifest == manifest
    assert reads == {str(DEMO): 1}


@pytest.mark.parametrize("parent", [PARENT, DatasetRef.parse("x/parent@1.0.0")], ids=["build-first", "read-first"])
def test_a_copy_that_a_build_needs_repairs_a_damaged_copy_that_the_read_found_first(tmp_path, monkeypatch, parent):
    # CHILD reads the parent and DEMO. The store comes first in the list and holds DEMO damaged, and origin
    # holds DEMO. Only the connectors build the parent, and its card declares DEMO. So the build copies DEMO
    # from origin over the damaged copy, and the read takes it there.
    parent_card = _card(parent, parents={ParentAlias("demo"): DEMO})
    store, origin = LocalRegistry(tmp_path / "store"), LocalRegistry(tmp_path / "origin")
    pins = _child_reading(store, parent_card, make_dataset().metadata)
    for registry in (store, origin):
        registry.put_dataset(make_dataset())
    (release_dir(store.root, DEMO) / "control.duckdb").write_bytes(b"cut short")
    _install_demo_backend(monkeypatch, parent_card)
    sources = [store, origin, InstalledConnectors()]

    plan = plan_read(CHILD.dataset_id, None, sources=sources, store=store)

    assert plan.steps == (
        CopyStep(DEMO, origin),
        _build(parent, depends_on=(DEMO,), pinned_by={CHILD: pins[parent]}),
        ReadStep(CHILD, store, depends_on=tuple(pins)),
    )
    version = read_release(CHILD.dataset_id, None, sources=sources, store=store)
    assert store.damage(DEMO) is None
    assert Path(version.payload_sources[SourceKey(str(DEMO))].root).is_relative_to(store.root)


_SPEC = TimeSeriesSpec(spec_type="sine", name="Sine", unit_value=ureg.dimensionless)


def _store_reading(registry: LocalRegistry, release: DatasetRef, source: DatasetRef) -> None:
    """Store ``release``, which keeps one Record of ``source`` and adds one Record of its own values."""
    dataset = TimeFDataset(metadata=_card(release, parents={ParentAlias("source"): source}))
    with TimeFReader(open_local_version(release_dir(registry.root, source))) as reader:
        dataset.import_from(reader, alias=ParentAlias("source"), records=[make_dataset().records[0].id])
    own = Signal.from_values(
        np.arange(16, dtype=np.float32), spec=_SPEC, name="own", time_axis=RegularAxis.from_rate_hz(16)
    )
    dataset.add_record(record=Record(sources=(Source(name="own", signals=(own,)),)))
    registry.put_dataset(dataset)


def test_a_payload_source_that_a_build_needs_as_a_parent_comes_with_its_own_payload_sources(tmp_path, monkeypatch):
    # CHILD reads the own Record of SOURCE, which reads DEMO, and a Record of the parent. Origin holds SOURCE
    # and DEMO. Only the connectors build the parent, and its card declares SOURCE. The read finds SOURCE in
    # origin first. The build then needs SOURCE in the store, so the copy takes DEMO along.
    source, parent = DatasetRef.parse("test/source@1.0.0"), DatasetRef.parse("x/parent@1.0.0")
    parent_card = _card(parent, parents={ParentAlias("source"): source})
    store, origin = LocalRegistry(tmp_path / "store"), LocalRegistry(tmp_path / "origin")
    store.put_dataset(make_dataset())
    _store_reading(store, source, DEMO)
    store.put_dataset(_demo(parent_card))
    child = TimeFDataset(metadata=_card(CHILD, parents={ParentAlias("source"): source, ParentAlias("parent"): parent}))
    for alias, reference, record in (("source", source, 2), ("parent", parent, make_dataset().records[0].id)):
        with TimeFReader(open_local_version(release_dir(store.root, reference))) as reader:
            child.import_from(reader, alias=ParentAlias(alias), records=[record])
    store.put_dataset(child)
    for reference in (DEMO, source, parent):
        shutil.copytree(release_dir(store.root, reference), release_dir(origin.root, reference))
        shutil.rmtree(release_dir(store.root, reference))
    shutil.rmtree(release_dir(origin.root, parent))
    pins = {pinned.release: pinned for pinned in store.get_manifest(CHILD.dataset_id).payload_sources.values()}
    assert list(pins) == [source, parent]
    _install_demo_backend(monkeypatch, parent_card)
    sources = [store, origin, InstalledConnectors()]

    plan = plan_read(CHILD.dataset_id, None, sources=sources, store=store)

    assert plan.steps == (
        CopyStep(source, origin),
        _build(parent, depends_on=(source,), pinned_by={CHILD: pins[parent]}),
        ReadStep(CHILD, store, depends_on=(source, parent)),
    )
    read_release(CHILD.dataset_id, None, sources=sources, store=store)
    assert store.exists(DEMO.dataset_id, str(DEMO.version))


def test_a_read_of_the_latest_version_asks_each_source_once(tmp_path, monkeypatch):
    # The first registry holds DEMO 1.0.0, and the second holds DEMO 2.0.0.
    newer = DatasetRef.parse(f"{DEMO.dataset_id}@2.0.0")
    registries = []
    for name, dataset in (("first", make_dataset()), ("second", _demo(_card(newer)))):
        LocalRegistry(tmp_path / name).put_dataset(dataset)
        transport, _ = build_service_fake(tmp_path / name)
        registries.append(RemoteRegistry(f"http://{name}.local", transport=transport, cache_dir=tmp_path / f"{name}-c"))
    reads = _count_reads(monkeypatch, registries[0])

    version = read_release(DEMO.dataset_id, "latest", sources=registries, store=None)

    assert version.manifest.metadata.dataset_version == newer.version
    # The plan takes the answer of the sources for the latest version, and asks for 2.0.0 nowhere.
    assert reads == {f"{DEMO.dataset_id}@None": 1}


def test_a_cycle_through_the_requested_release_starts_at_it(tmp_path, monkeypatch):
    # CHILD reads PARENT, which only the connectors build, and the card of PARENT declares CHILD.
    parent_card = _card(PARENT, parents={ParentAlias("child"): CHILD})
    store = LocalRegistry(tmp_path / "store")
    _child_reading(store, parent_card)
    _install_demo_backend(monkeypatch, parent_card)

    message = f"dataset parent cycle: {CHILD} -> {PARENT} -> {CHILD}"
    with pytest.raises(TimeFValidationError, match=f"^{re.escape(message)}$"):
        plan_read(CHILD.dataset_id, None, sources=[store, InstalledConnectors()], store=store)
