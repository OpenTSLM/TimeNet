import shutil

from _fake_registry import build_service_fake
import pytest

from timenet import builders
from timenet.builders import BuildOptions
from timenet.coordinator.build import plan_release_build, run_plan
from timenet.coordinator.plan import BuildStep, CopyStep, Plan
from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetAccessError, TimeNetRegistryError, TimeNetVersionUnavailableError
from timenet.progress import BytesMoved, StepFinished, StepStarted
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, RemoteRegistry, open_local_version
from timenet.registry._paths import release_dir
from timenet.registry.origin import OriginKind
from timenet.sources import InstalledConnectors
from timenet.testing import install_backends, make_dataset
from timenet.types import Access
from timenet.types.reference_keys import ParentAlias, SourceKey
from timenet.types.references import DatasetRef


REF_AB = DatasetRef.parse("a/b@1.0.0")
PARENT_REF = DatasetRef(dataset_id=make_dataset().metadata.dataset_id, version=make_dataset().metadata.dataset_version)
CHILD_REF = DatasetRef.parse("a/child@1.0.0")
GRANDCHILD_REF = DatasetRef.parse("a/grandchild@1.0.0")
_GATED = {"access": Access.CREDENTIALED, "access_url": "https://example.org/dua"}


def _build(release: DatasetRef, **fields) -> BuildStep:
    """Return the build step of ``release``, with the builder that is installed for it."""
    return BuildStep(release, builders.declared(release.dataset_id), **fields)


def _card(reference, **update):
    """Return the card of the demo dataset under another release."""
    return make_dataset().metadata.model_copy(
        update={"dataset_id": reference.dataset_id, "dataset_version": reference.version, **update}
    )


def _child_card(**update):
    return _card(CHILD_REF, parents={ParentAlias("parent"): PARENT_REF}, **update)


def _grandchild_card():
    return _card(GRANDCHILD_REF, parents={ParentAlias("child"): CHILD_REF})


def _demo(card):
    """Return the demo dataset under ``card``."""
    dataset = make_dataset()
    dataset._metadata = card
    return dataset


class _Backend:
    """A builder backend of another package. It builds the demo dataset under each card that it holds."""

    def __init__(self, *cards):
        self.cards = {card.dataset_id: card for card in cards}
        self.calls = []

    def dataset_ids(self):
        return list(self.cards)

    def card(self, dataset_id):
        return self.cards.get(dataset_id)

    def build(self, release, store, *, force, pinned_by, options):
        self.calls.append((release, force, dict(pinned_by)))
        store.put_dataset(_demo(self.cards[release.dataset_id]), force=force, pinned_by=pinned_by or None)


@pytest.fixture
def backends(monkeypatch):
    """Return the list of the installed builder backends, which a test fills."""
    return install_backends(monkeypatch)


@pytest.fixture
def target_registry(tmp_path):
    return LocalRegistry(tmp_path / "store")


@pytest.fixture
def fake(backends):
    backend = _Backend(_card(REF_AB))
    backends.append(backend)
    return backend


@pytest.fixture
def fake_parent(backends):
    backend = _Backend(make_dataset().metadata)
    backends.append(backend)
    return backend


@pytest.fixture
def fake_child(backends):
    backend = _Backend(_child_card())
    backends.append(backend)
    return backend


def _store_child_without_parent(registry):
    """Store the child, which reads a Record of the parent, without the parent.

    Returns:
        The pins of the child on the files of the parent.
    """
    parent = make_dataset()
    registry.put_dataset(parent)
    child = TimeFDataset(metadata=_child_card())
    with TimeFReader(open_local_version(release_dir(registry.root, PARENT_REF))) as reader:
        child.import_from(reader, alias=ParentAlias("parent"), records=[parent.records[0].id])
    registry.put_dataset(child)
    shutil.rmtree(registry.root / PARENT_REF.dataset_id)
    return registry.get_manifest(CHILD_REF.dataset_id).payload_sources[SourceKey(str(PARENT_REF))]


def test_plan_release_build_rejects_a_version_the_connector_does_not_declare(target_registry, fake):
    with pytest.raises(TimeNetVersionUnavailableError) as caught:
        plan_release_build("a/b@2.0.0", target_registry=target_registry)
    assert (caught.value.release, caught.value.declared) == (DatasetRef.parse("a/b@2.0.0"), REF_AB.version)


def test_a_plan_that_builds_refuses_a_read_only_local_registry_before_any_step(tmp_path, fake):
    target_registry = LocalRegistry(tmp_path / "store", read_only=True)
    with pytest.raises(TimeNetRegistryError, match=r"read_only=True, so it refuses the copies and builds of a plan\."):
        run_plan(Plan((_build(REF_AB),)), target_registry)
    assert fake.calls == []


def test_a_build_copies_a_missing_parent_and_reports_each_step_once(tmp_path, target_registry, fake_parent, fake_child):
    origin = LocalRegistry(tmp_path / "origin")
    origin.put_dataset(make_dataset())
    events = []

    plan = plan_release_build(CHILD_REF.dataset_id, target_registry=target_registry, parents_from=[origin])
    run_plan(plan, target_registry, options=BuildOptions(progress=events.append))

    assert plan.steps == (
        CopyStep(PARENT_REF, origin),
        _build(CHILD_REF, depends_on=(PARENT_REF,)),
    )
    # A copy also sends the bytes that it moves.
    assert [event for event in events if not isinstance(event, BytesMoved)] == [
        StepStarted(PARENT_REF, "copy"),
        StepFinished(PARENT_REF, "copy"),
        StepStarted(CHILD_REF, "build"),
        StepFinished(CHILD_REF, "build"),
    ]
    assert fake_parent.calls == []
    assert target_registry.exists(PARENT_REF.dataset_id, str(PARENT_REF.version))


def test_a_build_copies_a_release_whose_payload_source_only_the_target_registry_holds(
    tmp_path, target_registry, fake_parent, fake_child, backends
):
    # The registry to copy from holds the child, but not the parent whose files the child reads.
    origin = LocalRegistry(tmp_path / "origin")
    _store_child_without_parent(origin)
    target_registry.put_dataset(make_dataset())
    fake_grandchild = _Backend(_grandchild_card())
    backends.append(fake_grandchild)

    plan = plan_release_build(GRANDCHILD_REF.dataset_id, target_registry=target_registry, parents_from=[origin])
    run_plan(plan, target_registry)

    assert plan.steps == (
        CopyStep(CHILD_REF, origin),
        _build(GRANDCHILD_REF, depends_on=(CHILD_REF,)),
    )
    assert fake_parent.calls == fake_child.calls == []
    assert fake_grandchild.calls == [(GRANDCHILD_REF, False, {})]
    assert target_registry.get_manifest(CHILD_REF.dataset_id) == origin.get_manifest(CHILD_REF.dataset_id)


def test_a_build_builds_a_payload_source_of_a_release_that_it_copies(
    tmp_path, target_registry, fake_parent, fake_child, backends
):
    # The registry to copy from holds the child, but not the parent whose files the child reads.
    origin = LocalRegistry(tmp_path / "origin")
    pins = _store_child_without_parent(origin)
    fake_grandchild = _Backend(_grandchild_card())
    backends.append(fake_grandchild)

    plan = plan_release_build(
        GRANDCHILD_REF.dataset_id, target_registry=target_registry, parents_from=[origin, InstalledConnectors()]
    )
    run_plan(plan, target_registry)

    assert plan.steps == (
        _build(PARENT_REF, pinned_by={CHILD_REF: pins}),
        CopyStep(CHILD_REF, origin, depends_on=(PARENT_REF,)),
        _build(GRANDCHILD_REF, depends_on=(CHILD_REF,)),
    )
    assert fake_parent.calls == [(PARENT_REF, False, {CHILD_REF: pins})]
    assert fake_child.calls == []
    assert fake_grandchild.calls == [(GRANDCHILD_REF, False, {})]
    assert target_registry.get_manifest(CHILD_REF.dataset_id) == origin.get_manifest(CHILD_REF.dataset_id)


def test_run_plan_records_each_copy_and_each_build(tmp_path, target_registry, backends):
    # The registry to copy from holds the child and its payload source, the parent.
    origin = LocalRegistry(tmp_path / "origin")
    _store_child_without_parent(origin)
    origin.put_dataset(make_dataset())
    backends.extend([_Backend(_child_card()), _Backend(_grandchild_card())])
    plan = plan_release_build(GRANDCHILD_REF.dataset_id, target_registry=target_registry, parents_from=[origin])
    assert [(step.action, step.release) for step in plan.steps] == [("copy", CHILD_REF), ("build", GRANDCHILD_REF)]

    run_plan(plan, target_registry)

    records = {}
    for release in (PARENT_REF, CHILD_REF, GRANDCHILD_REF):
        record = target_registry.read_origin(release)
        records[release] = None if record is None else (record.kind, record.source)
    # The copy of the child brings its payload source along, and both name the registry. The build
    # names the fake distribution of the fake backend.
    assert records == {
        PARENT_REF: (OriginKind.DOWNLOAD, str(origin.root.resolve())),
        CHILD_REF: (OriginKind.DOWNLOAD, str(origin.root.resolve())),
        GRANDCHILD_REF: (OriginKind.BUILD, "timenet-testing 0"),
    }


def test_a_forced_build_forces_only_the_requested_release(target_registry, fake_parent, fake_child):
    target_registry.put_dataset(_demo(_child_card()))

    plan = plan_release_build(
        CHILD_REF.dataset_id, target_registry=target_registry, parents_from=[InstalledConnectors()], force=True
    )
    run_plan(plan, target_registry)

    assert plan.steps == (_build(PARENT_REF), _build(CHILD_REF, depends_on=(PARENT_REF,), force=True))
    assert fake_parent.calls == [(PARENT_REF, False, {})]
    assert fake_child.calls == [(CHILD_REF, True, {})]


def test_a_build_takes_a_parent_that_the_target_registry_holds_and_asks_a_later_registry_nothing(
    tmp_path, target_registry, fake_parent, fake_child
):
    target_registry.put_dataset(make_dataset())
    origin = LocalRegistry(tmp_path / "origin")
    origin.put_dataset(make_dataset())
    transport, requests = build_service_fake(origin.root)
    remote = RemoteRegistry("http://api.local", transport=transport)

    plan = plan_release_build(
        CHILD_REF.dataset_id, target_registry=target_registry, parents_from=[target_registry, remote]
    )
    run_plan(plan, target_registry)

    assert plan.steps == (_build(CHILD_REF),)
    assert requests == []


def test_a_build_skips_a_release_the_target_registry_holds(target_registry, fake):
    target_registry.put_dataset(_demo(_card(REF_AB)))
    assert plan_release_build("a/b", target_registry=target_registry).steps == ()


def test_a_damaged_copy_of_the_requested_release_gets_a_forced_build(target_registry, fake):
    target_registry.put_dataset(_demo(_card(REF_AB)))
    (release_dir(target_registry.root, REF_AB) / "control.duckdb").write_bytes(b"cut short")

    plan = plan_release_build("a/b", target_registry=target_registry)

    assert plan.steps == (_build(REF_AB, force=True),)
    run_plan(plan, target_registry)
    assert fake.calls == [(REF_AB, True, {})]
    assert target_registry.damage(REF_AB) is None


def test_plan_release_build_with_open_access_only_refuses_a_credentialed_card(target_registry, backends):
    fake_credentialed = _Backend(_card(REF_AB, **_GATED))
    backends.append(fake_credentialed)

    with pytest.raises(TimeNetAccessError) as caught:
        plan_release_build("a/b", target_registry=target_registry, open_access_only=True)

    assert caught.value.dataset == REF_AB
    # Without the flag, the target registry takes a release that is not open.
    run_plan(plan_release_build("a/b", target_registry=target_registry), target_registry)
    assert fake_credentialed.calls == [(REF_AB, False, {})]


def test_plan_release_build_with_open_access_only_refuses_a_held_release_with_a_credentialed_ancestor(
    target_registry, backends
):
    # The card of the child is open. The child keeps a Record of the open release a/b, which the target registry
    # lacks. It reads nothing from the credentialed parent, which is only in its lineage.
    child_card = _card(CHILD_REF, parents={ParentAlias("parent"): PARENT_REF, ParentAlias("open"): REF_AB})
    other, child = _Backend(_card(REF_AB)), _Backend(child_card)
    backends.extend([other, child])
    target_registry.put_dataset(_demo(make_dataset().metadata.model_copy(update=_GATED)))
    target_registry.put_dataset(_demo(_card(REF_AB)))
    dataset = TimeFDataset(metadata=child_card)
    with (
        TimeFReader(open_local_version(release_dir(target_registry.root, PARENT_REF))) as gated_reader,
        TimeFReader(open_local_version(release_dir(target_registry.root, REF_AB))) as open_reader,
    ):
        dataset.import_from(gated_reader, alias=ParentAlias("parent"))
        dataset.import_from(open_reader, alias=ParentAlias("open"), records=[make_dataset().records[0].id])
    target_registry.put_dataset(dataset)
    shutil.rmtree(target_registry.root / PARENT_REF.dataset_id)
    shutil.rmtree(release_dir(target_registry.root, REF_AB))
    held = target_registry.get_manifest(CHILD_REF.dataset_id)
    assert [card.source for card in held.lineage] == [PARENT_REF, REF_AB]
    assert list(held.payload_sources) == [SourceKey(str(REF_AB))]

    with pytest.raises(TimeNetAccessError) as caught:
        plan_release_build(
            CHILD_REF.dataset_id,
            target_registry=target_registry,
            parents_from=[InstalledConnectors()],
            open_access_only=True,
        )

    assert caught.value.dataset == CHILD_REF


def test_a_plan_runs_only_through_run_plan(target_registry, fake_parent, fake_child):
    plan = plan_release_build(
        CHILD_REF.dataset_id, target_registry=target_registry, parents_from=[InstalledConnectors()]
    )

    assert plan.steps == (
        _build(PARENT_REF),
        _build(CHILD_REF, depends_on=(PARENT_REF,)),
    )
    assert not target_registry.root.exists()
    assert fake_parent.calls == fake_child.calls == []
    run_plan(plan, target_registry)
    assert fake_parent.calls == [(PARENT_REF, False, {})]
    assert fake_child.calls == [(CHILD_REF, False, {})]


def test_a_plan_reads_the_card_of_each_connector_once(target_registry, backends):
    reads = []

    class _Counted(_Backend):
        def card(self, dataset_id):
            reads.append(dataset_id)
            return super().card(dataset_id)

    backends.extend([_Counted(make_dataset().metadata), _Counted(_child_card())])
    plan = plan_release_build(
        CHILD_REF.dataset_id, target_registry=target_registry, parents_from=[InstalledConnectors()]
    )
    run_plan(plan, target_registry)
    assert reads == [CHILD_REF.dataset_id, PARENT_REF.dataset_id]
    assert target_registry.exists(CHILD_REF.dataset_id, str(CHILD_REF.version))
