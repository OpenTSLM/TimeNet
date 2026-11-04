import shutil

import pytest

from timenet.builders import BuildOptions
from timenet.coordinator import build as build_module
from timenet.coordinator.build import plan_release_build, run_plan
from timenet.coordinator.plan import Step
from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetAccessError, TimeNetDatasetNotFoundError, TimeNetVersionUnavailableError
from timenet.progress import StepFinished, StepStarted
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.registry._paths import release_dir
from timenet.registry.shared import complete_copy
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
def store(tmp_path):
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


@pytest.fixture
def pins(store):
    """Store the child without the parent, and return the pins of the child on the files of the parent."""
    return _store_child_without_parent(store)


def test_a_planned_build_runs_into_the_store(store, fake):
    plan = plan_release_build("a/b", store=store)
    run_plan(plan, store)
    assert [(s.action, s.release) for s in plan.steps] == [("build", REF_AB)]
    assert store.exists("a/b", "1.0.0")
    assert fake.calls == [(REF_AB, False, {})]


def test_a_build_calls_the_claiming_backend_with_the_pins(store, fake_parent, fake_child, pins):
    plan = plan_release_build(CHILD_REF.dataset_id, store=store, parents_from=[InstalledConnectors()])
    run_plan(plan, store)
    # (release, force, pinned_by): the backend of another package gets the pins of the child.
    assert fake_parent.calls == [(PARENT_REF, False, {CHILD_REF: pins})]
    # The store holds the child, so the plan builds only its missing payload source.
    assert fake_child.calls == []
    assert plan.steps == (Step(PARENT_REF, "build", InstalledConnectors(), pinned_by={CHILD_REF: pins}),)
    assert store.exists(PARENT_REF.dataset_id, str(PARENT_REF.version))


def test_a_build_passes_the_pins_of_the_caller_to_the_backend(store, fake_parent, pins):
    plan = plan_release_build(PARENT_REF.dataset_id, store=store, pinned_by={CHILD_REF: pins})
    run_plan(plan, store)
    assert fake_parent.calls == [(PARENT_REF, False, {CHILD_REF: pins})]
    assert plan.steps == (Step(PARENT_REF, "build", InstalledConnectors(), pinned_by={CHILD_REF: pins}),)
    assert store.exists(PARENT_REF.dataset_id, str(PARENT_REF.version))


def test_plan_release_build_rejects_a_version_the_connector_does_not_declare(store, fake):
    with pytest.raises(TimeNetVersionUnavailableError) as caught:
        plan_release_build("a/b@2.0.0", store=store)
    assert (caught.value.release, caught.value.declared) == (DatasetRef.parse("a/b@2.0.0"), REF_AB.version)
    assert str(caught.value) == (
        "connector for 'a/b' builds version 1.0.0, not the requested 2.0.0. "
        "Ask for version 1.0.0, or leave out the version."
    )
    assert fake.calls == []


def test_a_build_accepts_the_version_that_the_connector_declares(store, fake):
    run_plan(plan_release_build("a/b@1.0.0", store=store), store)
    assert fake.calls == [(REF_AB, False, {})]


def test_a_build_copies_a_missing_parent_and_reports_each_step_once(tmp_path, store, fake_parent, fake_child):
    origin = LocalRegistry(tmp_path / "origin")
    origin.put_dataset(make_dataset())
    events = []

    plan = plan_release_build(CHILD_REF.dataset_id, store=store, parents_from=[origin])
    run_plan(plan, store, options=BuildOptions(progress=events.append))

    assert plan.steps == (
        Step(PARENT_REF, "copy", origin),
        Step(CHILD_REF, "build", InstalledConnectors(), depends_on=(PARENT_REF,)),
    )
    assert events == [
        StepStarted(PARENT_REF, "copy"),
        StepFinished(PARENT_REF, "copy"),
        StepStarted(CHILD_REF, "build"),
        StepFinished(CHILD_REF, "build"),
    ]
    assert fake_parent.calls == []
    assert store.exists(PARENT_REF.dataset_id, str(PARENT_REF.version))


@pytest.mark.usefixtures("fake_child")
@pytest.mark.parametrize("payload_source", ["missing", "damaged"])
def test_a_build_reuses_a_held_release_and_copies_only_its_payload_source(
    tmp_path, store, fake_parent, pins, payload_source
):
    # The store holds the child, but not an intact copy of its payload source. The registry to copy
    # from holds only the payload source.
    origin = LocalRegistry(tmp_path / "origin")
    origin.put_dataset(make_dataset())
    if payload_source == "damaged":
        store.put_dataset(make_dataset())
        (release_dir(store.root, PARENT_REF) / "control.duckdb").write_bytes(b"cut short")
    events = []

    plan = plan_release_build(CHILD_REF.dataset_id, store=store, parents_from=[origin])
    run_plan(plan, store, options=BuildOptions(progress=events.append))

    # The payload source gets its own copy step. The store keeps its copy of the child.
    assert plan.steps == (Step(PARENT_REF, "copy", origin),)
    assert events == [StepStarted(PARENT_REF, "copy"), StepFinished(PARENT_REF, "copy")]
    assert fake_parent.calls == []
    assert complete_copy(release_dir(store.root, PARENT_REF), origin.get_manifest(PARENT_REF.dataset_id))


def test_a_build_copies_a_release_whose_payload_source_only_the_store_holds(
    tmp_path, store, fake_parent, fake_child, backends
):
    # The registry to copy from holds the child, but not the parent whose files the child reads.
    origin = LocalRegistry(tmp_path / "origin")
    _store_child_without_parent(origin)
    store.put_dataset(make_dataset())
    fake_grandchild = _Backend(_grandchild_card())
    backends.append(fake_grandchild)

    plan = plan_release_build(GRANDCHILD_REF.dataset_id, store=store, parents_from=[origin])
    run_plan(plan, store)

    assert plan.steps == (
        Step(CHILD_REF, "copy", origin),
        Step(GRANDCHILD_REF, "build", InstalledConnectors(), depends_on=(CHILD_REF,)),
    )
    assert fake_parent.calls == fake_child.calls == []
    assert fake_grandchild.calls == [(GRANDCHILD_REF, False, {})]
    assert store.get_manifest(CHILD_REF.dataset_id) == origin.get_manifest(CHILD_REF.dataset_id)


def test_a_build_builds_a_payload_source_of_a_release_that_it_copies(
    tmp_path, store, fake_parent, fake_child, backends
):
    # The registry to copy from holds the child, but not the parent whose files the child reads.
    origin = LocalRegistry(tmp_path / "origin")
    pins = _store_child_without_parent(origin)
    fake_grandchild = _Backend(_grandchild_card())
    backends.append(fake_grandchild)

    plan = plan_release_build(GRANDCHILD_REF.dataset_id, store=store, parents_from=[origin, InstalledConnectors()])
    run_plan(plan, store)

    assert plan.steps == (
        Step(PARENT_REF, "build", InstalledConnectors(), pinned_by={CHILD_REF: pins}),
        Step(CHILD_REF, "copy", origin, depends_on=(PARENT_REF,)),
        Step(GRANDCHILD_REF, "build", InstalledConnectors(), depends_on=(CHILD_REF,)),
    )
    assert fake_parent.calls == [(PARENT_REF, False, {CHILD_REF: pins})]
    assert fake_child.calls == []
    assert fake_grandchild.calls == [(GRANDCHILD_REF, False, {})]
    assert store.get_manifest(CHILD_REF.dataset_id) == origin.get_manifest(CHILD_REF.dataset_id)


def test_a_forced_build_forces_only_the_requested_release(store, fake_parent, fake_child):
    store.put_dataset(_demo(_child_card()))

    plan = plan_release_build(CHILD_REF.dataset_id, store=store, parents_from=[InstalledConnectors()], force=True)
    run_plan(plan, store)

    assert [(s.action, s.release, s.force) for s in plan.steps] == [
        ("build", PARENT_REF, False),
        ("build", CHILD_REF, True),
    ]
    assert fake_parent.calls == [(PARENT_REF, False, {})]
    assert fake_child.calls == [(CHILD_REF, True, {})]


def test_a_build_skips_a_release_the_store_holds(store, fake):
    store.put_dataset(_demo(_card(REF_AB)))
    assert plan_release_build("a/b", store=store).steps == ()


def test_a_damaged_copy_of_the_requested_release_gets_a_forced_build(store, fake):
    store.put_dataset(_demo(_card(REF_AB)))
    (release_dir(store.root, REF_AB) / "control.duckdb").write_bytes(b"cut short")

    plan = plan_release_build("a/b", store=store)

    assert plan.steps == (Step(REF_AB, "build", InstalledConnectors(), force=True),)
    run_plan(plan, store)
    assert fake.calls == [(REF_AB, True, {})]
    assert store.damage(REF_AB) is None


def test_plan_release_build_with_open_access_only_refuses_a_credentialed_card(store, backends):
    fake_credentialed = _Backend(_card(REF_AB, **_GATED))
    backends.append(fake_credentialed)

    with pytest.raises(TimeNetAccessError) as caught:
        plan_release_build("a/b", store=store, open_access_only=True)

    assert caught.value.dataset == REF_AB
    assert fake_credentialed.calls == []
    # Without the flag, the local store takes a release that is not open.
    run_plan(plan_release_build("a/b", store=store), store)
    assert fake_credentialed.calls == [(REF_AB, False, {})]


@pytest.mark.parametrize("held", [False, True])
def test_plan_release_build_with_open_access_only_refuses_a_credentialed_parent(store, backends, held):
    gated = _Backend(make_dataset().metadata.model_copy(update=_GATED))
    child = _Backend(_child_card())
    backends.extend([gated, child])
    if held:
        store.put_dataset(_demo(gated.cards[PARENT_REF.dataset_id]))

    with pytest.raises(TimeNetAccessError) as caught:
        plan_release_build(
            CHILD_REF.dataset_id, store=store, parents_from=[InstalledConnectors()], open_access_only=True
        )

    assert caught.value.dataset == PARENT_REF
    assert gated.calls == child.calls == []


def test_plan_release_build_with_open_access_only_refuses_a_held_release_with_a_credentialed_ancestor(store, backends):
    # The card of the child is open. The child keeps a Record of the open release a/b, which the store
    # lacks. It reads nothing from the credentialed parent, which is only in its lineage.
    child_card = _card(CHILD_REF, parents={ParentAlias("parent"): PARENT_REF, ParentAlias("open"): REF_AB})
    other, child = _Backend(_card(REF_AB)), _Backend(child_card)
    backends.extend([other, child])
    store.put_dataset(_demo(make_dataset().metadata.model_copy(update=_GATED)))
    store.put_dataset(_demo(_card(REF_AB)))
    dataset = TimeFDataset(metadata=child_card)
    with (
        TimeFReader(open_local_version(release_dir(store.root, PARENT_REF))) as gated_reader,
        TimeFReader(open_local_version(release_dir(store.root, REF_AB))) as open_reader,
    ):
        dataset.import_from(gated_reader, alias=ParentAlias("parent"))
        dataset.import_from(open_reader, alias=ParentAlias("open"), records=[make_dataset().records[0].id])
    store.put_dataset(dataset)
    shutil.rmtree(store.root / PARENT_REF.dataset_id)
    shutil.rmtree(release_dir(store.root, REF_AB))
    held = store.get_manifest(CHILD_REF.dataset_id)
    assert [card.source for card in held.lineage] == [PARENT_REF, REF_AB]
    assert list(held.payload_sources) == [SourceKey(str(REF_AB))]

    with pytest.raises(TimeNetAccessError) as caught:
        plan_release_build(
            CHILD_REF.dataset_id, store=store, parents_from=[InstalledConnectors()], open_access_only=True
        )

    assert caught.value.dataset == CHILD_REF
    assert other.calls == child.calls == []


def test_a_plan_runs_only_through_run_plan(store, fake_parent, fake_child):
    plan = plan_release_build(CHILD_REF.dataset_id, store=store, parents_from=[InstalledConnectors()])

    assert plan.steps == (
        Step(PARENT_REF, "build", InstalledConnectors()),
        Step(CHILD_REF, "build", InstalledConnectors(), depends_on=(PARENT_REF,)),
    )
    assert not store.root.exists()
    assert fake_parent.calls == fake_child.calls == []
    run_plan(plan, store)
    assert fake_parent.calls == [(PARENT_REF, False, {})]
    assert fake_child.calls == [(CHILD_REF, False, {})]


@pytest.mark.parametrize("dataset", ["a/b", "a/b@2.0.0"])
def test_plan_release_build_checks_the_connector_before_it_plans(store, backends, monkeypatch, dataset):
    def no_plan(*args, **kwargs):
        raise AssertionError("the checks come before the plan")

    monkeypatch.setattr(build_module, "plan_steps", no_plan)
    if dataset == "a/b":
        with pytest.raises(TimeNetDatasetNotFoundError, match="no installed connector builds 'a/b'"):
            plan_release_build(dataset, store=store)
        return
    backends.append(_Backend(_card(REF_AB)))
    with pytest.raises(TimeNetVersionUnavailableError):
        plan_release_build(dataset, store=store)
