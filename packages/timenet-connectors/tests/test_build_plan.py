import json
from pathlib import Path
import shlex
import shutil

import click
import numpy as np
import pytest
from rich.console import Console
from typer.testing import CliRunner

import timenet
from timenet import builders, client as client_module
from timenet.cli import hints as hints_module
from timenet.client import default_store
from timenet.connectors import BaseConnector
from timenet.coordinator.build import plan_release_build, run_plan
from timenet.coordinator.plan import Plan, Step
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.engine import open_parents, run_pipeline
from timenet.errors import (
    TimeFFormatError,
    TimeFValidationError,
    TimeNetAccessError,
    TimeNetDatasetNotFoundError,
    TimeNetFileError,
    TimeNetInvalidCardError,
    TimeNetIsolationError,
    TimeNetMissingParentError,
    TimeNetVersionConflictError,
)
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, WritableRegistry, open_local_version
from timenet.registry._paths import release_dir
from timenet.sources import InstalledConnectors
from timenet.testing import install_backends, make_dataset
from timenet.types import Access, TimeSeriesSpec, ureg
from timenet.types.access import AccessRequirement
from timenet.types.parent_dataset_card import ParentDatasetCard
from timenet.types.reference_keys import ParentAlias, SourceKey
from timenet.types.references import DatasetRef
from timenet.values_backends import ValuesBackend
from timenet_connectors.builder import backend as backend_module, cli as cli_module, env as env_module
from timenet_connectors.builder.backend import ConnectorBuilder
from timenet_connectors.builder.cli import _plan_lines, app


GRAND = DatasetRef(dataset_id=make_dataset().metadata.dataset_id, version=make_dataset().metadata.dataset_version)
PARENT = DatasetRef.parse("test/parent@1.0.0")
CHILD = DatasetRef.parse("test/child@1.0.0")
runner = CliRunner()


def _parent_steps(dataset_id: str, store: LocalRegistry, **options) -> list[Step]:
    """Plan the build of ``dataset_id`` and return each step but its own build."""
    steps = plan_release_build(dataset_id, store=store, **options).steps
    return [step for step in steps if not (step.action == "build" and step.release.dataset_id == dataset_id)]


class _CardBackend:
    """A backend of another package, not ConnectorBuilder: it serves cards by dataset ID, and the plan never builds through it."""

    def __init__(self, cards):
        self.cards = cards

    def dataset_ids(self):
        return list(self.cards)

    def card(self, dataset_id):
        return self.cards.get(dataset_id)

    def build(self, release, store, *, force, pinned_by, options):
        raise AssertionError(f"the plan does not build {release}")


class _InRepoCards(ConnectorBuilder):
    """The builder backend of this package, with the cards of test datasets, so their builds take its path."""

    def __init__(self, cards):
        self.cards = cards

    def dataset_ids(self):
        return list(self.cards)

    def card(self, dataset_id):
        return self.cards.get(dataset_id)


def _cards(**parents: dict[str, DatasetRef]) -> dict:
    """Return connector cards by dataset ID; each keyword is ``<name>={alias: ref}`` for ``test/<name>``."""
    cards = {GRAND.dataset_id: make_dataset().metadata}
    for name, declared in parents.items():
        ref = DatasetRef.parse(f"test/{name}@1.0.0")
        cards[ref.dataset_id] = make_dataset().metadata.model_copy(
            update={"dataset_id": ref.dataset_id, "parents": declared}
        )
    return cards


def _install_cards(monkeypatch, **parents: dict[str, DatasetRef]) -> dict:
    """Register this package's builder backend, serving the cards of ``_cards(**parents)``.

    Returns:
        The cards by dataset ID, which a test may change.
    """
    cards = _cards(**parents)
    install_backends(monkeypatch, _InRepoCards(cards))
    return cards


@pytest.mark.parametrize(
    ("grand_in", "expected"),
    [
        ("output", lambda origin: [Step(PARENT, "build", InstalledConnectors())]),
        (
            "origin",
            lambda origin: [
                Step(GRAND, "copy", origin),
                Step(PARENT, "build", InstalledConnectors(), depends_on=(GRAND,)),
            ],
        ),
        (
            "nowhere",
            lambda origin: [
                Step(GRAND, "build", InstalledConnectors()),
                Step(PARENT, "build", InstalledConnectors(), depends_on=(GRAND,)),
            ],
        ),
    ],
)
def test_plan_copies_or_builds_each_parent_once_before_its_children(tmp_path, monkeypatch, grand_in, expected):
    _install_cards(monkeypatch, child={"parent": PARENT, "grand": GRAND}, parent={"grand": GRAND})
    registries = {name: LocalRegistry(tmp_path / name) for name in ("output", "origin")}
    if grand_in in registries:
        registries[grand_in].put_dataset(make_dataset())

    steps = _parent_steps(
        CHILD.dataset_id, registries["output"], parents_from=[registries["origin"], InstalledConnectors()]
    )

    assert steps == expected(registries["origin"])


@pytest.mark.parametrize(
    ("cards", "options", "error", "match"),
    [
        (
            {"child": {"parent": PARENT}, "parent": {}},
            {},
            TimeNetMissingParentError,
            r"^parent release test/parent@1\.0\.0 of test/child@1\.0\.0 is not in local registry",
        ),
        (
            {"child": {"parent": DatasetRef.parse("test/parent@2.0.0")}, "parent": {}},
            {"parents_from": [InstalledConnectors()]},
            TimeNetMissingParentError,
            # The installed connector of test/parent builds only the version that its card declares.
            r"^parent release test/parent@2\.0\.0 of test/child@1\.0\.0 is not in local registry .* or "
            r"installed connectors \(test/parent builds 1\.0\.0\)\.$",
        ),
        (
            {"child": {"parent": PARENT}, "parent": {"child": CHILD}},
            {"parents_from": [InstalledConnectors()]},
            TimeFValidationError,
            "cycle: test/child@1.0.0 -> test/parent@1.0.0 -> test/child@1.0.0",
        ),
    ],
    ids=["missing", "other-version", "cycle"],
)
# ruff: ignore[too-many-arguments, too-many-positional-arguments]
def test_plan_rejects_unobtainable_parents_before_anything_runs(tmp_path, monkeypatch, cards, options, error, match):
    _install_cards(monkeypatch, **cards)
    output = LocalRegistry(tmp_path / "output")

    with pytest.raises(error, match=match):
        _parent_steps(CHILD.dataset_id, output, **options)


_SPEC = TimeSeriesSpec(spec_type="sine", name="Sine", unit_value=ureg.dimensionless)
SOURCE = DatasetRef.parse("test/source@1.0.0")
DECLARED = DatasetRef.parse("test/declared@1.0.0")


def _child_of(registry: LocalRegistry, reference: DatasetRef, parent: DatasetRef, record_id: int) -> None:
    """Store ``reference``, which imports one Record of ``parent`` and adds one Record of its own values."""
    metadata = make_dataset().metadata.model_copy(
        update={"dataset_id": reference.dataset_id, "parents": {ParentAlias("parent"): parent}}
    )
    child = TimeFDataset(metadata=metadata)
    with TimeFReader(open_local_version(release_dir(registry.root, parent))) as reader:
        child.import_from(reader, records=[record_id])
        own = Signal.from_values(
            np.arange(16, dtype=np.float32), spec=_SPEC, name="own", time_axis=RegularAxis.from_rate_hz(16)
        )
        child.add_record(record=Record(sources=(Source(name="own", signals=(own,)),)))
        registry.put_dataset(child)


def _declared_without_its_sources_source(registry: LocalRegistry) -> None:
    """Store GRAND, then SOURCE, which reads a GRAND Record, then DECLARED, which reads only SOURCE's own values.

    DECLARED's payload sources are SOURCE alone; SOURCE's are GRAND. GRAND is removed afterwards.
    """
    registry.put_dataset(make_dataset())
    _child_of(registry, SOURCE, GRAND, make_dataset().records[0].id)
    # SOURCE's own Record, which holds values it stores itself, takes the second Record ID.
    _child_of(registry, DECLARED, SOURCE, 2)
    assert {str(source.release) for source in registry.get_manifest(DECLARED.dataset_id).payload_sources.values()} == {
        str(SOURCE)
    }
    shutil.rmtree(release_dir(registry.root, GRAND))


@pytest.mark.parametrize("where", ["output", "origin", "source-in-origin"])
def test_a_declared_parent_needs_only_its_direct_payload_sources(tmp_path, monkeypatch, child_builds, where):
    _install_cards(monkeypatch, child={"declared": DECLARED})
    output, origin = LocalRegistry(tmp_path / "output"), LocalRegistry(tmp_path / "origin")
    _declared_without_its_sources_source(origin if where == "origin" else output)
    if where == "source-in-origin":
        release_dir(origin.root, SOURCE).parent.mkdir(parents=True)
        shutil.move(release_dir(output.root, SOURCE), release_dir(origin.root, SOURCE))

    if where != "output":
        # A copy takes a release with each payload source that it needs, at every depth, so that the
        # store never holds a release without its payload sources. That includes GRAND, which SOURCE
        # reads and which no registry holds. So the plan fails before any step runs.
        with pytest.raises(TimeNetMissingParentError) as caught:
            _parent_steps(CHILD.dataset_id, output, parents_from=[origin])
        assert (caught.value.missing, caught.value.needed_by, caught.value.payload_source) == (GRAND, SOURCE, True)
        assert str(caught.value) == f"payload source {GRAND} of {SOURCE} is not in {output} or {origin}."
        return
    # DECLARED opens with SOURCE, its direct payload source, which the output holds. So the plan
    # does not need GRAND, SOURCE's own source.
    assert _parent_steps(CHILD.dataset_id, output, parents_from=[origin]) == []
    run_plan(plan_release_build(CHILD.dataset_id, store=output, parents_from=[origin]), output)
    for reference in (DECLARED, SOURCE):
        assert output.exists(reference.dataset_id, str(reference.version))
    assert not output.exists(GRAND.dataset_id, str(GRAND.version))
    # Only the requested release builds.
    assert child_builds == [(CHILD.dataset_id, False)]


def test_a_payload_source_that_is_later_a_declared_parent_brings_its_own_payload_sources(tmp_path, monkeypatch):
    # The child declares DECLARED first, then SOURCE. The walk finds SOURCE first as the payload source
    # of DECLARED. As a declared parent, SOURCE also needs GRAND, its own payload source, which is missing.
    _install_cards(monkeypatch, child={"declared": DECLARED, "source": SOURCE})
    output = LocalRegistry(tmp_path / "output")
    _declared_without_its_sources_source(output)
    pins = output.get_manifest(SOURCE.dataset_id).payload_sources[SourceKey(str(GRAND))]

    steps = _parent_steps(CHILD.dataset_id, output, parents_from=[InstalledConnectors()])

    assert steps == [Step(GRAND, "build", InstalledConnectors(), pinned_by={SOURCE: pins})]


def test_the_plan_reads_each_card_from_the_backend_that_claims_the_dataset(tmp_path, monkeypatch):
    cards = _cards(child={"parent": PARENT}, parent={"grand": GRAND})
    del cards[GRAND.dataset_id]
    # The in-repo backend claims GRAND (the hello-world connector) but none of the test datasets,
    # which only the second backend knows.
    install_backends(monkeypatch, ConnectorBuilder(), _CardBackend(cards))

    steps = _parent_steps(CHILD.dataset_id, LocalRegistry(tmp_path), parents_from=[InstalledConnectors()])

    assert steps == [
        Step(GRAND, "build", InstalledConnectors()),
        Step(PARENT, "build", InstalledConnectors(), depends_on=(GRAND,)),
    ]
    cards[PARENT.dataset_id] = cards[PARENT.dataset_id].model_copy(update={"dataset_version": "2.0.0"})
    with pytest.raises(
        TimeNetMissingParentError,
        match=r"^parent release test/parent@1\.0\.0 of test/child@1\.0\.0 is not in local registry .* or "
        r"installed connectors \(test/parent builds 2\.0\.0\)\.$",
    ):
        _parent_steps(CHILD.dataset_id, LocalRegistry(tmp_path), parents_from=[InstalledConnectors()])


def test_a_parent_whose_card_cannot_be_read_fails_the_plan_with_the_card_error(tmp_path, monkeypatch):
    (tmp_path / "dataset.yaml").write_text("dataset_id: [unclosed\n")
    # The in-repo backend claims PARENT, whose card cannot be read. The second backend serves CHILD.
    monkeypatch.setattr(backend_module, "known_ids", lambda: [PARENT.dataset_id])
    monkeypatch.setattr(backend_module, "connector_dir", lambda dataset_id: tmp_path)
    install_backends(monkeypatch, ConnectorBuilder(), _CardBackend(_cards(child={"parent": PARENT})))

    with pytest.raises(TimeNetInvalidCardError, match="could not read dataset card") as caught:
        _parent_steps(CHILD.dataset_id, LocalRegistry(tmp_path / "output"), parents_from=[InstalledConnectors()])

    # The plan does not report a card that cannot be read as a missing connector.
    assert not isinstance(caught.value, (LookupError, TimeNetDatasetNotFoundError))


class _Grand(BaseConnector[str]):
    """Rebuilds GRAND, the demo dataset."""

    def metadata(self):
        return make_dataset().metadata

    def download(self, cache_dir):
        return []

    def convert(self, raw_refs):
        return make_dataset()


def _parent_without_its_source(output: LocalRegistry) -> None:
    """Store GRAND and PARENT, which reads a GRAND Record, then remove GRAND."""
    output.put_dataset(make_dataset())
    _child_of(output, PARENT, GRAND, make_dataset().records[0].id)
    shutil.rmtree(release_dir(output.root, GRAND))


def test_a_missing_payload_source_names_the_release_that_reads_it(tmp_path, monkeypatch):
    _install_cards(monkeypatch, child={"parent": PARENT}, parent={"parent": GRAND})
    output = LocalRegistry(tmp_path / "output")
    _parent_without_its_source(output)

    with pytest.raises(TimeNetMissingParentError) as caught:
        _parent_steps(CHILD.dataset_id, output)

    assert (caught.value.missing, caught.value.needed_by, caught.value.payload_source) == (GRAND, PARENT, True)
    assert str(caught.value).startswith("payload source timenet/hello-world@1.0.0 of test/parent@1.0.0 is not in")


@pytest.mark.parametrize("backend", [ValuesBackend.PARQUET, ValuesBackend.ZARR])
def test_a_pinned_source_is_rebuilt_in_staging_and_published_only_if_it_matches(tmp_path, monkeypatch, backend):
    _install_cards(monkeypatch, child={"parent": PARENT}, parent={"parent": GRAND})
    monkeypatch.setattr(_Grand, "values_backend", backend)
    output = LocalRegistry(tmp_path / "output")
    _parent_without_its_source(output)
    pins = output.get_manifest(PARENT.dataset_id).payload_sources[SourceKey(str(GRAND))]

    steps = _parent_steps(CHILD.dataset_id, output, parents_from=[InstalledConnectors()])

    assert steps == [Step(GRAND, "build", InstalledConnectors(), pinned_by={PARENT: pins})]
    built = []

    def child(release, store_root, *, pinned_by, **_):
        built.append(release)
        if release == GRAND:
            run_pipeline(_Grand(), store_root, pinned_by=pinned_by)
        else:
            _store_from_card(release.dataset_id, store_root, force=False)

    monkeypatch.setattr(backend_module, "run_isolated_build", child)

    if backend == ValuesBackend.PARQUET:
        run_plan(plan_release_build(CHILD.dataset_id, store=output, parents_from=[InstalledConnectors()]), output)
        assert output.exists(GRAND.dataset_id, str(GRAND.version))
        assert built == [GRAND, CHILD]
        return
    # A Zarr rebuild writes other files than the Parquet files PARENT reads, so nothing is published.
    with pytest.raises(TimeFFormatError, match=r"Nothing was published\. Fetch the exact release .* or rebuild"):
        run_plan(plan_release_build(CHILD.dataset_id, store=output, parents_from=[InstalledConnectors()]), output)
    assert built == [GRAND]
    assert not output.exists(GRAND.dataset_id, str(GRAND.version))
    assert not list(release_dir(output.root, GRAND).parent.glob("*"))


def test_a_payload_source_that_two_releases_read_is_built_against_the_pins_of_both(tmp_path, monkeypatch):
    sibling = DatasetRef.parse("test/sibling@1.0.0")
    _install_cards(monkeypatch, child={"parent": PARENT, "sibling": sibling})
    output = LocalRegistry(tmp_path / "output")
    output.put_dataset(make_dataset())
    # PARENT and the sibling each read another Record of GRAND, so their pins differ.
    _child_of(output, PARENT, GRAND, make_dataset().records[0].id)
    _child_of(output, sibling, GRAND, make_dataset().records[1].id)
    shutil.rmtree(release_dir(output.root, GRAND))
    pins = {
        reader: output.get_manifest(reader.dataset_id).payload_sources[SourceKey(str(GRAND))]
        for reader in (PARENT, sibling)
    }

    steps = _parent_steps(CHILD.dataset_id, output, parents_from=[InstalledConnectors()])

    assert steps == [Step(GRAND, "build", InstalledConnectors(), pinned_by=pins)]


def test_a_built_release_needs_only_its_payload_sources(tmp_path, monkeypatch):
    _install_cards(monkeypatch, parent={"parent": GRAND})
    output = LocalRegistry(tmp_path / "output")
    _parent_without_its_source(output)
    pins = output.get_manifest(PARENT.dataset_id).payload_sources[SourceKey(str(GRAND))]

    with pytest.raises(TimeNetMissingParentError, match=r"^payload source timenet/hello-world@1\.0\.0 of test/parent"):
        _parent_steps(PARENT.dataset_id, output)
    connectors = [InstalledConnectors()]
    # The source that PARENT reads builds against its pins.
    assert _parent_steps(PARENT.dataset_id, output, parents_from=connectors) == [
        Step(GRAND, "build", InstalledConnectors(), pinned_by={PARENT: pins})
    ]
    # A rebuild of PARENT needs its declared parents instead, and pins the new files.
    assert _parent_steps(PARENT.dataset_id, output, parents_from=connectors, force=True) == [
        Step(GRAND, "build", InstalledConnectors())
    ]
    # A registry that holds the source fills the gap: the local PARENT is reused and GRAND is copied.
    origin = LocalRegistry(tmp_path / "origin")
    origin.put_dataset(make_dataset())
    assert _parent_steps(PARENT.dataset_id, output, parents_from=[origin]) == [Step(GRAND, "copy", origin)]


def test_a_damaged_parent_is_copied_again_but_never_built_over(tmp_path, monkeypatch, child_builds):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    output, origin = LocalRegistry(tmp_path / "output"), LocalRegistry(tmp_path / "origin")
    parent = make_dataset()
    parent._metadata = cards[PARENT.dataset_id]
    for registry in (output, origin):
        registry.put_dataset(parent)
    (release_dir(output.root, PARENT) / "control.duckdb").write_bytes(b"cut short")

    with pytest.raises(TimeNetFileError, match=r"'control\.duckdb' has 9 bytes, expected [\d,]+ bytes\.$") as caught:
        _parent_steps(CHILD.dataset_id, output, parents_from=[InstalledConnectors()])
    assert caught.value.release == PARENT

    assert _parent_steps(CHILD.dataset_id, output, parents_from=[origin, InstalledConnectors()]) == [
        Step(PARENT, "copy", origin)
    ]
    run_plan(plan_release_build(CHILD.dataset_id, store=output, parents_from=[origin]), output)
    assert _parent_steps(CHILD.dataset_id, output) == []
    assert child_builds == [(CHILD.dataset_id, False)]


class _Hosted(WritableRegistry):
    """A registry that is not local, over a local directory."""

    def __init__(self, root):
        self._local = LocalRegistry(root)

    @property
    def local_root(self):
        return self._local.root

    def list_datasets(self):
        return self._local.list_datasets()

    def get_manifest(self, dataset_id, version=None):
        return self._local.get_manifest(dataset_id, version)

    def open_file(self, dataset_id, version, relpath):
        return self._local.open_file(dataset_id, version, relpath)

    def open_release(self, reference, **options):
        return self._local.open_release(reference, **options)

    def put_dataset(self, dataset, **options):
        return self._local.put_dataset(dataset, **options)

    def put_release(self, from_directory):
        return self._local.put_release(from_directory)


_GATED = {"access": Access.CREDENTIALED, "access_url": "https://example.org/dua"}


@pytest.mark.parametrize("gated", ["child-card", "held-parent", "parent-card"])
def test_open_access_only_refuses_a_plan_that_visits_a_release_that_is_not_open(tmp_path, monkeypatch, gated):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    gated_ref = CHILD if gated == "child-card" else PARENT
    cards[gated_ref.dataset_id] = cards[gated_ref.dataset_id].model_copy(update=_GATED)
    store = LocalRegistry(tmp_path / "store")
    if gated == "held-parent":
        store.put_dataset(TimeFDataset(metadata=cards[PARENT.dataset_id]))

    with pytest.raises(TimeNetAccessError) as caught:
        _parent_steps(CHILD.dataset_id, store, parents_from=[InstalledConnectors()], open_access_only=True)

    error = caught.value
    requirement = AccessRequirement(gated_ref, Access.CREDENTIALED, _GATED["access_url"])
    assert (error.dataset, error.access, error.requirements) == (gated_ref, Access.CREDENTIALED, (requirement,))


def test_open_access_only_names_the_access_page_of_a_held_parent_from_its_lineage(tmp_path, monkeypatch):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    store = LocalRegistry(tmp_path / "store")
    # The card of the held parent stays open. Its lineage holds GRAND, which is credentialed.
    held = TimeFDataset(metadata=cards[PARENT.dataset_id])
    grand = make_dataset().metadata.model_copy(update=_GATED)
    held._import_state.lineage[str(GRAND)] = ParentDatasetCard(metadata=grand)
    store.put_dataset(held)

    with pytest.raises(TimeNetAccessError) as caught:
        _parent_steps(CHILD.dataset_id, store, parents_from=[InstalledConnectors()], open_access_only=True)

    assert (caught.value.dataset, caught.value.access) == (PARENT, Access.CREDENTIALED)
    assert caught.value.requirements == (AccessRequirement(GRAND, Access.CREDENTIALED, _GATED["access_url"]),)


def _store_from_card(dataset_id: str, root: Path, *, force: bool) -> Path:
    """Store a dataset from its card and return its version directory.

    A dataset with parents imports one Record of each; one without parents holds the demo dataset.
    """
    registry = LocalRegistry(Path(root))
    metadata = builders.find_card(dataset_id)
    assert metadata is not None
    with open_parents(metadata, registry) as context:
        dataset = TimeFDataset(metadata=metadata) if context.parents else make_dataset()
        dataset._metadata = metadata
        for handle in context.parents.values():
            dataset.import_from(handle, records=[make_dataset().records[0].id])
        registry.put_dataset(dataset, force=force)
    return registry.root / dataset_id / str(metadata.dataset_version)


def _prepared_builds(calls: list) -> object:
    """Return a stand-in for ``run_isolated_build`` that stores each release from its card and records the call."""

    def run_isolated_build(release, store_root, *, force=False, **_):
        calls.append((release.dataset_id, force))
        _store_from_card(release.dataset_id, store_root, force=force)

    return run_isolated_build


@pytest.fixture
def child_builds(monkeypatch) -> list:
    """Stand in for each isolated build with :func:`_prepared_builds`.

    Returns:
        The list that receives one ``(dataset_id, force)`` entry for each build.
    """
    calls = []
    monkeypatch.setattr(backend_module, "run_isolated_build", _prepared_builds(calls))
    return calls


def test_build_prepares_parents_before_the_isolated_child(tmp_path, monkeypatch):
    cards = _install_cards(monkeypatch, child={"parent": PARENT, "grand": GRAND}, parent={})
    monkeypatch.setattr(cli_module, "known_ids", lambda: list(cards))
    origin = LocalRegistry(tmp_path / "origin")
    origin.put_dataset(make_dataset())
    out = tmp_path / "out"
    calls = []
    monkeypatch.setattr(backend_module, "run_isolated_build", _prepared_builds(calls))

    result = runner.invoke(
        app,
        ["build", CHILD.dataset_id, "--local-registry", str(out), "--remote-registry", str(origin.root), "-f"],
    )

    assert result.exit_code == 0, result.output
    assert (out / GRAND.dataset_id / str(GRAND.version) / "manifest.json").is_file()
    # The parent builds first and never with --force; only the requested build's result reaches stdout.
    assert calls == [(PARENT.dataset_id, False), (CHILD.dataset_id, True)]
    assert result.stdout.strip() == str(out / CHILD.dataset_id / str(CHILD.version))


def test_every_build_entry_point_prepares_parents(tmp_path, monkeypatch, child_builds):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    out = tmp_path / "out"

    def build(origin=None):
        # The client and timenet.build() build through the coordinator.
        store = LocalRegistry(out)
        run_plan(
            plan_release_build(CHILD.dataset_id, store=store, parents_from=[] if origin is None else [origin]), store
        )

    with pytest.raises(TimeNetMissingParentError, match=r"parent release test/parent@1\.0\.0 of test/child@1\.0\.0"):
        build()
    assert child_builds == []

    # A registry instance to fetch from, as the SDK layers accept one.
    origin = LocalRegistry(tmp_path / "origin")
    parent = make_dataset()
    parent._metadata = cards[PARENT.dataset_id]
    origin.put_dataset(parent)
    build(origin)
    assert child_builds == [(CHILD.dataset_id, False)]
    assert LocalRegistry(out).exists(PARENT.dataset_id, str(PARENT.version))


class _NoDownloads(_Hosted):
    """A registry that is not local and fails the test if a file is read from it."""

    def open_file(self, dataset_id, version, relpath):
        pytest.fail(f"downloaded {dataset_id}@{version}/{relpath}")


def test_a_copy_of_a_release_that_is_not_open_fails_before_any_download(tmp_path, monkeypatch):
    cards = _install_cards(monkeypatch, child={"grand": GRAND, "parent": PARENT}, parent={})
    origin = _NoDownloads(tmp_path / "origin")
    # GRAND is open and would be copied first.
    origin.put_dataset(make_dataset())
    # The card of PARENT stays open. Its lineage holds an ancestor, which is credentialed.
    ancestor = DatasetRef.parse("test/ancestor@1.0.0")
    parent = TimeFDataset(metadata=cards[PARENT.dataset_id])
    card = make_dataset().metadata.model_copy(update={"dataset_id": ancestor.dataset_id, **_GATED})
    parent._import_state.lineage[str(ancestor)] = ParentDatasetCard(metadata=card)
    origin.put_dataset(parent)
    output = LocalRegistry(tmp_path / "out")

    with pytest.raises(TimeNetAccessError) as caught:
        run_plan(plan_release_build(CHILD.dataset_id, store=output, parents_from=[origin]), output)

    assert caught.value.dataset == PARENT
    assert caught.value.requirements == (AccessRequirement(ancestor, Access.CREDENTIALED, _GATED["access_url"]),)
    assert not output.exists(GRAND.dataset_id, str(GRAND.version))


def test_a_pinned_parent_build_hands_its_pins_to_the_isolated_child(tmp_path, monkeypatch):
    cards = _install_cards(monkeypatch, parent={"parent": GRAND})
    output = LocalRegistry(tmp_path / "output")
    _parent_without_its_source(output)
    pins = output.get_manifest(PARENT.dataset_id).payload_sources[SourceKey(str(GRAND))]
    received, pin_files = [], []

    def run_pipeline(connector, root, *, pinned_by, **_):
        received.append(pinned_by)
        output.put_dataset(make_dataset(), pinned_by=pinned_by)

    def child(command, env):
        # Stands in for the uv child: run the forwarded command line through the CLI in this process.
        argv = command[command.index("timenet-build") + 1 :]
        pin_files.extend(Path(argv[at + 1]) for at, arg in enumerate(argv) if arg == "--pinned-by")
        monkeypatch.setenv("TIMENET_ISOLATION", env["TIMENET_ISOLATION"])
        result = runner.invoke(app, argv)
        return "", result.exit_code

    monkeypatch.setattr(env_module, "_run_build", child)
    monkeypatch.setattr(cli_module, "known_ids", lambda: list(cards))
    monkeypatch.setattr(backend_module, "run_pipeline", run_pipeline)

    # The output holds PARENT, so the build gets only GRAND, the payload source of PARENT.
    result = runner.invoke(app, ["build", PARENT.dataset_id, "--local-registry", str(output.root)])

    assert result.exit_code == 0, result.output
    # The rebuild of GRAND gets the pins of PARENT, through a file that is gone afterwards.
    assert received == [{PARENT: pins}]
    assert len(pin_files) == 1
    assert not pin_files[0].exists()
    assert output.exists(GRAND.dataset_id, str(GRAND.version))


@pytest.mark.parametrize("case", ["missing", "missing-buildable", "damaged", "not-open", "not-served"])
def test_the_cli_hint_names_its_flags(tmp_path, monkeypatch, case):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    monkeypatch.setattr(cli_module, "known_ids", lambda: list(cards))
    store = LocalRegistry(tmp_path / "store")
    options = []
    if case in {"missing", "missing-buildable"}:
        # No registry holds the parent, and the list has no connectors.
        options = ["--no-auto-build"]
    if case == "missing":
        # No installed connector builds the parent.
        del cards[PARENT.dataset_id]
    if case == "damaged":
        store.put_dataset(TimeFDataset(metadata=cards[PARENT.dataset_id]))
        (release_dir(store.root, PARENT) / "control.duckdb").write_bytes(b"cut short")
        options = ["--remote-registry", str(tmp_path / "remote registry")]
    if case == "not-open":
        cards[CHILD.dataset_id] = cards[CHILD.dataset_id].model_copy(update=_GATED)
        monkeypatch.setattr(hints_module, "open_registry", lambda url, **_: _Hosted(tmp_path / "hosted"))
        options = ["--publish-to", "s3://bucket/registry"]
    if case == "not-served":
        # The remote registry lists the parent without serving its files, and the list has no connectors.
        hosted = _Hosted(tmp_path / "hosted")
        hosted.put_dataset(TimeFDataset(metadata=cards[PARENT.dataset_id].model_copy(update=_GATED)))
        monkeypatch.setattr(client_module, "open_registry", lambda uri, **_: hosted)
        options = ["--no-auto-build"]

    result = runner.invoke(app, ["build", CHILD.dataset_id, "--local-registry", str(store.root), *options])

    # The hint of a damaged copy repeats the registry flags of the command, quoted for the shell.
    remote = str(tmp_path / "remote registry")
    flags = f" --local-registry {shlex.quote(str(store.root))} --remote-registry {shlex.quote(remote)}"
    hints = {
        # No installed connector builds the parent.
        "missing": (TimeNetMissingParentError, "Pass a registry that holds it with --remote-registry <url>."),
        "missing-buildable": (
            TimeNetMissingParentError,
            "Pass a registry that holds it with --remote-registry <url>, or build it from its installed connector "
            "with --auto-build.",
        ),
        # No registry provides the parent, so the hint names no download.
        "damaged": (TimeNetFileError, f"Build it again with `timenet-build build test/parent{flags} --force`."),
        "not-open": (
            TimeNetAccessError,
            "_Hosted serves open releases only, so --publish-to takes open releases only. "
            "Build it into the store without --publish-to instead.",
        ),
        "not-served": (TimeNetAccessError, "Build it locally from its installed connector with --auto-build."),
    }
    error, hint = hints[case]
    assert isinstance(result.exception, error), result.output
    assert result.exception.hint == hint


def _publish_to(tmp_path, monkeypatch, cards) -> tuple[_Hosted, LocalRegistry, list]:
    """Make each ``--publish-to`` of the CLI open a registry that is not local, and record each isolated build.

    Returns:
        The registry that ``--publish-to`` opens, the store, and the list of the builds.
    """
    remote = _Hosted(tmp_path / "hosted")
    monkeypatch.setattr(hints_module, "open_registry", lambda url, **_: remote)
    monkeypatch.setattr(cli_module, "known_ids", lambda: list(cards))
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(tmp_path / "store"))
    calls = []
    monkeypatch.setattr(backend_module, "run_isolated_build", _prepared_builds(calls))
    return remote, LocalRegistry(tmp_path / "store"), calls


@pytest.mark.parametrize("gated", ["child", "parent"])
def test_publish_to_refuses_a_credentialed_release_before_any_download(tmp_path, monkeypatch, gated):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    gated_ref = CHILD if gated == "child" else PARENT
    cards[gated_ref.dataset_id] = cards[gated_ref.dataset_id].model_copy(update=_GATED)
    origin = LocalRegistry(tmp_path / "origin")
    origin.put_dataset(TimeFDataset(metadata=cards[PARENT.dataset_id]))
    _, store, calls = _publish_to(tmp_path, monkeypatch, cards)
    command = ["build", CHILD.dataset_id, "--publish-to", "s3://bucket/registry"]

    result = runner.invoke(app, [*command, "--remote-registry", str(origin.root), "--no-auto-build"])

    # The build into the store gets the access checks of the destination.
    assert isinstance(result.exception, TimeNetAccessError), result.output
    assert result.exception.dataset == gated_ref
    assert calls == []
    assert not store.exists(PARENT.dataset_id, str(PARENT.version))


def test_publish_to_builds_into_the_store_and_publishes(tmp_path, monkeypatch):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    origin = LocalRegistry(tmp_path / "origin")
    parent = make_dataset()
    parent._metadata = cards[PARENT.dataset_id]
    origin.put_dataset(parent)
    remote, store, calls = _publish_to(tmp_path, monkeypatch, cards)
    command = ["build", CHILD.dataset_id, "--publish-to", "s3://bucket/registry"]
    command += ["--remote-registry", str(origin.root), "--no-auto-build"]

    result = runner.invoke(app, command)

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{CHILD.version}\n"
    # The parent lands in the store first, and the child builds there. Then the publish takes the
    # child and its payload source, the parent, to the destination.
    assert calls == [(CHILD.dataset_id, False)]
    for registry in (store, remote):
        assert registry.exists(CHILD.dataset_id, str(CHILD.version))
        assert registry.exists(PARENT.dataset_id, str(PARENT.version))


def test_publish_to_copies_parents_only_from_the_sources(tmp_path, monkeypatch):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    remote, store, calls = _publish_to(tmp_path, monkeypatch, cards)
    parent = make_dataset()
    parent._metadata = cards[PARENT.dataset_id]
    remote.put_dataset(parent)
    command = ["build", CHILD.dataset_id, "--publish-to", "s3://bucket/registry", "--no-auto-build"]

    # The destination holds the parent, but the destination is no source of parents.
    result = runner.invoke(app, command)

    assert isinstance(result.exception, TimeNetMissingParentError), result.output
    assert "--remote-registry" in str(result.exception)
    assert calls == []
    # As the remote registry of the list, the same registry provides the parent.
    result = runner.invoke(app, [*command, "--remote-registry", str(remote.local_root)])
    assert result.exit_code == 0, result.output
    assert calls == [(CHILD.dataset_id, False)]
    assert store.exists(PARENT.dataset_id, str(PARENT.version))
    assert remote.exists(CHILD.dataset_id, str(CHILD.version))


def test_publish_to_a_local_directory_publishes_there(tmp_path, monkeypatch, child_builds):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    # A local directory takes releases that are not open, so the plan does not refuse this one.
    cards[CHILD.dataset_id] = cards[CHILD.dataset_id].model_copy(update=_GATED)
    monkeypatch.setattr(cli_module, "known_ids", lambda: list(cards))
    published = LocalRegistry(tmp_path / "published")

    result = runner.invoke(app, ["build", CHILD.dataset_id, "--publish-to", str(published.root)])

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{CHILD.version}\n"
    assert child_builds == [(PARENT.dataset_id, False), (CHILD.dataset_id, False)]
    # The build lands in the store, and the publish copies the child with its payload source.
    for registry in (default_store(), published):
        assert registry.exists(CHILD.dataset_id, str(CHILD.version))
        assert registry.exists(PARENT.dataset_id, str(PARENT.version))


@pytest.mark.parametrize("value", [" ", "ftp://host/registry"])
def test_publish_to_takes_a_registry_url_or_a_path(monkeypatch, child_builds, value):
    # A blank value names no registry. Unchecked, it would publish into the current directory.
    result = runner.invoke(app, ["build", GRAND.dataset_id, "--publish-to", value])

    # A usage error that names the flag.
    assert result.exit_code == 2, result.output
    assert "Invalidvalue:--publish-to" in "".join(click.unstyle(result.output).split())
    assert child_builds == []


def test_publish_to_a_destination_that_holds_another_card_fails_before_any_build(tmp_path, monkeypatch):
    cards = _install_cards(monkeypatch, child={}, parent={})
    remote, store, calls = _publish_to(tmp_path, monkeypatch, cards)
    # The destination holds the files of the store, under another card.
    published = release_dir(remote.local_root, CHILD)
    shutil.copytree(_store_from_card(CHILD.dataset_id, store.root, force=False), published)
    manifest = json.loads((published / "manifest.json").read_text())
    manifest["metadata"]["description"] = "Another card."
    (published / "manifest.json").write_text(json.dumps(manifest))

    result = runner.invoke(app, ["build", CHILD.dataset_id, "--publish-to", "s3://bucket/registry"])

    assert isinstance(result.exception, TimeNetVersionConflictError), result.output
    assert result.exception.difference == "card"
    assert result.exception.hint == "A published version cannot change, so a changed release needs a new version."
    assert "Nothing to do" not in click.unstyle(result.output)
    assert calls == []


def test_publish_to_a_destination_that_holds_the_version_builds_nothing_without_force(tmp_path, monkeypatch):
    cards = _install_cards(monkeypatch, child={}, parent={})
    remote, store, calls = _publish_to(tmp_path, monkeypatch, cards)
    # The line names a long path, so the console must not wrap it.
    monkeypatch.setattr(cli_module.console, "rich", Console(stderr=True, width=1000))
    # The destination holds the version, and the store does not.
    shutil.copytree(
        _store_from_card(CHILD.dataset_id, tmp_path / "elsewhere", force=False), release_dir(remote.local_root, CHILD)
    )

    result = runner.invoke(app, ["build", CHILD.dataset_id, "--publish-to", "s3://bucket/registry"])

    assert result.exit_code == 0, result.output
    assert click.unstyle(result.stdout) == f"{CHILD.version}\n"
    assert click.unstyle(result.stderr).splitlines() == [f"✅ Nothing to do: 'test/child' 1.0.0 is already in {remote}"]
    assert calls == []
    assert not store.exists(CHILD.dataset_id, str(CHILD.version))


def _files(root: Path) -> dict[Path, tuple[int, int]]:
    """Return the size and the modification time of each file under ``root``, by its path."""
    return {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in root.rglob("*") if path.is_file()}


def _change_the_next_build(monkeypatch) -> None:
    """Make each later build store its dataset with one more Record, so that it writes other files."""
    put_dataset = LocalRegistry.put_dataset

    def put_one_more_record(self, dataset, **options):
        extra = Signal.from_values(
            np.arange(16, dtype=np.float32), spec=_SPEC, name="extra", time_axis=RegularAxis.from_rate_hz(16)
        )
        dataset.add_record(record=Record(sources=(Source(name="extra", signals=(extra,)),)))
        return put_dataset(self, dataset, **options)

    monkeypatch.setattr(LocalRegistry, "put_dataset", put_one_more_record)


def test_force_with_publish_to_rebuilds_and_the_publish_refuses_other_files(tmp_path, monkeypatch):
    cards = _install_cards(monkeypatch, child={}, parent={})
    remote, _, calls = _publish_to(tmp_path, monkeypatch, cards)
    command = ["build", CHILD.dataset_id, "--publish-to", "s3://bucket/registry"]
    assert runner.invoke(app, command).exit_code == 0
    published = _files(remote.local_root)
    _change_the_next_build(monkeypatch)  # the next build writes other files
    result = runner.invoke(app, [*command, "--force"])
    assert isinstance(result.exception, TimeNetVersionConflictError)
    assert calls == [(CHILD.dataset_id, False), (CHILD.dataset_id, True)]
    # A published version cannot change, so the destination keeps its files.
    assert _files(remote.local_root) == published


def test_force_with_publish_to_does_nothing_for_the_same_files(tmp_path, monkeypatch):
    cards = _install_cards(monkeypatch, child={}, parent={})
    remote, _, calls = _publish_to(tmp_path, monkeypatch, cards)
    saved = LocalRegistry(tmp_path / "saved")
    _store_from_card(CHILD.dataset_id, saved.root, force=False)

    def copy_the_saved_release(release, store_root, *, force=False, **_):
        # Each build puts the same release directory into the store, so both builds write the same files.
        calls.append((release.dataset_id, force))
        LocalRegistry(store_root).put_release(release_dir(saved.root, CHILD))

    monkeypatch.setattr(backend_module, "run_isolated_build", copy_the_saved_release)
    command = ["build", CHILD.dataset_id, "--publish-to", "s3://bucket/registry"]
    assert runner.invoke(app, command).exit_code == 0
    published = _files(remote.local_root)

    result = runner.invoke(app, [*command, "--force"])

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{CHILD.version}\n"
    assert calls == [(CHILD.dataset_id, False), (CHILD.dataset_id, True)]
    # The destination holds the same files, so the publish copied nothing.
    assert _files(remote.local_root) == published


@pytest.mark.parametrize(
    ("entry", "hint"),
    [
        ("cli", "Build in this interpreter with --no-isolation."),
        ("sdk", "Build in this interpreter with TIMENET_ISOLATION=off."),
    ],
)
def test_an_environment_that_cannot_be_made_names_the_build_in_this_interpreter(tmp_path, monkeypatch, entry, hint):
    cards = _install_cards(monkeypatch, parent={})
    monkeypatch.setattr(cli_module, "known_ids", lambda: list(cards))
    monkeypatch.setattr(env_module, "BASE_DISTRIBUTIONS", ("timenet-not-installed",))
    store = tmp_path / "store"

    if entry == "cli":
        error = runner.invoke(app, ["build", PARENT.dataset_id, "--local-registry", str(store)]).exception
    else:
        with pytest.raises(TimeNetIsolationError) as caught:
            timenet.build(PARENT.dataset_id, store=store)
        error = caught.value

    assert isinstance(error, TimeNetIsolationError)
    assert error.hint == hint


def test_a_store_in_parents_from_is_not_copied_onto_itself(tmp_path, monkeypatch, child_builds):
    _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    out, origin = LocalRegistry(tmp_path / "out"), LocalRegistry(tmp_path / "origin")
    _store_from_card(PARENT.dataset_id, origin.root, force=False)

    plan = plan_release_build(CHILD.dataset_id, store=out, parents_from=[out, LocalRegistry(out.root), origin])
    run_plan(plan, out)

    assert plan.steps == (
        Step(PARENT, "copy", origin),
        Step(CHILD, "build", InstalledConnectors(), depends_on=(PARENT,)),
    )
    # The plan drops the store from the list, so an error names the store once.
    empty = LocalRegistry(tmp_path / "empty")
    with pytest.raises(TimeNetMissingParentError) as missing:
        _parent_steps(CHILD.dataset_id, empty, parents_from=[empty, LocalRegistry(empty.root)])
    assert missing.value.searched == str(empty)


def test_a_parent_that_a_registry_lists_without_serving_is_built_by_the_connectors(tmp_path, monkeypatch):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    hosted = _Hosted(tmp_path / "hosted")
    hosted.put_dataset(TimeFDataset(metadata=cards[PARENT.dataset_id].model_copy(update=_GATED)))
    output = LocalRegistry(tmp_path / "out")

    steps = _parent_steps(CHILD.dataset_id, output, parents_from=[hosted, InstalledConnectors()])

    assert steps == [Step(PARENT, "build", InstalledConnectors())]
    # Without the connectors, the plan names the access page of the release.
    with pytest.raises(TimeNetAccessError) as caught:
        _parent_steps(CHILD.dataset_id, output, parents_from=[hosted])
    assert caught.value.requirements == (AccessRequirement(PARENT, Access.CREDENTIALED, _GATED["access_url"]),)


@pytest.mark.parametrize("answer", ["not-found", "not-served"])
def test_a_damaged_copy_in_the_output_stops_the_plan_if_no_registry_serves_the_release(tmp_path, monkeypatch, answer):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    output, hosted = LocalRegistry(tmp_path / "output"), _Hosted(tmp_path / "hosted")
    parent = make_dataset()
    parent._metadata = cards[PARENT.dataset_id]
    output.put_dataset(parent)
    (release_dir(output.root, PARENT) / "control.duckdb").write_bytes(b"cut short")
    if answer == "not-served":
        hosted.put_dataset(TimeFDataset(metadata=cards[PARENT.dataset_id].model_copy(update=_GATED)))

    # The damage, not the missing release or its access page, is what the caller must fix.
    with pytest.raises(TimeNetFileError) as caught:
        _parent_steps(CHILD.dataset_id, output, parents_from=[hosted])

    assert caught.value.release == PARENT


def test_a_copy_gets_a_payload_source_at_depth_two_from_another_registry_first(tmp_path, monkeypatch, child_builds):
    _install_cards(monkeypatch, child={"declared": DECLARED})
    output, origin, mirror = (LocalRegistry(tmp_path / name) for name in ("output", "origin", "mirror"))
    _declared_without_its_sources_source(origin)
    mirror.put_dataset(make_dataset())

    steps = _parent_steps(CHILD.dataset_id, output, parents_from=[origin, mirror])

    # The copy of DECLARED brings SOURCE from origin. Origin lacks GRAND, which SOURCE reads, so GRAND
    # comes first, from mirror.
    assert steps == [Step(GRAND, "copy", mirror), Step(DECLARED, "copy", origin, depends_on=(GRAND,))]
    run_plan(plan_release_build(CHILD.dataset_id, store=output, parents_from=[origin, mirror]), output)
    for reference in (GRAND, SOURCE, DECLARED):
        assert output.exists(reference.dataset_id, str(reference.version))
    assert child_builds == [(CHILD.dataset_id, False)]


def test_a_payload_source_at_depth_two_builds_against_the_pins_of_its_reader(tmp_path, monkeypatch):
    _install_cards(monkeypatch, child={"declared": DECLARED})
    output, origin = LocalRegistry(tmp_path / "output"), LocalRegistry(tmp_path / "origin")
    _declared_without_its_sources_source(origin)
    pins = origin.get_manifest(SOURCE.dataset_id).payload_sources[SourceKey(str(GRAND))]

    steps = _parent_steps(CHILD.dataset_id, output, parents_from=[origin, InstalledConnectors()])

    assert steps == [
        Step(GRAND, "build", InstalledConnectors(), pinned_by={SOURCE: pins}),
        Step(DECLARED, "copy", origin, depends_on=(GRAND,)),
    ]


def test_a_copy_whose_payload_source_no_source_provides_fails_before_any_step_runs(tmp_path, monkeypatch, child_builds):
    cards = _install_cards(monkeypatch, child={"parent": PARENT, "declared": DECLARED}, parent={})
    del cards[GRAND.dataset_id]
    output, origin = LocalRegistry(tmp_path / "output"), LocalRegistry(tmp_path / "origin")
    _declared_without_its_sources_source(origin)

    # The plan builds PARENT first. Then the copy of DECLARED needs GRAND, which nothing provides.
    with pytest.raises(
        TimeNetMissingParentError, match=r"^payload source timenet/hello-world@1\.0\.0 of test/source@1\.0\.0"
    ):
        run_plan(
            plan_release_build(CHILD.dataset_id, store=output, parents_from=[origin, InstalledConnectors()]), output
        )

    assert child_builds == []
    assert not output.root.exists()


def test_the_requested_build_waits_for_a_payload_source_of_a_held_parent(tmp_path, monkeypatch):
    _install_cards(monkeypatch, child={"declared": DECLARED})
    output, origin = LocalRegistry(tmp_path / "output"), LocalRegistry(tmp_path / "origin")
    origin.put_dataset(make_dataset())
    _child_of(origin, SOURCE, GRAND, make_dataset().records[0].id)
    _child_of(origin, DECLARED, SOURCE, 2)
    # The output holds DECLARED, but not SOURCE, whose files DECLARED reads.
    shutil.copytree(release_dir(origin.root, DECLARED), release_dir(output.root, DECLARED))

    plan = plan_release_build(CHILD.dataset_id, store=output, parents_from=[origin])

    # DECLARED has no step, so the build of CHILD waits for the step of SOURCE behind it.
    assert plan.steps == (
        Step(SOURCE, "copy", origin),
        Step(CHILD, "build", InstalledConnectors(), depends_on=(SOURCE,)),
    )


def _count_reads(monkeypatch, registry) -> dict[str, int]:
    """Count each manifest that is read from a registry, by release."""
    reads, get_manifest = {}, registry.get_manifest

    def counted(dataset_id, version=None):
        reads[f"{dataset_id}@{version}"] = reads.get(f"{dataset_id}@{version}", 0) + 1
        return get_manifest(dataset_id, version)

    monkeypatch.setattr(registry, "get_manifest", counted)
    return reads


def test_a_plan_reads_each_manifest_once_from_each_registry(tmp_path, monkeypatch):
    # SOURCE is a declared parent and a payload source of DECLARED, so the walk meets it twice. The
    # registry answers for SOURCE first as the payload source.
    parents = {"declared": DECLARED, "source": SOURCE}
    _install_cards(monkeypatch, child=parents)
    layout = LocalRegistry(tmp_path / "hosted")
    layout.put_dataset(make_dataset())
    _child_of(layout, SOURCE, GRAND, make_dataset().records[0].id)
    _child_of(layout, DECLARED, SOURCE, 2)
    hosted, output = _Hosted(layout.root), LocalRegistry(tmp_path / "output")
    hosted_reads, output_reads = _count_reads(monkeypatch, hosted), _count_reads(monkeypatch, output)

    steps = _parent_steps(CHILD.dataset_id, output, parents_from=[hosted])

    assert steps == [Step(reference, "copy", hosted) for reference in parents.values()]
    # The registry that is not local answers through its manifest, and the plan reads that manifest
    # only once.
    assert hosted_reads == {str(reference): 1 for reference in (DECLARED, SOURCE, GRAND)}
    # The output is read for the requested release too, which it does not hold.
    assert output_reads == {str(reference): 1 for reference in (CHILD, DECLARED, SOURCE)}


def test_plan_lines_draw_a_shared_release_once(tmp_path):
    hosted = LocalRegistry(tmp_path / "hosted")
    plan = Plan(
        (
            Step(GRAND, "build", InstalledConnectors()),
            Step(PARENT, "copy", hosted, depends_on=(GRAND,)),
            Step(CHILD, "build", InstalledConnectors(), depends_on=(PARENT, GRAND)),
        )
    )

    assert _plan_lines(plan) == [
        "test/child@1.0.0: build with installed connectors",
        f"├── test/parent@1.0.0: copy from {hosted}",
        "│   └── timenet/hello-world@1.0.0: build with installed connectors",
        "└── timenet/hello-world@1.0.0: see above",
    ]


def test_plan_lines_name_read_steps(tmp_path):
    hosted = LocalRegistry(tmp_path / "hosted")
    # Neither step depends on the other, so each is a root of its own, the last step first.
    plan = Plan((Step(GRAND, "read", hosted), Step(PARENT, "read", hosted)))

    assert _plan_lines(plan) == [
        f"test/parent@1.0.0: read from {hosted}",
        f"timenet/hello-world@1.0.0: read from {hosted}",
    ]


def test_dry_run_prints_the_plan_and_changes_nothing(tmp_path, monkeypatch):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    remote, store, calls = _publish_to(tmp_path, monkeypatch, cards)
    result = runner.invoke(app, ["build", CHILD.dataset_id, "--dry-run", "--publish-to", "s3://bucket/registry"])
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "test/child@1.0.0: build with installed connectors",
        "└── test/parent@1.0.0: build with installed connectors",
        f"then publish test/child@1.0.0 to {remote}",
    ]
    assert calls == [] and not store.root.exists() and not remote.exists(CHILD.dataset_id, "1.0.0")


def test_the_dry_run_lines_equal_the_plan_lines_of_plan_build(tmp_path, monkeypatch, child_builds):
    cards = _install_cards(monkeypatch, child={"parent": PARENT, "grand": GRAND}, parent={})
    monkeypatch.setattr(cli_module, "known_ids", lambda: list(cards))
    origin = LocalRegistry(tmp_path / "origin")
    _store_from_card(PARENT.dataset_id, origin.root, force=False)
    monkeypatch.setenv("TIMENET_REMOTE_REGISTRY", str(origin.root))

    result = runner.invoke(app, ["build", CHILD.dataset_id, "--dry-run"])

    # The command and the public API plan alike: a copy of PARENT, a build of GRAND, and the build of CHILD.
    assert result.exit_code == 0, result.output
    lines = _plan_lines(timenet.plan_build(CHILD.dataset_id))
    assert len(lines) == 3
    assert result.stdout.splitlines() == lines
    assert child_builds == []


def test_a_build_prints_the_plan_before_the_first_step(tmp_path, monkeypatch, child_builds):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    monkeypatch.setattr(cli_module, "known_ids", lambda: list(cards))
    # The plan names a long path, so the console must not wrap the lines around it.
    monkeypatch.setattr(cli_module.console, "rich", Console(stderr=True, width=1000))
    origin = LocalRegistry(tmp_path / "origin")
    _store_from_card(PARENT.dataset_id, origin.root, force=False)

    result = runner.invoke(app, ["build", CHILD.dataset_id, "--remote-registry", str(origin.root)])

    assert result.exit_code == 0, result.output
    lines = result.stderr.splitlines()
    first_step = next(index for index, line in enumerate(lines) if "Building" in line or "Copying" in line)
    assert lines[:first_step] == [
        "📋 Plan for 'test/child' 1.0.0:",
        "   test/child@1.0.0: build with installed connectors",
        f"   └── test/parent@1.0.0: copy from {origin}",
    ]
    assert child_builds == [(CHILD.dataset_id, False)]


@pytest.mark.parametrize("quiet", [False, True])
def test_quiet_hides_the_plan(tmp_path, monkeypatch, child_builds, quiet):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    monkeypatch.setattr(cli_module, "known_ids", lambda: list(cards))

    result = runner.invoke(app, [*(["--quiet"] if quiet else []), "build", CHILD.dataset_id])

    assert result.exit_code == 0, result.output
    if quiet:
        assert result.stderr == ""
    else:
        assert "📋 Plan for 'test/child' 1.0.0:" in result.stderr.splitlines()
    assert child_builds == [(PARENT.dataset_id, False), (CHILD.dataset_id, False)]


@pytest.mark.parametrize("held_in", ["store", "both", "store-only", "destination-only"])
def test_a_dry_run_of_a_held_release_says_nothing_to_do(tmp_path, monkeypatch, held_in):
    cards = _install_cards(monkeypatch, child={}, parent={})
    remote, store, calls = _publish_to(tmp_path, monkeypatch, cards)
    # The line names a long path, so the console must not wrap it.
    monkeypatch.setattr(cli_module.console, "rich", Console(stderr=True, width=1000))
    directory = _store_from_card(CHILD.dataset_id, store.root, force=False)
    if held_in in {"both", "destination-only"}:
        shutil.copytree(directory, release_dir(remote.local_root, CHILD))
    if held_in == "destination-only":
        shutil.rmtree(store.root)
    options = [] if held_in == "store" else ["--publish-to", "s3://bucket/registry"]

    result = runner.invoke(app, ["build", CHILD.dataset_id, "--dry-run", *options])

    assert result.exit_code == 0, result.output
    if held_in == "store-only":
        # The store holds the release and the destination does not, so only the publish is left to do.
        assert result.stdout.splitlines() == [f"then publish test/child@1.0.0 to {remote}"]
        assert result.stderr == ""
    else:
        held_by = store if held_in == "store" else remote
        assert result.stdout == ""
        assert result.stderr.splitlines() == [f"✅ Nothing to do: 'test/child' 1.0.0 is already in {held_by}"]
    assert calls == []
