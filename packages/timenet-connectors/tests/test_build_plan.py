from pathlib import Path
import shutil

import click
import numpy as np
import pytest
from typer.testing import CliRunner

from timenet import builders
from timenet.connectors import BaseConnector
from timenet.coordinator.build import build_release
from timenet.coordinator.plan import Step, plan_parents
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.engine import open_parents, run_pipeline
from timenet.errors import (
    TimeFFormatError,
    TimeFValidationError,
    TimeNetAccessError,
    TimeNetDatasetNotFoundError,
    TimeNetFileError,
    TimeNetInvalidCardError,
    TimeNetMissingParentError,
)
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, WritableRegistry
from timenet.registry._paths import release_dir
from timenet.testing import install_backends, make_dataset
from timenet.types import Access, TimeSeriesSpec, ureg
from timenet.types.access import AccessRequirement
from timenet.types.parent_dataset_card import ParentDatasetCard
from timenet.types.reference_keys import ParentAlias, SourceKey
from timenet.types.references import DatasetRef
from timenet.values_backends import ValuesBackend
import timenet_connectors
from timenet_connectors import api as api_module
from timenet_connectors.builder import backend as backend_module, cli as cli_module, env as env_module
from timenet_connectors.builder.backend import ConnectorBuilder
from timenet_connectors.builder.cli import app


GRAND = DatasetRef(dataset_id=make_dataset().metadata.dataset_id, version=make_dataset().metadata.dataset_version)
PARENT = DatasetRef.parse("test/parent@1.0.0")
CHILD = DatasetRef.parse("test/child@1.0.0")
runner = CliRunner()


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
        ("output", [Step(PARENT, "build")]),
        ("origin", [Step(GRAND, "copy"), Step(PARENT, "build", depends_on=(GRAND,))]),
        ("nowhere", [Step(GRAND, "build"), Step(PARENT, "build", depends_on=(GRAND,))]),
    ],
)
def test_plan_uses_fetches_or_builds_each_parent_once_before_its_children(tmp_path, monkeypatch, grand_in, expected):
    _install_cards(monkeypatch, child={"parent": PARENT, "grand": GRAND}, parent={"grand": GRAND})
    registries = {name: LocalRegistry(tmp_path / name) for name in ("output", "origin")}
    if grand_in in registries:
        registries[grand_in].put_dataset(make_dataset())

    steps = plan_parents(CHILD.dataset_id, registries["output"], fetch_from=registries["origin"], build_parents=True)

    assert steps == expected


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
            {"build_parents": True},
            TimeFValidationError,
            "not that exact version",
        ),
        (
            {"child": {"parent": PARENT}, "parent": {"child": CHILD}},
            {"build_parents": True},
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
        plan_parents(CHILD.dataset_id, output, **options)


_SPEC = TimeSeriesSpec(spec_type="sine", name="Sine", unit_value=ureg.dimensionless)
SOURCE = DatasetRef.parse("test/source@1.0.0")
DECLARED = DatasetRef.parse("test/declared@1.0.0")


def _child_of(registry: LocalRegistry, reference: DatasetRef, parent: DatasetRef, record_id: int) -> None:
    """Store ``reference``, which imports one Record of ``parent`` and adds one Record of its own values."""
    metadata = make_dataset().metadata.model_copy(
        update={"dataset_id": reference.dataset_id, "parents": {ParentAlias("parent"): parent}}
    )
    child = TimeFDataset(metadata=metadata)
    with TimeFReader(registry.open_version(parent.dataset_id, str(parent.version))) as reader:
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

    steps = plan_parents(CHILD.dataset_id, output, fetch_from=origin)

    # DECLARED opens with SOURCE, its direct payload source, which a copy of DECLARED brings. So the
    # plan does not name GRAND, SOURCE's own source.
    assert steps == ([] if where == "output" else [Step(DECLARED, "copy")])
    if where != "output":
        # But a copy takes a release with each payload source that it needs, at every depth, so that
        # the store never holds a release without its payload sources. That includes GRAND. The
        # registry to copy from lacks GRAND, so the copy fails before it copies anything.
        with pytest.raises(
            TimeNetMissingParentError, match=r"^payload source timenet/hello-world@1\.0\.0 of test/source"
        ):
            build_release(CHILD.dataset_id, store=output, fetch_from=origin)
        assert child_builds == []
        return
    build_release(CHILD.dataset_id, store=output, fetch_from=origin)
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

    steps = plan_parents(CHILD.dataset_id, output, build_parents=True)

    assert steps == [Step(GRAND, "build", pinned_by={SOURCE: pins})]


def test_the_plan_reads_each_card_from_the_backend_that_claims_the_dataset(tmp_path, monkeypatch):
    cards = _cards(child={"parent": PARENT}, parent={"grand": GRAND})
    del cards[GRAND.dataset_id]
    # The in-repo backend claims GRAND (the hello-world connector) but none of the test datasets,
    # which only the second backend knows.
    install_backends(monkeypatch, ConnectorBuilder(), _CardBackend(cards))

    steps = plan_parents(CHILD.dataset_id, LocalRegistry(tmp_path), build_parents=True)

    assert steps == [Step(GRAND, "build"), Step(PARENT, "build", depends_on=(GRAND,))]
    cards[PARENT.dataset_id] = cards[PARENT.dataset_id].model_copy(update={"dataset_version": "2.0.0"})
    with pytest.raises(TimeFValidationError, match=r"test/parent@1\.0\.0 is missing .* builds version 2\.0\.0"):
        plan_parents(CHILD.dataset_id, LocalRegistry(tmp_path), build_parents=True)


def test_a_parent_whose_card_cannot_be_read_fails_the_plan_with_the_card_error(tmp_path, monkeypatch):
    (tmp_path / "dataset.yaml").write_text("dataset_id: [unclosed\n")
    # The in-repo backend claims PARENT, whose card cannot be read. The second backend serves CHILD.
    monkeypatch.setattr(backend_module, "known_ids", lambda: [PARENT.dataset_id])
    monkeypatch.setattr(backend_module, "connector_dir", lambda dataset_id: tmp_path)
    install_backends(monkeypatch, ConnectorBuilder(), _CardBackend(_cards(child={"parent": PARENT})))

    with pytest.raises(TimeNetInvalidCardError, match="could not read dataset card") as caught:
        plan_parents(CHILD.dataset_id, LocalRegistry(tmp_path / "output"), build_parents=True)

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
        plan_parents(CHILD.dataset_id, output)

    assert (caught.value.missing, caught.value.needed_by, caught.value.payload_source) == (GRAND, PARENT, True)
    assert str(caught.value).startswith("payload source timenet/hello-world@1.0.0 of test/parent@1.0.0 is not in")


@pytest.mark.parametrize("backend", [ValuesBackend.PARQUET, ValuesBackend.ZARR])
def test_a_pinned_source_is_rebuilt_in_staging_and_published_only_if_it_matches(tmp_path, monkeypatch, backend):
    _install_cards(monkeypatch, child={"parent": PARENT}, parent={"parent": GRAND})
    monkeypatch.setattr(_Grand, "values_backend", backend)
    output = LocalRegistry(tmp_path / "output")
    _parent_without_its_source(output)
    pins = output.get_manifest(PARENT.dataset_id).payload_sources[SourceKey(str(GRAND))]

    steps = plan_parents(CHILD.dataset_id, output, build_parents=True)

    assert steps == [Step(GRAND, "build", pinned_by={PARENT: pins})]
    built = []

    def child(release, store_root, *, pinned_by, **_):
        built.append(release)
        if release == GRAND:
            run_pipeline(_Grand(), store_root, pinned_by=pinned_by)
        else:
            _store_from_card(release.dataset_id, store_root, force=False)

    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    monkeypatch.setattr(backend_module, "run_isolated_build", child)

    if backend == ValuesBackend.PARQUET:
        build_release(CHILD.dataset_id, store=output, build_parents=True)
        assert output.exists(GRAND.dataset_id, str(GRAND.version))
        assert built == [GRAND, CHILD]
        return
    # A Zarr rebuild writes other files than the Parquet files PARENT reads, so nothing is published.
    with pytest.raises(TimeFFormatError, match=r"Nothing was published\. Fetch the exact release .* or rebuild"):
        build_release(CHILD.dataset_id, store=output, build_parents=True)
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

    steps = plan_parents(CHILD.dataset_id, output, build_parents=True)

    assert steps == [Step(GRAND, "build", pinned_by=pins)]


def test_a_built_release_needs_only_its_payload_sources(tmp_path, monkeypatch, child_builds):
    _install_cards(monkeypatch, parent={"parent": GRAND})
    output = LocalRegistry(tmp_path / "output")
    _parent_without_its_source(output)
    pins = output.get_manifest(PARENT.dataset_id).payload_sources[SourceKey(str(GRAND))]

    with pytest.raises(TimeNetMissingParentError, match=r"^payload source timenet/hello-world@1\.0\.0 of test/parent"):
        plan_parents(PARENT.dataset_id, output)
    # The source that PARENT reads builds against its pins.
    assert plan_parents(PARENT.dataset_id, output, build_parents=True) == [
        Step(GRAND, "build", pinned_by={PARENT: pins})
    ]
    # A rebuild of PARENT needs its declared parents instead, and pins the new files.
    assert plan_parents(PARENT.dataset_id, output, build_parents=True, force_build=True) == [Step(GRAND, "build")]
    # A registry that holds the source fills the gap: the local PARENT is reused and GRAND is copied.
    origin = LocalRegistry(tmp_path / "origin")
    origin.put_dataset(make_dataset())
    assert plan_parents(PARENT.dataset_id, output, fetch_from=origin) == [Step(PARENT, "copy")]
    build_release(PARENT.dataset_id, store=output, fetch_from=origin)
    assert output.exists(GRAND.dataset_id, str(GRAND.version))
    assert child_builds == []


def test_a_damaged_parent_is_fetched_again_but_never_built_over(tmp_path, monkeypatch, child_builds):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    output, origin = LocalRegistry(tmp_path / "output"), LocalRegistry(tmp_path / "origin")
    parent = make_dataset()
    parent._metadata = cards[PARENT.dataset_id]
    for registry in (output, origin):
        registry.put_dataset(parent)
    (release_dir(output.root, PARENT) / "control.duckdb").write_bytes(b"cut short")

    with pytest.raises(TimeNetFileError, match=r"'control\.duckdb' has 9 bytes, expected [\d,]+ bytes\.$") as caught:
        plan_parents(CHILD.dataset_id, output, build_parents=True)
    assert caught.value.release == PARENT

    assert plan_parents(CHILD.dataset_id, output, fetch_from=origin) == [Step(PARENT, "copy")]
    build_release(CHILD.dataset_id, store=output, fetch_from=origin)
    assert plan_parents(CHILD.dataset_id, output) == []
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

    def open_version(self, dataset_id, version=None):
        return self._local.open_version(dataset_id, version)

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
        plan_parents(CHILD.dataset_id, store, build_parents=True, open_access_only=True)

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
        plan_parents(CHILD.dataset_id, store, build_parents=True, open_access_only=True)

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
    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    calls = []
    monkeypatch.setattr(backend_module, "run_isolated_build", _prepared_builds(calls))
    return calls


def test_build_prepares_parents_before_the_isolated_child(tmp_path, monkeypatch):
    cards = _install_cards(monkeypatch, child={"parent": PARENT, "grand": GRAND}, parent={})
    monkeypatch.setattr(cli_module, "known_ids", lambda: list(cards))
    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    origin = LocalRegistry(tmp_path / "origin")
    origin.put_dataset(make_dataset())
    out = tmp_path / "out"
    calls = []
    monkeypatch.setattr(backend_module, "run_isolated_build", _prepared_builds(calls))

    result = runner.invoke(
        app,
        ["build", CHILD.dataset_id, "--out", str(out), "--fetch-from", str(origin.root), "--build-parents", "-f"],
    )

    assert result.exit_code == 0, result.output
    assert (out / GRAND.dataset_id / str(GRAND.version) / "manifest.json").is_file()
    # The parent builds first and never with --force; only the requested build's result reaches stdout.
    assert calls == [(PARENT.dataset_id, False), (CHILD.dataset_id, True)]
    assert result.stdout.strip() == str(out / CHILD.dataset_id / str(CHILD.version))


def test_a_built_version_starts_no_isolated_child(tmp_path, monkeypatch):
    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    monkeypatch.setattr(env_module.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("started a subprocess"))
    out = LocalRegistry(tmp_path / "out")
    out.put_dataset(make_dataset())

    result = runner.invoke(app, ["build", GRAND.dataset_id, "--out", str(out.root)])

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == str(release_dir(out.root, GRAND))


@pytest.mark.parametrize("entry", ["api", "coordinator"])
def test_every_build_entry_point_prepares_parents(tmp_path, monkeypatch, child_builds, entry):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    out = tmp_path / "out"

    def build(fetch_from=None):
        if entry == "api":
            timenet_connectors.build(CHILD.dataset_id, out=out, fetch_from=fetch_from)
        else:
            # The client builds through the coordinator.
            build_release(CHILD.dataset_id, store=LocalRegistry(out), fetch_from=fetch_from)

    with pytest.raises(
        TimeNetMissingParentError, match=r"parent release test/parent@1\.0\.0 of test/child@1\.0\.0"
    ) as caught:
        build()
    assert child_builds == []
    if entry == "api":
        assert str(caught.value).endswith(". Copy it with fetch_from=<registry> or build it with build_parents=True.")

    # A registry instance to fetch from, as the SDK layers accept one.
    origin = LocalRegistry(tmp_path / "origin")
    parent = make_dataset()
    parent._metadata = cards[PARENT.dataset_id]
    origin.put_dataset(parent)
    build(fetch_from=origin)
    assert child_builds == [(CHILD.dataset_id, False)]
    assert LocalRegistry(out).exists(PARENT.dataset_id, str(PARENT.version))


@pytest.mark.parametrize("obtain", ["fetch_from", "build_parents"])
def test_load_obtains_a_missing_parent_before_it_builds_and_reads(tmp_path, monkeypatch, obtain):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TIMENET_REGISTRY", str(tmp_path / "out"))
    calls = []
    monkeypatch.setattr(backend_module, "run_isolated_build", _prepared_builds(calls))
    origin = LocalRegistry(tmp_path / "origin")
    parent = make_dataset()
    parent._metadata = cards[PARENT.dataset_id]
    origin.put_dataset(parent)

    with pytest.raises(TimeNetMissingParentError, match=r"^parent release test/parent@1\.0\.0 of test/child"):
        timenet_connectors.load(CHILD.dataset_id)
    if obtain == "fetch_from":
        dataset = timenet_connectors.load(CHILD.dataset_id, fetch_from=origin)
    else:
        dataset = timenet_connectors.load(CHILD.dataset_id, build_parents=True)

    assert dataset.metadata.dataset_id == CHILD.dataset_id
    # The child holds the Record it imported from the parent.
    assert [record.id for record in dataset.records] == [make_dataset().records[0].id]
    builds = [(CHILD.dataset_id, False)]
    assert calls == (builds if obtain == "fetch_from" else [(PARENT.dataset_id, False), *builds])


def test_load_copies_a_missing_payload_source_through_fetch_from(tmp_path, monkeypatch):
    out = LocalRegistry(tmp_path / "out")
    _parent_without_its_source(out)
    origin = LocalRegistry(tmp_path / "origin")
    origin.put_dataset(make_dataset())
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TIMENET_REGISTRY", str(out.root))
    # The build has nothing to do, and no builder backend can build the payload source, so only the
    # read can copy it.
    monkeypatch.setattr(api_module, "build", lambda dataset_id, **options: None)
    monkeypatch.setattr(builders, "entry_points", lambda group: [])

    dataset = timenet_connectors.load(PARENT.dataset_id, fetch_from=origin)

    assert out.exists(GRAND.dataset_id, str(GRAND.version))
    assert len(dataset.records) == 2


class _NoDownloads(_Hosted):
    """A registry that is not local and fails the test if a file is read from it."""

    def open_file(self, dataset_id, version, relpath):
        pytest.fail(f"downloaded {dataset_id}@{version}/{relpath}")


@pytest.mark.parametrize("page", ["own", "ancestor"])
def test_a_fetch_of_a_release_that_is_not_open_fails_before_any_download(tmp_path, monkeypatch, child_builds, page):
    cards = _install_cards(monkeypatch, child={"grand": GRAND, "parent": PARENT}, parent={})
    origin = _NoDownloads(tmp_path / "origin")
    # GRAND is open and would be fetched first.
    origin.put_dataset(make_dataset())
    ancestor = DatasetRef.parse("test/ancestor@1.0.0")
    if page == "own":
        parent, requirement = TimeFDataset(metadata=cards[PARENT.dataset_id].model_copy(update=_GATED)), PARENT
    else:
        # The card of PARENT stays open. Its lineage holds an ancestor, which is credentialed.
        parent = TimeFDataset(metadata=cards[PARENT.dataset_id])
        card = make_dataset().metadata.model_copy(update={"dataset_id": ancestor.dataset_id, **_GATED})
        parent._import_state.lineage[str(ancestor)] = ParentDatasetCard(metadata=card)
        requirement = ancestor
    origin.put_dataset(parent)
    output = LocalRegistry(tmp_path / "out")

    with pytest.raises(TimeNetAccessError) as caught:
        build_release(CHILD.dataset_id, store=output, fetch_from=origin)

    assert caught.value.dataset == PARENT
    assert child_builds == []
    assert caught.value.requirements == (AccessRequirement(requirement, Access.CREDENTIALED, _GATED["access_url"]),)
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

    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    monkeypatch.setattr(env_module, "_run_build", child)
    monkeypatch.setattr(cli_module, "known_ids", lambda: list(cards))
    monkeypatch.setattr(backend_module, "run_pipeline", run_pipeline)

    # The output holds PARENT, so the build gets only GRAND, the payload source of PARENT.
    result = runner.invoke(app, ["build", PARENT.dataset_id, "--out", str(output.root), "--build-parents"])

    assert result.exit_code == 0, result.output
    # The rebuild of GRAND gets the pins of PARENT, through a file that is gone afterwards.
    assert received == [{PARENT: pins}]
    assert len(pin_files) == 1
    assert not pin_files[0].exists()
    assert output.exists(GRAND.dataset_id, str(GRAND.version))


STORED = DatasetRef.parse("test/stored@1.0.0")


class _CopyBackend(_CardBackend):
    """A second builder backend that rebuilds STORED from a saved copy, and the other releases from their cards.

    It records each call.
    """

    def __init__(self, cards, saved, calls):
        super().__init__(cards)
        self.saved = saved
        self.calls = calls

    def build_prepared(self, release, store, *, force, pinned_by, options):
        self.calls.append((release.dataset_id, dict(pinned_by)))
        if release == STORED:
            shutil.copytree(self.saved, release_dir(store.root, STORED))
        else:
            _store_from_card(release.dataset_id, store.root, force=force)


def test_a_parent_that_another_backend_claims_builds_there_with_its_pins(tmp_path, monkeypatch, child_builds):
    output = LocalRegistry(tmp_path / "output")
    stored = make_dataset()
    stored._metadata = stored.metadata.model_copy(update={"dataset_id": STORED.dataset_id})
    output.put_dataset(stored)
    _child_of(output, PARENT, STORED, make_dataset().records[0].id)
    pins = output.get_manifest(PARENT.dataset_id).payload_sources[SourceKey(str(STORED))]
    saved = tmp_path / "saved"
    shutil.move(release_dir(output.root, STORED), saved)
    calls = []
    # The in-repo backend comes first but claims none of the test datasets.
    backends = [ConnectorBuilder(), _CopyBackend(_cards(child={"parent": PARENT}, parent={}, stored={}), saved, calls)]
    install_backends(monkeypatch, *backends)

    build_release(CHILD.dataset_id, store=output, build_parents=True)

    # The second backend rebuilds STORED against the pins of PARENT. It gets no options to plan or
    # fetch the parents of STORED. Then it builds CHILD, which it claims too.
    assert calls == [(STORED.dataset_id, {PARENT: pins}), (CHILD.dataset_id, {})]
    assert child_builds == []


class _OwnBuild(_InRepoCards):
    """A subclass of the builder backend of this package that overrides ``build_prepared()``, and records each call."""

    def __init__(self, cards, calls):
        super().__init__(cards)
        self.calls = calls

    def build_prepared(self, release, store, *, force, pinned_by, options):
        self.calls.append((release.dataset_id, dict(pinned_by)))
        _store_from_card(release.dataset_id, store.root, force=force)


@pytest.mark.parametrize("overrides", [True, False], ids=["overrides-build", "inherits-build"])
def test_a_subclass_of_the_in_repo_backend_builds_through_its_own_build_prepared(
    tmp_path, monkeypatch, child_builds, overrides
):
    cards = _cards(child={"parent": PARENT}, parent={})
    own = []
    backend = _OwnBuild(cards, own) if overrides else _InRepoCards(cards)
    install_backends(monkeypatch, backend)
    output = LocalRegistry(tmp_path / "output")

    build_release(CHILD.dataset_id, store=output, build_parents=True)

    assert output.exists(PARENT.dataset_id, str(PARENT.version))
    if overrides:
        # Each release builds through the build_prepared() of the subclass.
        assert own == [(PARENT.dataset_id, {}), (CHILD.dataset_id, {})]
        assert child_builds == []
    else:
        # Without an own build_prepared(), each release takes the built-in path.
        assert child_builds == [(PARENT.dataset_id, False), (CHILD.dataset_id, False)]


class _Proxy:
    """A builder backend that is not a ``ConnectorBuilder``, and whose methods come from ``__getattr__``."""

    def __init__(self, backend):
        self._backend = backend

    def __getattr__(self, name):
        return getattr(self._backend, name)


def test_a_proxy_backend_builds_through_the_build_prepared_of_its_instance(tmp_path, monkeypatch, child_builds):
    own = []
    # The class of the proxy has no build_prepared(). The instance forwards it to a backend that records it.
    backend = _Proxy(_OwnBuild(_cards(child={"parent": PARENT}, parent={}), own))
    assert not hasattr(_Proxy, "build_prepared")
    install_backends(monkeypatch, backend)
    output = LocalRegistry(tmp_path / "output")

    build_release(CHILD.dataset_id, store=output, build_parents=True)

    assert output.exists(PARENT.dataset_id, str(PARENT.version))
    assert own == [(PARENT.dataset_id, {}), (CHILD.dataset_id, {})]
    assert child_builds == []


@pytest.mark.parametrize("case", ["missing", "damaged", "not-open"])
def test_the_cli_hint_names_its_flags(tmp_path, monkeypatch, case):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    monkeypatch.setattr(cli_module, "known_ids", lambda: list(cards))
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    out = LocalRegistry(tmp_path / "out")
    if case == "damaged":
        out.put_dataset(TimeFDataset(metadata=cards[PARENT.dataset_id]))
        (release_dir(out.root, PARENT) / "control.duckdb").write_bytes(b"cut short")
    if case == "not-open":
        cards[CHILD.dataset_id] = cards[CHILD.dataset_id].model_copy(update=_GATED)
        monkeypatch.setattr(cli_module, "_resolve_target", lambda target: _Hosted(tmp_path / "hosted"))

    # A damaged parent is never built over, even with --build-parents.
    options = ["--build-parents"] if case == "damaged" else []
    result = runner.invoke(app, ["build", CHILD.dataset_id, "--out", str(out.root), *options])

    hints = {
        "missing": (
            TimeNetMissingParentError,
            ". Fetch it with --fetch-from <registry> or build it with --build-parents.",
        ),
        # A rebuild of a release that others read changes the files they pin.
        "damaged": (
            TimeNetFileError,
            ". Fetch an intact copy with --fetch-from <registry>, or rebuild it with "
            "`timenet-build build test/parent --force`. "
            "A rebuild changes its files, and the releases that pin the old files then fail to open.",
        ),
        "not-open": (
            TimeNetAccessError,
            ". A remote --out holds open releases only. Build it into a local --out instead.",
        ),
    }
    error, hint = hints[case]
    assert isinstance(result.exception, error)
    assert str(result.exception).endswith(hint)


def _remote_out(tmp_path, monkeypatch, cards) -> tuple[_Hosted, LocalRegistry, list]:
    """Make the ``--out`` of the CLI a registry that is not local, and record each isolated build.

    Returns:
        The registry that ``--out`` names, the default local registry, and the list of the builds.
    """
    remote = _Hosted(tmp_path / "hosted")
    monkeypatch.setattr(cli_module, "_resolve_target", lambda target: remote)
    monkeypatch.setattr(cli_module, "known_ids", lambda: list(cards))
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    calls = []
    monkeypatch.setattr(backend_module, "run_isolated_build", _prepared_builds(calls))
    return remote, LocalRegistry(tmp_path / "home/registry"), calls


def test_build_to_a_remote_out_refuses_a_credentialed_release_before_any_download(tmp_path, monkeypatch):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    cards[PARENT.dataset_id] = cards[PARENT.dataset_id].model_copy(update=_GATED)
    origin = LocalRegistry(tmp_path / "origin")
    origin.put_dataset(TimeFDataset(metadata=cards[PARENT.dataset_id]))
    _, local, calls = _remote_out(tmp_path, monkeypatch, cards)

    command = ["build", CHILD.dataset_id, "--out", "s3://bucket/registry", "--fetch-from", str(origin.root)]

    result = runner.invoke(app, command)

    # The build into the local registry gets the access checks of the remote registry.
    assert isinstance(result.exception, TimeNetAccessError), result.output
    assert result.exception.dataset == PARENT
    assert calls == []
    assert not local.exists(PARENT.dataset_id, str(PARENT.version))


def test_build_to_a_remote_out_builds_locally_and_copies(tmp_path, monkeypatch):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    origin = LocalRegistry(tmp_path / "origin")
    parent = make_dataset()
    parent._metadata = cards[PARENT.dataset_id]
    origin.put_dataset(parent)
    remote, local, calls = _remote_out(tmp_path, monkeypatch, cards)
    command = ["build", CHILD.dataset_id, "--out", "s3://bucket/registry", "--fetch-from", str(origin.root)]

    result = runner.invoke(app, command)

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == str(CHILD.version)
    # The parent lands in the local registry first, and the child builds there. Then the copy takes
    # the child and its payload source, the parent, to the remote registry.
    assert calls == [(CHILD.dataset_id, False)]
    for registry in (local, remote):
        assert registry.exists(CHILD.dataset_id, str(CHILD.version))
        assert registry.exists(PARENT.dataset_id, str(PARENT.version))
    # A remote registry that holds the version gets nothing, and nothing builds.
    shutil.rmtree(local.root)
    result = runner.invoke(app, command)
    assert (result.exit_code, result.stdout.strip()) == (0, str(CHILD.version))
    assert result.stderr.splitlines() == ["✅ Nothing to do: 'test/child' 1.0.0 is already in the remote --out"]
    assert calls == [(CHILD.dataset_id, False)]
    assert not local.root.exists()


def test_build_to_a_remote_out_copies_a_missing_parent_from_that_registry(tmp_path, monkeypatch):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    remote, local, calls = _remote_out(tmp_path, monkeypatch, cards)
    parent = make_dataset()
    parent._metadata = cards[PARENT.dataset_id]
    remote.put_dataset(parent)

    result = runner.invoke(app, ["build", CHILD.dataset_id, "--out", "s3://bucket/registry"])

    # Without --fetch-from, the build copies the parent from the remote --out, which holds it, into
    # the local registry. So only the child builds.
    assert result.exit_code == 0, result.output
    assert calls == [(CHILD.dataset_id, False)]
    assert local.exists(PARENT.dataset_id, str(PARENT.version))
    assert remote.exists(CHILD.dataset_id, str(CHILD.version))


def _files(root: Path) -> dict[Path, tuple[int, int]]:
    """Return the size and the modification time of each file under ``root``, by its path."""
    return {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in root.rglob("*") if path.is_file()}


def test_force_is_a_usage_error_if_the_remote_out_holds_the_version(tmp_path, monkeypatch):
    cards = _install_cards(monkeypatch, child={}, parent={})
    _, local, calls = _remote_out(tmp_path, monkeypatch, cards)
    command = ["build", CHILD.dataset_id, "--out", "s3://bucket/registry"]
    assert runner.invoke(app, command).exit_code == 0
    held = _files(local.root)

    result = runner.invoke(app, [*command, "--force"])

    # A published version never changes. So the command refuses before anything downloads or builds.
    assert result.exit_code == 2, result.output
    output = click.unstyle(result.output)
    assert "Invalid value for --force" in output
    # The error box wraps the message and draws a border on each line.
    assert "a published version never changes" in " ".join(output.replace("│", " ").split())
    assert calls == [(CHILD.dataset_id, False)]
    assert _files(local.root) == held
