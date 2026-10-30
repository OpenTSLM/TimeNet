import shutil

import numpy as np
import pytest

from timenet import builders
from timenet.connectors import BaseConnector
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.engine import run_pipeline
from timenet.errors import (
    TimeFFormatError,
    TimeFValidationError,
    TimeNetAccessError,
    TimeNetFileError,
    TimeNetMissingParentError,
)
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, S3Registry, WritableRegistry
from timenet.registry._paths import release_dir
from timenet.testing import make_dataset
from timenet.types import Access, TimeSeriesSpec, ureg
from timenet.types.access import AccessRequirement
from timenet.types.parent_dataset_card import ParentDatasetCard
from timenet.types.reference_keys import ParentAlias, SourceKey
from timenet.types.references import DatasetRef
from timenet_connectors.builder.backend import ConnectorBuilder
from timenet_connectors.builder.plan import ParentStep, plan_parents, prepare_parents


GRAND = DatasetRef(dataset_id=make_dataset().metadata.dataset_id, version=make_dataset().metadata.dataset_version)
PARENT = DatasetRef.parse("test/parent@1.0.0")
CHILD = DatasetRef.parse("test/child@1.0.0")


class _CardBackend:
    """A builder backend that serves connector cards by dataset ID."""

    def __init__(self, cards):
        self.cards = cards

    def knows(self, dataset_id):
        return dataset_id in self.cards

    def declared_version(self, dataset_id):
        return str(self.cards[dataset_id].dataset_version)

    def card(self, dataset_id):
        return self.cards.get(dataset_id)

    def build(self, dataset_id, root, *, force=False):
        raise AssertionError(f"the plan does not build {dataset_id}")


class _Entry:
    def __init__(self, backend):
        self._backend = backend

    def load(self):
        return lambda: self._backend


def _cards(**parents: dict[str, DatasetRef]) -> dict:
    """Return connector cards by dataset ID; each keyword is ``<name>={alias: ref}`` for ``test/<name>``."""
    cards = {GRAND.dataset_id: make_dataset().metadata}
    for name, declared in parents.items():
        ref = DatasetRef.parse(f"test/{name}@1.0.0")
        cards[ref.dataset_id] = make_dataset().metadata.model_copy(
            update={"dataset_id": ref.dataset_id, "parents": declared}
        )
    return cards


def _no_build(step: ParentStep) -> None:
    pytest.fail(f"the plan built {step.reference}")


def _install_cards(monkeypatch, **parents: dict[str, DatasetRef]) -> dict:
    """Register one builder backend that serves the cards of ``_cards(**parents)``.

    Returns:
        The cards by dataset ID, which a test may change.
    """
    cards = _cards(**parents)
    monkeypatch.setattr(builders, "entry_points", lambda group: [_Entry(_CardBackend(cards))])
    return cards


@pytest.mark.parametrize(
    ("grand_in", "expected"),
    [
        ("output", [ParentStep(PARENT, "build")]),
        ("origin", [ParentStep(GRAND, "fetch"), ParentStep(PARENT, "build")]),
        ("nowhere", [ParentStep(GRAND, "build"), ParentStep(PARENT, "build")]),
    ],
)
def test_plan_uses_fetches_or_builds_each_parent_once_before_its_children(tmp_path, monkeypatch, grand_in, expected):
    _install_cards(monkeypatch, child={"parent": PARENT, "grand": GRAND}, parent={"grand": GRAND})
    registries = {name: LocalRegistry(tmp_path / name) for name in ("output", "origin")}
    if grand_in in registries:
        registries[grand_in].store(make_dataset())

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
        ({"child": {"parent": PARENT}}, {"remote_output": True}, TimeFValidationError, "local output"),
    ],
    ids=["missing", "other-version", "cycle", "fetch-into-remote"],
)
def test_plan_rejects_unobtainable_parents_before_anything_runs(  # noqa: PLR0913, PLR0917
    tmp_path, monkeypatch, cards, options, error, match
):
    _install_cards(monkeypatch, **cards)
    output = LocalRegistry(tmp_path / "output")
    fetch_from = None
    if options.pop("remote_output", False):
        output = S3Registry("s3://bucket/registry", cache_dir=tmp_path / "cache")
        fetch_from = LocalRegistry(tmp_path / "origin")

    with pytest.raises(error, match=match):
        plan_parents(CHILD.dataset_id, output, fetch_from=fetch_from, **options)


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
        registry.store(child)


def _declared_without_its_sources_source(registry: LocalRegistry) -> None:
    """Store GRAND, then SOURCE, which reads a GRAND Record, then DECLARED, which reads only SOURCE's own values.

    DECLARED's payload sources are SOURCE alone; SOURCE's are GRAND. GRAND is removed afterwards.
    """
    registry.store(make_dataset())
    _child_of(registry, SOURCE, GRAND, make_dataset().records[0].id)
    # SOURCE's own Record, which holds values it stores itself, takes the second Record ID.
    _child_of(registry, DECLARED, SOURCE, 2)
    assert {str(source.release) for source in registry.get_manifest(DECLARED.dataset_id).payload_sources.values()} == {
        str(SOURCE)
    }
    shutil.rmtree(release_dir(registry.root, GRAND))


@pytest.mark.parametrize("where", ["output", "origin", "source-in-origin"])
def test_a_declared_parent_needs_only_its_direct_payload_sources(tmp_path, monkeypatch, where):
    _install_cards(monkeypatch, child={"declared": DECLARED})
    output, origin = LocalRegistry(tmp_path / "output"), LocalRegistry(tmp_path / "origin")
    _declared_without_its_sources_source(origin if where == "origin" else output)
    if where == "source-in-origin":
        release_dir(origin.root, SOURCE).parent.mkdir(parents=True)
        shutil.move(release_dir(output.root, SOURCE), release_dir(origin.root, SOURCE))

    steps = plan_parents(CHILD.dataset_id, output, fetch_from=origin)

    # DECLARED opens with SOURCE, its direct payload source, which a fetch of DECLARED brings; GRAND,
    # SOURCE's own source, is not needed.
    assert steps == ([] if where == "output" else [ParentStep(DECLARED, "fetch")])
    prepare_parents(CHILD.dataset_id, output, build=_no_build, fetch_from=origin)
    for reference in (DECLARED, SOURCE):
        assert output.exists(reference.dataset_id, str(reference.version))
    assert not output.exists(GRAND.dataset_id, str(GRAND.version))


def test_the_plan_reads_each_card_from_the_backend_that_claims_the_dataset(tmp_path, monkeypatch):
    cards = _cards(child={"parent": PARENT}, parent={"grand": GRAND})
    del cards[GRAND.dataset_id]
    # The in-repo backend claims GRAND (the hello-world connector) but none of the test datasets,
    # which only the second backend knows.
    entries = [_Entry(ConnectorBuilder()), _Entry(_CardBackend(cards))]
    monkeypatch.setattr(builders, "entry_points", lambda group: entries)

    steps = plan_parents(CHILD.dataset_id, LocalRegistry(tmp_path), build_parents=True)

    assert steps == [ParentStep(GRAND, "build"), ParentStep(PARENT, "build")]
    cards[PARENT.dataset_id] = cards[PARENT.dataset_id].model_copy(update={"dataset_version": "2.0.0"})
    with pytest.raises(TimeFValidationError, match=r"test/parent@1\.0\.0 is missing .* builds version 2\.0\.0"):
        plan_parents(CHILD.dataset_id, LocalRegistry(tmp_path), build_parents=True)


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
    output.store(make_dataset())
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


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_a_pinned_source_is_rebuilt_in_staging_and_published_only_if_it_matches(tmp_path, monkeypatch, backend):
    _install_cards(monkeypatch, child={"parent": PARENT}, parent={"parent": GRAND})
    output = LocalRegistry(tmp_path / "output")
    _parent_without_its_source(output)
    pins = output.get_manifest(PARENT.dataset_id).payload_sources[SourceKey(str(GRAND))]

    steps = plan_parents(CHILD.dataset_id, output, build_parents=True)

    assert steps == [ParentStep(GRAND, "build", pinned_by={PARENT: pins})]

    def build(step):
        run_pipeline(_Grand(), output.root, values_backend=backend, pinned_by=step.pinned_by)

    if backend == "parquet":
        prepare_parents(CHILD.dataset_id, output, build=build, build_parents=True)
        assert output.exists(GRAND.dataset_id, str(GRAND.version))
        return
    # A Zarr rebuild writes other files than the Parquet files PARENT reads, so nothing is published.
    with pytest.raises(TimeFFormatError, match=r"Nothing was published\. Fetch the exact release .* or rebuild"):
        prepare_parents(CHILD.dataset_id, output, build=build, build_parents=True)
    assert not output.exists(GRAND.dataset_id, str(GRAND.version))
    assert not list(release_dir(output.root, GRAND).parent.glob("*"))


def test_a_damaged_parent_is_fetched_again_but_never_built_over(tmp_path, monkeypatch):
    _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    output, origin = LocalRegistry(tmp_path / "output"), LocalRegistry(tmp_path / "origin")
    for registry in (output, origin):
        registry.store(TimeFDataset(metadata=_cards(parent={})[PARENT.dataset_id]))
    (release_dir(output.root, PARENT) / "control.duckdb").write_bytes(b"cut short")

    with pytest.raises(TimeNetFileError, match=r"'control\.duckdb' has 9 bytes, expected [\d,]+ bytes\.$") as caught:
        plan_parents(CHILD.dataset_id, output, build_parents=True)
    assert caught.value.release == PARENT

    assert plan_parents(CHILD.dataset_id, output, fetch_from=origin) == [ParentStep(PARENT, "fetch")]
    prepare_parents(CHILD.dataset_id, output, build=_no_build, fetch_from=origin)
    assert plan_parents(CHILD.dataset_id, output) == []


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

    def store(self, dataset, **options):
        return self._local.store(dataset, **options)


_GATED = {"access": Access.CREDENTIALED, "access_url": "https://example.org/dua"}


@pytest.mark.parametrize("gated", ["child-card", "held-parent", "parent-card"])
def test_a_remote_output_refuses_a_plan_that_visits_a_release_that_is_not_open(tmp_path, monkeypatch, gated):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    gated_ref = CHILD if gated == "child-card" else PARENT
    cards[gated_ref.dataset_id] = cards[gated_ref.dataset_id].model_copy(update=_GATED)
    output = _Hosted(tmp_path / "hosted")
    if gated == "held-parent":
        output.store(TimeFDataset(metadata=cards[PARENT.dataset_id]))

    with pytest.raises(TimeNetAccessError) as caught:
        plan_parents(CHILD.dataset_id, output, build_parents=True)

    error = caught.value
    requirement = AccessRequirement(gated_ref, Access.CREDENTIALED, _GATED["access_url"])
    assert (error.dataset, error.access, error.requirements) == (gated_ref, Access.CREDENTIALED, (requirement,))


def test_a_remote_output_names_the_access_page_of_a_held_parent_from_its_lineage(tmp_path, monkeypatch):
    cards = _install_cards(monkeypatch, child={"parent": PARENT}, parent={})
    output = _Hosted(tmp_path / "hosted")
    # The card of the held parent stays open. Its lineage holds GRAND, which is credentialed.
    held = TimeFDataset(metadata=cards[PARENT.dataset_id])
    grand = make_dataset().metadata.model_copy(update=_GATED)
    held.lineage[str(GRAND)] = ParentDatasetCard(metadata=grand)
    output.store(held)

    with pytest.raises(TimeNetAccessError) as caught:
        plan_parents(CHILD.dataset_id, output, build_parents=True)

    assert (caught.value.dataset, caught.value.access) == (PARENT, Access.CREDENTIALED)
    assert caught.value.requirements == (AccessRequirement(GRAND, Access.CREDENTIALED, _GATED["access_url"]),)
