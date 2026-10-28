from pathlib import Path
import shutil

import numpy as np
import pytest
from typer.testing import CliRunner

from timenet.connectors import BuildContext
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError, TimeNetDatasetNotFoundError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, S3Registry
from timenet.registry._paths import release_dir
from timenet.testing import make_dataset
from timenet.types import TimeSeriesSpec, ureg
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef
from timenet_connectors.builder import cli as cli_module, plan as plan_module
from timenet_connectors.builder.cli import app
from timenet_connectors.builder.plan import ParentStep, plan_parents, prepare_parents


GRAND = DatasetRef(dataset_id=make_dataset().metadata.dataset_id, version=make_dataset().metadata.dataset_version)
PARENT = DatasetRef.parse("test/parent@1.0.0")
CHILD = DatasetRef.parse("test/child@1.0.0")
runner = CliRunner()


def _install_cards(monkeypatch, **parents: dict[str, DatasetRef]) -> None:
    """Serve connector cards by dataset ID; each keyword is ``<name>={alias: ref}`` for ``test/<name>``."""
    cards = {GRAND.dataset_id: make_dataset().metadata}
    for name, declared in parents.items():
        ref = DatasetRef.parse(f"test/{name}@1.0.0")
        cards[ref.dataset_id] = make_dataset().metadata.model_copy(
            update={"dataset_id": ref.dataset_id, "parents": declared}
        )

    def read_card(dataset_id):
        if dataset_id not in cards:
            raise LookupError(dataset_id)
        return cards[dataset_id]

    monkeypatch.setattr(plan_module, "read_card", read_card)


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
        ({"child": {"parent": PARENT}, "parent": {}}, {}, TimeNetDatasetNotFoundError, "--build-parents"),
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
    prepare_parents(CHILD.dataset_id, output, build=pytest.fail, fetch_from=origin)
    for reference in (DECLARED, SOURCE):
        assert output.exists(reference.dataset_id, str(reference.version))
    assert not output.exists(GRAND.dataset_id, str(GRAND.version))


def test_build_prepares_parents_before_the_isolated_child(tmp_path, monkeypatch):
    _install_cards(monkeypatch, child={"parent": PARENT, "grand": GRAND}, parent={})
    monkeypatch.setattr(cli_module, "connector_dir", lambda dataset_id: tmp_path)
    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    origin = LocalRegistry(tmp_path / "origin")
    origin.store(make_dataset())
    out = tmp_path / "out"
    calls = []

    def fake_isolated(dataset_id, root, *, force, **_):
        calls.append((dataset_id, force))
        registry = LocalRegistry(Path(root))
        metadata = plan_module.read_card(dataset_id)
        with BuildContext.open(metadata, registry) as context:
            dataset = TimeFDataset(metadata=metadata) if context.parents else make_dataset()
            dataset._metadata = metadata
            for handle in context.parents.values():
                dataset.import_from(handle, records=[make_dataset().records[0].id])
            context.check_imports(dataset)
            registry.store(dataset, force=force)
        return str(registry.root / dataset_id / str(metadata.dataset_version))

    monkeypatch.setattr(cli_module, "run_isolated", fake_isolated)
    result = runner.invoke(
        app,
        ["build", CHILD.dataset_id, "--out", str(out), "--fetch-parents", str(origin.root), "--build-parents", "-f"],
    )

    assert result.exit_code == 0, result.output
    assert (out / GRAND.dataset_id / str(GRAND.version) / "manifest.json").is_file()
    # The parent builds first and never with --force; only the requested build's result reaches stdout.
    assert calls == [(PARENT.dataset_id, False), (CHILD.dataset_id, True)]
    assert result.stdout.strip() == str(out / CHILD.dataset_id / str(CHILD.version))
