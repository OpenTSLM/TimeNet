from pathlib import Path

import pytest
from typer.testing import CliRunner

from timenet.connectors import BuildContext
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError, TimeNetDatasetNotFoundError
from timenet.registry import LocalRegistry, S3Registry
from timenet.testing import make_dataset
from timenet.types.references import DatasetRef
from timenet_connectors.builder import cli as cli_module, plan as plan_module
from timenet_connectors.builder.cli import app
from timenet_connectors.builder.plan import ParentStep, plan_parents


GRAND = DatasetRef(dataset_id=make_dataset().metadata.dataset_id, version=make_dataset().metadata.dataset_version)
PARENT = DatasetRef.parse("test/parent@1.0.0")
CHILD = DatasetRef.parse("test/child@1.0.0")
runner = CliRunner()


def _install_cards(monkeypatch, **inputs: dict[str, DatasetRef]) -> None:
    """Serve connector cards by dataset ID; each keyword is ``<name>={alias: ref}`` for ``test/<name>``."""
    cards = {GRAND.dataset_id: make_dataset().metadata}
    for name, declared in inputs.items():
        ref = DatasetRef.parse(f"test/{name}@1.0.0")
        cards[ref.dataset_id] = make_dataset().metadata.model_copy(
            update={"dataset_id": ref.dataset_id, "inputs": declared}
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
            dataset = TimeFDataset(metadata=metadata) if context.inputs else make_dataset()
            dataset._metadata = metadata
            for handle in context.inputs.values():
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
