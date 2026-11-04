from pathlib import Path
import tempfile
from typing import Any, ClassVar, cast

import httpx
import pytest

from timenet.config import settings
from timenet.connectors import BaseConnector, BuildContext
from timenet.dataset import TimeFDataset
from timenet.engine import open_parents, publish_pipeline, run_pipeline, store_dataset
from timenet.errors import TimeFValidationError, TimeNetAccessError, TimeNetDatasetNotFoundError
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, RemoteRegistry
from timenet.registry.writable import WritableRegistry
from timenet.testing import make_dataset
from timenet.types import Access, Version
from timenet.types.access import AccessRequirement, effective_access
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef
from timenet.values_backends import ValuesBackend
from timenet.writer import TimeFWriter


def _write_demo_card() -> Path:
    """Write a dataset.yaml mirroring ``make_dataset()``'s metadata, for the demo connector's card.

    The demo connector declares its identity in a card like a real connector, rather than overriding
    ``metadata()``. Generating the card from ``make_dataset().metadata`` keeps the two in step as the
    fixture's id/version change across the stack.

    Returns:
        Path to the written card.
    """
    m = make_dataset().metadata
    lines = [
        f"dataset_id: {m.dataset_id}",
        f"dataset_version: {m.dataset_version}",
        f'name: "{m.name}"',
        f'description: "{m.description}"',
        f"license: {m.license}",
    ]
    if m.domains:
        lines.append("domains:")
        lines += [f"  - {d}" for d in m.domains]
    if m.tags:
        lines.append("tags:")
        lines += [f"  - {t}" for t in m.tags]
    card = Path(tempfile.mkdtemp()) / "dataset.yaml"
    card.write_text("\n".join(lines) + "\n")
    return card


_DEMO_CARD = _write_demo_card()


class _DemoConnector(BaseConnector[str]):
    CARD = _DEMO_CARD

    def download(self, cache_dir: Path) -> list[str]:
        return ["ref"]

    def convert(self, raw_refs: list[str]) -> TimeFDataset:
        return make_dataset()


def test_store_writes_a_readable_layout(tmp_path):
    connector = _DemoConnector()
    dataset = connector.convert(connector.download(tmp_path))
    version_dir = store_dataset(dataset, tmp_path)
    assert (version_dir / "manifest.json").exists()
    assert version_dir == tmp_path / "timenet/hello-world" / "1.0.0"


def test_store_derives_schema_if_needed(tmp_path):
    connector = _DemoConnector()
    dataset = connector.convert(connector.download(tmp_path))
    assert dataset.schema is None
    store_dataset(dataset, tmp_path)  # should derive schema itself
    assert dataset.schema is not None


def test_run_pipeline_end_to_end(tmp_path):
    version_dir = run_pipeline(_DemoConnector(), tmp_path, cache_dir=tmp_path / "cache")
    manifest = Manifest.model_validate_json((version_dir / "manifest.json").read_text())
    assert manifest.counts.records == 3
    assert manifest.dataset_id == "timenet/hello-world"


def test_connector_defaults_to_the_parquet_values_backend():
    assert _DemoConnector().values_backend == ValuesBackend.PARQUET


def test_run_pipeline_writes_the_requested_values_backend(tmp_path):
    pytest.importorskip("zarr")

    class _ZarrConnector(_DemoConnector):
        values_backend = ValuesBackend.ZARR

    version_dir = run_pipeline(_ZarrConnector(), tmp_path, cache_dir=tmp_path / "cache")

    manifest = Manifest.model_validate_json((version_dir / "manifest.json").read_text())
    assert manifest.values_backend == "zarr"


def test_publish_pipeline_passes_the_requested_values_backend_to_the_registry(tmp_path):
    class _Registry:
        def __init__(self) -> None:
            self.values_backend: str | None = None

        def exists(self, dataset_id: str, version: str) -> bool:
            return False

        def put_dataset(self, dataset, *, force: bool, values_backend: str, progress_cb) -> None:
            self.values_backend = values_backend

    class _ZarrConnector(_DemoConnector):
        values_backend = ValuesBackend.ZARR

    registry = _Registry()
    publish_pipeline(_ZarrConnector(), cast(WritableRegistry, registry), cache_dir=tmp_path / "cache")

    assert registry.values_backend == "zarr"


class _CountingConnector(_DemoConnector):
    def __init__(self) -> None:
        self.downloads = 0

    def download(self, cache_dir: Path) -> list[str]:
        self.downloads += 1
        return ["ref"]


def test_run_pipeline_is_idempotent(tmp_path):
    connector = _CountingConnector()
    first = run_pipeline(connector, tmp_path, cache_dir=tmp_path / "cache")
    second = run_pipeline(connector, tmp_path, cache_dir=tmp_path / "cache")
    assert first == second
    assert connector.downloads == 1  # the second run skipped download/convert/store


def test_run_pipeline_force_rebuilds(tmp_path):
    connector = _CountingConnector()
    run_pipeline(connector, tmp_path, cache_dir=tmp_path / "cache")
    version_dir = run_pipeline(connector, tmp_path, cache_dir=tmp_path / "cache", force=True)
    assert connector.downloads == 2
    assert (version_dir / "manifest.json").exists()
    assert [path.name for path in version_dir.parent.iterdir()] == [version_dir.name]


def test_a_failed_forced_rebuild_keeps_the_committed_version(tmp_path, monkeypatch):
    connector = _CountingConnector()
    version_dir = run_pipeline(connector, tmp_path, cache_dir=tmp_path / "cache")
    before = (version_dir / "manifest.json").read_bytes()

    def fail(self):
        raise RuntimeError("write failed")

    monkeypatch.setattr(TimeFWriter, "write", fail)
    with pytest.raises(RuntimeError, match="write failed"):
        run_pipeline(connector, tmp_path, cache_dir=tmp_path / "cache", force=True)
    assert (version_dir / "manifest.json").read_bytes() == before
    assert [path.name for path in version_dir.parent.iterdir()] == [version_dir.name]


def test_invalid_backend_preserves_committed_version(tmp_path):
    connector = _CountingConnector()
    version_dir = run_pipeline(connector, tmp_path, cache_dir=tmp_path / "cache")
    before = {p.relative_to(version_dir): p.read_bytes() for p in version_dir.rglob("*") if p.is_file()}
    connector.values_backend = cast("ValuesBackend", "invalid")
    with pytest.raises(TimeFValidationError, match="backend"):
        run_pipeline(connector, tmp_path, cache_dir=tmp_path / "cache", force=True)
    assert {p.relative_to(version_dir): p.read_bytes() for p in version_dir.rglob("*") if p.is_file()} == before
    assert connector.downloads == 1


def test_clean_cache_keeps_caller_supplied_dir(tmp_path):
    cache = tmp_path / "mine"
    run_pipeline(_DemoConnector(), tmp_path / "root", cache_dir=cache, keep_cache=False)
    assert cache.is_dir()  # a caller-owned cache_dir is never deleted, even when asked to clean it


def _isolated_home(tmp_path, monkeypatch) -> None:
    """Point the settings at a scratch home so the default cache lands under tmp_path."""
    for var in ("TIMENET_STORAGE", "TIMENET_CACHE", "TIMENET_REGISTRY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))


def test_clean_cache_removes_the_auto_created_default(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    run_pipeline(_DemoConnector(), tmp_path / "root")  # cache_dir=None -> the engine made it
    assert not (settings().cache_dir / make_dataset().metadata.dataset_id).exists()


def test_run_pipeline_keeps_the_cache_when_asked(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    run_pipeline(_DemoConnector(), tmp_path / "root", keep_cache=True)
    assert (settings().cache_dir / make_dataset().metadata.dataset_id).exists()


def test_run_pipeline_keeps_the_cache_when_download_fails(tmp_path, monkeypatch):
    # Downloads resume: http writes a .part and renames it, so a file that landed is complete and the
    # next run skips it. Deleting the cache here would throw away good bytes.
    _isolated_home(tmp_path, monkeypatch)

    class _HalfDownloadedConnector(_DemoConnector):
        def download(self, cache_dir: Path) -> list[str]:
            (cache_dir / "first.bin").write_bytes(b"complete")
            raise RuntimeError("download blew up")

    with pytest.raises(RuntimeError, match="download blew up"):
        run_pipeline(_HalfDownloadedConnector(), tmp_path / "root")
    cache = settings().cache_dir / make_dataset().metadata.dataset_id
    assert (cache / "first.bin").read_bytes() == b"complete"  # the finished file survives the retry


def test_run_pipeline_keeps_the_cache_when_convert_fails(tmp_path, monkeypatch):
    # The rmtree runs after store_dataset, so a convert that raises keeps what download fetched.
    _isolated_home(tmp_path, monkeypatch)

    class _FailingConnector(_DemoConnector):
        def download(self, cache_dir: Path) -> list[str]:
            (cache_dir / "source.bin").write_bytes(b"raw")
            return ["ref"]

        def convert(self, raw_refs: list[str]) -> TimeFDataset:
            raise RuntimeError("convert blew up")

    with pytest.raises(RuntimeError, match="convert blew up"):
        run_pipeline(_FailingConnector(), tmp_path / "root")
    assert (settings().cache_dir / make_dataset().metadata.dataset_id).exists()


_PARENT = DatasetRef(dataset_id=make_dataset().metadata.dataset_id, version=make_dataset().metadata.dataset_version)


def _store_credentialed_parent(root: Path) -> None:
    """Store the demo dataset as a credentialed release, so the effective access of a child is credentialed."""
    parent = make_dataset()
    parent._metadata = parent.metadata.model_copy(
        update={"access": Access.CREDENTIALED, "access_url": "https://example.org/dua"}
    )
    LocalRegistry(root).put_dataset(parent)


class _ChildConnector(BaseConnector[str]):
    """Declares the demo dataset under the parent alias ``parent`` and imports one record with ``IMPORT``."""

    IMPORT: ClassVar[dict[str, Any] | None] = {}

    def metadata(self):
        return make_dataset().metadata.model_copy(
            update={"dataset_id": "test/child", "parents": {ParentAlias("parent"): _PARENT}}
        )

    def download(self, cache_dir):
        return []

    def compose(self, raw_refs, context):
        dataset = TimeFDataset(metadata=self.metadata())
        if self.IMPORT is not None:
            dataset.import_from(
                context.parents[ParentAlias("parent")], records=[make_dataset().records[0].id], **self.IMPORT
            )
        return dataset


@pytest.mark.parametrize(
    "imported", [{}, {"alias": ParentAlias("parent")}, None], ids=["inferred", "explicit", "nothing-imported"]
)
def test_run_pipeline_records_each_declared_parent_with_its_access(tmp_path, monkeypatch, imported):
    _store_credentialed_parent(tmp_path)
    monkeypatch.setattr(_ChildConnector, "IMPORT", imported)

    version_dir = run_pipeline(_ChildConnector(), tmp_path, cache_dir=tmp_path / "cache")

    manifest = Manifest.model_validate_json((version_dir / "manifest.json").read_text())
    assert {alias: card.source for alias, card in manifest.parent_cards.items()} == {"parent": _PARENT}
    assert [card.source for card in manifest.lineage] == [_PARENT]
    # The card of a child that only reads a credentialed parent stays open. Its effective level is
    # credentialed, and the access page stays in the card of the parent.
    assert (manifest.metadata.access, manifest.metadata.access_url) == (Access.OPEN, None)
    assert effective_access(manifest.metadata, manifest.lineage) is Access.CREDENTIALED


class _ConvertOnlyChild(_ChildConnector):
    """Declares a parent but implements only ``convert``, so it cannot receive the opened parents."""

    compose = BaseConnector.compose

    def convert(self, raw_refs):
        return TimeFDataset(metadata=self.metadata())


@pytest.mark.parametrize("pipeline", ["run", "publish"])
def test_a_connector_without_compose_fails_before_opening_parents_or_downloading(tmp_path, monkeypatch, pipeline):
    monkeypatch.setattr(_ConvertOnlyChild, "download", lambda self, cache_dir: pytest.fail("downloaded"))
    # The parent is missing too: the check runs before the engine looks for it.
    with pytest.raises(TimeFValidationError, match=r"declares parents \['parent'\].*implement compose"):
        if pipeline == "run":
            run_pipeline(_ConvertOnlyChild(), tmp_path, cache_dir=tmp_path / "cache")
        else:
            publish_pipeline(_ConvertOnlyChild(), LocalRegistry(tmp_path), cache_dir=tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


class _DroppedParentChild(_ChildConnector):
    """Returns a dataset whose card leaves out the declared parent."""

    def compose(self, raw_refs, context):
        return TimeFDataset(metadata=self.metadata().model_copy(update={"parents": {}}))


class _OtherVersionChild(_ChildConnector):
    """Returns a dataset whose card has another version than the card that the engine read."""

    def compose(self, raw_refs, context):
        return TimeFDataset(metadata=self.metadata().model_copy(update={"dataset_version": Version(2, 0, 0)}))


class _InPlaceDroppedParentChild(_ChildConnector):
    """Returns the same card object from each ``metadata()`` call, and removes the parent from it in place."""

    def __init__(self) -> None:
        self._card = super().metadata()

    def metadata(self):
        return self._card

    def compose(self, raw_refs, context):
        dataset = TimeFDataset(metadata=self.metadata())
        del dataset.metadata.parents[ParentAlias("parent")]
        return dataset


@pytest.mark.parametrize("pipeline", ["run", "publish"])
@pytest.mark.parametrize(
    ("connector", "differ"),
    [
        (_DroppedParentChild, "parents"),
        (_OtherVersionChild, "dataset_version"),
        (_InPlaceDroppedParentChild, "parents"),
    ],
    ids=["dropped-parent", "other-version", "dropped-parent-in-place"],
)
def test_a_dataset_with_another_card_than_the_engine_read_fails_and_stores_no_release(
    tmp_path, pipeline, connector, differ
):
    root = tmp_path / "root"
    LocalRegistry(root).put_dataset(make_dataset())
    before = sorted(root.rglob("*"))
    with pytest.raises(
        TimeFValidationError,
        match=rf"the dataset that compose\(\) returned has another card than test/child: {differ}$",
    ):
        if pipeline == "run":
            run_pipeline(connector(), root, cache_dir=tmp_path / "cache")
        else:
            publish_pipeline(connector(), LocalRegistry(root), cache_dir=tmp_path / "cache")
    assert sorted(root.rglob("*")) == before


def test_run_pipeline_fails_on_a_missing_parent_before_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(_ChildConnector, "download", lambda self, cache_dir: pytest.fail("downloaded"))

    with pytest.raises(TimeNetDatasetNotFoundError, match=str(_PARENT)):
        run_pipeline(_ChildConnector(), tmp_path, cache_dir=tmp_path / "cache")


def test_open_parents_closes_the_parents_it_opened_when_a_later_one_is_missing(tmp_path, monkeypatch):
    LocalRegistry(tmp_path).put_dataset(make_dataset())
    missing = DatasetRef.parse("test/missing@1.0.0")
    metadata = (
        _ChildConnector()
        .metadata()
        .model_copy(update={"parents": {ParentAlias("parent"): _PARENT, ParentAlias("missing"): missing}})
    )
    closed = []
    close = TimeFReader.close

    def record_close(reader):
        closed.append(reader.metadata.dataset_id)
        close(reader)

    monkeypatch.setattr(TimeFReader, "close", record_close)

    with pytest.raises(TimeNetDatasetNotFoundError, match=str(missing)):
        open_parents(metadata, LocalRegistry(tmp_path))

    assert closed == [_PARENT.dataset_id]


def test_a_parent_reader_that_fails_to_close_does_not_keep_the_other_open(tmp_path, monkeypatch):
    second = make_dataset()
    second._metadata = second.metadata.model_copy(update={"dataset_id": "test/second"})
    for dataset in (make_dataset(), second):
        LocalRegistry(tmp_path).put_dataset(dataset)
    parents = {ParentAlias("parent"): _PARENT, ParentAlias("second"): DatasetRef.parse("test/second@1.0.0")}
    context = open_parents(
        _ChildConnector().metadata().model_copy(update={"parents": parents}), LocalRegistry(tmp_path)
    )
    closed = []
    close = TimeFReader.close

    def fail_the_first_close(reader):
        dataset_id = reader.metadata.dataset_id
        close(reader)
        closed.append(dataset_id)
        if len(closed) == 1:
            raise RuntimeError(f"{dataset_id} failed to close")

    monkeypatch.setattr(TimeFReader, "close", fail_the_first_close)

    with pytest.raises(RuntimeError, match="test/second failed to close"):
        context.close()

    # The reader opened last closes first. The other reader still closes.
    assert closed == ["test/second", _PARENT.dataset_id]


def test_a_build_context_made_directly_closes_its_parent_readers(tmp_path, monkeypatch):
    second = make_dataset()
    second._metadata = second.metadata.model_copy(update={"dataset_id": "test/second"})
    registry = LocalRegistry(tmp_path)
    for dataset in (make_dataset(), second):
        registry.put_dataset(dataset)
    readers = {
        ParentAlias("parent"): TimeFReader(registry.open_version(_PARENT.dataset_id, str(_PARENT.version))),
        ParentAlias("second"): TimeFReader(registry.open_version("test/second", "1.0.0")),
    }
    closed = []
    close = TimeFReader.close

    def record_close(reader):
        closed.append(reader.metadata.dataset_id)
        close(reader)

    monkeypatch.setattr(TimeFReader, "close", record_close)

    with BuildContext(parents=readers):
        assert closed == []

    assert closed == ["test/second", _PARENT.dataset_id]


class _GatedConnector(_DemoConnector):
    """The demo dataset, credentialed."""

    def metadata(self):
        return make_dataset().metadata.model_copy(
            update={"access": Access.CREDENTIALED, "access_url": "https://example.org/dua"}
        )

    def convert(self, raw_refs: list[str]) -> TimeFDataset:
        dataset = make_dataset()
        dataset._metadata = self.metadata()
        return dataset


def test_publish_pipeline_refuses_a_card_that_is_not_open_before_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(_GatedConnector, "download", lambda self, cache_dir: pytest.fail("downloaded"))
    remote = RemoteRegistry(
        "http://api.local", transport=httpx.MockTransport(lambda request: httpx.Response(404)), cache_dir=tmp_path
    )

    with pytest.raises(TimeNetAccessError) as caught:
        publish_pipeline(_GatedConnector(), remote, cache_dir=tmp_path / "cache")

    # A card has no lineage, so its own page is the only one.
    assert (caught.value.dataset, caught.value.requirements) == (
        _PARENT,
        (AccessRequirement(_PARENT, Access.CREDENTIALED, "https://example.org/dua"),),
    )
    # A local registry is the user's own, so it may hold the credentialed release.
    monkeypatch.undo()
    publish_pipeline(_GatedConnector(), LocalRegistry(tmp_path / "local"), cache_dir=tmp_path / "cache")
    assert LocalRegistry(tmp_path / "local").exists(_PARENT.dataset_id, str(_PARENT.version))


class _HostedRegistry:
    """A registry that is not local. It serves the parents from a local registry and refuses writes."""

    def __init__(self, root: Path) -> None:
        self.local = LocalRegistry(root)

    def exists(self, dataset_id, version):
        return False

    def open_version(self, dataset_id, version=None):
        return self.local.open_version(dataset_id, version)

    def put_dataset(self, dataset, **kwargs):
        pytest.fail("wrote a child whose parent is credentialed")


def test_publish_pipeline_refuses_an_open_child_of_a_credentialed_parent_before_the_write(tmp_path):
    _store_credentialed_parent(tmp_path)
    registry = cast(WritableRegistry, _HostedRegistry(tmp_path))

    with pytest.raises(TimeNetAccessError) as caught:
        publish_pipeline(_ChildConnector(), registry, cache_dir=tmp_path / "cache")

    # The card of the child is open, so the check before the download passes. Its lineage makes it
    # credentialed.
    child = DatasetRef(dataset_id="test/child", version=_PARENT.version)
    assert (caught.value.dataset, caught.value.access, caught.value.requirements) == (
        child,
        Access.CREDENTIALED,
        (AccessRequirement(_PARENT, Access.CREDENTIALED, "https://example.org/dua"),),
    )
