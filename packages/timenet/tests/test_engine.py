from pathlib import Path
import tempfile
from typing import Any, ClassVar, cast

import pytest

from timenet.config import is_cache_folder, settings
from timenet.connectors import BaseConnector, BuildContext
from timenet.dataset import TimeFDataset
from timenet.engine import open_parents, run_pipeline, store_dataset
from timenet.errors import TimeFValidationError, TimeNetDatasetNotFoundError
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.registry._paths import release_dir
from timenet.testing import make_dataset
from timenet.types import Access, Version
from timenet.types.access import effective_access
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


def test_clean_cache_removes_the_auto_created_default(tmp_path):
    run_pipeline(_DemoConnector(), tmp_path / "root")  # cache_dir=None -> the engine made it
    assert not (settings().downloads_dir / make_dataset().metadata.dataset_id).exists()


def test_run_pipeline_keeps_the_cache_when_asked(tmp_path):
    run_pipeline(_DemoConnector(), tmp_path / "root", keep_cache=True)
    assert (settings().downloads_dir / make_dataset().metadata.dataset_id).exists()


def test_the_engine_marks_the_downloads_folder_that_it_creates(tmp_path):
    run_pipeline(_DemoConnector(), tmp_path / "root")
    assert is_cache_folder(settings().downloads_dir)


def test_a_build_into_the_downloads_that_it_removes_fails_before_the_download(tmp_path, monkeypatch):
    monkeypatch.setattr(_DemoConnector, "download", lambda self, cache_dir: pytest.fail("downloaded"))
    root = settings().downloads_dir / make_dataset().metadata.dataset_id / "registry"
    with pytest.raises(TimeFValidationError, match=r"^the local registry .* overlaps .*, which TimeNet deletes"):
        run_pipeline(_DemoConnector(), root)
    assert not root.exists()


def test_the_engine_deletes_nothing_in_a_downloads_folder_without_the_marker(tmp_path):
    # The folder existed before TimeNet used it, for example ~/Downloads on a volume that ignores case.
    # So the engine leaves it unmarked, and the folder of the dataset in it can hold files of the user.
    folder = settings().downloads_dir / make_dataset().metadata.dataset_id
    folder.mkdir(parents=True)
    (folder / "notes.txt").write_text("mine")
    run_pipeline(_DemoConnector(), tmp_path / "root")
    assert (folder / "notes.txt").read_text() == "mine"


def test_run_pipeline_keeps_the_cache_when_download_fails(tmp_path):
    # Downloads resume: http writes a .part and renames it, so a file that landed is complete and the
    # next run skips it. Deleting the cache here would throw away good bytes.
    class _HalfDownloadedConnector(_DemoConnector):
        def download(self, cache_dir: Path) -> list[str]:
            (cache_dir / "first.bin").write_bytes(b"complete")
            raise RuntimeError("download blew up")

    with pytest.raises(RuntimeError, match="download blew up"):
        run_pipeline(_HalfDownloadedConnector(), tmp_path / "root")
    cache = settings().downloads_dir / make_dataset().metadata.dataset_id
    assert (cache / "first.bin").read_bytes() == b"complete"  # the finished file survives the retry


def test_run_pipeline_keeps_the_cache_when_convert_fails(tmp_path):
    # The rmtree runs after store_dataset, so a convert that raises keeps what download fetched.
    class _FailingConnector(_DemoConnector):
        def download(self, cache_dir: Path) -> list[str]:
            (cache_dir / "source.bin").write_bytes(b"raw")
            return ["ref"]

        def convert(self, raw_refs: list[str]) -> TimeFDataset:
            raise RuntimeError("convert blew up")

    with pytest.raises(RuntimeError, match="convert blew up"):
        run_pipeline(_FailingConnector(), tmp_path / "root")
    assert (settings().downloads_dir / make_dataset().metadata.dataset_id).exists()


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


def test_a_connector_without_compose_fails_before_opening_parents_or_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(_ConvertOnlyChild, "download", lambda self, cache_dir: pytest.fail("downloaded"))
    # The parent is missing too: the check runs before the engine looks for it.
    with pytest.raises(TimeFValidationError, match=r"declares parents \['parent'\].*implement compose"):
        run_pipeline(_ConvertOnlyChild(), tmp_path, cache_dir=tmp_path / "cache")
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


@pytest.mark.parametrize(
    ("connector", "differ"),
    [
        (_DroppedParentChild, "parents"),
        (_OtherVersionChild, "dataset_version"),
        (_InPlaceDroppedParentChild, "parents"),
    ],
    ids=["dropped-parent", "other-version", "dropped-parent-in-place"],
)
def test_a_dataset_with_another_card_than_the_engine_read_fails_and_stores_no_release(tmp_path, connector, differ):
    root = tmp_path / "root"
    LocalRegistry(root).put_dataset(make_dataset())
    before = sorted(root.rglob("*"))
    with pytest.raises(
        TimeFValidationError,
        match=rf"the dataset that compose\(\) returned has another card than test/child: {differ}$",
    ):
        run_pipeline(connector(), root, cache_dir=tmp_path / "cache")
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
        ParentAlias("parent"): TimeFReader(open_local_version(release_dir(registry.root, _PARENT))),
        ParentAlias("second"): TimeFReader(open_local_version(registry.root / "test/second/1.0.0")),
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
