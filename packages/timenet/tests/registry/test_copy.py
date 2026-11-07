"""``copy_release()`` copies a release with its payload sources and never changes a published version.

The ``source`` registry holds a parent from ``make_dataset()`` and a child that keeps one of its
Records, so the child reads payloads that the parent stores: the parent is a payload source of the
child. The ``chain_source`` registry adds one level: a grandchild reads a middle release, which reads
a grandparent.
"""

from pathlib import Path
import shutil
import tempfile

from _fake_registry import build_service_fake
import numpy as np
import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.errors import (
    TimeFFormatError,
    TimeFValidationError,
    TimeNetAccessError,
    TimeNetMissingParentError,
    TimeNetVersionConflictError,
)
from timenet.progress import BytesMoved, ProgressEvent, StepFinished, StepStarted
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, RemoteRegistry, WritableRegistry, copy_release, open_local_version
from timenet.registry.shared import complete_copy
from timenet.testing import make_dataset
from timenet.types import Access, Annotation, DatasetMetadata, DatasetSchema, License, TimeSeriesSpec, Version, ureg
from timenet.types.references import DatasetRef
from timenet.writer import TimeFWriter


parent_ref = DatasetRef.parse("timenet/hello-world@1.0.0")
child_ref = DatasetRef.parse("test/child@1.0.0")
parent = (parent_ref.dataset_id, str(parent_ref.version))
child = (child_ref.dataset_id, str(child_ref.version))

grandparent_ref = DatasetRef.parse("test/grandparent@1.0.0")
middle_ref = DatasetRef.parse("test/middle@1.0.0")
grandchild_ref = DatasetRef.parse("test/grandchild@1.0.0")
chain = [(ref.dataset_id, str(ref.version)) for ref in (grandparent_ref, middle_ref, grandchild_ref)]

_SPEC = TimeSeriesSpec(spec_type="env", name="env", unit_value=ureg.dimensionless)
_CREDENTIALED = {"access": Access.CREDENTIALED, "access_url": "https://example.org/access"}
_OPEN = {"access": Access.OPEN, "access_url": None}


def _derive(loaded: TimeFDataset, dataset_id: str, records, **card) -> TimeFDataset:
    """Return a release under ``dataset_id`` that keeps read Records and their ids, as an edit does.

    ``card`` sets other fields of the card of the release.
    """
    return TimeFDataset.from_parts(
        metadata=loaded.metadata.model_copy(update={"dataset_id": dataset_id, **card}),
        records=records,
        tasks=(),
        schema=DatasetSchema(),
        last_ids=loaded.last_ids,
    )


def _write(root: Path, dataset: TimeFDataset) -> Path:
    dataset.derive_schema()
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _write_parent_and_child(root: Path, *, credentialed: DatasetRef | None = None) -> None:
    """Write the parent and a child that reads payloads of the parent under ``root``.

    With ``credentialed``, the card of that release declares credentialed access, and the other card
    declares open access.
    """
    dataset = make_dataset()
    if credentialed == parent_ref:
        dataset._metadata = dataset.metadata.model_copy(update=_CREDENTIALED)
    parent_dir = _write(root, dataset)
    with TimeFReader(open_local_version(parent_dir)) as reader:
        loaded = reader.read()
        card = _CREDENTIALED if credentialed == child_ref else _OPEN
        _write(root, _derive(loaded, child_ref.dataset_id, [loaded.records[0]], **card))


def _release(dataset_id: str, records: list[Record]) -> TimeFDataset:
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id=dataset_id,
            dataset_version=Version(1, 0, 0),
            name=dataset_id,
            description="d",
            license=License.MIT,
        )
    )
    for record in records:
        dataset.add_record(record=record)
    return dataset


def _record(name: str) -> Record:
    """Return a named Record whose Signal values are held in memory."""
    signal = Signal.from_irregular(
        np.arange(3, dtype=np.float32), time_offsets_us=[0, 1_000, 2_000], spec=_SPEC, name=name
    )
    return Record(sources=(Source(name="env", signals=(signal,)),), metadata={"name": name})


def _write_chain(root: Path, *, grandchild_reads_grandparent: bool) -> None:
    """Write a grandparent, a middle release that reads it, and a grandchild that reads the middle release.

    The middle release keeps the grandparent's Record ``"a"`` and stores its own Record ``"b"``. The
    grandchild keeps ``"b"``, so it reads only the middle release. With ``grandchild_reads_grandparent``,
    it also keeps ``"a"``, so it reads the grandparent too.
    """
    grandparent_dir = _write(root, _release(grandparent_ref.dataset_id, [_record("a")]))
    with TimeFReader(open_local_version(grandparent_dir)) as reader:
        loaded = reader.read()
        middle = _derive(loaded, middle_ref.dataset_id, loaded.records)
        middle.add_record(record=_record("b"))
        middle_dir = _write(root, middle)
    with TimeFReader(open_local_version(middle_dir)) as reader:
        loaded = reader.read()
        kept = [record for record in loaded.records if grandchild_reads_grandparent or record.metadata["name"] == "b"]
        _write(root, _derive(loaded, grandchild_ref.dataset_id, kept))


def _payload_sources(registry: WritableRegistry, reference: DatasetRef) -> list[DatasetRef]:
    manifest = registry.get_manifest(reference.dataset_id, str(reference.version))
    return [pinned.release for pinned in manifest.payload_sources.values()]


def _other_build(dataset_id: str) -> TimeFDataset:
    """Return a release of ``dataset_id`` at version 1.0.0 whose files differ from the source's copy."""
    dataset = make_dataset()
    dataset._metadata = dataset.metadata.model_copy(update={"dataset_id": dataset_id})
    dataset.records[0].annotate(Annotation(key="build", value="another"))
    return dataset


def _remote(tmp_path: Path, name: str, *, token: str | None = None) -> tuple[RemoteRegistry, Path]:
    """Return a remote registry over the service fake, and the directory that the fake serves."""
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    transport, _ = build_service_fake(root, token=token)
    registry = RemoteRegistry(
        f"http://{name}.local", token=token, local_registry=tmp_path / f"{name}-cache", transport=transport
    )
    return registry, root


def _steps(events: list[ProgressEvent]) -> list[ProgressEvent]:
    """Return the step events, without the byte events of each copy."""
    return [event for event in events if not isinstance(event, BytesMoved)]


@pytest.fixture
def source(tmp_path) -> LocalRegistry:
    root = tmp_path / "source"
    _write_parent_and_child(root)
    return LocalRegistry(root)


@pytest.fixture
def chain_source(tmp_path) -> LocalRegistry:
    root = tmp_path / "chain"
    _write_chain(root, grandchild_reads_grandparent=False)
    registry = LocalRegistry(root)
    assert _payload_sources(registry, grandchild_ref) == [middle_ref]
    assert _payload_sources(registry, middle_ref) == [grandparent_ref]
    return registry


@pytest.fixture(params=["local", "remote"])
def dest(request, tmp_path) -> WritableRegistry:
    if request.param == "local":
        return LocalRegistry(tmp_path / "dest")
    registry, _ = _remote(tmp_path, "dest", token="tok_rw")
    return registry


def test_copy_release_copies_a_release_and_its_payload_sources(source, dest):
    copy_release(child_ref, source, dest)
    assert dest.exists(*child) and dest.exists(*parent)
    assert dest.get_manifest(*child) == source.get_manifest(*child)
    assert dest.get_manifest(*parent) == source.get_manifest(*parent)


def test_copy_release_does_nothing_for_the_same_release(source, dest, monkeypatch):
    copy_release(child_ref, source, dest)

    def unexpected(*args, **kwargs):
        pytest.fail("a copy of the same release must not download or upload it again")

    monkeypatch.setattr(dest, "put_release", unexpected)
    monkeypatch.setattr(source, "download_release", unexpected)
    events: list[ProgressEvent] = []
    copy_release(child_ref, source, dest, progress=events.append)  # no error, no second upload
    assert events == []


def _rewrite_child_manifest(dest: LocalRegistry, **update) -> Path:
    """Rewrite the manifest of the child in ``dest`` with ``update``; return the manifest file."""
    path = dest.root / child_ref.dataset_id / str(child_ref.version) / "manifest.json"
    path.write_text(dest.get_manifest(*child).model_copy(update=update).to_json())
    return path


def test_copy_release_refuses_a_release_with_the_same_files_and_another_card(source, tmp_path):
    dest = LocalRegistry(tmp_path / "dest")
    copy_release(child_ref, source, dest)
    card = dest.get_manifest(*child).metadata.model_copy(update={"description": "Another description"})
    path = _rewrite_child_manifest(dest, metadata=card)
    before = path.read_bytes()
    # A fresh registry: the rewrite can keep the mtime of the first write, so a cached manifest could stay.
    with pytest.raises(TimeNetVersionConflictError, match="another card than") as raised:
        copy_release(child_ref, source, LocalRegistry(dest.root))
    assert raised.value.difference == "card"
    assert path.read_bytes() == before


def test_copy_release_ignores_the_build_environment(source, tmp_path):
    dest = LocalRegistry(tmp_path / "dest")
    copy_release(child_ref, source, dest)
    _rewrite_child_manifest(dest, build_env={"python": "0.0.0"})
    events: list[ProgressEvent] = []
    copy_release(child_ref, source, LocalRegistry(dest.root), progress=events.append)
    assert events == []


def test_copy_release_refuses_another_build_of_the_version(source, dest):
    # the destination holds child_ref built from other records
    dest.put_dataset(_other_build(child_ref.dataset_id))
    before = dest.get_manifest(*child)
    with pytest.raises(TimeNetVersionConflictError, match="new version") as raised:
        copy_release(child_ref, source, dest)
    assert (raised.value.release, raised.value.needed_by) == (child_ref, None)
    assert (raised.value.registry, raised.value.source) == (str(dest), str(source))
    assert raised.value.difference == "files"  # the card differs too, but the files come first
    assert dest.get_manifest(*child) == before
    assert not dest.exists(*parent)  # nothing is copied, not even the payload source


def test_copy_release_refuses_a_payload_source_with_other_files(source, tmp_path):
    dest = LocalRegistry(tmp_path / "dest")
    # the destination holds parent_ref with other files
    dest.put_dataset(_other_build(parent_ref.dataset_id))
    before = dest.get_manifest(*parent)
    with pytest.raises(TimeNetVersionConflictError, match=str(parent_ref)) as raised:
        copy_release(child_ref, source, dest)
    assert (raised.value.release, raised.value.needed_by) == (parent_ref, child_ref)
    assert not dest.exists(*child)
    assert dest.get_manifest(*parent) == before


def _without_parent(source: LocalRegistry) -> LocalRegistry:
    """Remove the parent from ``source``, and return a new registry over its directory.

    The new registry does not keep a cached manifest of the parent.
    """
    shutil.rmtree(source.root / parent_ref.dataset_id)
    return LocalRegistry(source.root)


def test_copy_release_reads_a_payload_source_that_only_dest_holds(source, dest):
    copy_release(parent_ref, source, dest)
    before = dest.get_manifest(*parent)
    events: list[ProgressEvent] = []
    copy_release(child_ref, _without_parent(source), dest, progress=events.append)
    assert dest.get_manifest(*child) == source.get_manifest(*child)
    assert dest.get_manifest(*parent) == before
    assert _steps(events) == [StepStarted(child_ref, "copy"), StepFinished(child_ref, "copy")]


def test_copy_release_refuses_a_payload_source_that_only_dest_holds_with_other_files(source, dest):
    # the destination holds parent_ref with other payload files than the child pins
    dest.put_dataset(_release(parent_ref.dataset_id, [_record("other")]))
    before = dest.get_manifest(*parent)
    with pytest.raises(TimeFFormatError, match=f"payload source {parent_ref} differs"):
        copy_release(child_ref, _without_parent(source), dest)
    assert not dest.exists(*child)
    assert dest.get_manifest(*parent) == before


def test_copy_release_checks_the_access_of_a_payload_source_that_only_dest_holds(tmp_path):
    root = tmp_path / "source"
    _write_parent_and_child(root, credentialed=parent_ref)
    dest, dest_root = _remote(tmp_path, "dest", token="tok_rw")
    # the service holds the parent, whose card declares credentialed access
    shutil.copytree(root / parent_ref.dataset_id, dest_root / parent_ref.dataset_id)
    with pytest.raises(TimeNetAccessError) as raised:
        copy_release(child_ref, _without_parent(LocalRegistry(root)), dest)
    assert raised.value.dataset == parent_ref
    assert not dest.exists(*child)


def test_copy_release_refuses_a_payload_source_that_no_registry_holds(source, dest):
    origin = _without_parent(source)
    with pytest.raises(TimeNetMissingParentError) as raised:
        copy_release(child_ref, origin, dest)
    assert (raised.value.missing, raised.value.needed_by) == (parent_ref, child_ref)
    # The error names both registries that the copy looked in.
    assert raised.value.searched == f"{origin} or {dest}"
    assert str(raised.value) == f"payload source {parent_ref} of {child_ref} is not in {origin} or {dest}."
    assert not dest.exists(*child)


def test_copy_release_refuses_a_payload_source_that_only_dest_holds_damaged(source, tmp_path):
    dest = LocalRegistry(tmp_path / "dest")
    copy_release(parent_ref, source, dest)
    parent_dir = dest.root / parent_ref.dataset_id / str(parent_ref.version)
    (parent_dir / dest.get_manifest(*parent).files.control[0].path).unlink()
    with pytest.raises(TimeNetMissingParentError) as raised:
        copy_release(child_ref, _without_parent(source), dest)
    assert (raised.value.missing, raised.value.needed_by) == (parent_ref, child_ref)
    assert not dest.exists(*child)


def test_copy_release_repairs_a_damaged_local_copy(source, tmp_path):
    dest = LocalRegistry(tmp_path / "dest")
    copy_release(child_ref, source, dest)
    child_dir = dest.root / child_ref.dataset_id / str(child_ref.version)
    manifest = dest.get_manifest(*child)
    (child_dir / manifest.files.control[0].path).unlink()
    assert not complete_copy(child_dir, manifest)
    events: list[ProgressEvent] = []
    copy_release(child_ref, source, dest, progress=events.append)
    assert complete_copy(child_dir, manifest)
    assert _steps(events) == [StepStarted(child_ref, "copy"), StepFinished(child_ref, "copy")]


def test_copy_release_with_force_copies_a_complete_local_copy_again(source, tmp_path):
    dest = LocalRegistry(tmp_path / "dest")
    copy_release(child_ref, source, dest)
    child_dir = dest.root / child_ref.dataset_id / str(child_ref.version)
    parent_dir = dest.root / parent_ref.dataset_id / str(parent_ref.version)
    for directory in (child_dir, parent_dir):
        (directory / "stale.txt").write_text("left over")
    events: list[ProgressEvent] = []
    copy_release(child_ref, source, dest, force=True, progress=events.append)
    # The requested release is copied again as a whole. Its complete payload source stays.
    assert not (child_dir / "stale.txt").exists()
    assert (parent_dir / "stale.txt").exists()
    assert complete_copy(child_dir, dest.get_manifest(*child))
    assert _steps(events) == [StepStarted(child_ref, "copy"), StepFinished(child_ref, "copy")]
    # force=True does not replace a copy with other files.
    other = LocalRegistry(tmp_path / "other")
    other.put_dataset(_other_build(child_ref.dataset_id))
    with pytest.raises(TimeNetVersionConflictError, match="new version"):
        copy_release(child_ref, source, other, force=True)


def test_copy_release_with_force_refuses_a_registry_that_is_not_local(source, tmp_path):
    dest, _ = _remote(tmp_path, "dest", token="tok_rw")
    with pytest.raises(TimeFValidationError, match=r"^force=True needs a local dest"):
        copy_release(child_ref, source, dest, force=True)
    assert not dest.exists(*child)
    assert not dest.exists(*parent)


def test_copy_release_reports_each_copied_release(source, tmp_path):
    events: list[ProgressEvent] = []
    copy_release(child_ref, source, LocalRegistry(tmp_path / "dest"), progress=events.append)
    assert _steps(events) == [
        StepStarted(parent_ref, "copy"),
        StepFinished(parent_ref, "copy"),
        StepStarted(child_ref, "copy"),
        StepFinished(child_ref, "copy"),
    ]


@pytest.mark.parametrize("kind", ["local", "remote"])
def test_copy_release_reports_the_bytes_of_each_release_that_it_copies_to_a_local_registry(kind, tmp_path):
    if kind == "local":
        _write_parent_and_child(tmp_path / "source")
        origin: LocalRegistry | RemoteRegistry = LocalRegistry(tmp_path / "source")
    else:
        origin, root = _remote(tmp_path, "source")
        _write_parent_and_child(root)
    events: list[ProgressEvent] = []
    copy_release(child_ref, origin, LocalRegistry(tmp_path / "dest"), progress=events.append)
    for release in (parent_ref, child_ref):
        # The bytes of a release arrive between the start and the end of its step.
        start, end = events.index(StepStarted(release, "copy")), events.index(StepFinished(release, "copy"))
        moved = events[start + 1 : end]
        manifest = origin.get_manifest(release.dataset_id, str(release.version))
        total = sum(part.size for part in manifest.files.all_files())
        done = [event.done for event in moved if isinstance(event, BytesMoved)]
        assert done
        assert moved == [BytesMoved(str(release), count, total) for count in done]
        assert done == sorted(done)
        assert done[-1] == total


def test_copy_release_copies_only_the_payload_sources_that_the_release_lists(chain_source, dest):
    # The grandchild reads the middle release only. The source lacks the grandparent, which the middle reads.
    shutil.rmtree(chain_source.root / grandparent_ref.dataset_id)
    events: list[ProgressEvent] = []
    copy_release(grandchild_ref, LocalRegistry(chain_source.root), dest, progress=events.append)
    assert [e.release for e in events if isinstance(e, StepStarted)] == [middle_ref, grandchild_ref]
    assert not dest.exists(*chain[0])


def test_copy_release_without_payload_sources_copies_the_release_alone(chain_source, dest):
    copy_release(middle_ref, chain_source, dest, with_payload_sources=False)
    assert dest.get_manifest(*chain[1]) == chain_source.get_manifest(*chain[1])
    assert not dest.exists(*chain[0])


def test_copy_release_copies_a_shared_payload_source_once(tmp_path):
    root = tmp_path / "chain"
    _write_chain(root, grandchild_reads_grandparent=True)
    shared_source = LocalRegistry(root)
    assert set(_payload_sources(shared_source, grandchild_ref)) == {grandparent_ref, middle_ref}
    events: list[ProgressEvent] = []
    copy_release(grandchild_ref, shared_source, LocalRegistry(tmp_path / "dest"), progress=events.append)
    assert [e.release for e in events if isinstance(e, StepStarted)] == [grandparent_ref, middle_ref, grandchild_ref]


def test_copy_release_between_remote_registries_removes_its_download(tmp_path, monkeypatch):
    remote_source, source_root = _remote(tmp_path, "source")
    _write_parent_and_child(source_root)
    dest, _ = _remote(tmp_path, "dest", token="tok_rw")
    downloads = tmp_path / "tmp"
    downloads.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(downloads))
    copy_release(child_ref, remote_source, dest)
    assert dest.get_manifest(*child) == remote_source.get_manifest(*child)
    assert dest.get_manifest(*parent) == remote_source.get_manifest(*parent)
    assert list(downloads.iterdir()) == []


def test_copy_release_between_remote_registries_downloads_only_the_releases_that_dest_lacks(tmp_path, monkeypatch):
    remote_source, source_root = _remote(tmp_path, "source")
    _write_parent_and_child(source_root)
    dest, _ = _remote(tmp_path, "dest", token="tok_rw")
    copy_release(parent_ref, remote_source, dest)
    downloads: list[DatasetRef] = []
    download = remote_source.download_release

    def counting(reference, *args, **kwargs):
        downloads.append(reference)
        return download(reference, *args, **kwargs)

    monkeypatch.setattr(remote_source, "download_release", counting)
    copy_release(child_ref, remote_source, dest)
    assert downloads == [child_ref]  # dest already holds the parent
    assert dest.get_manifest(*child) == remote_source.get_manifest(*child)


def test_copy_release_removes_its_download_after_an_error(tmp_path, monkeypatch):
    remote_source, source_root = _remote(tmp_path, "source")
    _write_parent_and_child(source_root)
    dest, _ = _remote(tmp_path, "dest", token="tok_rw")
    downloads = tmp_path / "tmp"
    downloads.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(downloads))

    def refuse(directory):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(dest, "put_release", refuse)
    with pytest.raises(RuntimeError, match="upload failed"):
        copy_release(child_ref, remote_source, dest)
    assert list(downloads.iterdir()) == []


@pytest.mark.parametrize("remote", [False, True], ids=["local", "remote"])
def test_a_forced_download_keeps_the_old_copy_if_the_final_rename_fails(remote, tmp_path, monkeypatch):
    if remote:
        registry, root = _remote(tmp_path, "source")
    else:
        root = tmp_path / "source"
        registry = LocalRegistry(root)
    _write_parent_and_child(root)
    manifest = registry.get_manifest(*parent)
    directory = tmp_path / "copy" / parent_ref.dataset_id / str(parent_ref.version)
    registry.download_release(parent_ref, directory, manifest=manifest)
    rename = Path.replace

    def replace(self, target):
        if ".tmp-" in self.name:
            raise OSError("rename failed")
        return rename(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError, match="rename failed"):
        registry.download_release(parent_ref, directory, force=True, manifest=manifest)
    assert complete_copy(directory, manifest)


@pytest.mark.parametrize("credentialed", [child_ref, parent_ref], ids=["release", "payload-source"])
def test_copy_release_refuses_a_release_that_is_not_open_for_a_registry_that_is_not_local(credentialed, tmp_path):
    root = tmp_path / "source"
    _write_parent_and_child(root, credentialed=credentialed)
    dest, _ = _remote(tmp_path, "dest", token="tok_rw")
    with pytest.raises(TimeNetAccessError) as raised:
        copy_release(child_ref, LocalRegistry(root), dest)
    assert raised.value.dataset == credentialed
    assert not dest.exists(*child)
    assert not dest.exists(*parent)


@pytest.mark.parametrize("credentialed", [child_ref, parent_ref], ids=["release", "payload-source"])
def test_copy_release_copies_a_release_that_is_not_open_between_local_registries(credentialed, tmp_path):
    root = tmp_path / "source"
    _write_parent_and_child(root, credentialed=credentialed)
    dest = LocalRegistry(tmp_path / "dest")
    copy_release(child_ref, LocalRegistry(root), dest)
    assert dest.get_manifest(*child).metadata.access is (
        Access.CREDENTIALED if credentialed == child_ref else Access.OPEN
    )
    assert dest.exists(*parent)


def test_copy_release_refuses_to_fetch_a_release_that_is_not_open_from_a_registry_that_is_not_local(tmp_path):
    remote_source, source_root = _remote(tmp_path, "source")
    _write_parent_and_child(source_root, credentialed=child_ref)
    dest = LocalRegistry(tmp_path / "dest")
    with pytest.raises(TimeNetAccessError) as raised:
        copy_release(child_ref, remote_source, dest)
    assert raised.value.dataset == child_ref
    assert not dest.exists(*child)
    assert not dest.exists(*parent)  # the open parent is not copied either: every check runs first
