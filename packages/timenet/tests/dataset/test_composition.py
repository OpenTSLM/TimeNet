import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import make_dataset
from timenet.types import Access, Version
from timenet.types.object_kind import ObjectKind
from timenet.types.reference_keys import InputAlias
from timenet.types.references import DatasetRef
from timenet.writer import TimeFWriter


def store(root, dataset):
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def test_selected_signal_then_record_preserves_edits_and_aliases(tmp_path, monkeypatch):
    parent = make_dataset()
    path = store(tmp_path, parent)
    handle = TimeFReader(open_local_version(path)).get_handle()
    ref = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
    child = TimeFDataset(
        metadata=parent.metadata.model_copy(update={"dataset_id": "test/child", "inputs": {InputAlias("base"): ref}})
    )
    original = parent.records[0]
    child.import_from(handle, signals=[original.signals[0].id])
    assert len(child.records) == 1
    assert len(child.records[0].signals) == 1
    first_source = child.records[0].sources[0]
    first_signal = first_source.signals[0]
    first_source.metadata["edited"] = True
    first_signal.metadata["edited"] = True
    child.records[0].metadata["edited"] = True
    imported = child.import_from(handle, records=[original.id])
    assert list(child.input_locks) == [InputAlias("base")]
    assert imported.records[original.id].metadata["edited"] is True
    assert child.records[0].sources[0] is first_source
    assert imported.signals[original.signals[0].id] is first_signal
    assert first_source.metadata["edited"] is True
    assert first_signal.metadata["edited"] is True
    assert len(child.records[0].signals) == len(original.signals)
    child_path = store(tmp_path, child)
    stored = TimeFReader(open_local_version(child_path)).get_handle()

    def unexpected_catalog():
        raise AssertionError("a single-object handle must not hydrate the full catalog")

    monkeypatch.setattr(stored.reader, "_read_catalog", unexpected_catalog)
    assert stored.get_handle(ObjectKind.RECORD, original.id, alias=InputAlias("base")).read().metadata["edited"] is True
    assert stored.get_handle(ObjectKind.RECORD, original.id, parent=ref).read().id == imported.records[original.id].id
    with pytest.raises(TimeFValidationError, match="not both"):
        stored.get_handle(ObjectKind.RECORD, original.id, alias=InputAlias("base"), parent=ref)
    for reader in (handle, stored):
        reader.close()


def test_importing_two_children_of_same_parent_keeps_metadata_separate_and_reuses_payloads(tmp_path):
    original = make_dataset()
    original_id = original.records[0].id
    base_path = store(tmp_path, original)
    with TimeFReader(open_local_version(base_path)) as reader:
        children = []
        for name in ("left", "right"):
            child = TimeFDataset(metadata=original.metadata.model_copy(update={"dataset_id": f"test/{name}"}))
            imported = child.import_from(reader.get_handle(), alias=InputAlias("base"), records=[original_id])
            imported.records[original_id].metadata["branch"] = name
            children.append((store(tmp_path, child), imported.records[original_id].id))
    merged = TimeFDataset(metadata=original.metadata.model_copy(update={"dataset_id": "test/merged"}))
    readers = [TimeFReader(open_local_version(path)) for path, _ in children]
    try:
        for name, reader, (_, identity) in zip(("left", "right"), readers, children, strict=True):
            merged.import_from(reader.get_handle(), alias=InputAlias(name), records=[identity])
        assert len({record.id for record in merged.records}) == 2
        merged.records[0].metadata["branch"] = "edited"
        assert merged.records[1].metadata["branch"] == "right"
        merged_path = store(tmp_path, merged)
    finally:
        for reader in readers:
            reader.close()
    version = open_local_version(merged_path)
    assert not version.manifest.files.time_series
    assert {source.dataset_id for source in version.manifest.payload_sources.values()} == {original.metadata.dataset_id}
    for path in [base_path, *(path for path, _ in children)]:
        (path / "control.duckdb").rename(path / "unavailable.duckdb")
    with TimeFReader(version) as reader:
        for record in reader.read().read().records:
            assert record.signals[0].to_arrow().equals(original.records[0].signals[0].to_arrow())


def test_import_keeps_parent_version_pins_and_access_restrictions(tmp_path):
    parent = make_dataset()
    parent._metadata = parent.metadata.model_copy(
        update={"access": Access.RESTRICTED, "access_url": "https://example.org/access"}
    )
    original_path = store(tmp_path, parent)
    newer = make_dataset()
    newer._metadata = parent.metadata.model_copy(update={"dataset_version": Version(1, 0, 1)})
    newer_path = store(tmp_path, newer)
    child = TimeFDataset(metadata=make_dataset().metadata.model_copy(update={"dataset_id": "test/child"}))
    with TimeFReader(open_local_version(original_path)) as first, TimeFReader(open_local_version(newer_path)) as second:
        child.import_from(first.get_handle(), alias=InputAlias("base"), records=[parent.records[0].id])
        assert child.metadata.access == Access.RESTRICTED
        assert child.metadata.access_url == parent.metadata.access_url
        with pytest.raises(TimeFValidationError, match="different parent snapshot"):
            child.import_from(second.get_handle(), alias=InputAlias("base"), records=[newer.records[0].id])


def test_import_rejects_a_declared_alias_mismatch_before_copying_objects(tmp_path):
    parent = make_dataset()
    parent_path = store(tmp_path, parent)
    ref = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
    child = TimeFDataset(
        metadata=parent.metadata.model_copy(update={"dataset_id": "test/child", "inputs": {InputAlias("base"): ref}})
    )
    with (
        TimeFReader(open_local_version(parent_path)) as reader,
        pytest.raises(TimeFValidationError, match="does not declare parent"),
    ):
        child.import_from(reader.get_handle(), alias=InputAlias("other"), records=[parent.records[0].id])
    assert child.records == ()
    assert child.input_locks == {}
