from timenet.dataset import TimeFDataset
from timenet.dataset.edit import edit_version
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import Version
from timenet.writer import TimeFWriter


def store(root, dataset):
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def test_selected_signal_then_task_preserves_edits_and_remaps_dependencies(tmp_path, monkeypatch):
    parent = make_dataset()
    path = store(tmp_path, parent)
    handle = TimeFReader(DatasetVersion.open_local(path)).get_handle()
    child = TimeFDataset(metadata=parent.metadata.model_copy(update={"dataset_id": "test/child"}))
    original = parent.records[0]
    imported = child.import_from(handle, namespace="base", signals=[original.signals[0].id])
    assert len(child.records) == 1
    assert len(child.records[0].signals) == 1
    child.records[0].metadata["edited"] = True
    chosen = parent.tasks[0]
    imported = child.import_from(handle, namespace="base", tasks=[chosen.id])
    assert imported.records[original.id].metadata["edited"] is True
    assert imported.tasks[chosen.id].inputs[0] is imported.records[original.id]
    assert len(child.records[0].signals) == len(original.signals)
    assert "edited" not in original.metadata
    child_path = store(tmp_path, child)
    stored = TimeFReader(DatasetVersion.open_local(child_path)).get_handle()
    with monkeypatch.context() as scoped:

        def unexpected_catalog():
            raise AssertionError("a single-object handle must not hydrate the full catalog")

        scoped.setattr(stored.reader, "_read_catalog", unexpected_catalog)
        assert stored.get_handle("Record", original.id, namespace="base").read().metadata["edited"] is True
        assert stored.get_handle("Task", chosen.id, namespace="base").read().inputs[0].metadata["edited"] is True
    grandchild = TimeFDataset(metadata=parent.metadata.model_copy(update={"dataset_id": "test/grandchild"}))
    grandchild.import_from(stored, records=[imported.records[original.id].id])
    grandchild_path = store(tmp_path, grandchild)
    for reader in (handle, stored):
        reader.close()
    version = DatasetVersion.open_local(grandchild_path)
    (path / "control.duckdb").rename(path / "unavailable.duckdb")
    (child_path / "control.duckdb").rename(child_path / "unavailable.duckdb")
    with TimeFReader(version) as reader:
        restored = reader.read()
        assert restored.records[0].signals[0].to_arrow().equals(original.signals[0].to_arrow())


def test_diamond_keeps_parent_namespaces_distinct_and_shares_ultimate_storage(tmp_path):
    original = make_dataset()
    original_id = original.records[0].id
    base_path = store(tmp_path, original)
    with TimeFReader(DatasetVersion.open_local(base_path)) as reader:
        children = []
        for name in ("left", "right"):
            child = TimeFDataset(metadata=original.metadata.model_copy(update={"dataset_id": f"test/{name}"}))
            imported = child.import_from(reader.get_handle(), namespace="base", records=[original_id])
            imported.records[original_id].metadata["branch"] = name
            children.append((store(tmp_path, child), imported.records[original_id].id))
    merged = TimeFDataset(metadata=original.metadata.model_copy(update={"dataset_id": "test/merged"}))
    readers = [TimeFReader(DatasetVersion.open_local(path)) for path, _ in children]
    try:
        for name, reader, (_, identity) in zip(("left", "right"), readers, children, strict=True):
            merged.import_from(reader.get_handle(), namespace=name, records=[identity])
        assert len({record.id for record in merged.records}) == 2
        merged.records[0].metadata["branch"] = "edited"
        assert merged.records[1].metadata["branch"] == "right"
        merged_path = store(tmp_path, merged)
    finally:
        for reader in readers:
            reader.close()
    version = DatasetVersion.open_local(merged_path)
    assert not version.manifest.files.time_series
    assert {source.dataset_id for source in version.manifest.payload_sources.values()} == {original.metadata.dataset_id}
    for path in [base_path, *(path for path, _ in children)]:
        (path / "control.duckdb").rename(path / "unavailable.duckdb")
    with TimeFReader(version) as reader:
        for record in reader.read().records:
            assert record.signals[0].to_arrow().equals(original.records[0].signals[0].to_arrow())


def test_edit_preserves_namespace_lookup_and_input_provenance(tmp_path):
    parent = make_dataset()
    path = store(tmp_path, parent)
    child = TimeFDataset(metadata=parent.metadata.model_copy(update={"dataset_id": "test/child"}))
    with TimeFReader(DatasetVersion.open_local(path)) as reader:
        child.import_from(reader.get_handle(), namespace="base", records=[record.id for record in parent.records])
        child_path = store(tmp_path, child)
    edited_path = edit_version(child_path, tmp_path, dataset_version=Version(1, 0, 1))
    with TimeFReader(DatasetVersion.open_local(edited_path)) as reader:
        restored = reader.get_handle().get_handle("Record", parent.records[0].id, namespace="base").read()
        assert restored.signals[0].to_arrow().equals(parent.records[0].signals[0].to_arrow())
        assert reader._manifest.input_locks == DatasetVersion.open_local(child_path).manifest.input_locks
