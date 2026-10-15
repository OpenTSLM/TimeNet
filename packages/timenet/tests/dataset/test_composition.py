from timenet.dataset import TimeFDataset
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import make_dataset
from timenet.writer import TimeFWriter


def store(root, dataset):
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def test_selected_signal_then_task_preserves_edits_and_remaps_dependencies(tmp_path):
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
    assert stored.get_handle("Record", original.id, namespace="base").read().metadata["edited"] is True
    grandchild = TimeFDataset(metadata=parent.metadata.model_copy(update={"dataset_id": "test/grandchild"}))
    grandchild.import_from(stored, records=[imported.records[original.id].id])
    grandchild_path = store(tmp_path, grandchild)
    for reader in (handle, stored):
        reader.close()
    (path / "control.duckdb").rename(path / "unavailable.duckdb")
    (child_path / "control.duckdb").rename(child_path / "unavailable.duckdb")
    with TimeFReader(DatasetVersion.open_local(grandchild_path)) as reader:
        restored = reader.read()
        assert restored.records[0].signals[0].to_arrow().equals(original.signals[0].to_arrow())
