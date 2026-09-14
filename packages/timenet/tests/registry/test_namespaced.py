import dataclasses

import numpy as np

from timenet.control_plane import DeclarativeDataset, TimeFReader, TimeFWriter
from timenet.registry import DatasetVersion, LocalRegistry
from timenet.testing import make_dataset, make_metadata
from timenet.types import License


def _namespaced_dataset() -> DeclarativeDataset:
    dataset = make_dataset(n_records=1, n_values=8)
    dataset.metadata = dataclasses.replace(
        make_metadata(dataset_id="ChengsenWang/TSQA"),
        name="TSQA",
        description="A namespaced dataset.",
        license=License.APACHE_2_0,
    )
    return dataset


def _write(root, dataset):
    with TimeFWriter(root, dataset.metadata) as writer:
        writer.write(dataset)


def test_list_datasets_finds_multiple_namespaced(tmp_path):
    _write(tmp_path, make_dataset(n_records=1, n_values=8))  # namespaced id: test/bedside
    _write(tmp_path, _namespaced_dataset())  # namespaced id: ChengsenWang/TSQA
    ids = {m.dataset_id for m in LocalRegistry(tmp_path).list_datasets()}
    assert ids == {"test/bedside", "ChengsenWang/TSQA"}


def test_get_manifest_and_open_file_namespaced(tmp_path):
    _write(tmp_path, _namespaced_dataset())
    registry = LocalRegistry(tmp_path)
    manifest = registry.get_manifest("ChengsenWang/TSQA")
    assert manifest.dataset_id == "ChengsenWang/TSQA"
    with registry.open_file("ChengsenWang/TSQA", "1.0.0", "manifest.json") as handle:
        assert b"TSQA" in handle.read()


def test_namespaced_round_trip(tmp_path):
    original = _namespaced_dataset()
    _write(tmp_path, _namespaced_dataset())
    version_dir = tmp_path / "ChengsenWang" / "TSQA" / "1.0.0"
    assert version_dir.exists()
    with TimeFReader.open_version(DatasetVersion.open_local(version_dir)) as reader:
        assert reader.record_ids() == sorted(record.id for record in original.records)
        assert reader.task_ids() == sorted(task.id for task in original.tasks)
        for record in original.records:
            view = reader.record(record.id)
            assert [signal.signal_id for signal in view.signals()] == [signal.id for signal in record.signals()]
            for signal in record.signals():
                np.testing.assert_array_equal(reader.values(signal.id), signal.values)


def test_cache_dir_is_ignored_by_list(tmp_path):
    _write(tmp_path, _namespaced_dataset())
    (tmp_path / ".cache" / "ChengsenWang").mkdir(parents=True)
    ids = {m.dataset_id for m in LocalRegistry(tmp_path).list_datasets()}
    assert ids == {"ChengsenWang/TSQA"}
