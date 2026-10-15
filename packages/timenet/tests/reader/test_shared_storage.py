from copy import deepcopy
import json

import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import make_dataset
from timenet.writer import TimeFWriter


def store(root, dataset, **kwargs):
    dataset.derive_schema()
    with TimeFWriter(root, dataset, **kwargs) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_prepared_shared_waveforms_do_not_open_parent_catalogs(tmp_path, backend):
    original = make_dataset()
    expected = original.records[0].signals[0].to_arrow()
    parent_dir = store(tmp_path, original, values_backend=backend)
    reader = TimeFReader(DatasetVersion.open_local(parent_dir))
    parent = reader.get_handle().reader._read_catalog()
    child = TimeFDataset(metadata=original.metadata.model_copy(update={"dataset_id": "test/child"}))
    child.add_record(record=deepcopy(parent.records[0]))
    child_dir = store(tmp_path, child)
    reader.close()
    manifest = json.loads((child_dir / "manifest.json").read_text())
    assert manifest["files"]["time_series"] == []
    assert len(manifest["payload_sources"]) == 1
    # ID/version is the pin; descriptive metadata does not change the reference.
    source = json.loads((parent_dir / "manifest.json").read_text())
    source["metadata"]["description"] = "Updated description"
    (parent_dir / "manifest.json").write_text(json.dumps(source))
    prepared = DatasetVersion.open_local(child_dir)
    # Preparation requires the full release, but processing never opens parent catalogs.
    (parent_dir / "control.duckdb").rename(parent_dir / "unavailable.duckdb")
    (parent_dir / "manifest.json").unlink()
    with TimeFReader(prepared) as child_reader:
        assert child_reader.read().records[0].signals[0].to_arrow().equals(expected)


def test_shared_payloads_cannot_reference_the_output_release(tmp_path):
    path = store(tmp_path, make_dataset())
    original_manifest = (path / "manifest.json").read_bytes()
    with TimeFReader(DatasetVersion.open_local(path)) as reader:
        loaded = reader._read_catalog()
        with pytest.raises(TimeFValidationError, match="also the output release"):
            store(tmp_path / "copy", loaded)
    assert (path / "manifest.json").read_bytes() == original_manifest
    with TimeFReader(DatasetVersion.open_local(path)) as reader:
        assert reader.read().records
