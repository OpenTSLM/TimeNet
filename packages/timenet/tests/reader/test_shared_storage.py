from copy import deepcopy
import json

import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError
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
def test_shared_waveforms_need_no_parent_catalog(tmp_path, backend):
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
    (parent_dir / "control.duckdb").rename(parent_dir / "unavailable.duckdb")
    with TimeFReader(DatasetVersion.open_local(child_dir)) as child_reader:
        assert child_reader.read().records[0].signals[0].to_arrow().equals(expected)
    source = json.loads((parent_dir / "manifest.json").read_text())
    source["metadata"]["description"] = "Changed release"
    (parent_dir / "manifest.json").write_text(json.dumps(source))
    with (
        TimeFReader(DatasetVersion.open_local(child_dir)) as child_reader,
        pytest.raises(TimeFFormatError, match="manifest changed"),
    ):
        child_reader.read().records[0].signals[0].to_arrow()
