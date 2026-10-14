import pickle

import pytest

from timenet.reader import TimeFReader
from timenet.registry.version import DatasetVersion
from timenet.testing import make_dataset
from timenet.writer import TimeFWriter


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_array_handles_read_without_reopening_their_catalog(tmp_path, backend):
    dataset = make_dataset()
    with TimeFWriter(tmp_path, dataset, values_backend=backend) as writer:
        writer.write()
    path = tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
    with TimeFReader(DatasetVersion.open_local(path)) as reader:
        signal = next(reader.iter_records()).signals[0]
        expected = signal.to_arrow()
        assert signal.values.reference is not None
        assert signal.values.reference.source is not None
        assert signal.values.reference.source.dataset.dataset_id == dataset.metadata.dataset_id
    # A storage handle retains physical locators, rather than a deferred catalog query.
    (path / "control.duckdb").unlink()
    restored = pickle.loads(pickle.dumps(signal.values))
    assert restored.read().equals(expected)
    assert restored.read_steps(1, 4).equals(expected.slice(1, 3))
