from dataclasses import replace
import pickle

import pytest

from timenet.dataset import Record, Signal, Source
from timenet.errors import TimeFFormatError
from timenet.payloads import reader as payload_reader
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


def test_cached_timestamps_skip_repeat_scans_but_validate_another_descriptor(tmp_path, monkeypatch):
    dataset = make_dataset()
    signal = Signal.from_irregular(
        [1.0, 2.0, 3.0], time_offsets_us=[0, 10, 20], spec=dataset.records[0].signals[0].spec, name="irregular"
    )
    dataset.add_record(record=Record(record_id="irregular", sources=(Source(name="sensor", signals=(signal,)),)))
    path = dataset.write(path=tmp_path)
    calls = []
    original = payload_reader.validate_offsets

    def tracked(values, descriptor):
        calls.append(descriptor)
        original(values, descriptor)

    monkeypatch.setattr(payload_reader, "validate_offsets", tracked)
    with TimeFReader(DatasetVersion.open_local(path)) as reader:
        stored = next(reader.iter_records(["irregular"])).signals[0].time_offsets
        assert stored is not None and stored.reference is not None
        assert stored.read().to_pylist() == stored.read().to_pylist() == [0, 10, 20]
        assert len(calls) == 1
        wrong = reader._payload_resolver.payload(replace(stored.descriptor, first_us=5), stored.reference)
        with pytest.raises(TimeFFormatError, match="endpoints"):
            wrong.read()
        assert len(calls) == 2
