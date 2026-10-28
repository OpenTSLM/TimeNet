import copy
import pickle
from typing import get_type_hints

import numpy as np
import pyarrow as pa
import pytest

from timenet.format.control_reader import DuckDBControlReader
from timenet.format.stored_array import ChunkLocation, ChunkReader, StoredArray
from timenet.reader import TimeFReader
import timenet.reader.reader as reader_module
from timenet.registry import open_local_version
from timenet.testing import make_dataset
from timenet.types import ArrayHandle, TimeSeriesSpec
from timenet.types.reference_keys import SourceKey
from timenet.values_backends.types import StoredChunk
from timenet.values_backends.zarr.reader import ZarrValuesReader
from timenet.writer import TimeFWriter


def _write(root, **kwargs):
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(root, dataset, **kwargs) as writer:
        writer.write()
    return dataset, root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _arrays(reader):
    signals = [signal for record in reader.iter_records() for signal in record.signals]
    for signal in signals:
        assert isinstance(signal._values, StoredArray)
    return signals


@pytest.fixture(params=["parquet", "zarr"])
def stored(tmp_path, request):
    # Small chunks split the long series into several Parquet chunks.
    dataset, version_dir = _write(tmp_path, values_backend=request.param, chunk_max_bytes=64)
    expected = {signal.id: signal.to_arrow() for record in dataset.records for signal in record.signals}
    with TimeFReader(open_local_version(version_dir)) as reader:
        yield reader, _arrays(reader), expected


@pytest.fixture
def array_reads(monkeypatch):
    reads = []
    for name in ("_read_array", "_read_array_steps"):
        original = getattr(TimeFReader, name)

        def counting(self, location, *args, original=original):
            reads.append(location)
            return original(self, location, *args)

        monkeypatch.setattr(TimeFReader, name, counting)
    return reads


def test_a_stored_array_reads_its_chunks_and_is_never_loaded(stored):
    _, signals, expected = stored
    for signal in signals:
        handle = signal._values
        assert isinstance(handle, ArrayHandle)
        assert not handle.is_loaded
        assert handle.read().equals(expected[signal.id])
        assert handle.read_steps(3, 10).equals(expected[signal.id].slice(3, 7))
        assert not handle.is_loaded


def test_a_stored_array_locates_its_chunks_without_reading_them(stored, array_reads):
    _, signals, _ = stored
    locations = {signal.id: signal._values._location for signal in signals}
    assert not array_reads
    for signal in signals:
        location = locations[signal.id]
        assert isinstance(location, ChunkLocation)
        assert location.source is None
        assert [chunk.chunk_idx for chunk in location.chunks] == list(range(len(location.chunks)))
        assert sum(chunk.n_values for chunk in location.chunks) == signal.n_values


def test_stored_arrays_locate_their_chunks_a_batch_of_signals_at_a_time(tmp_path, monkeypatch):
    _, version_dir = _write(tmp_path)
    monkeypatch.setattr(reader_module, "_CHUNK_LOCATOR_BATCH_SIZE", 2)
    batches = []
    original = DuckDBControlReader.chunk_locations_by_ids

    def counting_batch(reader, signal_ids):
        batches.append(tuple(signal_ids))
        return original(reader, signal_ids)

    monkeypatch.setattr(DuckDBControlReader, "chunk_locations_by_ids", counting_batch)
    with TimeFReader(open_local_version(version_dir)) as reader:
        handles = [signal._values for signal in _arrays(reader)]
        assert not batches
        locations = [handle._location for handle in handles]
        # The reader keeps each batch's chunk locations until it closes, so asking again queries nothing.
        assert [handle._location for handle in handles] == locations
    assert len(handles) > 2
    assert [len(batch) for batch in batches] == [len(handles[start : start + 2]) for start in range(0, len(handles), 2)]


def test_a_stored_array_locates_and_reads_its_chunks_after_its_reader_closes(stored):
    reader, signals, expected = stored
    handle = signals[0]._values
    location = handle._location
    reader.close()
    # The reader reopens its catalog and values files on demand.
    assert handle._location == location
    assert handle.read().equals(expected[signals[0].id])
    assert handle.read_steps(1, 3).equals(expected[signals[0].id].slice(1, 2))


def test_stored_arrays_from_one_reader_for_one_signal_are_equal(tmp_path):
    _, version_dir = _write(tmp_path)
    with TimeFReader(open_local_version(version_dir)) as reader:
        first, other = (signal._values for signal in _arrays(reader)[:2])
        again = _arrays(reader)[0]._values
        assert again is not first
        assert again == first
        assert other != first


def test_a_stored_array_pickles_its_reader_and_signal_without_values(stored, array_reads):
    _, signals, expected = stored
    handle = signals[0]._values
    assert isinstance(handle, StoredArray)
    pickled = pickle.dumps(handle)
    handle.read()
    assert pickle.dumps(handle) == pickled

    back = pickle.loads(pickled)
    assert type(back) is StoredArray
    assert (back.signal_id, back.spec) == (handle.signal_id, handle.spec)
    assert back._location == handle._location
    assert not back.is_loaded
    assert back.read().equals(expected[signals[0].id])
    assert len(array_reads) == 2


def test_a_deep_copy_shares_the_stored_array(stored):
    _, signals, _ = stored
    handle = signals[0]._values
    assert copy.deepcopy(handle) is handle


def test_type_hints_resolve_on_the_stored_array():
    hints = get_type_hints(StoredArray)
    assert hints["reader"] is ChunkReader
    assert hints["spec"] is TimeSeriesSpec
    assert get_type_hints(ChunkLocation) == {"source": SourceKey | None, "chunks": tuple[StoredChunk, ...]}


def _address_ranges(values):
    if isinstance(values, np.ndarray):
        start = values.__array_interface__["data"][0]
        return [(start, start + values.nbytes)]
    return [(buffer.address, buffer.address + buffer.size) for buffer in values.buffers() if buffer is not None]


def test_read_arrays_keep_no_decoded_payload_alive(stored):
    reader, signals, _ = stored
    arrays = {signal.id: signal._values.read() for signal in signals}
    backend = reader._values_reader()
    # The reader still caches the row groups or Zarr chunks it decoded for the reads.
    cached = [*getattr(backend, "_chunk_cache", {}).values()]
    parquet = getattr(backend, "_parquet", None)
    if parquet is not None:
        cached.extend(parquet._groups.values())
    assert cached
    decoded = [address for values in cached for address in _address_ranges(values)]
    for signal_id, kept in arrays.items():
        assert isinstance(kept, pa.Array)
        for start, stop in _address_ranges(kept):
            assert not any(start < end and begin < stop for begin, end in decoded), signal_id


def test_read_step_ranges_keep_no_decoded_payload_alive(stored):
    reader, signals, _ = stored
    backend = reader._values_reader()
    for signal in signals:
        backend.close()
        kept = signal._values.read_steps(3, 10)
        assert isinstance(kept, pa.Array)
        parquet = getattr(backend, "_parquet", None)
        # Steps 3 to 10 of each series lie inside one Zarr storage chunk or one Parquet row group.
        (decoded,) = [*getattr(backend, "_chunk_cache", {}).values(), *(parquet._groups.values() if parquet else ())]
        for start, stop in _address_ranges(kept):
            assert not any(start < end and begin < stop for begin, end in _address_ranges(decoded)), signal.id


def test_a_zarr_series_across_storage_chunks_is_copied_once(tmp_path, monkeypatch):
    _, version_dir = _write(tmp_path, values_backend="zarr", chunk_max_bytes=64)
    ranges = []
    original = ZarrValuesReader._read_range

    def recording(self, version, rel_path, start, stop):
        ranges.append(original(self, version, rel_path, start, stop))
        return ranges[-1]

    monkeypatch.setattr(ZarrValuesReader, "_read_range", recording)
    with TimeFReader(open_local_version(version_dir)) as reader:
        signal = max(_arrays(reader), key=lambda signal: signal.n_values)
        kept = signal._values.read()
        backend = reader._values_reader()
        assert isinstance(backend, ZarrValuesReader)
        # The series is one run, and the read decodes several storage chunks.
        assert len(backend._chunk_cache) > 1
    (joined,) = ranges
    # The reader joins the chunks into a new array, and the result uses that array without a second copy.
    assert np.shares_memory(kept.to_numpy(), joined)
