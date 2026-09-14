"""The Zarr values backend, and what has to stay true when a version swaps backends.

The control plane is the same database either way. Only the values plane changes, so these tests
ask two things: that a Zarr-written version reads back byte for byte with its dtype intact, and that
the same hierarchy written to either backend comes back identical.
"""

import numpy as np
import pytest

from timenet.control_plane import TimeFReader, TimeFWriter
from timenet.control_plane.zarr_values import STORE_DIR
from timenet.testing import make_dataset


pytest.importorskip("zarr")


def _write(root, backend, **options):
    """Write the fixture dataset with one backend and return its version directory."""
    dataset = make_dataset(n_records=3, n_values=256)
    with TimeFWriter(root, dataset.metadata, values_backend=backend, **options) as writer:
        writer.write(dataset)
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _signal(reader, record_external_id, signal_external_id):
    """Return one signal's view, found by the id it was built under."""
    signals = reader.record(record_external_id).signals()
    (found,) = [signal for signal in signals if signal.external_id == signal_external_id]
    return found


@pytest.fixture
def zarr_version(tmp_path):
    """A version whose values live in a Zarr store."""
    return _write(tmp_path / "zarr", "zarr")


@pytest.fixture
def parquet_version(tmp_path):
    """The same dataset, written to Parquet shards."""
    return _write(tmp_path / "parquet", "parquet")


def test_values_round_trip_with_their_dtype(zarr_version):
    """float32 in, float32 out. A backend that widens to float64 doubles every tensor downstream."""
    with TimeFReader(zarr_version) as reader:
        values = reader.values(_signal(reader, "record-000", "record-000-lead-i").signal_id)
    assert values.shape == (256,)
    assert values.dtype == np.float32


def test_every_signal_matches_the_parquet_version_exactly(zarr_version, parquet_version):
    """Signals are matched by the name they were built under, not by the surrogate id."""
    with TimeFReader(zarr_version) as zarred, TimeFReader(parquet_version) as parqueted:
        assert zarred.record_ids() == parqueted.record_ids()
        for record_id in parqueted.record_ids():
            here = zarred.record(record_id).signals()
            there = parqueted.record(record_id).signals()
            assert [signal.external_id for signal in here] == [signal.external_id for signal in there]
            for from_store, from_shard in zip(here, there, strict=True):
                from_zarr = zarred.values(from_store.signal_id)
                from_parquet = parqueted.values(from_shard.signal_id)
                assert from_zarr.dtype == from_parquet.dtype
                np.testing.assert_array_equal(from_zarr, from_parquet)


def test_the_hierarchy_is_unchanged(zarr_version, parquet_version):
    """The values backend must not reach the control plane. Everything but the locator is equal."""
    with TimeFReader(zarr_version) as zarred, TimeFReader(parquet_version) as parqueted:
        assert zarred.record_ids() == parqueted.record_ids()
        assert zarred.task_ids() == parqueted.task_ids()
        for record_id in parqueted.record_ids():
            here, there = zarred.record(record_id), parqueted.record(record_id)
            assert here.record_id == there.record_id
            assert [(s.source_id, s.external_id, s.name, s.depth) for s in here.walk_sources()] == [
                (s.source_id, s.external_id, s.name, s.depth) for s in there.walk_sources()
            ]
            assert [
                (s.signal_id, s.external_id, s.spec_type, s.dtype, s.axis_type, s.n_values) for s in here.signals()
            ] == [(s.signal_id, s.external_id, s.spec_type, s.dtype, s.axis_type, s.n_values) for s in there.signals()]
            assert [a.to_text() for a in here.annotations] == [a.to_text() for a in there.annotations]
            assert [a.to_text() for s in here.signals() for a in s.annotations] == [
                a.to_text() for s in there.signals() for a in s.annotations
            ]


def test_an_irregular_signal_keeps_its_time_offsets_beside_its_values(zarr_version):
    """Irregular series write two parallel arrays, so both must exist and share a length."""
    with TimeFReader(zarr_version) as reader:
        chest = _signal(reader, "record-000", "record-000-chest")
        assert chest.axis_type == "irregular"
        assert reader.values(chest.signal_id).shape == (10,)
        artifacts = reader.values_backends()
    irregular = [path for path in artifacts if "/_irregular/" in path]
    offsets = [path for path in artifacts if "/_time_offsets/" in path]
    assert len(irregular) == len(offsets) == 1


def test_a_locator_addresses_an_element_offset_not_a_row_group(zarr_version):
    """The whole point of the neutral locator: no row group, and still one addressed read."""
    with TimeFReader(zarr_version) as reader:
        lead_i = _signal(reader, "record-000", "record-000-lead-i")
        locators = reader.chunk_locators(lead_i.signal_id)
        declared = reader.values_backends()
    assert len(locators) == 1
    (locator,) = locators
    assert locator.signal_id == lead_i.signal_id
    assert locator.chunk_minor_idx is None
    assert locator.chunk_file.startswith(f"{STORE_DIR}/")
    assert declared[locator.chunk_file] == "zarr"
    assert locator.n_values == 256


def test_signals_of_one_modality_share_one_array_at_distinct_offsets(zarr_version):
    """A modality is one array, so the offsets have to partition it rather than all start at zero."""
    with TimeFReader(zarr_version) as reader:
        ecg = [
            reader.chunk_locators(signal.signal_id)[0]
            for record in reader.iter_records()
            for signal in record.signals()
            if signal.spec_type == "ecg-voltage"
        ]
    assert len({locator.chunk_file for locator in ecg}) == 1
    offsets = sorted(locator.chunk_major_idx for locator in ecg)
    assert offsets == sorted(set(offsets))
    assert offsets[0] == 0


def test_the_manifest_says_which_backend_wrote_the_values(tmp_path):
    """A consumer with only manifest.json has to know whether it needs the zarr extra."""
    dataset = make_dataset(n_records=1, n_values=64)
    with TimeFWriter(tmp_path, dataset.metadata, values_backend="zarr") as writer:
        manifest = writer.write(dataset)
    assert manifest.values_backend == "zarr"
    assert manifest.to_dict()["values_backend"] == "zarr"
    assert all(part.path.startswith(f"{STORE_DIR}/") for part in manifest.files.time_series)


def test_an_unknown_backend_is_refused(tmp_path):
    dataset = make_dataset(n_records=1, n_values=32)
    with (
        pytest.raises(ValueError, match="unknown values backend"),
        TimeFWriter(tmp_path, dataset.metadata, values_backend="hdf5") as writer,
    ):
        writer.write(dataset)


def test_a_small_chunk_budget_does_not_change_what_comes_back(tmp_path):
    """Zarr chunking is a storage decision, so a signal that spans several chunks reads the same."""
    whole = _write(tmp_path / "whole", "zarr", chunk_max_bytes=1024 * 1024)
    split = _write(tmp_path / "split", "zarr", chunk_max_bytes=256)
    with TimeFReader(whole) as one, TimeFReader(split) as other:
        here = one.values(_signal(one, "record-000", "record-000-lead-i").signal_id)
        there = other.values(_signal(other, "record-000", "record-000-lead-i").signal_id)
        np.testing.assert_array_equal(here, there)
        assert here.dtype == there.dtype
