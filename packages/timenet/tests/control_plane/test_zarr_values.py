"""The Zarr values backend, and what has to stay true when a version swaps backends.

The control plane is the same database either way. Only the values plane changes, so these tests
ask two things: that a Zarr-written version reads back byte for byte with its dtype intact, and that
the same hierarchy written to either backend comes back identical.
"""

from fractions import Fraction

import numpy as np
import pytest

from timenet.control_plane import DeclarativeDataset, Record, Signal, Source, TimeFReader, TimeFWriter
from timenet.control_plane.values import PendingSignal
from timenet.control_plane.zarr_values import STORE_DIR, ZarrValuesPlaneWriter
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFValidationError
from timenet.testing import make_dataset, make_metadata
from timenet.types import TimeSeriesSpec, ureg


pytest.importorskip("zarr")

STAGE_SPEC = TimeSeriesSpec(spec_type="sleep-stage", name="Sleep stage", unit_value=ureg.dimensionless, dtype="str")
STAGE_AXIS = RegularAxis(period_us=Fraction(30_000_000, 1))


def _string_dataset():
    """Build a dataset whose one signal carries string values rather than numbers."""
    stages = Signal(
        id="stages",
        name="Sleep stage",
        values=np.array(["W", "N1", "N2", "N3", "REM"]),
        time_axis=STAGE_AXIS,
        spec=STAGE_SPEC,
    )
    return DeclarativeDataset(
        metadata=make_metadata(),
        records=[Record(id="night-000", sources=[Source(id="psg", name="PSG", signals=[stages])])],
    )


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


def test_a_string_signal_is_refused_by_name(tmp_path):
    """``np.dtype('str').itemsize`` is 0, so the chunk budget used to divide by zero instead.

    Naming a backend that cannot hold the dtype is caller input, not a corrupt artifact, so the
    refusal is the same class as the partition's below it and a caller catching ``ValueError``
    around the write sees both.
    """
    dataset = _string_dataset()
    with (
        pytest.raises(TimeFValidationError) as raised,
        TimeFWriter(tmp_path, dataset.metadata, values_backend="zarr") as writer,
    ):
        writer.write(dataset)
    assert isinstance(raised.value, ValueError)
    message = str(raised.value)
    assert "'str'" in message
    assert "'zarr'" in message
    assert "'parquet'" in message


def test_a_pending_signal_whose_two_arrays_disagree_is_refused(tmp_path):
    """The partition advances both arrays together, and it is reached without passing through Signal.

    ``Signal`` refuses the same pair at construction, but a backend implementing the public
    ``ValuesWriter`` protocol is handed a ``PendingSignal``, which nothing else checks.
    """
    writer = ZarrValuesPlaneWriter(tmp_path)
    pending = PendingSignal(
        signal_id=1,
        name="Chest temperature",
        spec_type="temperature",
        dtype="float32",
        values=np.arange(100, dtype=np.float32),
        time_offsets_us=np.arange(99, dtype=np.int64),
    )
    with pytest.raises(TimeFValidationError, match="one offset per value"):
        writer.add(pending)


def test_an_enum_signal_is_refused_by_name_too(tmp_path):
    """'enum' is no NumPy dtype at all, so asking NumPy for its width raised a TypeError instead."""
    stage_spec = TimeSeriesSpec(
        spec_type="sleep-stage",
        name="Sleep stage",
        unit_value=ureg.dimensionless,
        dtype="enum",
        categories=("W", "N1", "N2", "N3", "REM"),
    )
    stages = Signal(
        id="stages",
        name="Sleep stage",
        values=np.array(["W", "N1", "N2", "N3", "REM"]),
        time_axis=STAGE_AXIS,
        spec=stage_spec,
    )
    dataset = DeclarativeDataset(
        metadata=make_metadata(),
        records=[Record(id="night-000", sources=[Source(id="psg", name="PSG", signals=[stages])])],
    )
    with (
        pytest.raises(TimeFValidationError) as raised,
        TimeFWriter(tmp_path, dataset.metadata, values_backend="zarr") as writer,
    ):
        writer.write(dataset)
    message = str(raised.value)
    assert "'enum'" in message
    assert "'parquet'" in message


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


def test_a_window_reads_the_same_steps_a_slice_of_the_whole_signal_holds(zarr_version, parquet_version):
    """A window is a range read on this backend too, so both backends must return the same bytes."""
    with TimeFReader(zarr_version) as zarred, TimeFReader(parquet_version) as parqueted:
        from_store = _signal(zarred, "record-001", "record-001-lead-ii").signal_id
        from_shard = _signal(parqueted, "record-001", "record-001-lead-ii").signal_id
        whole = zarred.values(from_store)
        for start, stop in ((0, 0), (0, 256), (17, 33), (200, 4_000)):
            window = zarred.values_window(from_store, start, stop)
            assert window.dtype == whole.dtype
            assert window.tobytes() == whole[start:stop].tobytes(), f"window [{start}, {stop})"
            assert window.tobytes() == parqueted.values_window(from_shard, start, stop).tobytes()


def test_a_window_crossing_a_storage_chunk_is_stitched_back(tmp_path):
    """A 256 byte Zarr chunk holds 64 float32, so this window crosses several of them."""
    version = _write(tmp_path / "split", "zarr", chunk_max_bytes=256)
    with TimeFReader(version) as reader:
        signal_id = _signal(reader, "record-000", "record-000-lead-i").signal_id
        whole = reader.values(signal_id)
        assert reader.values_window(signal_id, 30, 200).tobytes() == whole[30:200].tobytes()
