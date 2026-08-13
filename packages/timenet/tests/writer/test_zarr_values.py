from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.dataset.edit import edit_version
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import assert_datasets_equal, make_dataset
from timenet.types import DatasetMetadata, Domain, License, TimeSeriesSpec, Version, ureg
from timenet.values_backends.zarr import reader as zarr_reader_module
from timenet.values_backends.zarr.reader import ZarrValuesReader
from timenet.values_backends.zarr.writer import _array_name
from timenet.writer import TimeFWriter


def _write(tmp_path, **kwargs) -> Path:
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset, values_backend="zarr", **kwargs) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def test_manifest_records_zarr_backend_and_store_files(tmp_path):
    version_dir = _write(tmp_path)
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    assert manifest.values_backend == "zarr"
    assert manifest.files.time_series, "expected the zarr store's files to be listed"
    for part in manifest.files.time_series:
        assert part.path.startswith("time_series.zarr/")
        assert (version_dir / part.path).is_file()
        assert part.checksum.startswith("sha256:")


def test_one_array_per_spec_type(tmp_path):
    version_dir = _write(tmp_path)
    group = zarr.open_group(store=version_dir / "time_series.zarr", mode="r")
    assert set(group.array_keys()) == {"sine", "cosine"}


def test_spec_types_encode_to_distinct_single_path_segments():
    logical = ("camera/front", "camera\\front", "/camera/front", "camera%2Ffront")
    encoded = tuple(_array_name(name) for name in logical)
    assert len(set(encoded)) == len(logical)
    assert all("/" not in name and "\\" not in name for name in encoded)


def test_index_locator_resolves_to_values(tmp_path):
    version_dir = _write(tmp_path, chunk_max_bytes=64)
    index = pq.read_table(version_dir / "time_series_index.parquet").to_pylist()
    row = index[0]
    # Zarr locator: chunk_file=array path, chunk_major_idx=element start, chunk_minor_idx unused.
    assert row["chunk_minor_idx"] is None
    array = zarr.open_array(store=version_dir / row["chunk_file"], mode="r")
    chunk = np.asarray(array[row["chunk_major_idx"] : row["chunk_major_idx"] + row["n_values"]])
    assert len(chunk) == row["n_values"]


def test_one_index_row_per_series(tmp_path):
    # Zarr chunks the storage itself, so even a tiny chunk_max_bytes must not multiply index rows:
    # each (sample, series) pair gets exactly one placement spanning the series' full length.
    version_dir = _write(tmp_path, chunk_max_bytes=64)
    index = pq.read_table(version_dir / "time_series_index.parquet").to_pylist()
    keys = [(row["sample_id"], row["time_series_id"]) for row in index]
    assert len(keys) == len(set(keys))
    long_series = next(row for row in index if row["time_series_id"] == "ts-long-1")
    assert long_series["n_values"] == 512  # the fixture's long series, unsplit


def test_shard_aligned_appends_round_trip(tmp_path):
    # Tiny chunks and shards force many mid-stream shard flushes plus trailing partial shards.
    version_dir = _write(tmp_path, chunk_max_bytes=64, shard_target_bytes=128)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        assert_datasets_equal(make_dataset(), reader.read())


def test_copy_on_write_edit_keeps_zarr_backend(tmp_path):
    version_dir = _write(tmp_path)
    out = edit_version(version_dir, tmp_path / "out", dataset_version=Version(1, 0, 1), remove_sample_ids=("sample-1",))
    manifest = Manifest.from_json((out / "manifest.json").read_text())
    assert manifest.values_backend == "zarr"
    with TimeFReader(DatasetVersion.open_local(out)) as reader:
        sample = next(iter(reader.iter_samples()))
        assert len(sample.time_series[0].to_arrow()) > 0


def test_unknown_backend_rejected(tmp_path):
    dataset = make_dataset()
    dataset.derive_schema()
    with pytest.raises(ValueError, match="values_backend"), TimeFWriter(tmp_path, dataset, values_backend="hdf5") as w:
        w.write()


def test_decoded_chunk_cache_is_byte_bounded(monkeypatch):
    class FakeArray:
        shape = (4,)

        def __getitem__(self, item):
            return np.arange(4, dtype=np.int64)[item]

    reader = ZarrValuesReader()
    monkeypatch.setattr(zarr_reader_module, "_CHUNK_CACHE_MAX_BYTES", 40)
    reader._chunk("first", FakeArray(), 0, 4)
    reader._chunk("second", FakeArray(), 0, 4)
    assert list(reader._chunk_cache) == [("second", 0)]
    assert reader._chunk_cache_bytes == 32


def test_oversized_decoded_chunk_is_not_cached(monkeypatch):
    class FakeArray:
        shape = (4,)

        def __getitem__(self, item):
            return np.arange(4, dtype=np.int64)[item]

    reader = ZarrValuesReader()
    monkeypatch.setattr(zarr_reader_module, "_CHUNK_CACHE_MAX_BYTES", 16)
    reader._chunk("large", FakeArray(), 0, 4)
    assert not reader._chunk_cache
    assert reader._chunk_cache_bytes == 0


def test_nd_uint8_round_trip_and_range_read(tmp_path):
    frames = np.arange(7 * 4 * 5 * 3, dtype=np.uint8).reshape(7, 4, 5, 3)
    spec = TimeSeriesSpec(
        spec_type="camera",
        name="RGB camera",
        unit_value=ureg.dimensionless,
        dtype="uint8",
        value_shape=(4, 5, 3),
        dimension_names=("height", "width", "color"),
    )
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="bench/camera",
            dataset_version=Version(1, 0, 0),
            name="Camera",
            description="N-D fixture",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    dataset.add_sample(
        time_series=(
            TimeSeries(
                spec=spec,
                channel="rgb",
                time_axis=RegularAxis.from_rate_hz(30),
                n_values=len(frames),
                loader=lambda: pa.FixedShapeTensorArray.from_numpy_ndarray(frames, dim_names=spec.dimension_names),
                time_series_id="camera-1",
            ),
        ),
        sample_id="sample-camera",
    )
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset, values_backend="zarr", chunk_max_bytes=120) as writer:
        writer.write()
    version_dir = tmp_path / "bench/camera/1.0.0"
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    assert manifest.timef_format_version == 1
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = next(iter(reader.iter_samples())).time_series[0]
        assert isinstance(restored.to_arrow(), pa.FixedShapeTensorArray)
        np.testing.assert_array_equal(restored.to_numpy(), frames)
        np.testing.assert_array_equal(restored.read_steps(2, 5).to_numpy_ndarray(), frames[2:5])


def test_zarr_empty_range_read_returns_typed_empty_arrays(tmp_path):
    # Covers ZarrValuesReader.load_range's `start >= bounded_stop` branch, which builds a
    # (0, *value_shape) array: an empty range must yield a correctly typed zero-length Arrow array for
    # both scalar and N-D specs (pa.FixedShapeTensorArray.from_numpy_ndarray rejects a 0-length ndarray).
    frames = np.arange(6 * 2 * 3, dtype=np.uint8).reshape(6, 2, 3)
    nd_spec = TimeSeriesSpec(
        spec_type="camera",
        name="cam",
        unit_value=ureg.dimensionless,
        dtype="uint8",
        value_shape=(2, 3),
        dimension_names=("h", "w"),
    )
    scalar_spec = TimeSeriesSpec(
        spec_type="sine",
        name="sine",
        unit_value=ureg.millivolt,
    )
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="bench/empty",
            dataset_version=Version(1, 0, 0),
            name="Empty range",
            description="empty-range fixture",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    dataset.add_sample(
        time_series=(
            TimeSeries(
                spec=nd_spec,
                channel="rgb",
                time_axis=RegularAxis.from_rate_hz(10),
                time_series_id="cam-1",
                n_values=len(frames),
                loader=lambda: pa.FixedShapeTensorArray.from_numpy_ndarray(frames, dim_names=nd_spec.dimension_names),
            ),
            TimeSeries(
                spec=scalar_spec,
                channel="i",
                time_axis=RegularAxis.from_rate_hz(10),
                time_series_id="sig-1",
                n_values=6,
                loader=lambda: pa.array(np.arange(6, dtype=np.float32)),
            ),
        ),
        sample_id="sample-0",
    )
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset, values_backend="zarr") as writer:
        writer.write()
    with TimeFReader(DatasetVersion.open_local(tmp_path / "bench/empty/1.0.0")) as reader:
        series = {ts.spec.spec_type: ts for ts in next(iter(reader.iter_samples())).time_series}

    nd_empty = series["camera"].read_steps(2, 2)  # valid but empty half-open range
    assert isinstance(nd_empty, pa.FixedShapeTensorArray)
    assert len(nd_empty) == 0
    assert nd_empty.to_numpy_ndarray().shape == (0, 2, 3)

    scalar_empty = series["sine"].read_steps(6, 6)  # start == total
    assert len(scalar_empty) == 0
    assert scalar_empty.type == pa.float32()
