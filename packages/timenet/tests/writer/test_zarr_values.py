from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import zarr

from timenet.dataset.edit import edit_version
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.testing import assert_datasets_equal, make_dataset
from timenet.types import Version
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
    for rel in manifest.files.time_series:
        assert rel.startswith("time_series.zarr/")
        assert (version_dir / rel).is_file()
        assert manifest.checksums[rel].startswith("sha256:")


def test_one_array_per_spec_type(tmp_path):
    version_dir = _write(tmp_path)
    group = zarr.open_group(store=version_dir / "time_series.zarr", mode="r")
    assert set(group.array_keys()) == {"sine", "cosine"}


def test_index_locator_resolves_to_values(tmp_path):
    version_dir = _write(tmp_path, chunk_max_bytes=64)
    index = pq.read_table(version_dir / "time_series_index.parquet").to_pylist()
    row = index[0]
    # Zarr locator: chunk_file=array path, chunk_offset0=element start, chunk_offset1 unused.
    assert row["chunk_offset1"] is None
    array = zarr.open_array(store=version_dir / row["chunk_file"], mode="r")
    chunk = np.asarray(array[row["chunk_offset0"] : row["chunk_offset0"] + row["n_values"]])
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
    with TimeFReader(version_dir) as reader:
        assert_datasets_equal(make_dataset(), reader.read())


def test_copy_on_write_edit_keeps_zarr_backend(tmp_path):
    version_dir = _write(tmp_path)
    out = edit_version(version_dir, tmp_path / "out", dataset_version=Version(1, 0, 1), remove_sample_ids=("sample-1",))
    manifest = Manifest.from_json((out / "manifest.json").read_text())
    assert manifest.values_backend == "zarr"
    with TimeFReader(out) as reader:
        sample = next(iter(reader.iter_samples()))
        assert len(sample.time_series[0].to_arrow()) > 0


def test_unknown_backend_rejected(tmp_path):
    dataset = make_dataset()
    dataset.derive_schema()
    with pytest.raises(ValueError, match="values_backend"), TimeFWriter(tmp_path, dataset, values_backend="hdf5") as w:
        w.write()
