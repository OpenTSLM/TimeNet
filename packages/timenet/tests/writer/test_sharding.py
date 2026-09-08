"""Sharded write/read round-trips and targeted-read verification.

Small byte targets split every artifact type into many parts. These tests assert the round-trip is
lossless and, crucially, that a read only touches the shards and row groups it must: materializing one
series opens only the value shard(s) its chunks live in and reads only those chunks' row groups, never
the whole plane. The control-plane tables are still read eagerly on this branch; their targeted (pruned)
reads land with the pruned reader, and their assertions belong here too once that exists.
"""

import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import assert_datasets_equal
from timenet.types import (
    Annotation,
    ClassificationTask,
    DatasetMetadata,
    License,
    TimeSeriesSpec,
    Version,
    ureg,
)
import timenet.values_backends.parquet.reader as values_reader
from timenet.writer import TimeFWriter


# Small enough that a modest dataset shards every artifact type into several parts, tiny on disk.
_SMALL_TARGETS = {
    "control_shard_target_bytes": 128,
    "shard_target_bytes": 256,
    "row_group_target_bytes": 128,
    "chunk_max_bytes": 64,
}


def _sharded_dataset(n_records: int, series_len: int) -> TimeFDataset:
    """Build a synthetic dataset large enough to shard every artifact type under small byte targets."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/sharding",
            dataset_version=Version(1, 0, 0),
            name="Sharding",
            description="synthetic",
            license=License.MIT,
        )
    )
    spec = TimeSeriesSpec(spec_type="s", name="S", unit_value=ureg.dimensionless)
    axis = RegularAxis.from_rate_hz(16)
    for i in range(n_records):
        values = [float((i + j) % 11) for j in range(series_len)]
        ts = TimeSeries.from_values(values, spec=spec, signal="a", time_axis=axis, time_series_id=f"ts-{i:03d}")
        record = dataset.add_record(time_series=(ts,), subject_ids=(f"subj-{i}",), record_id=f"record-{i:03d}")
        record.add_annotation(Annotation(key="label", value=f"cls-{i % 3}", id=f"ann-{i:03d}"))
        dataset.add_task(record, ClassificationTask(target=f"c{i % 2}", id=f"task-{i:03d}"))
    return dataset


def _write(tmp_path, dataset, **targets):
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset, **targets) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _index_rows(version_dir, manifest):
    return [r for part in manifest.files.time_series_index for r in pq.read_table(version_dir / part.path).to_pylist()]


# ---- round trip ------------------------------------------------------------------------------


@pytest.mark.parametrize(("n_records", "series_len"), [(12, 128), (5, 300), (20, 40)])
def test_all_artifact_types_shard_and_round_trip(tmp_path, n_records, series_len):
    original = _sharded_dataset(n_records, series_len)
    version_dir = _write(tmp_path, _sharded_dataset(n_records, series_len), **_SMALL_TARGETS)
    files = Manifest.from_json((version_dir / "manifest.json").read_text()).files
    classification_parts = [part for part in files.tasks if "task=classification" in part.path]
    assert len(files.records) >= 3
    assert len(files.annotations) >= 3
    assert len(files.time_series_index) >= 3
    assert len(classification_parts) >= 3
    assert len(files.time_series) >= 3  # values-plane shards
    for rel in files.all_parts():
        assert (version_dir / rel).exists()
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
    assert_datasets_equal(original, restored)


# ---- metadata --------------------------------------------------------------------------------


def test_index_is_globally_sorted_across_parts(tmp_path):
    # The bounded-memory index writer emits rows in encoded (record_id, time_series_id, chunk_idx) order
    # by nested iteration; across dozens of parts the concatenation must stay globally sorted, which is
    # what lets the reader concatenate parts in manifest order and bisect.
    version_dir = _write(tmp_path, _sharded_dataset(12, 128), **_SMALL_TARGETS)
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    assert len(manifest.files.time_series_index) >= 3
    rows = _index_rows(version_dir, manifest)
    keys = [(r["record_id"], r["time_series_id"], r["chunk_idx"]) for r in rows]
    assert keys == sorted(keys)
    assert len(rows) == manifest.counts.time_series_index_rows


def test_index_metadata_locates_every_series_in_the_shards(tmp_path):
    version_dir = _write(tmp_path, _sharded_dataset(12, 128), **_SMALL_TARGETS)
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    shards = {part.path for part in manifest.files.time_series}
    row_group_counts = {rel: pq.ParquetFile(version_dir / rel).metadata.num_row_groups for rel in shards}
    rows = _index_rows(version_dir, manifest)
    assert len(rows) == manifest.counts.time_series_index_rows
    for row in rows:
        assert row["chunk_file"] in shards  # locator points at a listed shard
        assert 0 <= row["chunk_major_idx"] < row_group_counts[row["chunk_file"]]  # a real row group in it
        assert row["n_values"] > 0


# ---- targeted reads: only the shards / row groups we need ------------------------------------


def test_open_and_build_records_reads_no_value_shard(tmp_path, monkeypatch):
    version_dir = _write(tmp_path, _sharded_dataset(12, 128), **_SMALL_TARGETS)
    opened: list[str] = []
    original = pq.ParquetFile
    monkeypatch.setattr(pq, "ParquetFile", lambda p, *a, **k: opened.append(str(p)) or original(p, *a, **k))
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        dataset = reader.read()
        _ = [s.time_series for s in dataset.records]  # touch every record's series metadata
    # values are lazy; building records may open control-plane tables but no value shard (under time_series/)
    assert not [p for p in opened if "/time_series/" in p]


@pytest.mark.parametrize("series_id", ["ts-000", "ts-006", "ts-011"])
def test_reading_a_series_opens_only_its_value_shards(tmp_path, monkeypatch, series_id):
    version_dir = _write(tmp_path, _sharded_dataset(12, 128), **_SMALL_TARGETS)
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    expected_shards = {r["chunk_file"] for r in _index_rows(version_dir, manifest) if r["time_series_id"] == series_id}
    assert expected_shards <= {part.path for part in manifest.files.time_series}
    assert len(expected_shards) < len(manifest.files.time_series)  # the series lives in only some shards

    opened: list[str] = []
    original = pq.ParquetFile
    monkeypatch.setattr(pq, "ParquetFile", lambda p, *a, **k: opened.append(str(p)) or original(p, *a, **k))
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        dataset = reader.read()
        assert not [p for p in opened if "/time_series/" in p]  # read touches control tables, not value shards
        series = next(ts for s in dataset.records for ts in s.time_series if ts.time_series_id == series_id)
        series.to_arrow()
    opened_rel = {p.removeprefix(f"{version_dir}/") for p in opened if "/time_series/" in p}
    assert opened_rel == expected_shards  # exactly the series' value shards, nothing else


def test_reading_a_series_reads_only_its_row_groups(tmp_path, monkeypatch):
    version_dir = _write(tmp_path, _sharded_dataset(12, 128), **_SMALL_TARGETS)
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    expected = {
        (r["chunk_file"], r["chunk_major_idx"])
        for r in _index_rows(version_dir, manifest)
        if r["time_series_id"] == "ts-006"
    }

    read: list[tuple[str, int]] = []
    original = values_reader.ParquetValuesReader._row_group

    def spy(self, root, rel_path, row_group):
        read.append((rel_path, row_group))
        return original(self, root, rel_path, row_group)

    monkeypatch.setattr(values_reader.ParquetValuesReader, "_row_group", spy)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        series = next(ts for s in reader.read().records for ts in s.time_series if ts.time_series_id == "ts-006")
        series.to_arrow()
    assert set(read) == expected  # only ts-006's chunks' row groups were decoded, not whole shards
