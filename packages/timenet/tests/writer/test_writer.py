import json

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFValidationError
from timenet.manifest import Manifest
from timenet.testing import make_dataset
from timenet.types import (
    Annotation,
    ClassificationTask,
    DatasetMetadata,
    License,
    TimeSeriesSpec,
    Version,
    ureg,
)
from timenet.writer import TimeFWriter
from timenet.writer.encodings import applied_matches, values_encoding_of
from timenet.writer.value_encoding import ValueEncoding


def _written(tmp_path, dataset=None, **kwargs):
    dataset = dataset if dataset is not None else make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset, **kwargs) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _query(version_dir, sql):
    """Run one query against a committed version's control database."""
    connection = duckdb.connect(str(version_dir / "control.duckdb"), read_only=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def _chunks(version_dir):
    """Return every chunk locator, as the control plane stores them."""
    rows = _query(
        version_dir,
        "SELECT s.external_id, c.chunk_idx, v.chunk_file, c.chunk_major_idx, c.chunk_minor_idx, c.n_values "
        "FROM signal_chunks c "
        "JOIN time_series s ON s.time_series_id = c.time_series_id "
        "JOIN values_artifacts v ON v.artifact_id = c.artifact_id "
        "ORDER BY s.external_id, c.chunk_idx",
    )
    names = ("time_series_id", "chunk_idx", "chunk_file", "chunk_major_idx", "chunk_minor_idx", "n_values")
    return [dict(zip(names, row, strict=True)) for row in rows]


# ---- layout & manifest ------------------------------------------------------------------------


def test_writes_expected_layout(tmp_path):
    version_dir = _written(tmp_path)
    assert (version_dir / "manifest.json").exists()
    files = Manifest.from_json((version_dir / "manifest.json").read_text()).files
    assert files.control_db is not None and files.control_db.path == "control.duckdb"
    assert any(part.path.startswith("time_series/") for part in files.time_series)
    for rel in files.all_parts():
        assert (version_dir / rel).exists()


def test_the_version_directory_holds_only_the_two_planes(tmp_path):
    # The control plane is one file and the values plane is a directory of shards. Nothing else is
    # published, so a stray Parquet control table would show up here.
    version_dir = _written(tmp_path)
    assert {entry.name for entry in version_dir.iterdir()} == {"manifest.json", "control.duckdb", "time_series"}


def test_manifest_is_valid_and_matches_dataset(tmp_path):
    version_dir = _written(tmp_path)
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    assert manifest.dataset_id == "timenet/hello-world"
    assert manifest.timef_format_version == 2
    assert manifest.counts.records == 3
    assert len(manifest.schema.time_series_specs) == 2
    # files listed in the manifest all exist
    for rel in manifest.files.all_parts():
        assert (version_dir / rel).exists()


def test_manifest_has_per_file_checksum_and_size(tmp_path):
    version_dir = _written(tmp_path)
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    parts = manifest.files.all_files()
    assert parts, "expected file descriptors"
    assert all(part.checksum.startswith("sha256:") for part in parts)
    assert all(part.size > 0 for part in parts)


def test_manifest_data_files_are_lists_of_parts(tmp_path):
    version_dir = _written(tmp_path)
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    assert isinstance(manifest.files.time_series, tuple)
    assert len(manifest.files.time_series) >= 1
    raw_files = json.loads((version_dir / "manifest.json").read_text())["files"]
    assert isinstance(raw_files["time_series"], list), "time_series should serialize as a JSON array"
    for entry in (*raw_files["time_series"], raw_files["control_db"]):
        assert set(entry) == {"path", "checksum", "size"}


# ---- shard schema & encodings -----------------------------------------------------------------


def test_shard_has_time_series_id_column(tmp_path):
    version_dir = _written(tmp_path)
    shard = next(version_dir.glob("time_series/part-*.parquet"))
    names = set(pq.ParquetFile(shard).schema_arrow.names)
    assert "time_series_id" in names  # self-describing shards (re-indexable)
    assert "values" in names


def test_values_carry_the_selected_encoding(tmp_path):
    version_dir = _written(tmp_path)
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    assert manifest.value_encoding, "the manifest should record what each modality was encoded with"
    for shard in version_dir.glob("time_series/part-*.parquet"):
        pf = pq.ParquetFile(shard)
        spec_types = set(pf.read(columns=["spec_type"]).column("spec_type").to_pylist())
        assert len(spec_types) == 1, "shards are single-modality so one encoding always fits"
        selected = ValueEncoding(manifest.value_encoding[spec_types.pop()])
        assert applied_matches(selected, values_encoding_of(str(shard)))


# ---- chunk splitting --------------------------------------------------------------------------


def test_long_series_splits_into_multiple_chunks(tmp_path):
    # Tiny chunk cap forces the long hello_world series to split.
    version_dir = _written(tmp_path, chunk_max_bytes=64, row_group_target_bytes=64)
    chunks_per_series: dict[str, set] = {}
    for row in _chunks(version_dir):
        chunks_per_series.setdefault(row["time_series_id"], set()).add(row["chunk_idx"])
    assert any(len(chunks) > 1 for chunks in chunks_per_series.values())


def test_chunk_locators_resolve_to_values(tmp_path):
    version_dir = _written(tmp_path, chunk_max_bytes=64, row_group_target_bytes=64)
    row = _chunks(version_dir)[0]
    shard = pq.ParquetFile(version_dir / row["chunk_file"])
    table = shard.read_row_group(row["chunk_major_idx"])
    chunk = table.column("values")[row["chunk_minor_idx"]]
    assert len(chunk) == row["n_values"]


def test_shard_rotation_leaves_no_empty_trailing_shard(tmp_path):
    # Tiny shard cap forces a rotation on essentially every row group, including the last one.
    version_dir = _written(tmp_path, chunk_max_bytes=64, row_group_target_bytes=64, shard_target_bytes=64)
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    assert manifest.files.time_series, "expected at least one shard"
    for part in manifest.files.time_series:
        assert pq.ParquetFile(version_dir / part.path).metadata.num_rows > 0, f"empty shard {part.path} published"


# ---- validation & commit protocol -------------------------------------------------------------


def test_write_derives_schema_automatically(tmp_path):
    dataset = make_dataset()
    assert dataset.schema is None
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    assert dataset.schema is not None


def test_unknown_values_backend_rejected_before_staging(tmp_path):
    dataset = make_dataset()
    with pytest.raises(ValueError, match="values_backend"):
        TimeFWriter(tmp_path, dataset, values_backend="hdf5")
    assert not list(tmp_path.rglob("*.tmp-*"))


def test_recommit_raises_file_exists(tmp_path):
    _written(tmp_path)
    dataset = make_dataset()
    dataset.derive_schema()
    with pytest.raises(FileExistsError):
        TimeFWriter(tmp_path, dataset).__enter__()


def test_manifest_written_last(tmp_path):
    seen_before_manifest = []
    dataset = make_dataset()
    dataset.derive_schema()

    def probe(event):
        version_dir = tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
        if event.stage != "commit":
            seen_before_manifest.append((version_dir / "manifest.json").exists())

    with TimeFWriter(tmp_path, dataset, progress_cb=probe) as writer:
        writer.write()
    # the committed manifest never appeared before the commit event
    assert not any(seen_before_manifest)


def test_abort_leaves_no_partial_dir(tmp_path):
    calls = {"n": 0}

    def bad_loader():
        calls["n"] += 1
        raise RuntimeError("boom")

    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="demo/boom",
            dataset_version=Version(1, 0, 0),
            name="Boom",
            description="d",
            license=License.MIT,
        )
    )
    spec = TimeSeriesSpec(
        spec_type="s",
        name="S",
        unit_value=ureg.dimensionless,
    )
    record = dataset.add_record(
        time_series=(
            TimeSeries(spec=spec, signal="c", time_axis=RegularAxis.from_rate_hz(1), n_values=1, loader=bad_loader),
        ),
    )
    record.add_annotation(Annotation(key="k", value=1))
    dataset.add_task(record, ClassificationTask(target="x"))
    dataset.derive_schema()

    with pytest.raises(RuntimeError), TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    # no committed dir and no staging leftovers
    assert not (tmp_path / "demo/boom" / "1.0.0").exists()
    assert not list((tmp_path / "demo/boom").glob("*.tmp-*")) if (tmp_path / "demo/boom").exists() else True


def test_per_series_array_contract_enforced(tmp_path):
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="demo/bad",
            dataset_version=Version(1, 0, 0),
            name="Bad",
            description="d",
            license=License.MIT,
        )
    )
    spec = TimeSeriesSpec(
        spec_type="s",
        name="S",
        unit_value=ureg.dimensionless,
    )
    import pyarrow as pa  # noqa: PLC0415

    # declares 5 observations, loader returns 3
    ts = TimeSeries(
        spec=spec,
        signal="c",
        time_axis=RegularAxis.from_rate_hz(1),
        loader=lambda: pa.array([1.0, 2.0, 3.0], type=pa.float32()),
        n_values=5,
    )
    dataset.add_record(time_series=(ts,))
    dataset.derive_schema()
    with pytest.raises(TimeFValidationError), TimeFWriter(tmp_path, dataset) as writer:
        writer.write()


def test_control_plane_records_content(tmp_path):
    version_dir = _written(tmp_path)
    rows = {row[0]: row[1] for row in _query(version_dir, "SELECT external_id, subject_ids FROM records")}
    assert set(rows) == {"record-0", "record-1", "record-2"}
    # shared series appears in both record-0 and record-1
    links = _query(
        version_dir,
        "SELECT r.external_id, s.external_id FROM record_time_series l "
        "JOIN records r ON r.record_id = l.record_id "
        "JOIN time_series s ON s.time_series_id = l.time_series_id",
    )
    carrying = {record for record, series in links if series == "ts-shared"}
    assert carrying == {"record-0", "record-1"}


def test_shared_series_stored_once(tmp_path):
    version_dir = _written(tmp_path)
    # the shared series has one chunk row in the shards and one in the control plane, though two
    # records link to it
    shard_rows = [
        r
        for shard in version_dir.glob("time_series/part-*.parquet")
        for r in pq.read_table(shard).to_pylist()
        if r["time_series_id"] == "ts-shared"
    ]
    assert len(shard_rows) == 1
    assert len([row for row in _chunks(version_dir) if row["time_series_id"] == "ts-shared"]) == 1


def test_tasks_carry_their_type(tmp_path):
    version_dir = _written(tmp_path)
    stored = {row[0] for row in _query(version_dir, "SELECT DISTINCT task_type FROM tasks")}
    assert stored == {"classification", "answer", "scalar_prediction", "temporal_localization"}


def test_int32_guard_is_exposed():
    # The guard is a documented invariant; verify the constant/limit exists.
    from timenet.values_backends.parquet.writer import MAX_ELEMENTS_PER_ROW_GROUP  # noqa: PLC0415

    assert MAX_ELEMENTS_PER_ROW_GROUP == 2**31


def _dup_dataset():
    return TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="demo/dup",
            dataset_version=Version(1, 0, 0),
            name="Dup",
            description="d",
            license=License.MIT,
        )
    )


def test_same_id_different_series_rejected(tmp_path):
    # Two records reaching different series under one time_series_id: only one can be written, so the
    # other record would silently read back the wrong signal. Must fail loudly, not first-wins.
    spec = TimeSeriesSpec(
        spec_type="s",
        name="S",
        unit_value=ureg.dimensionless,
    )
    dataset = _dup_dataset()
    a = TimeSeries(
        spec=spec,
        signal="a",
        time_axis=RegularAxis.from_rate_hz(1),
        n_values=2,
        loader=lambda: pa.array([1.0, 2.0], type=pa.float32()),
        time_series_id="ts-x",
    )
    b = TimeSeries(  # same id, different signal and window
        spec=spec,
        signal="b",
        time_axis=RegularAxis.from_rate_hz(1),
        n_values=2,
        loader=lambda: pa.array([9.0, 9.0], type=pa.float32()),
        time_series_id="ts-x",
    )
    dataset.add_record(time_series=(a,), record_id="s-a")
    dataset.add_record(time_series=(b,), record_id="s-b")
    dataset.derive_schema()
    with (
        pytest.raises(TimeFValidationError, match="claimed by two different series"),
        TimeFWriter(tmp_path, dataset) as writer,
    ):
        writer.write()


def test_same_series_shared_across_records_still_dedupes(tmp_path):
    # The supported sharing path: the same instance in two records must still collapse to one shard.
    spec = TimeSeriesSpec(
        spec_type="s",
        name="S",
        unit_value=ureg.dimensionless,
    )
    dataset = _dup_dataset()
    shared = TimeSeries(
        spec=spec,
        signal="a",
        time_axis=RegularAxis.from_rate_hz(1),
        n_values=2,
        loader=lambda: pa.array([1.0, 2.0], type=pa.float32()),
        time_series_id="ts-shared",
    )
    dataset.add_record(time_series=(shared,), record_id="s-a")
    dataset.add_record(time_series=(shared,), record_id="s-b")
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    manifest = Manifest.from_json((tmp_path / "demo" / "dup" / "1.0.0" / "manifest.json").read_text())
    # the shared series is written once, though two records reference it
    assert manifest.counts.time_series_chunks == 1
    assert manifest.counts.records == 2
