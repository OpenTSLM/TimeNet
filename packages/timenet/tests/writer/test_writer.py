import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFValidationError
from timenet.format.constants import ANNOTATIONS_TEMPLATE, INDEX_TEMPLATE, SAMPLES_TEMPLATE, part_path
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


# ---- layout & manifest ------------------------------------------------------------------------


def test_writes_expected_layout(tmp_path):
    version_dir = _written(tmp_path)
    assert (version_dir / "manifest.json").exists()
    parts = Manifest.from_json((version_dir / "manifest.json").read_text()).files.all_parts()
    # Read the paths from the manifest rather than hard-coding part numbers; check each artifact lands
    # under its own directory and every listed part exists on disk.
    for prefix in ("samples/", "annotations/", "time_series_index/", "time_series/", "tasks/task="):
        assert any(rel.startswith(prefix) for rel in parts), f"expected a part under {prefix!r}"
    for rel in parts:
        assert (version_dir / rel).exists()


def test_control_tables_route_part_names_through_part_path(tmp_path, monkeypatch):
    # samples/annotations/time_series_index must render part names through part_path (which guards the
    # 8-digit ceiling), like tasks and shards, not by formatting the template directly.
    seen: list[str] = []

    def spy(template, index, **fields):
        seen.append(template)
        return part_path(template, index, **fields)

    monkeypatch.setattr("timenet.writer.writer.part_path", spy)
    _written(tmp_path)
    assert {SAMPLES_TEMPLATE, ANNOTATIONS_TEMPLATE, INDEX_TEMPLATE} <= set(seen)


def test_manifest_is_valid_and_matches_dataset(tmp_path):
    version_dir = _written(tmp_path)
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    assert manifest.dataset_id == "timenet/hello-world"
    assert manifest.timef_format_version == 1
    assert manifest.counts.samples == 3
    assert len(manifest.schema.time_series_specs) == 2
    # files listed in the manifest all exist
    for rel in (*manifest.files.samples, *manifest.files.annotations, *manifest.files.time_series_index):
        assert (version_dir / rel).exists()
    for rel in (*manifest.files.tasks, *manifest.files.time_series):
        assert (version_dir / rel).exists()


def test_manifest_has_per_file_checksums(tmp_path):
    version_dir = _written(tmp_path)
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    assert manifest.checksums, "expected per-file checksums"
    assert all(v.startswith("sha256:") for v in manifest.checksums.values())
    assert manifest.files.samples[0] in manifest.checksums


def test_manifest_data_files_are_lists_of_parts(tmp_path):
    version_dir = _written(tmp_path)
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    for parts in (manifest.files.samples, manifest.files.annotations, manifest.files.time_series_index):
        assert isinstance(parts, tuple)
        assert len(parts) >= 1
    raw_files = json.loads((version_dir / "manifest.json").read_text())["files"]
    for key in ("samples", "annotations", "time_series_index", "tasks", "time_series"):
        assert isinstance(raw_files[key], list), f"{key} should serialize as a JSON array"


# ---- shard schema & encodings -----------------------------------------------------------------


def test_shard_has_time_series_id_column(tmp_path):
    version_dir = _written(tmp_path)
    shard = next(version_dir.glob("time_series/shard-*.parquet"))
    names = set(pq.ParquetFile(shard).schema_arrow.names)
    assert "time_series_id" in names  # self-describing shards (re-indexable)
    assert "values" in names


def test_values_carry_the_selected_encoding(tmp_path):
    version_dir = _written(tmp_path)
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    assert manifest.value_encoding, "the manifest should record what each modality was encoded with"
    for shard in version_dir.glob("time_series/shard-*.parquet"):
        pf = pq.ParquetFile(shard)
        spec_types = set(pf.read(columns=["spec_type"]).column("spec_type").to_pylist())
        assert len(spec_types) == 1, "shards are single-modality so one encoding always fits"
        selected = ValueEncoding(manifest.value_encoding[spec_types.pop()])
        assert applied_matches(selected, values_encoding_of(str(shard)))


# ---- chunk splitting --------------------------------------------------------------------------


def test_long_series_splits_into_multiple_chunks(tmp_path):
    # Tiny chunk cap forces the long hello_world series to split.
    version_dir = _written(tmp_path, chunk_max_bytes=64, row_group_target_bytes=64)
    index = pq.read_table(version_dir / "time_series_index/part-00000000.parquet").to_pylist()
    chunks_per_series: dict[str, set] = {}
    for row in index:
        chunks_per_series.setdefault(row["time_series_id"], set()).add(row["chunk_idx"])
    assert any(len(chunks) > 1 for chunks in chunks_per_series.values())


def test_index_offsets_resolve_to_values(tmp_path):
    version_dir = _written(tmp_path, chunk_max_bytes=64, row_group_target_bytes=64)
    index = pq.read_table(version_dir / "time_series_index/part-00000000.parquet").to_pylist()
    row = index[0]
    shard = pq.ParquetFile(version_dir / row["chunk_file"])
    table = shard.read_row_group(row["chunk_major_idx"])
    chunk = table.column("values")[row["chunk_minor_idx"]]
    assert len(chunk) == row["n_values"]


def test_shard_rotation_leaves_no_empty_trailing_shard(tmp_path):
    # Tiny shard cap forces a rotation on essentially every row group, including the last one.
    version_dir = _written(tmp_path, chunk_max_bytes=64, row_group_target_bytes=64, shard_target_bytes=64)
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    assert manifest.files.time_series, "expected at least one shard"
    for rel in manifest.files.time_series:
        assert pq.ParquetFile(version_dir / rel).metadata.num_rows > 0, f"empty shard {rel} was published"


# ---- validation & commit protocol -------------------------------------------------------------


def test_write_without_schema_raises(tmp_path):
    dataset = make_dataset()  # schema not derived
    with pytest.raises(TimeFValidationError), TimeFWriter(tmp_path, dataset) as writer:
        writer.write()


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
    sample = dataset.add_sample(
        time_series=(
            TimeSeries(spec=spec, channel="c", time_axis=RegularAxis.from_rate_hz(1), n_values=1, loader=bad_loader),
        ),
    )
    sample.add_annotation(Annotation(key="k", value=1))
    dataset.add_task(sample, ClassificationTask(target="x"))
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
        channel="c",
        time_axis=RegularAxis.from_rate_hz(1),
        loader=lambda: pa.array([1.0, 2.0, 3.0], type=pa.float32()),
        n_values=5,
    )
    dataset.add_sample(time_series=(ts,))
    dataset.derive_schema()
    with pytest.raises(TimeFValidationError), TimeFWriter(tmp_path, dataset) as writer:
        writer.write()


def test_samples_parquet_content(tmp_path):
    version_dir = _written(tmp_path)
    rows = {r["sample_id"]: r for r in pq.read_table(version_dir / "samples/part-00000000.parquet").to_pylist()}
    assert set(rows) == {"sample-0", "sample-1", "sample-2"}
    # shared series appears in both sample-0 and sample-1
    ids0 = {ts["time_series_id"] for ts in rows["sample-0"]["time_series"]}
    ids1 = {ts["time_series_id"] for ts in rows["sample-1"]["time_series"]}
    assert "ts-shared" in ids0 & ids1


def test_shared_series_stored_once(tmp_path):
    version_dir = _written(tmp_path)
    # the shared series has one chunk row in the shards, but two index rows (one per sample)
    shard_rows = [
        r
        for shard in version_dir.glob("time_series/shard-*.parquet")
        for r in pq.read_table(shard).to_pylist()
        if r["time_series_id"] == "ts-shared"
    ]
    index_rows = [
        r
        for r in pq.read_table(version_dir / "time_series_index/part-00000000.parquet").to_pylist()
        if r["time_series_id"] == "ts-shared"
    ]
    assert len(shard_rows) == 1
    assert len(index_rows) == 2


def test_tasks_partitioned_by_type(tmp_path):
    version_dir = _written(tmp_path)
    parts = {p.parent.name for p in version_dir.glob("tasks/task=*/part-*.parquet")}
    assert parts == {"task=classification", "task=answer", "task=scalar_prediction", "task=temporal_localization"}


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
    # Two samples reaching different series under one time_series_id: only one can be written, so the
    # other sample would silently read back the wrong channel. Must fail loudly, not first-wins.
    spec = TimeSeriesSpec(
        spec_type="s",
        name="S",
        unit_value=ureg.dimensionless,
    )
    dataset = _dup_dataset()
    a = TimeSeries(
        spec=spec,
        channel="a",
        time_axis=RegularAxis.from_rate_hz(1),
        n_values=2,
        loader=lambda: pa.array([1.0, 2.0], type=pa.float32()),
        time_series_id="ts-x",
    )
    b = TimeSeries(  # same id, different channel and window
        spec=spec,
        channel="b",
        time_axis=RegularAxis.from_rate_hz(1),
        n_values=2,
        loader=lambda: pa.array([9.0, 9.0], type=pa.float32()),
        time_series_id="ts-x",
    )
    dataset.add_sample(time_series=(a,), sample_id="s-a")
    dataset.add_sample(time_series=(b,), sample_id="s-b")
    dataset.derive_schema()
    with (
        pytest.raises(TimeFValidationError, match="claimed by two different series"),
        TimeFWriter(tmp_path, dataset) as writer,
    ):
        writer.write()


def test_same_series_shared_across_samples_still_dedupes(tmp_path):
    # The supported sharing path: the same instance in two samples must still collapse to one shard.
    spec = TimeSeriesSpec(
        spec_type="s",
        name="S",
        unit_value=ureg.dimensionless,
    )
    dataset = _dup_dataset()
    shared = TimeSeries(
        spec=spec,
        channel="a",
        time_axis=RegularAxis.from_rate_hz(1),
        n_values=2,
        loader=lambda: pa.array([1.0, 2.0], type=pa.float32()),
        time_series_id="ts-shared",
    )
    dataset.add_sample(time_series=(shared,), sample_id="s-a")
    dataset.add_sample(time_series=(shared,), sample_id="s-b")
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    manifest = Manifest.from_json((tmp_path / "demo" / "dup" / "1.0.0" / "manifest.json").read_text())
    # the shared series is written once, though two samples reference it
    assert manifest.counts.time_series_chunks == 1
    assert manifest.counts.samples == 2
