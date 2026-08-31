"""End-to-end tests for the ``enum`` scalar value dtype.

An enum channel stores its values as a PyArrow dictionary array. Parquet writes the dictionary
natively in each shard, so the codebook is self-contained. Reading back returns a dictionary array
whose ``.dictionary`` holds the labels and ``.indices`` holds the integer codes.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFValidationError
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import DatasetMetadata, Domain, License, TimeSeriesSpec, Version, ureg
from timenet.writer import TimeFWriter


pytestmark = pytest.mark.value_dtypes


def _spec(categories: tuple[str, ...] = ("awake", "light", "deep", "rem")) -> TimeSeriesSpec:
    return TimeSeriesSpec(
        spec_type="sleep_stage",
        name="Sleep stage",
        unit_value=ureg.dimensionless,
        dtype="enum",
        categories=categories,
    )


def _dataset(
    values: list[str],
    *,
    n_samples: int = 1,
) -> TimeFDataset:
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/enum",
            dataset_version=Version(1, 0, 0),
            name="Enum round-trip fixture",
            description="A categorical scalar series.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    spec = _spec()
    per_sample = len(values) // n_samples
    for i in range(n_samples):
        start = i * per_sample
        end = start + per_sample if i < n_samples - 1 else len(values)
        sample_values = values[start:end]
        ts = TimeSeries.from_values(
            sample_values,
            spec=spec,
            channel="stage",
            time_axis=RegularAxis.from_rate_hz(1),
        )
        dataset.add_sample(time_series=(ts,), sample_id=f"sample-{i}")
    dataset.derive_schema()
    return dataset


def _write(tmp_path: Path, dataset: TimeFDataset, **writer_kwargs) -> Path:
    with TimeFWriter(tmp_path, dataset, **writer_kwargs) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def test_enum_spec_construction():
    spec = _spec()
    assert spec.dtype == "enum"
    assert spec.categories == ("awake", "light", "deep", "rem")


def test_enum_spec_rejects_empty_categories():
    with pytest.raises(TimeFValidationError, match="must be non-empty"):
        TimeSeriesSpec(
            spec_type="stage",
            name="s",
            unit_value=ureg.dimensionless,
            dtype="enum",
        )


def test_enum_spec_rejects_duplicates():
    with pytest.raises(TimeFValidationError, match="must be unique"):
        TimeSeriesSpec(
            spec_type="stage",
            name="s",
            unit_value=ureg.dimensionless,
            dtype="enum",
            categories=("a", "b", "a"),
        )


def test_non_enum_rejects_categories():
    with pytest.raises(TimeFValidationError, match="only for dtype 'enum'"):
        TimeSeriesSpec(
            spec_type="hr",
            name="h",
            unit_value=ureg.dimensionless,
            dtype="float32",
            categories=("a",),
        )


def test_enum_rejects_values_outside_codebook():
    spec = _spec()
    with pytest.raises(TimeFValidationError, match="outside its categories"):
        TimeSeries.from_values(
            ["awake", "unknown"],
            spec=spec,
            channel="stage",
            time_axis=RegularAxis.from_rate_hz(1),
        )


def test_enum_round_trips_labels(tmp_path):
    labels = ["awake", "deep", "awake", "rem"]
    version_dir = _write(tmp_path, _dataset(labels))
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        series = reader.read().samples[0].time_series[0]
    result = series.to_arrow()
    assert pa.types.is_dictionary(result.type)
    assert result.cast(pa.string()).to_pylist() == labels


def test_enum_shard_leaf_is_dictionary(tmp_path):
    labels = ["awake", "deep", "awake", "rem"]
    version_dir = _write(tmp_path, _dataset(labels))
    shard = next(version_dir.glob("time_series/part-*.parquet"))
    leaf = pq.ParquetFile(shard).schema_arrow.field("values").type.value_type
    assert pa.types.is_dictionary(leaf)
    assert leaf.value_type == pa.string()


def test_enum_manifest_records_dtype_and_categories(tmp_path):
    version_dir = _write(tmp_path, _dataset(["awake", "deep"]))
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    spec = manifest.schema.time_series_specs[0]
    assert spec.dtype == "enum"
    assert spec.categories == ("awake", "light", "deep", "rem")


def test_enum_read_range_returns_labels(tmp_path):
    labels = ["awake", "light", "deep", "rem", "awake"]
    version_dir = _write(tmp_path, _dataset(labels))
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        series = reader.read().samples[0].time_series[0]
    result = series.read_steps(1, 4)
    assert result.cast(pa.string()).to_pylist() == ["light", "deep", "rem"]


def test_enum_rejected_on_zarr(tmp_path):
    dataset = _dataset(["awake"])
    with (
        pytest.raises(TimeFValidationError, match="does not support the enum dtype"),
        TimeFWriter(tmp_path, dataset, values_backend="zarr") as writer,
    ):
        writer.write()


def test_enum_multi_shard_different_value_subsets(tmp_path):
    """Force multiple shards with tiny shard_target_bytes. Each shard sees a different subset of
    labels, so the per-shard dictionaries differ. Reading back must still produce correct labels.
    """
    labels = ["awake"] * 20 + ["deep"] * 20 + ["rem"] * 20 + ["light"] * 20
    dataset = _dataset(labels, n_samples=4)  # default categories cover all four labels
    version_dir = _write(
        tmp_path,
        dataset,
        shard_target_bytes=64,
        row_group_target_bytes=32,
        chunk_max_bytes=32,
    )
    shards = sorted(version_dir.glob("time_series/part-*.parquet"))
    assert len(shards) > 1, f"expected multiple shards, got {len(shards)}"

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        loaded = reader.read()
    for i, sample in enumerate(loaded.samples):
        ts = sample.time_series[0]
        result = ts.to_arrow()
        assert pa.types.is_dictionary(result.type)
        result_labels = result.cast(pa.string()).to_pylist()
        start = i * 20
        end = start + 20
        assert result_labels == labels[start:end]


def test_enum_multi_row_group_dictionary_consistency(tmp_path):
    """Force multiple row groups within a shard. Shard 1 sees only ["awake", "deep"], shard 2 sees
    only ["rem", "light"]. After reading, all labels must decode correctly.
    """
    labels = ["awake", "deep"] * 10 + ["rem", "light"] * 10
    dataset = _dataset(labels, n_samples=2)
    version_dir = _write(
        tmp_path,
        dataset,
        shard_target_bytes=2**30,
        row_group_target_bytes=32,
        chunk_max_bytes=32,
    )

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        loaded = reader.read()
    all_labels = []
    for sample in loaded.samples:
        ts = sample.time_series[0]
        result = ts.to_arrow()
        assert pa.types.is_dictionary(result.type)
        all_labels.extend(result.cast(pa.string()).to_pylist())
    assert all_labels == labels


def test_enum_streaming_write_read(tmp_path):
    """Write many samples with overlapping but non-identical label sets. Verify that streaming
    write (small chunks forcing buffer flushes) and streaming read produce correct results.
    """
    samples_labels = [
        ["awake", "light", "deep"],
        ["rem", "n1", "n2"],
        ["n3", "awake", "rem"],
        ["deep", "light", "n1", "n2", "n3"],
        ["awake", "awake", "awake"],
    ]
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/enum-stream",
            dataset_version=Version(1, 0, 0),
            name="Streaming enum fixture",
            description="Tests streaming write/read with varied label subsets.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    spec = _spec(categories=("awake", "light", "deep", "rem", "n1", "n2", "n3"))
    for i, labels in enumerate(samples_labels):
        ts = TimeSeries.from_values(labels, spec=spec, channel="stage", time_axis=RegularAxis.from_rate_hz(1))
        dataset.add_sample(time_series=(ts,), sample_id=f"s-{i}")
    dataset.derive_schema()

    version_dir = _write(
        tmp_path,
        dataset,
        shard_target_bytes=64,
        row_group_target_bytes=32,
        chunk_max_bytes=16,
    )

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        loaded = reader.read()
    for i, sample in enumerate(loaded.samples):
        ts = sample.time_series[0]
        result = ts.to_arrow()
        assert pa.types.is_dictionary(result.type)
        assert result.cast(pa.string()).to_pylist() == samples_labels[i]


def test_torch_maps_enum_to_integer_codes():
    torch = pytest.importorskip("torch")
    from timenet.torch import TimeFTorchDataset  # noqa: PLC0415

    dataset = _dataset(["awake", "deep", "rem", "awake"])
    item = TimeFTorchDataset(dataset)[0]
    codes = item["series"][0]
    assert codes.dtype == torch.int64
    assert len(codes) == 4
    assert codes.tolist() == [0, 2, 3, 0]


def test_enum_indices_consistent_across_shards_with_different_subsets(tmp_path):
    """Shard 1 encodes only ["a", "b", "c"], shard 2 encodes only ["c", "d", "e"]. After reading,
    the torch bridge produces stable integer codes from the codebook order.
    """
    pytest.importorskip("torch")
    from timenet.torch import _series_tensor  # noqa: PLC0415

    spec = _spec(categories=("a", "b", "c", "d", "e"))
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/enum-cross",
            dataset_version=Version(1, 0, 0),
            name="Cross-shard enum",
            description="Tests codebook-based indices across shards.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    ts1 = TimeSeries.from_values(["a", "b", "c"], spec=spec, channel="stage", time_axis=RegularAxis.from_rate_hz(1))
    ts2 = TimeSeries.from_values(["c", "d", "e"], spec=spec, channel="stage", time_axis=RegularAxis.from_rate_hz(1))
    dataset.add_sample(time_series=(ts1,), sample_id="s-0")
    dataset.add_sample(time_series=(ts2,), sample_id="s-1")
    dataset.derive_schema()

    version_dir = _write(tmp_path, dataset, shard_target_bytes=64, chunk_max_bytes=32)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        loaded = reader.read()

    s0 = loaded.samples[0].time_series[0]
    s1 = loaded.samples[1].time_series[0]

    arr0 = s0.to_arrow()
    arr1 = s1.to_arrow()
    assert arr0.cast(pa.string()).to_pylist() == ["a", "b", "c"]
    assert arr1.cast(pa.string()).to_pylist() == ["c", "d", "e"]

    codes0 = _series_tensor(s0)
    codes1 = _series_tensor(s1)

    idx_c_in_s0 = codes0[2].item()
    idx_c_in_s1 = codes1[0].item()
    assert idx_c_in_s0 == idx_c_in_s1, f"'c' has index {idx_c_in_s0} in sample 0 but {idx_c_in_s1} in sample 1"
