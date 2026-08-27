"""End-to-end tests for the ``enum`` scalar value dtype.

An enum channel stores a category codebook in the spec's ``categories`` and its values as label
strings on the Parquet dictionary leaf. Reading back returns the labels; the torch bridge maps
labels to integer codes.
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


_SLEEP_CATEGORIES = ("awake", "light", "deep", "rem")


def _spec(dtype: str, categories: tuple[str, ...] = ()) -> TimeSeriesSpec:
    return TimeSeriesSpec(
        spec_type=f"chan_{dtype}",
        name=dtype,
        unit_value=ureg.dimensionless,
        dtype=dtype,
        categories=categories,
    )


def _dataset(spec: TimeSeriesSpec, values, sample_id: str = "sample-0") -> TimeFDataset:
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
    ts = TimeSeries.from_values(values, spec=spec, channel="stage", time_axis=RegularAxis.from_rate_hz(1))
    dataset.add_sample(time_series=(ts,), sample_id=sample_id)
    dataset.derive_schema()
    return dataset


def _write(tmp_path, dataset) -> Path:
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def test_enum_dtype_requires_categories():
    with pytest.raises(TimeFValidationError, match="categories"):
        TimeSeriesSpec(spec_type="s", name="S", unit_value=ureg.dimensionless, dtype="enum")


@pytest.mark.parametrize(
    "categories",
    [(), ("", "a"), ("a", "a")],
)
def test_enum_dtype_rejects_unusable_categories(categories):
    with pytest.raises(TimeFValidationError, match="categories"):
        TimeSeriesSpec(spec_type="s", name="S", unit_value=ureg.dimensionless, dtype="enum", categories=categories)


def test_non_enum_dtype_rejects_categories():
    with pytest.raises(TimeFValidationError, match="categories"):
        TimeSeriesSpec(spec_type="s", name="S", unit_value=ureg.dimensionless, dtype="int16", categories=("a", "b"))


def test_enum_round_trips_labels(tmp_path):
    labels = ["awake", "deep", "awake", "rem"]
    version_dir = _write(tmp_path, _dataset(_spec("enum", _SLEEP_CATEGORIES), labels))
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        series = reader.read().samples[0].time_series[0]
    assert series.to_arrow().type == pa.string()
    assert series.to_arrow().to_pylist() == labels


def test_enum_shard_leaf_is_dictionary(tmp_path):
    labels = ["awake", "deep", "awake", "rem"]
    version_dir = _write(tmp_path, _dataset(_spec("enum", _SLEEP_CATEGORIES), labels))
    shard = next(version_dir.glob("time_series/part-*.parquet"))
    leaf = pq.ParquetFile(shard).schema_arrow.field("values").type.value_type
    assert pa.types.is_dictionary(leaf)
    assert leaf.value_type == pa.string()


def test_enum_manifest_records_categories(tmp_path):
    labels = ["awake", "deep"]
    version_dir = _write(tmp_path, _dataset(_spec("enum", _SLEEP_CATEGORIES), labels))
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    spec = manifest.schema.time_series_specs[0]
    assert spec.dtype == "enum"
    assert spec.categories == _SLEEP_CATEGORIES


def test_enum_rejects_unknown_label_at_write_time(tmp_path):
    labels = ["awake", "hacker"]
    with pytest.raises(TimeFValidationError, match="outside its categories"):
        _write(tmp_path, _dataset(_spec("enum", _SLEEP_CATEGORIES), labels))


def test_enum_from_values_rejects_unknown_label():
    spec = _spec("enum", _SLEEP_CATEGORIES)
    with pytest.raises(TimeFValidationError, match="outside its categories"):
        TimeSeries.from_values(["awake", "nope"], spec=spec, channel="stage", time_axis=RegularAxis.from_rate_hz(1))


def test_enum_read_range_returns_labels(tmp_path):
    labels = ["awake", "light", "deep", "rem", "awake"]
    version_dir = _write(tmp_path, _dataset(_spec("enum", _SLEEP_CATEGORIES), labels))
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        series = reader.read().samples[0].time_series[0]
    assert series.read_steps(1, 4).to_pylist() == ["light", "deep", "rem"]


def test_enum_rejected_on_zarr(tmp_path):
    dataset = _dataset(_spec("enum", _SLEEP_CATEGORIES), ["awake"])
    with (
        pytest.raises(TimeFValidationError, match="does not support the str or enum dtype"),
        TimeFWriter(tmp_path, dataset, values_backend="zarr") as writer,
    ):
        writer.write()


def test_torch_maps_enum_to_integer_codes():
    torch = pytest.importorskip("torch")
    from timenet.torch import TimeFTorchDataset  # noqa: PLC0415

    dataset = _dataset(_spec("enum", _SLEEP_CATEGORIES), ["awake", "deep", "rem", "awake"])
    item = TimeFTorchDataset(dataset)[0]
    codes = item["series"][0]
    assert codes.dtype == torch.int64
    assert codes.tolist() == [0, 2, 3, 0]
