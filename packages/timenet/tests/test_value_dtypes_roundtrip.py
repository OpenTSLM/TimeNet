"""End-to-end round-trip of every scalar value dtype through the default Parquet values plane.

The Parquet backend once rejected any non-float32 spec. This file proves each supported dtype (and
the ``str`` dtype) writes, reads, and survives a copy-on-write edit with its Arrow type and values
intact.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.dataset.edit import edit_version
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import DatasetMetadata, Domain, License, TimeSeriesSpec, Version, ureg
from timenet.writer import TimeFWriter


pytestmark = pytest.mark.value_dtypes


SCALAR_DTYPES = [
    "bool",
    "int8",
    "int16",
    "int32",
    "uint8",
    "uint16",
    "uint32",
    "float32",
    "float64",
]


def _values(dtype: str) -> np.ndarray:
    if dtype == "bool":
        return np.array([True, False, True, True])
    if "float" in dtype:
        return np.array([0.1, -0.5, 2.0, 9.99], dtype=np.dtype(dtype))
    return np.array([1, 2, 3, 4], dtype=np.dtype(dtype))


def _spec(dtype: str = "float32") -> TimeSeriesSpec:
    return TimeSeriesSpec(
        spec_type=f"chan_{dtype}",
        name=dtype,
        unit_value=ureg.dimensionless,
        dtype=dtype,
    )


def _dataset(spec: TimeSeriesSpec, values, channel: str = "c", sample_id: str = "sample-0") -> TimeFDataset:
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/dtypes",
            dataset_version=Version(1, 0, 0),
            name="Dtype round-trip fixture",
            description="A scalar series of one dtype.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    ts = TimeSeries.from_values(
        values,
        spec=spec,
        channel=channel,
        time_axis=RegularAxis.from_rate_hz(1),
        time_series_id=f"ts-{channel}",
    )
    dataset.add_sample(time_series=(ts,), sample_id=sample_id)
    dataset.derive_schema()
    return dataset


def _write(tmp_path, dataset, **kwargs) -> Path:
    with TimeFWriter(tmp_path, dataset, **kwargs) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _first_series(version_dir):
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        return reader.read().samples[0].time_series[0]


@pytest.mark.parametrize("dtype", SCALAR_DTYPES)
def test_scalar_dtype_round_trips(tmp_path, dtype):
    values = _values(dtype)
    version_dir = _write(tmp_path, _dataset(_spec(dtype), values))
    restored = _first_series(version_dir)
    expected = pa.from_numpy_dtype(np.dtype(dtype))
    assert restored.to_arrow().type == expected
    assert restored.to_numpy().tolist() == values.tolist()


@pytest.mark.parametrize("dtype", SCALAR_DTYPES)
def test_scalar_dtype_shard_leaf(tmp_path, dtype):
    version_dir = _write(tmp_path, _dataset(_spec(dtype), _values(dtype)))
    shard = next(version_dir.glob("time_series/part-*.parquet"))
    leaf = pq.ParquetFile(shard).schema_arrow.field("values").type.value_type
    assert leaf == pa.from_numpy_dtype(np.dtype(dtype))


def test_str_round_trips_labels(tmp_path):
    labels = ["normal", "afib", "normal", "vt"]
    version_dir = _write(tmp_path, _dataset(_spec(dtype="str"), labels))
    restored = _first_series(version_dir)
    assert restored.to_arrow().type == pa.string()
    assert restored.to_arrow().to_pylist() == labels


def test_str_shard_leaf_is_string(tmp_path):
    version_dir = _write(tmp_path, _dataset(_spec(dtype="str"), ["normal", "afib"]))
    shard = next(version_dir.glob("time_series/part-*.parquet"))
    leaf = pq.ParquetFile(shard).schema_arrow.field("values").type.value_type
    assert leaf == pa.string()


def test_str_dtype_round_trips_manifest(tmp_path):
    version_dir = _write(tmp_path, _dataset(_spec(dtype="str"), ["normal", "afib"]))
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    assert manifest.schema.time_series_specs[0].dtype == "str"


def test_copy_on_write_edit_keeps_bool_and_str(tmp_path):
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/dtypes",
            dataset_version=Version(1, 0, 0),
            name="Two samples",
            description="Two samples so a copy-on-write edit can drop one.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    bool_spec = _spec(dtype="bool")
    bool0 = TimeSeries.from_values(
        [True, False, True], spec=bool_spec, channel="active", time_axis=RegularAxis.from_rate_hz(1)
    )
    str0 = TimeSeries.from_values(
        ["normal", "afib"],
        spec=_spec(dtype="str"),
        channel="stage",
        time_axis=RegularAxis.from_rate_hz(1),
    )
    bool1 = TimeSeries.from_values(
        [False, False, True], spec=bool_spec, channel="active", time_axis=RegularAxis.from_rate_hz(1)
    )
    str1 = TimeSeries.from_values(
        ["vt", "normal"],
        spec=_spec(dtype="str"),
        channel="stage",
        time_axis=RegularAxis.from_rate_hz(1),
    )
    dataset.add_sample(time_series=(bool0, str0), sample_id="sample-0")
    dataset.add_sample(time_series=(bool1, str1), sample_id="sample-1")
    dataset.derive_schema()
    version_dir = _write(tmp_path / "base", dataset)
    edited = edit_version(
        version_dir, tmp_path / "out", dataset_version=Version(1, 1, 0), remove_sample_ids=["sample-1"]
    )
    with TimeFReader(DatasetVersion.open_local(edited)) as reader:
        sample = reader.read().samples[0]
    series = {ts.channel: ts for ts in sample.time_series}
    assert series["active"].to_arrow().type == pa.bool_()
    assert series["active"].to_numpy().tolist() == [True, False, True]
    assert series["stage"].to_arrow().type == pa.string()
    assert series["stage"].to_arrow().to_pylist() == ["normal", "afib"]


_BACKEND_IDS = ["parquet", "zarr"]


def test_log_book_strings_round_trip_byte_identical(tmp_path):
    # A log book mixes tiny notes and very long entries. Strings are variable-width, so storage must
    # preserve every byte of every entry, no matter how long one is. Strings live on Parquet (the
    # dictionary backend); Zarr rejects the str dtype.
    entries = [
        "hi",  # two characters
        "Alarm cleared by operator 7 at 09:14.",  # a short note
        " ".join(["the"] * 40),  # a sentence-length entry (139 chars)
        " ".join(["word"] * 300),  # a ~1500-char paragraph
        "e" * 5000,  # a 5000-char log line
        "überstraße €\nnext line",  # unicode + newline
        "",  # empty entry
    ]
    original = pa.array(entries, type=pa.string())

    version_dir = _write(tmp_path, _dataset(_spec(dtype="str"), entries), values_backend="parquet")
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        series = reader.read().samples[0].time_series[0]

    # Byte-identical: the whole series and a ranged read must equal the original utf-8 bytes.
    assert series.to_arrow().equals(original)
    assert series.to_arrow().to_pylist() == entries
    assert series.read_steps(1, 4).equals(original.slice(1, 3))


@pytest.mark.parametrize("values_backend", _BACKEND_IDS)
def test_mixed_dtypes_byte_identical_across_backends(tmp_path, values_backend):
    # One sample with float32, int16, and bool channels must read back byte-identical on both
    # backends, so a dataset's logical content does not depend on the storage engine. (str is
    # excluded here because Zarr does not support it; str channels are covered on Parquet above.)
    channels = [
        ("acc", _spec("float32"), np.array([0.1, 0.2, 0.3], dtype=np.float32)),
        ("steps", _spec("int16"), np.array([1, 2, 3], dtype=np.int16)),
        ("active", _spec("bool"), np.array([True, False, True])),
    ]
    series = tuple(
        TimeSeries.from_values(values, spec=spec, channel=name, time_axis=RegularAxis.from_rate_hz(1))
        for name, spec, values in channels
    )
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/mixed",
            dataset_version=Version(1, 0, 0),
            name="Mixed",
            description="Mixed-dtype channels on any backend.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    dataset.add_sample(time_series=series, sample_id="sample-0")
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset, values_backend=values_backend)

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        sample = reader.read().samples[0]
    by_channel = {ts.channel: ts for ts in sample.time_series}
    for name, spec, values in channels:
        expected = pa.array(np.asarray(values), type=pa.from_numpy_dtype(np.dtype(spec.dtype)))
        assert by_channel[name].to_arrow().equals(expected), name
