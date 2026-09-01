"""One sample may carry channels of different dtypes, each its own spec_type.

Each dtype must be its own Parquet shard (one shard per spec_type, since a shard's ``values``
column is a single Arrow type). Reading a sample back returns every channel with its correct dtype
and values.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import DatasetMetadata, Domain, License, TimeSeriesSpec, Version, ureg
from timenet.writer import TimeFWriter


pytestmark = pytest.mark.value_dtypes


def _spec(dtype: str) -> TimeSeriesSpec:
    return TimeSeriesSpec(spec_type=f"chan_{dtype}", name=dtype, unit_value=ureg.dimensionless, dtype=dtype)


def _dataset(channels) -> TimeFDataset:
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/dtypes",
            dataset_version=Version(1, 0, 0),
            name="Mixed dtype channels",
            description="A sample with channels of several dtypes.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    series = tuple(
        TimeSeries.from_values(values, spec=spec, channel=name, time_axis=RegularAxis.from_rate_hz(1))
        for name, spec, values in channels
    )
    dataset.add_sample(time_series=series, sample_id="sample-0")
    dataset.derive_schema()
    return dataset


def _write(tmp_path, dataset) -> Path:
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _mixed_channels():
    return [
        ("acc", _spec("float32"), np.array([0.1, 0.2, 0.3], dtype=np.float32)),
        ("steps", _spec("int16"), np.array([1, 2, 3], dtype=np.int16)),
        ("active", _spec("bool"), np.array([True, False, True])),
        ("stage", _spec("str"), ["normal", "afib", "normal"]),
    ]


def test_each_dtype_persists_to_its_own_shard(tmp_path):
    version_dir = _write(tmp_path, _dataset(_mixed_channels()))
    # One shard per spec_type, so four dtypes become four files with four distinct value types.
    shards = sorted(version_dir.glob("time_series/part-*.parquet"))
    assert len(shards) == 4
    leaf_types = {pq.ParquetFile(s).schema_arrow.field("values").type.value_type for s in shards}
    assert leaf_types == {pa.float32(), pa.int16(), pa.bool_(), pa.string()}


def test_mixed_dtype_channels_read_back_typed(tmp_path):
    version_dir = _write(tmp_path, _dataset(_mixed_channels()))
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        sample = reader.read().samples[0]
    by_channel = {ts.channel: ts for ts in sample.time_series}
    assert by_channel["acc"].to_arrow().type == pa.float32()
    assert by_channel["acc"].to_numpy().tolist() == np.array([0.1, 0.2, 0.3], dtype=np.float32).tolist()
    assert by_channel["steps"].to_arrow().type == pa.int16()
    assert by_channel["steps"].to_numpy().tolist() == [1, 2, 3]
    assert by_channel["active"].to_arrow().type == pa.bool_()
    assert by_channel["active"].to_numpy().tolist() == [True, False, True]
    assert by_channel["stage"].to_arrow().type == pa.string()
    assert by_channel["stage"].to_arrow().to_pylist() == ["normal", "afib", "normal"]
