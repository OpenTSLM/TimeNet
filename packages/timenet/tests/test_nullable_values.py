from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import DatasetMetadata, License, TimeSeriesSpec, Version, ureg
from timenet.writer import TimeFWriter


pytest.importorskip("torch")

from timenet.torch import TimeFTorchDataset


def _spec(dtype="float32", **kwargs):
    return TimeSeriesSpec(spec_type="x", name="X", unit_value=ureg.dimensionless, dtype=dtype, **kwargs)


@pytest.mark.parametrize("nullable", [1, 0, None, "true", np.bool_(True)])
def test_nullable_requires_a_real_bool(nullable):
    with pytest.raises(TimeFValidationError, match="nullable"):
        _spec(nullable=nullable)


@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize(
    ("dtype", "observations", "categories"),
    [
        ("float32", [1.0, None, 3.0], ()),
        ("int16", [1, None, 3], ()),
        ("bool", [True, None, False], ()),
        ("str", ["a", None, "b"], ()),
        ("enum", ["a", None, "b"], ("a", "b")),
    ],
)
def test_scalar_constructors_preserve_nulls_and_declared_dtype(irregular, dtype, observations, categories):
    spec = _spec(dtype, categories=categories, nullable=True)
    if irregular:
        series = TimeSeries.from_irregular(observations, time_offsets_us=[0, 2, 4], spec=spec, signal="x")
    else:
        series = TimeSeries.from_values(observations, spec=spec, signal="x", time_axis=OrdinalAxis())
    assert series.to_arrow().to_pylist() == observations
    values, valid = series.to_numpy_and_mask()
    assert valid.dtype == np.bool_
    assert valid.tolist() == [True, False, True]
    assert values.shape == (3,)
    assert values[valid].tolist() == [observations[0], observations[2]]
    if dtype not in {"str", "enum"}:
        assert values.dtype == np.dtype(dtype)


@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "int16", "bool", "str", "enum"])
def test_nonnullable_constructors_reject_none(irregular, dtype):
    spec = _spec(dtype, categories=("a",) if dtype == "enum" else ())
    with pytest.raises(TimeFValidationError, match="null"):
        if irregular:
            TimeSeries.from_irregular([None], time_offsets_us=[0], spec=spec, signal="x")
        else:
            TimeSeries.from_values([None], spec=spec, signal="x", time_axis=OrdinalAxis())


def _lazy_series(spec, array):
    return TimeSeries(spec=spec, signal="x", time_axis=OrdinalAxis(), loader=lambda: array, n_values=len(array))


@pytest.mark.parametrize("dtype", ["float32", "int16", "bool", "str", "enum"])
def test_writer_checks_nullability_before_numpy_conversion(tmp_path, dtype):
    spec = _spec(dtype, categories=("a",) if dtype == "enum" else (), nullable=True)
    arrow_type = pa.string() if dtype in {"str", "enum"} else pa.from_numpy_dtype(np.dtype(dtype))
    array = pa.array([None], type=arrow_type)
    if dtype == "enum":
        array = array.dictionary_encode()
    writer = TimeFWriter(tmp_path, make_dataset())
    assert writer._read_and_validate(_lazy_series(spec, array)).equals(array)
    with pytest.raises(TimeFValidationError, match="null"):
        writer._read_and_validate(_lazy_series(replace(spec, nullable=False), array))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_float_accepts_nonfinite_observations(tmp_path, value):
    # An IEEE NaN or infinity is a measured payload, not an absent measurement. Nullability governs
    # Arrow nulls only, so a channel may carry a null and a NaN side by side and they stay distinct.
    series = TimeSeries.from_values([None, value], spec=_spec(nullable=True), signal="x", time_axis=OrdinalAxis())
    validated = TimeFWriter(tmp_path, make_dataset())._read_and_validate(series)

    assert validated.null_count == 1
    observed = validated[1].as_py()
    assert np.isnan(observed) or np.isinf(observed)


def test_nonfinite_values_are_allowed_when_not_nullable(tmp_path):
    # The finiteness rejection is gone entirely, not merely relaxed for nullable specs.
    series = TimeSeries.from_values([np.nan], spec=_spec(), signal="x", time_axis=OrdinalAxis())
    TimeFWriter(tmp_path, make_dataset())._read_and_validate(series)


def test_nullable_enum_still_rejects_unknown_observations(tmp_path):
    spec = _spec("enum", categories=("a",), nullable=True)
    array = pa.array([None, "unknown"]).dictionary_encode()
    with pytest.raises(TimeFValidationError, match="categories"):
        TimeFWriter(tmp_path, make_dataset())._read_and_validate(_lazy_series(spec, array))


@pytest.mark.parametrize(
    ("dtype", "observations"),
    [("float32", [[1.0, 2.0], None, [3.0, 4.0]]), ("bool", [[True, False], None, [False, True]])],
)
def test_tensor_mask_marks_whole_timesteps_and_preserves_shape(tmp_path, dtype, observations):
    spec = _spec(dtype, value_shape=(2,), nullable=True)
    value_type = pa.from_numpy_dtype(np.dtype(dtype))
    storage = pa.array(observations, type=pa.list_(value_type, 2))
    array = pa.ExtensionArray.from_storage(pa.fixed_shape_tensor(value_type, (2,)), storage)
    series = _lazy_series(spec, array)
    values, valid = series.to_numpy_and_mask()
    assert values.shape == (3, 2)
    assert values.dtype == np.dtype(dtype)
    assert valid.tolist() == [True, False, True]
    assert values[valid].tolist() == [observations[0], observations[2]]
    assert TimeFWriter(tmp_path, make_dataset())._read_and_validate(series).equals(array)


def test_writer_accepts_all_null_tensor(tmp_path):
    spec = _spec(value_shape=(2,), nullable=True)
    storage = pa.array([None], type=pa.list_(pa.float32(), 2))
    array = pa.ExtensionArray.from_storage(pa.fixed_shape_tensor(pa.float32(), (2,)), storage)
    series = _lazy_series(spec, array)
    assert TimeFWriter(tmp_path, make_dataset())._read_and_validate(series).equals(array)


@pytest.mark.parametrize("dtype", ["float32", "int16"])
def test_writer_rejects_partial_tensor_nulls(tmp_path, dtype):
    spec = _spec(dtype, value_shape=(2,), nullable=True)
    value_type = pa.from_numpy_dtype(np.dtype(dtype))
    storage = pa.array([[1, None]], type=pa.list_(value_type, 2))
    array = pa.ExtensionArray.from_storage(pa.fixed_shape_tensor(value_type, (2,)), storage)
    with pytest.raises(TimeFValidationError, match="whole timesteps"):
        TimeFWriter(tmp_path, make_dataset())._read_and_validate(_lazy_series(spec, array))


# ---- round trip ------------------------------------------------------------------------------


def _write_read(tmp_path, series, *, values_backend):
    """Write one series through a backend and read it back out of the committed version."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/nullable",
            dataset_version=Version(1, 0, 0),
            name="Nullable",
            description="A series with missing timesteps.",
            license=License.CC_BY_4_0,
        )
    )
    dataset.add_record(time_series=(series,), record_id="record-0")
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset, values_backend=values_backend) as writer:
        writer.write()
    version_dir = tmp_path / "timenet/nullable/1.0.0"
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        return reader.read().records[0].time_series[0]


@pytest.mark.parametrize("values_backend", ["parquet", "zarr"])
@pytest.mark.parametrize(
    ("dtype", "observations", "categories"),
    [
        ("float32", [1.0, None, 3.0], ()),
        ("int16", [1, None, 3], ()),
        ("bool", [True, None, False], ()),
        ("str", ["a", None, "b"], ()),
        ("enum", ["a", None, "b"], ("a", "b")),
    ],
)
def test_nullable_scalar_round_trips(tmp_path, values_backend, dtype, observations, categories):
    spec = _spec(dtype, categories=categories, nullable=True)
    series = TimeSeries.from_values(observations, spec=spec, signal="x", time_axis=OrdinalAxis())
    restored = _write_read(tmp_path, series, values_backend=values_backend)

    values = restored.to_arrow()
    assert values.null_count == 1, "the missing timestep survived as a null"
    assert values.is_valid().to_pylist() == [True, False, True]
    assert values[0].as_py() == observations[0]
    assert values[2].as_py() == observations[2]


def test_nullable_tensor_round_trips_whole_timesteps(tmp_path):
    # Only Zarr carries N-D values. Nullability applies to the whole timestep, so one absent frame
    # is one null row, not three null components.
    frames = np.arange(12, dtype=np.float32).reshape(4, 3)
    storage = pa.FixedSizeListArray.from_arrays(
        pa.array(frames.ravel(), type=pa.float32()), 3, mask=pa.array([False, True, False, False])
    )
    tensor = pa.FixedShapeTensorArray.from_storage(pa.fixed_shape_tensor(pa.float32(), [3]), storage)
    series = TimeSeries(
        spec=_spec(value_shape=(3,), nullable=True),
        signal="camera",
        time_axis=OrdinalAxis(),
        loader=lambda: tensor,
        n_values=4,
    )
    restored = _write_read(tmp_path, series, values_backend="zarr")

    back = restored.to_arrow()
    assert isinstance(back, pa.FixedShapeTensorArray)
    assert back.null_count == 1
    assert back.is_valid().to_pylist() == [True, False, True, True]
    np.testing.assert_array_equal(back.to_numpy_ndarray()[0], frames[0])


def test_nonnullable_zarr_writes_no_validity_array(tmp_path):
    # A dataset that never opts in must stay byte-for-byte the layout it has today.
    series = TimeSeries.from_values([1.0, 2.0, 3.0], spec=_spec(), signal="x", time_axis=OrdinalAxis())
    _write_read(tmp_path, series, values_backend="zarr")

    store = tmp_path / "timenet/nullable/1.0.0" / "time_series.zarr"
    assert store.is_dir()
    assert not [path.as_posix() for path in store.rglob("*") if "_validity" in path.as_posix()]


def test_torch_exposes_values_and_mask():
    series = TimeSeries.from_values([1.0, None, 3.0], spec=_spec(nullable=True), signal="x", time_axis=OrdinalAxis())
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/nullable",
            dataset_version=Version(1, 0, 0),
            name="Nullable",
            description="A series with missing timesteps.",
            license=License.CC_BY_4_0,
        )
    )
    dataset.add_record(time_series=(series,), record_id="record-0")
    dataset.derive_schema()
    item = TimeFTorchDataset(dataset)[0]

    values, mask = item["series"][0], item["series_masks"][0]
    assert values.tolist() == [1.0, 0.0, 3.0]  # the null slot holds a placeholder, not an observation
    assert mask.tolist() == [True, False, True]


def test_torch_omits_the_mask_for_a_series_that_cannot_be_null():
    series = TimeSeries.from_values([1.0, 2.0], spec=_spec(), signal="x", time_axis=OrdinalAxis())
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/nullable",
            dataset_version=Version(1, 0, 0),
            name="Nullable",
            description="A series with no missing timesteps.",
            license=License.CC_BY_4_0,
        )
    )
    dataset.add_record(time_series=(series,), record_id="record-0")
    dataset.derive_schema()
    item = TimeFTorchDataset(dataset)[0]

    assert item["series_masks"] == (None,)
