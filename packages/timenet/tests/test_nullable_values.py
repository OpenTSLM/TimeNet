from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import IrregularAxis, OrdinalAxis, RegularAxis
from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import DatasetMetadata, License, TimeSeriesSpec, Version, ureg
from timenet.values_backends.zarr.reader import ZarrValuesReader
from timenet.writer import TimeFWriter


pytest.importorskip("torch")

from torch.utils.data import DataLoader

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
    # NaN and infinity are present values, not missing measurements.
    # The nullable flag controls Arrow nulls only.
    series = TimeSeries.from_values([None, value], spec=_spec(nullable=True), signal="x", time_axis=OrdinalAxis())
    validated = TimeFWriter(tmp_path, make_dataset())._read_and_validate(series)

    assert validated.null_count == 1
    observed = validated[1].as_py()
    assert np.isnan(observed) or np.isinf(observed)


def test_nonfinite_values_are_allowed_when_not_nullable(tmp_path):
    # Floating-point specs allow NaN and infinity, including non-nullable specs.
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


def _write_read(tmp_path, series, *, values_backend, leading_series=None, **writer_kwargs):
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
    time_series = (series,) if leading_series is None else (leading_series, series)
    dataset.add_record(time_series=time_series, record_id="record-0")
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset, values_backend=values_backend, **writer_kwargs) as writer:
        writer.write()
    version_dir = tmp_path / "timenet/nullable/1.0.0"
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        return reader.read().records[0].time_series[-1]


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
    # Only Zarr stores multidimensional values. A missing frame is one null row.
    # Individual components cannot be null.
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
    # Non-nullable datasets must not write validity arrays.
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


def test_torch_returns_an_all_true_mask_for_a_series_that_cannot_be_null():
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

    assert item["series_masks"][0].tolist() == [True, True]


@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize("tensor", [False, True])
def test_zarr_nullable_ranges_preserve_nulls_and_bound_validity_reads(tmp_path, monkeypatch, irregular, tensor):
    observations = [0.0, None, 2.0, None, 4.0, 5.0, None, 7.0]
    spec = _spec(nullable=True, value_shape=(2,) if tensor else ())
    if tensor:
        storage = pa.array(
            [None if value is None else [value, value] for value in observations],
            type=pa.list_(pa.float32(), 2),
        )
        array = pa.ExtensionArray.from_storage(pa.fixed_shape_tensor(pa.float32(), (2,)), storage)
    else:
        array = pa.array(observations, type=pa.float32())
    offsets = pa.array([0, 2, 5, 9, 14, 20, 27, 35], type=pa.int64())
    series = replace(
        _lazy_series(spec, array),
        time_axis=IrregularAxis(first_us=0, last_us=35) if irregular else RegularAxis.from_rate_hz(1),
        time_offsets_loader=(lambda: offsets) if irregular else None,
    )
    restored = _write_read(
        tmp_path,
        series,
        values_backend="zarr",
        leading_series=replace(series, time_series_id="prefix", signal="prefix"),
        chunk_max_bytes=4,
        shard_target_bytes=32,
    )
    expected = restored.to_arrow()
    assert expected.is_valid().to_pylist() == [True, False, True, False, True, True, False, True]
    reads = []
    original_read = ZarrValuesReader._read_range

    def track_read(self, version, rel, start, stop):
        reads.append((rel, start, stop))
        return original_read(self, version, rel, start, stop)

    monkeypatch.setattr(ZarrValuesReader, "_read_range", track_read)
    for start, stop in [(0, 2), (1, 6), (3, 8), (6, 99), (2, 2), (8, 99), (10, 11)]:
        reads.clear()
        assert restored.read_steps(start, stop).equals(expected.slice(start, stop - start))
        if start < min(stop, 8):
            validity_reads = [(lo, hi) for rel, lo, hi in reads if "/_validity/" in rel]
            assert validity_reads == [(8 + start, 8 + min(stop, 8))]
        else:
            assert not reads


def test_zarr_nullable_range_combines_noncontiguous_runs_in_order(tmp_path, monkeypatch):
    spec = _spec(nullable=True)
    series = TimeSeries.from_values(
        [9.0, 0.0, None, 99.0, 99.0, None, 0.0, 7.0], spec=spec, signal="x", time_axis=OrdinalAxis()
    )
    _write_read(tmp_path, series, values_backend="zarr", chunk_max_bytes=4, shard_target_bytes=32)
    version = DatasetVersion.open_local(tmp_path / "timenet/nullable/1.0.0")
    rows = [
        {"chunk_file": "time_series.zarr/x", "chunk_major_idx": 0, "n_values": 3},
        {"chunk_file": "time_series.zarr/x", "chunk_major_idx": 5, "n_values": 3},
    ]
    reader = ZarrValuesReader()
    full = reader.load(version, rows, spec)
    assert full.to_pylist() == [9.0, 0.0, None, None, 0.0, 7.0]
    reads = []
    original_read = reader._read_range

    def track_read(version, rel, start, stop):
        reads.append((rel, start, stop))
        return original_read(version, rel, start, stop)

    monkeypatch.setattr(reader, "_read_range", track_read)
    selected = reader.load_range(version, rows, 1, 5, spec)
    assert selected.equals(full.slice(1, 4))
    assert selected.to_pylist() == [0.0, None, None, 0.0]
    validity_reads = [(lo, hi) for rel, lo, hi in reads if "/_validity/" in rel]
    assert validity_reads == [(1, 3), (5, 7)]
    reader.close()


@pytest.mark.parametrize("tensor", [False, True])
def test_to_numpy_rejects_actual_nulls(tensor):
    spec = _spec(nullable=True, value_shape=(2,) if tensor else ())
    if tensor:
        storage = pa.array([[0.0, 1.0], None], type=pa.list_(pa.float32(), 2))
        array = pa.ExtensionArray.from_storage(pa.fixed_shape_tensor(pa.float32(), (2,)), storage)
    else:
        array = pa.array([0.0, None], type=pa.float32())
    with pytest.raises(TimeFValidationError, match=r"to_numpy_and_mask.*to_arrow"):
        _lazy_series(spec, array).to_numpy()


def test_to_numpy_allows_nullable_without_nulls_and_preserves_special_floats():
    observations = np.array([0.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    series = TimeSeries.from_values(observations, spec=_spec(nullable=True), signal="x", time_axis=OrdinalAxis())
    np.testing.assert_array_equal(series.to_numpy(), observations)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_torch_default_batches_mixed_nullable_series(batch_size):
    dataset = TimeFDataset(metadata=make_dataset().metadata)
    for index in range(2):
        nullable = TimeSeries.from_values(
            [0.0, None], spec=_spec(nullable=True), signal="missing", time_axis=OrdinalAxis()
        )
        present = TimeSeries.from_values([0.0, 1.0], spec=_spec(), signal="present", time_axis=OrdinalAxis())
        dataset.add_record(time_series=(nullable, present), record_id=f"record-{index}")
    batch = next(iter(DataLoader(TimeFTorchDataset(dataset), batch_size=batch_size)))
    assert batch["series_masks"][0].tolist() == [[True, False]] * batch_size
    assert batch["series_masks"][1].tolist() == [[True, True]] * batch_size
    assert batch["series"][0].tolist() == [[0.0, 0.0]] * batch_size


@pytest.mark.parametrize("nullable", [False, True])
def test_torch_enum_direct_loader_rejects_unknown_labels(nullable):
    series = _lazy_series(
        _spec("enum", categories=("a", "b"), nullable=nullable),
        pa.array(["a", "unknown"]).dictionary_encode(),
    )
    dataset = TimeFDataset(metadata=make_dataset().metadata)
    dataset.add_record(time_series=(series,), record_id="record-0")
    with pytest.raises(TimeFValidationError, match="categories"):
        TimeFTorchDataset(dataset)[0]


@pytest.mark.parametrize("nullable", [False, True])
def test_torch_enum_direct_loader_distinguishes_category_zero_from_null(nullable):
    series = _lazy_series(
        _spec("enum", categories=("a", "b"), nullable=nullable),
        pa.array(["a", None, "b"]).dictionary_encode(),
    )
    dataset = TimeFDataset(metadata=make_dataset().metadata)
    dataset.add_record(time_series=(series,), record_id="record-0")
    if not nullable:
        with pytest.raises(TimeFValidationError, match="null"):
            TimeFTorchDataset(dataset)[0]
    else:
        item = TimeFTorchDataset(dataset)[0]
        assert item["series"][0].tolist() == [0, 0, 1]
        assert item["series_masks"][0].tolist() == [True, False, True]


@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize("container", [list, np.asarray, lambda values: np.array(values, dtype=object)])
@pytest.mark.parametrize("observations", [[1, 0], [2, -1], [True, False]])
def test_bool_constructors_use_numeric_truth_conversion(irregular, container, observations):
    values = container(observations)
    spec = _spec("bool")
    if irregular:
        series = TimeSeries.from_irregular(values, time_offsets_us=[0, 1], spec=spec, signal="x")
    else:
        series = TimeSeries.from_values(values, time_axis=OrdinalAxis(), spec=spec, signal="x")
    assert series.to_arrow().to_pylist() == [bool(value) for value in observations]


@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize("container", [list, lambda values: np.array(values, dtype=object)])
@pytest.mark.parametrize("nullable", [False, True])
@pytest.mark.parametrize(
    ("observations", "expected"),
    [([2, None, 0, -1], [True, None, False, True]), ([None, None, None, None], [None, None, None, None])],
)
def test_bool_constructors_preserve_none_separately_from_false(irregular, container, nullable, observations, expected):
    values = container(observations)
    spec = _spec("bool", nullable=nullable)

    def construct():
        if irregular:
            return TimeSeries.from_irregular(values, time_offsets_us=[0, 1, 2, 3], spec=spec, signal="x")
        return TimeSeries.from_values(values, time_axis=OrdinalAxis(), spec=spec, signal="x")

    if nullable:
        series = construct()
        assert series.to_arrow().to_pylist() == expected
        assert series.to_numpy_and_mask()[0].dtype == np.bool_
    else:
        with pytest.raises(TimeFValidationError, match="null"):
            construct()
