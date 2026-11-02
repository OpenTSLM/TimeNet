import pickle
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from timenet.dataset import Signal
from timenet.dataset.axis import IrregularAxis, OrdinalAxis, RegularAxis
from timenet.dataset.time_series import _StepWindow
from timenet.errors import TimeFNullValuesError, TimeFValidationError
from timenet.types import (
    Annotation,
    ArrayHandle,
    StepInterval,
    StepPoint,
    TimeInterval,
    TimePoint,
    TimeSeriesSpec,
    ureg,
)
from timenet.types.handles import ArrowArray, CallableArray, NumpyArray


def _spec():
    return TimeSeriesSpec(
        spec_type="ecg_lead",
        name="ECG Lead",
        unit_value=ureg.millivolt,
    )


def _series(**overrides):
    fields: dict[str, Any] = {
        "spec": _spec(),
        "name": "II",
        "time_axis": RegularAxis.from_rate_hz(500),
        "n_values": 3,
        "loader": lambda: pa.array([1.0, 2.0, 3.0], type=pa.float32()),
        **overrides,
    }
    if "signal" in overrides:
        fields["name"] = fields.pop("signal")
    return Signal.from_loader(**fields)


def test_to_arrow_returns_loader_output():
    ts = _series()
    arr = ts.to_arrow()
    assert isinstance(arr, pa.Array)
    assert arr.type == pa.float32()
    assert arr.to_pylist() == [1.0, 2.0, 3.0]


def test_to_numpy():
    arr = _series().to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.tolist() == [1.0, 2.0, 3.0]


def test_loader_is_lazy():
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return pa.array([1.0], type=pa.float32())

    ts = _series(loader=loader)
    assert calls["n"] == 0
    ts.to_arrow()
    assert calls["n"] == 1


def test_signal_delegates_to_explicit_handle_methods():
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    offsets = np.array([0, 2, 9], dtype=np.int64)
    calls = []

    class ExplicitHandle(ArrayHandle):
        @property
        def is_loaded(self):
            return False

        def read(self):
            calls.append("arrow")
            return pa.array(values)

        def read_numpy(self):
            calls.append("numpy")
            return values

        def read_steps(self, start, stop):
            calls.append((start, stop))
            return pa.array(values[start:stop])

    handle = ExplicitHandle()
    signal = Signal.from_loader(
        spec=_spec(),
        name="events",
        time_axis=IrregularAxis(first_us=0, last_us=9),
        n_values=len(values),
        loader=handle,
        time_offsets_loader=NumpyArray(offsets),
    )
    assert calls == []
    assert signal._values is handle
    assert signal.to_arrow().to_pylist() == values.tolist()
    assert signal.to_numpy() is values
    assert signal.read_steps(1, 3).to_pylist() == [2.0, 3.0]
    assert calls == ["arrow", "numpy", (1, 3)]
    assert np.shares_memory(signal.time_offsets_us(), offsets)


def test_a_handle_with_its_own_range_read_keeps_range_reads_lazy():
    values = pa.array([1.0, 2.0, 3.0], type=pa.float32())
    calls = []

    class RangeHandle(ArrayHandle):
        @property
        def is_loaded(self):
            return False

        def read(self):
            calls.append("full")
            return values

        def read_steps(self, start, stop):
            calls.append((start, stop))
            return values.slice(start, stop - start)

    signal = _series(loader=RangeHandle())
    assert calls == []
    assert signal.read_steps(1, 3).to_pylist() == [2.0, 3.0]
    assert calls == [(1, 3)]
    assert signal.to_arrow() is values
    assert signal.to_numpy().tolist() == [1.0, 2.0, 3.0]
    assert calls == [(1, 3), "full", "full"]


def test_identity_equality():
    a = _series()
    b = _series()
    assert a != b  # eq=False: two distinct instances never compare equal
    assert a == a


def test_frozen():
    ts = _series()
    with pytest.raises(AttributeError):
        ts.name = "V1"


def test_signal_accepts_declarative_eager_data():
    signal = Signal(
        name="I",
        data=[1.0, 2.0],
        spec=_spec(),
        time_axis=RegularAxis.from_rate_hz(500),
    )

    assert signal.name == "I"
    assert signal.n_values == 2
    assert signal.to_arrow().to_pylist() == [1.0, 2.0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"signal": ""},
        # The window used to be stated and could contradict the values; now it is derived from a
        # count, so the count is what has to be sound.
        {"n_values": 0},
        {"n_values": -1},
        {"n_values": 1.5},
        {"n_values": True},
    ],
)
def test_validation_rejects(overrides):
    with pytest.raises(ValueError):
        _series(**overrides)


def test_the_window_is_derived_from_the_axis_and_the_count():
    # Nothing stores the end, so it cannot disagree with the values.
    assert _series(n_values=5000).span_us == (0, 10_000_000)  # 5000 values at 500 Hz


def test_from_values_casts_to_float32_and_derives_the_window():
    ts = Signal.from_values([1.0, 2.0, 3.0, 4.0], spec=_spec(), name="II", time_axis=RegularAxis.from_rate_hz(2))
    values = ts.to_numpy()
    assert values.dtype == np.float32
    assert values.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert ts.span_us == (0, 2_000_000)  # 4 values at 2 Hz


def test_from_values_counts_the_array_it_was_given():
    # The window is derived, so a caller cannot hand it one that disagrees with the values.
    ts = Signal.from_values(
        np.array([1.0, 2.0]), spec=_spec(), name="II", time_axis=RegularAxis.from_rate_hz(4).at_index(4)
    )
    assert ts.n_values == 2
    assert ts.span_us == (1_000_000, 1_500_000)  # starts 4 values into a 4 Hz axis


@pytest.mark.parametrize("values", [[1, 2, 3], np.array([1.9, 2.9, 3.9])], ids=["list", "numpy"])
def test_from_values_casts_to_the_specified_numpy_dtype(values):
    spec = _spec().model_copy(update={"dtype": "int16"})
    arr = Signal.from_values(values, spec=spec, name="II", time_axis=RegularAxis.from_rate_hz(2)).to_arrow()
    assert arr.type == pa.int16()
    assert arr.to_pylist() == [1, 2, 3]


def test_from_values_str_spec_keeps_string_labels():
    spec = _spec().model_copy(update={"dtype": "str"})
    arr = Signal.from_values(["normal", "afib"], spec=spec, name="II", time_axis=RegularAxis.from_rate_hz(2)).to_arrow()
    assert arr.type == pa.string()
    assert arr.to_pylist() == ["normal", "afib"]


def test_numpy_backing_keeps_dtype_conversion_and_arrow_output_consistent():
    values = np.array([1.9, 2.9, 3.9, 4.9, 5.9, 6.9], dtype=np.float64)[::2]
    spec = _spec().model_copy(update={"dtype": "int16"})
    signal = Signal.from_values(values, spec=spec, name="events", time_axis=OrdinalAxis())
    assert signal.to_numpy().dtype == np.int16
    assert signal.to_numpy().tolist() == [1, 3, 5]
    assert not np.shares_memory(signal.to_numpy(), values)
    assert signal.to_arrow().type == pa.int16()
    assert signal.to_arrow().to_pylist() == [1, 3, 5]


def test_numpy_tensor_values_stay_numpy_with_the_specs_value_shape():
    frames = np.arange(12, dtype=np.float64).reshape(3, 2, 2)
    spec = _spec().model_copy(update={"value_shape": (2, 2)})
    signal = Signal.from_values(frames, spec=spec, name="frames", time_axis=OrdinalAxis())
    assert isinstance(signal._values, NumpyArray)
    assert signal.n_values == 3
    assert signal.to_numpy().dtype == np.float32
    np.testing.assert_array_equal(signal.to_numpy(), frames)
    tensor = signal.to_arrow()
    assert isinstance(tensor, pa.FixedShapeTensorArray)
    np.testing.assert_array_equal(tensor.to_numpy_ndarray(), frames)


@pytest.mark.parametrize(("value_shape", "values"), [((), np.zeros((3, 2))), ((2,), np.zeros(3)), ((2,), np.zeros(()))])
def test_numpy_values_need_the_specs_value_shape(value_shape, values):
    spec = _spec().model_copy(update={"value_shape": value_shape})
    with pytest.raises(TimeFValidationError, match=r"needs shape \(n_steps"):
        Signal.from_values(values, spec=spec, name="frames", time_axis=OrdinalAxis())


@pytest.mark.parametrize("dtype", ["bool", "int8", "int64", "uint16", "float32", "float64"])
def test_to_numpy_is_read_only_for_every_dtype(dtype):
    spec = _spec().model_copy(update={"dtype": dtype})
    values = np.array([1, 0, 1]).astype(dtype)
    arrow = pa.array(values)
    signals = [
        Signal.from_values(values, spec=spec, name="s", time_axis=OrdinalAxis()),
        Signal.from_values(values.tolist(), spec=spec, name="s", time_axis=OrdinalAxis()),
        Signal.from_loader(name="s", spec=spec, time_axis=OrdinalAxis(), n_values=3, loader=lambda: arrow),
    ]
    for signal in signals:
        read = signal.to_numpy()
        assert read.tolist() == values.tolist()
        with pytest.raises(ValueError, match="read-only"):
            read[0] = read[1]
        assert signal.to_arrow().to_pylist() == values.tolist()


def test_from_irregular_converts_and_checks_arrow_values_against_the_spec():
    offsets = [0, 5, 9]
    signal = Signal.from_irregular(pa.array([1.0, 2.0, 3.0]), time_offsets_us=offsets, spec=_spec(), name="c")
    assert signal.to_arrow().type == pa.float32()
    with pytest.raises(TimeFValidationError, match="nullable=False"):
        Signal.from_irregular(pa.array([1.0, None, 3.0]), time_offsets_us=offsets, spec=_spec(), name="c")
    spec = _spec().model_copy(update={"dtype": "enum", "categories": ("a", "b")})
    with pytest.raises(TimeFValidationError, match="outside its categories"):
        Signal.from_irregular(pa.array(["a", "c", "b"]), time_offsets_us=offsets, spec=spec, name="c")


def _counting(n, axis=None):
    return _series(
        n_values=n,
        time_axis=axis or RegularAxis.from_rate_hz(500),
        loader=lambda: pa.array([float(i) for i in range(n)], type=pa.float32()),
    )


def test_step_range_of_a_steps_span_is_the_span():
    ts = _counting(6)
    assert ts.step_range(StepInterval(time_series=ts, start=2, stop=5)) == (2, 5)


def test_step_range_of_a_seconds_span_on_a_regular_axis():
    # 1 Hz: value k sits at k seconds, so [2 s, 5 s) is steps 2..5.
    ts = _counting(6, axis=RegularAxis.from_rate_hz(1))
    assert ts.step_range(TimeInterval.seconds(2.0, 5.0)) == (2, 5)


def test_step_range_of_a_seconds_span_on_an_irregular_axis():
    ts = Signal.from_irregular(
        [0.0, 1.0, 2.0, 3.0], time_offsets_us=[0, 1_000_000, 2_000_000, 5_000_000], spec=_spec(), name="c"
    )
    # [1 s, 5 s) picks the values at 1 s and 2 s, not the one at 5 s (exclusive end).
    assert ts.step_range(TimeInterval.seconds(1.0, 5.0)) == (1, 3)


def test_irregular_full_validation_rejects_unordered_offsets():
    with pytest.raises(TimeFValidationError, match="non-decreasing"):
        Signal.from_irregular([1.0, 2.0, 3.0], time_offsets_us=[0, 10, 9], spec=_spec(), name="events")


def test_irregular_basic_validation_shares_buffers_without_scanning(monkeypatch: pytest.MonkeyPatch):
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    offsets = np.array([0, 2, 9], dtype=np.int64)

    def reject_scan(*_args, **_kwargs):
        raise AssertionError("basic validation must not scan timestamp ordering")

    monkeypatch.setattr("timenet.dataset.axis.np.diff", reject_scan)
    signal = Signal.from_irregular(values, time_offsets_us=offsets, spec=_spec(), name="events", full_validation=False)
    assert np.shares_memory(signal.to_numpy(), values)
    assert np.shares_memory(signal.time_offsets_us(), offsets)
    assert isinstance(signal.time_axis, IrregularAxis)
    assert signal.time_axis.first_us == 0
    assert signal.time_axis.last_us == 9
    assert signal.time_offsets_us().tolist() == [0, 2, 9]


@pytest.mark.parametrize(
    ("offsets", "message"),
    [([0, 9], "one time offset per value"), ([0.0, 2.0, 9.0], "whole microseconds")],
)
def test_irregular_basic_validation_rejects_invalid_offsets(offsets, message):
    with pytest.raises(TimeFValidationError, match=message):
        Signal.from_irregular(
            [1.0, 2.0, 3.0], time_offsets_us=offsets, spec=_spec(), name="events", full_validation=False
        )


def test_step_range_horizon_matches_n_steps():
    ts = _counting(12)
    span = StepInterval(time_series=ts, start=0, stop=12)
    start, stop = ts.step_range(span)
    assert stop - start == span.n_steps


def test_step_range_rejects_a_point():
    with pytest.raises(TimeFValidationError, match="not a point"):
        ts = _counting(3)
        ts.step_range(StepPoint(time_series=ts, start=1))


def test_step_range_rejects_a_steps_span_naming_another_series():
    with pytest.raises(TimeFValidationError, match="not this series"):
        _counting(6).step_range(StepInterval(time_series=_counting(6), start=0, stop=3))


def test_step_range_rejects_a_steps_span_past_the_series():
    with pytest.raises(TimeFValidationError, match="runs past"):
        ts = _counting(3)
        ts.step_range(StepInterval(time_series=ts, start=0, stop=4))


def test_step_range_rejects_a_seconds_span_on_an_ordinal_series():
    ts = Signal.from_values([1.0, 2.0, 3.0], spec=_spec(), name="c", time_axis=OrdinalAxis())
    with pytest.raises(TimeFValidationError, match="no timeline"):
        ts.step_range(TimeInterval.seconds(0.0, 1.0))


_VALUES = [1.0, 2.0, 3.0, 4.0]
_OFFSETS_US = [0, 5, 9, 20]


def _no_arrow(*_args, **_kwargs):
    raise AssertionError("built Arrow")


def _signal_from(source, *, irregular=False, nullable=False):
    """Build the same series from NumPy values, a list, or a loader callable."""
    spec = _spec().model_copy(update={"dtype": "float32", "nullable": nullable})
    if source == "loader":
        return Signal.from_loader(
            spec=spec,
            name="c",
            time_axis=IrregularAxis.spanning(_OFFSETS_US) if irregular else RegularAxis.from_rate_hz(2),
            n_values=len(_VALUES),
            loader=lambda: pa.array(_VALUES, type=pa.float32()),
            time_offsets_loader=(lambda: pa.array(_OFFSETS_US, type=pa.int64())) if irregular else None,
        )
    values = np.array(_VALUES) if source == "numpy" else list(_VALUES)
    if irregular:
        return Signal.from_irregular(values, time_offsets_us=_OFFSETS_US, spec=spec, name="c")
    return Signal.from_values(values, spec=spec, name="c", time_axis=RegularAxis.from_rate_hz(2))


@pytest.mark.parametrize("nullable", [False, True])
@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize("source", ["numpy", "list", "loader"])
def test_accessors_read_the_same_series_from_every_source(source, irregular, nullable):
    ts = _signal_from(source, irregular=irregular, nullable=nullable)
    assert ts.to_arrow().equals(pa.array(_VALUES, type=pa.float32()))
    values = ts.to_numpy()
    assert values.dtype == np.float32
    assert values.tolist() == _VALUES
    assert ts.read_steps(1, 3).to_pylist() == [2.0, 3.0]
    assert ts.read_steps(2, 99).to_pylist() == [3.0, 4.0]
    if irregular:
        assert ts.time_offsets_us().dtype == np.int64
        assert ts.time_offsets_us().tolist() == _OFFSETS_US
        assert ts.step_range(TimeInterval(start_us=5, end_us=20)) == (1, 3)
    else:
        with pytest.raises(TimeFValidationError, match="only an IrregularAxis"):
            ts.time_offsets_us()


def test_in_memory_values_stay_numpy_or_arrow_as_given():
    assert isinstance(_signal_from("numpy")._values, NumpyArray)
    assert isinstance(_signal_from("list")._values, ArrowArray)
    arrow = pa.array(_VALUES, type=pa.float32())
    signal = Signal(name="c", data=arrow, spec=_spec(), time_axis=RegularAxis.from_rate_hz(2))
    assert isinstance(signal._values, ArrowArray)
    assert signal.to_arrow() is arrow
    assert isinstance(_signal_from("numpy", irregular=True)._time_offsets, NumpyArray)


@pytest.mark.parametrize("nullable", [False, True])
def test_numpy_values_read_as_numpy_without_building_arrow(monkeypatch, nullable):
    ts = _signal_from("numpy", irregular=True, nullable=nullable)
    monkeypatch.setattr(pa, "array", _no_arrow)
    assert ts.to_numpy().tolist() == _VALUES
    assert ts.time_offsets_us().tolist() == _OFFSETS_US


@pytest.mark.parametrize(
    ("dtype", "source"),
    [("bool", np.array([True, False, True])), ("float64", np.arange(6.0)[::2])],
)
def test_values_that_arrow_copies_do_not_follow_later_changes_to_the_source(dtype, source):
    spec = _spec().model_copy(update={"dtype": dtype})
    ts = Signal.from_values(source, spec=spec, name="c", time_axis=OrdinalAxis())
    expected = source.tolist()
    source[0] = source[1]
    assert ts.to_numpy().tolist() == expected
    assert ts.to_arrow().to_pylist() == expected


def test_from_loader_wraps_a_callable_to_read_through_it():
    calls = []

    def loader():
        calls.append(None)
        return pa.array([1.0, 2.0, 3.0], type=pa.float32())

    ts = _series(loader=loader)
    values = ts._values
    assert isinstance(values, CallableArray)
    assert values.function is loader
    ts.to_arrow()
    ts.to_numpy()
    ts.read_steps(0, 1)
    assert len(calls) == 3


def test_from_loader_keeps_value_handles_as_they_are():
    values = ArrowArray(pa.array([1.0, 2.0, 3.0], type=pa.float32()))
    offsets = NumpyArray(np.array([0, 5, 9]))
    ts = Signal.from_loader(
        spec=_spec(),
        name="c",
        time_axis=IrregularAxis.spanning([0, 5, 9]),
        n_values=3,
        loader=values,
        time_offsets_loader=offsets,
    )
    assert ts._values is values
    assert ts._time_offsets is offsets
    assert ts.to_arrow() is values.read()
    assert ts.time_offsets_us().tolist() == [0, 5, 9]


class _RangeLoader:
    """A plain callable that also reads step ranges, without subclassing ArrayHandle."""

    def __call__(self) -> pa.Array:
        return pa.array([0, 5, 9], type=pa.int64())

    def read_steps(self, start: int, stop: int) -> pa.Array:
        return self().slice(start, stop - start)


@pytest.mark.parametrize("field", ["loader", "time_offsets_loader"])
def test_from_loader_rejects_a_callable_with_its_own_range_read(field):
    with pytest.raises(TimeFValidationError, match="subclass ArrayHandle"):
        _series(time_axis=IrregularAxis.spanning([0, 5, 9]), **{field: _RangeLoader()})


# NumPy 2.5 deprecates setting a shape, but it still reshapes the array in place.
@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")
@pytest.mark.parametrize("dtype", ["float32", "bool"])
def test_reshaping_a_read_array_leaves_the_stored_values(dtype):
    spec = _spec().model_copy(update={"dtype": dtype})
    ts = Signal.from_values(np.arange(4).astype(dtype), spec=spec, name="c", time_axis=OrdinalAxis())
    read = ts.to_numpy()
    read.shape = (2, 2)  # ty: ignore[deprecated]
    assert ts.to_numpy().shape == (4,)


def test_to_numpy_explains_a_partly_null_tensor_step_in_full_sentences():
    storage = pa.FixedSizeListArray.from_arrays(pa.array([1.0, None, 3.0, 4.0], type=pa.float32()), 2)
    tensor = pa.ExtensionArray.from_storage(pa.fixed_shape_tensor(pa.float32(), (2,)), storage)
    spec = _spec().model_copy(update={"value_shape": (2,), "nullable": True})
    ts = _series(spec=spec, n_values=2, loader=lambda: tensor)
    with pytest.raises(TimeFNullValuesError) as raised:
        ts.to_numpy()
    assert str(raised.value) == (
        "series contains null values. Tensor values may have nulls only for whole timesteps. "
        "Use to_arrow() to preserve missingness"
    )


@pytest.mark.parametrize("nullable", [False, True])
def test_to_numpy_points_null_values_to_arrow(nullable):
    ts = _series(
        spec=_spec().model_copy(update={"nullable": nullable}),
        loader=lambda: pa.array([1.0, None, 3.0], type=pa.float32()),
    )
    with pytest.raises(TimeFNullValuesError, match=r"1 of 3 values are null.*Use to_arrow\(\)"):
        ts.to_numpy()


def _permuted_tensor():
    frames = np.arange(24, dtype=np.float32).reshape(4, 6)
    storage = pa.FixedSizeListArray.from_arrays(pa.array(frames.ravel()), 6)
    tensor_type = pa.fixed_shape_tensor(pa.float32(), (3, 2), permutation=[1, 0])
    return pa.ExtensionArray.from_storage(tensor_type, storage)


@pytest.mark.parametrize("nullable", [False, True])
@pytest.mark.parametrize("sliced", [False, True])
def test_permuted_and_sliced_tensors_read_as_pyarrow_converts_them(sliced, nullable):
    tensor = _permuted_tensor()
    if sliced:
        tensor = tensor.slice(1, 2)
    expected = tensor.to_numpy_ndarray()
    spec = _spec().model_copy(update={"value_shape": expected.shape[1:], "nullable": nullable})
    ts = Signal(name="c", data=tensor, spec=spec, time_axis=OrdinalAxis())
    np.testing.assert_array_equal(ts.to_numpy(), expected)


@pytest.mark.parametrize("nullable", [False, True])
def test_boolean_tensors_read_as_numpy_with_their_value_shape(nullable):
    frames = np.array([[True, False], [False, False], [True, True]])
    storage = pa.FixedSizeListArray.from_arrays(pa.array(frames.ravel()), 2)
    tensor = pa.ExtensionArray.from_storage(pa.fixed_shape_tensor(pa.bool_(), (2,)), storage)
    spec = _spec().model_copy(update={"dtype": "bool", "value_shape": (2,), "nullable": nullable})
    ts = Signal(name="c", data=tensor, spec=spec, time_axis=OrdinalAxis())
    np.testing.assert_array_equal(ts.to_numpy(), frames)


@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize("source", ["numpy", "list"])
def test_a_signal_built_in_memory_survives_pickling(source, irregular):
    ts = _signal_from(source, irregular=irregular)
    back = pickle.loads(pickle.dumps(ts))
    assert (back.name, back.n_values, back.time_axis) == (ts.name, ts.n_values, ts.time_axis)
    assert back.to_arrow().equals(ts.to_arrow())
    np.testing.assert_array_equal(back.to_numpy(), ts.to_numpy())
    if irregular:
        np.testing.assert_array_equal(back.time_offsets_us(), ts.time_offsets_us())


def _at(offset_us: int, n: int = 4) -> Signal:
    # 1 Hz, so value k sits at offset_us + k seconds.
    return _counting(n, axis=RegularAxis.from_rate_hz(1, offset_us=offset_us))


@pytest.mark.parametrize(
    ("start_s", "end_s", "overlaps"),
    [
        (0, 2, False),  # ends where the series starts
        (0, 11, True),  # partly before the series
        (11, 13, True),  # inside the series
        (13, 20, True),  # partly after the series
        (14, 20, False),  # starts one period after the last value
        (0, 20, True),  # covers the whole series
    ],
)
def test_overlaps_with_on_a_regular_axis(start_s, end_s, overlaps):
    # Values at 10, 11, 12, 13 s; the series covers [10 s, 14 s).
    signal = _at(10_000_000)
    assert signal.overlaps_with(TimeInterval.seconds(start_s, end_s)) is overlaps


@pytest.mark.parametrize(
    ("start_us", "end_us", "overlaps"),
    [(0, 10, False), (0, 11, True), (20, 21, True), (21, 30, False)],
)
def test_overlaps_with_on_an_irregular_axis(start_us, end_us, overlaps):
    # The series covers its first time offset through its last one, inclusive.
    signal = Signal.from_irregular([1.0, 2.0], time_offsets_us=[10, 20], spec=_spec(), name="events")
    assert signal.overlaps_with(TimeInterval(start_us=start_us, end_us=end_us)) is overlaps


def test_overlaps_with_rejects_an_ordinal_series():
    signal = Signal.from_values([1.0, 2.0], spec=_spec(), name="c", time_axis=OrdinalAxis())
    with pytest.raises(TimeFValidationError, match="no timeline"):
        signal.overlaps_with(TimeInterval.seconds(0.0, 1.0))


def _range_reads(n: int, calls: list, axis=None) -> Signal:
    class RangeHandle(ArrayHandle):
        @property
        def is_loaded(self):
            return False

        def read(self):
            calls.append("full")
            return pa.array([float(i) for i in range(n)], type=pa.float32())

        def read_steps(self, start, stop):
            calls.append((start, stop))
            return pa.array([float(i) for i in range(start, stop)], type=pa.float32())

    return _series(n_values=n, time_axis=axis or RegularAxis.from_rate_hz(1), loader=RangeHandle())


def test_slice_reads_no_values_until_it_is_read():
    calls = []
    signal = _range_reads(10, calls)
    part = signal.slice(TimeInterval.seconds(2, 6))
    assert calls == []
    assert part.to_arrow().to_pylist() == [2.0, 3.0, 4.0, 5.0]
    assert calls == [(2, 6)]


def test_a_range_read_of_a_slice_reads_only_that_range():
    calls = []
    part = _range_reads(10, calls).slice(TimeInterval.seconds(2, 6))
    assert part.read_steps(1, 3).to_pylist() == [3.0, 4.0]
    # A stop past the slice's end reads to the slice's end, never past it.
    assert part.read_steps(2, 10).to_pylist() == [4.0, 5.0]
    assert calls == [(3, 5), (4, 6)]


def test_slice_of_a_regular_series_keeps_its_place_on_the_timeline():
    signal = Signal(
        spec=_spec(),
        name="II",
        time_axis=RegularAxis.from_rate_hz(2, offset_us=1_000_000),
        data=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        metadata={"lead": "II"},
    )
    # 2 Hz from 1 s: [2 s, 3 s) holds the values at 2 s and 2.5 s.
    part = signal.slice(TimeInterval.seconds(2, 3))
    assert part is not signal
    assert part.to_numpy().tolist() == [2.0, 3.0]
    assert part.n_values == 2
    assert part.span_us == (2_000_000, 3_000_000)
    assert part.name == signal.name
    assert part.spec is signal.spec
    assert part.metadata == {"lead": "II"}


def test_slice_drops_annotations():
    signal = _counting(6, axis=RegularAxis.from_rate_hz(1))
    signal.annotate(Annotation(key="quality", value="good"))
    assert signal.slice(TimeInterval.seconds(2, 4)).annotations == ()


def test_slice_of_an_irregular_series_keeps_the_sliced_time_offsets():
    signal = Signal.from_irregular(
        [10.0, 11.0, 12.0, 13.0],
        time_offsets_us=[0, 1_500_000, 2_500_000, 5_000_000],
        spec=_spec(),
        name="events",
    )
    part = signal.slice(TimeInterval.seconds(1, 4))
    assert part.to_numpy().tolist() == [11.0, 12.0]
    assert part.time_offsets_us().tolist() == [1_500_000, 2_500_000]
    assert part.time_axis == IrregularAxis(first_us=1_500_000, last_us=2_500_000)


def test_slice_of_an_ordinal_series_by_steps_keeps_the_axis():
    axis = OrdinalAxis()
    signal = Signal.from_values([0.0, 1.0, 2.0, 3.0], spec=_spec(), name="c", time_axis=axis)
    part = signal.slice(StepInterval(time_series=signal, start=1, stop=3))
    assert part.to_numpy().tolist() == [1.0, 2.0]
    assert part.time_axis is axis
    assert part.span_us is None


def test_slice_by_steps_on_a_regular_series():
    signal = _counting(6, axis=RegularAxis.from_rate_hz(1))
    part = signal.slice(StepInterval(time_series=signal, start=4, stop=6))
    assert part.to_numpy().tolist() == [4.0, 5.0]
    assert part.span_us == (4_000_000, 6_000_000)


def test_a_slice_of_a_slice_reads_the_original_series_once():
    calls = []
    signal = _range_reads(10, calls)
    outer = signal.slice(TimeInterval.seconds(2, 8))
    inner = outer.slice(StepInterval(time_series=outer, start=1, stop=3))
    assert inner.span_us == (3_000_000, 5_000_000)
    # The inner slice reads the original handle, not the outer slice's window.
    window = inner._values
    assert isinstance(window, _StepWindow)
    assert window.base is signal._values
    assert inner.to_arrow().to_pylist() == [3.0, 4.0]
    assert calls == [(3, 5)]


def test_a_slice_of_a_sliced_irregular_series_keeps_its_time_offsets():
    signal = Signal.from_irregular(
        [0.0, 1.0, 2.0, 3.0, 4.0], time_offsets_us=[0, 10, 20, 30, 40], spec=_spec(), name="events"
    )
    inner = signal.slice(TimeInterval(start_us=10, end_us=41)).slice(TimeInterval(start_us=20, end_us=31))
    assert inner.to_numpy().tolist() == [2.0, 3.0]
    assert inner.time_offsets_us().tolist() == [20, 30]


@pytest.mark.parametrize(
    "make_span",
    [
        lambda _: TimeInterval.seconds(1, 20),
        lambda ts: StepInterval(time_series=ts, start=0, stop=7),
    ],
)
def test_slice_rejects_a_span_past_the_series(make_span):
    signal = _counting(6, axis=RegularAxis.from_rate_hz(1))
    with pytest.raises(TimeFValidationError, match="runs past"):
        signal.slice(make_span(signal))


# A regular series with values at 10, 11, 12, 13 s covers [10 s, 14 s). An irregular one with values
# at 10, 20, 30 s covers [10 s, 30 s + 1 us).
_REGULAR_AT_10_S = "regular"
_IRREGULAR_AT_10_S = "irregular"


def _at_10_s(axis: str) -> Signal:
    if axis == _REGULAR_AT_10_S:
        return _at(10_000_000)
    return Signal.from_irregular(
        [0.0, 1.0, 2.0], time_offsets_us=[10_000_000, 20_000_000, 30_000_000], spec=_spec(), name="c"
    )


@pytest.mark.parametrize(
    ("axis", "start_s", "end_s", "clipped"),
    [
        (_REGULAR_AT_10_S, 0, 11, [0.0]),  # starts before the series
        (_REGULAR_AT_10_S, 13, 20, [3.0]),  # ends after the series
        (_IRREGULAR_AT_10_S, 0, 15, [0.0]),
        (_IRREGULAR_AT_10_S, 25, 40, [2.0]),
    ],
)
def test_slice_of_a_partly_overlapping_interval_raises_unless_clipped(axis, start_s, end_s, clipped):
    signal = _at_10_s(axis)
    interval = TimeInterval.seconds(start_s, end_s)
    assert signal.overlaps_with(interval)
    with pytest.raises(TimeFValidationError, match="runs past"):
        signal.slice(interval)
    assert signal.slice(interval, clip=True).to_numpy().tolist() == clipped


@pytest.mark.parametrize(("axis", "start_s", "end_s"), [(_REGULAR_AT_10_S, 10.2, 10.8), (_IRREGULAR_AT_10_S, 12, 18)])
@pytest.mark.parametrize("clip", [False, True])
def test_slice_of_an_interval_between_two_values_raises(axis, start_s, end_s, clip):
    signal = _at_10_s(axis)
    interval = TimeInterval.seconds(start_s, end_s)
    assert signal.overlaps_with(interval)
    with pytest.raises(TimeFValidationError, match="covers no steps"):
        signal.slice(interval, clip=clip)


@pytest.mark.parametrize("axis", [_REGULAR_AT_10_S, _IRREGULAR_AT_10_S])
def test_clip_rejects_an_interval_outside_the_series(axis):
    with pytest.raises(TimeFValidationError, match="does not overlap"):
        _at_10_s(axis).slice(TimeInterval.seconds(40, 50), clip=True)


def test_clip_cuts_a_step_span_at_the_series_end():
    signal = _counting(6, axis=RegularAxis.from_rate_hz(1))
    part = signal.slice(StepInterval(time_series=signal, start=4, stop=9), clip=True)
    assert part.to_numpy().tolist() == [4.0, 5.0]


def test_clip_rejects_a_step_span_that_starts_past_the_series():
    signal = _counting(6, axis=RegularAxis.from_rate_hz(1))
    with pytest.raises(TimeFValidationError, match="covers no steps"):
        signal.slice(StepInterval(time_series=signal, start=6, stop=9), clip=True)


@pytest.mark.parametrize("clip", [False, True])
def test_slice_rejects_a_step_span_naming_another_series(clip):
    with pytest.raises(TimeFValidationError, match="not this series"):
        _counting(6).slice(StepInterval(time_series=_counting(6), start=0, stop=3), clip=clip)


@pytest.mark.parametrize("make_point", [lambda _: TimePoint.seconds(1), lambda ts: StepPoint(time_series=ts, start=1)])
@pytest.mark.parametrize("clip", [False, True])
def test_slice_rejects_a_point(make_point, clip):
    signal = _counting(6)
    with pytest.raises(TimeFValidationError, match="not a point"):
        signal.slice(make_point(signal), clip=clip)


@pytest.mark.parametrize("clip", [False, True])
def test_slice_rejects_a_time_span_on_an_ordinal_series(clip):
    ordinal = Signal.from_values([1.0, 2.0], spec=_spec(), name="c", time_axis=OrdinalAxis())
    with pytest.raises(TimeFValidationError, match="no timeline"):
        ordinal.slice(TimeInterval.seconds(0.0, 1.0), clip=clip)


def _counting_offset_reads(reads: list) -> Signal:
    def offsets():
        reads.append(1)
        return pa.array([0, 10, 20, 30], type=pa.int64())

    return Signal.from_loader(
        spec=_spec(),
        name="events",
        time_axis=IrregularAxis(first_us=0, last_us=30),
        n_values=4,
        loader=lambda: pa.array([0.0, 1.0, 2.0, 3.0], type=pa.float32()),
        time_offsets_loader=offsets,
    )


def test_slice_by_a_time_span_reads_irregular_time_offsets_once():
    reads = []
    part = _counting_offset_reads(reads).slice(TimeInterval(start_us=5, end_us=25))
    assert len(reads) == 1
    assert part.time_axis == IrregularAxis(first_us=10, last_us=20)


def test_slice_by_a_step_span_reads_irregular_time_offsets_once():
    reads = []
    signal = _counting_offset_reads(reads)
    part = signal.slice(StepInterval(time_series=signal, start=1, stop=3))
    assert len(reads) == 1
    assert part.time_axis == IrregularAxis(first_us=10, last_us=20)


def test_a_time_span_scoped_to_other_series_does_not_overlap_this_one():
    signal, other = _counting(6, axis=RegularAxis.from_rate_hz(1)), _counting(6, axis=RegularAxis.from_rate_hz(1))
    assert not signal.overlaps_with(TimeInterval.seconds(1, 3, time_series=(other,)))


@pytest.mark.parametrize("clip", [False, True])
def test_slice_rejects_a_time_span_scoped_to_other_series(clip):
    signal, other = _counting(6, axis=RegularAxis.from_rate_hz(1)), _counting(6, axis=RegularAxis.from_rate_hz(1))
    with pytest.raises(TimeFValidationError, match="scoped to other series"):
        signal.slice(TimeInterval.seconds(1, 3, time_series=(other,)), clip=clip)


def test_a_time_span_scoped_to_other_series_skips_an_ordinal_series():
    ordinal = Signal.from_values([1.0, 2.0], spec=_spec(), name="c", time_axis=OrdinalAxis())
    assert not ordinal.overlaps_with(TimeInterval.seconds(0, 1, time_series=(_counting(3),)))


@pytest.mark.parametrize("with_others", [False, True])
def test_a_time_span_scoped_to_this_series_selects_it(with_others):
    signal, other = _counting(6, axis=RegularAxis.from_rate_hz(1)), _counting(6, axis=RegularAxis.from_rate_hz(1))
    interval = TimeInterval.seconds(1, 3, time_series=(other, signal) if with_others else (signal,))
    assert signal.overlaps_with(interval)
    assert signal.slice(interval).to_numpy().tolist() == [1.0, 2.0]
    assert signal.slice(interval, clip=True).to_numpy().tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    "make_span", [lambda ts: StepInterval(time_series=ts, start=0, stop=2), lambda _: TimePoint.seconds(1)]
)
def test_overlaps_with_rejects_a_span_that_is_not_a_time_interval(make_span):
    signal = _counting(6, axis=RegularAxis.from_rate_hz(1))
    with pytest.raises(TimeFValidationError, match="needs a time interval"):
        signal.overlaps_with(make_span(signal))


def test_a_slice_of_an_irregular_series_copies_only_its_time_offsets():
    signal = Signal.from_irregular([0.0, 1.0, 2.0, 3.0], time_offsets_us=[0, 10, 20, 30], spec=_spec(), name="c")
    part = signal.slice(TimeInterval(start_us=10, end_us=25))
    assert part.time_offsets_us().tolist() == [10, 20]
    assert not np.shares_memory(part.time_offsets_us(), signal.time_offsets_us())
