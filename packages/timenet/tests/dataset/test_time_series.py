import pickle
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from timenet.dataset import Signal
from timenet.dataset.axis import IrregularAxis, OrdinalAxis, RegularAxis
from timenet.errors import TimeFNullValuesError, TimeFValidationError
from timenet.types import StepInterval, StepPoint, TimeInterval, TimeSeriesSpec, ureg
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
    if "time_series_id" in overrides:
        fields["id"] = fields.pop("time_series_id")
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


def test_identity_equality():
    a = _series()
    b = _series()
    assert a != b  # eq=False: two distinct instances never compare equal
    assert a == a


def test_frozen():
    ts = _series()
    with pytest.raises(AttributeError):
        ts.name = "V1"


def test_default_id_unique_explicit_id_kept():
    assert _series().id != _series().id
    assert _series(time_series_id="fixed").id == "fixed"


def test_signal_accepts_declarative_eager_data():
    signal = Signal(
        id="lead-i",
        name="I",
        data=[1.0, 2.0],
        spec=_spec(),
        time_axis=RegularAxis.from_rate_hz(500),
    )

    assert signal.id == "lead-i"
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


def test_from_values_generates_unique_id_unless_given():
    a = Signal.from_values([1.0], spec=_spec(), name="II", time_axis=RegularAxis.from_rate_hz(1))
    b = Signal.from_values([1.0], spec=_spec(), name="II", time_axis=RegularAxis.from_rate_hz(1))
    assert a.id != b.id
    fixed = Signal.from_values([1.0], spec=_spec(), name="II", time_axis=RegularAxis.from_rate_hz(1), id="x")
    assert fixed.id == "x"


def test_from_values_casts_to_the_specified_numpy_dtype():
    spec = _spec().model_copy(update={"dtype": "int16"})
    arr = Signal.from_values([1, 2, 3], spec=spec, name="II", time_axis=RegularAxis.from_rate_hz(2)).to_arrow()
    assert arr.type == pa.int16()
    assert arr.to_pylist() == [1, 2, 3]


def test_from_values_str_spec_keeps_string_labels():
    spec = _spec().model_copy(update={"dtype": "str"})
    arr = Signal.from_values(["normal", "afib"], spec=spec, name="II", time_axis=RegularAxis.from_rate_hz(2)).to_arrow()
    assert arr.type == pa.string()
    assert arr.to_pylist() == ["normal", "afib"]


def _counting(n, tsid="s", axis=None):
    return _series(
        n_values=n,
        time_series_id=tsid,
        time_axis=axis or RegularAxis.from_rate_hz(500),
        loader=lambda: pa.array([float(i) for i in range(n)], type=pa.float32()),
    )


def test_step_range_of_a_steps_span_is_the_span():
    ts = _counting(6)
    assert ts.step_range(StepInterval(time_series_id="s", start=2, stop=5)) == (2, 5)


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


def test_step_range_horizon_matches_n_steps():
    ts = _counting(12)
    span = StepInterval(time_series_id="s", start=0, stop=12)
    start, stop = ts.step_range(span)
    assert stop - start == span.n_steps


def test_step_range_rejects_a_point():
    with pytest.raises(TimeFValidationError, match="not a point"):
        _counting(3).step_range(StepPoint(time_series_id="s", start=1))


def test_step_range_rejects_a_steps_span_naming_another_series():
    with pytest.raises(TimeFValidationError, match="not this series"):
        _counting(6).step_range(StepInterval(time_series_id="other", start=0, stop=3))


def test_step_range_rejects_a_steps_span_past_the_series():
    with pytest.raises(TimeFValidationError, match="runs past"):
        _counting(3).step_range(StepInterval(time_series_id="s", start=0, stop=4))


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
    dense, valid = ts.to_numpy_and_mask()
    assert dense.dtype == np.float32
    assert dense.tolist() == _VALUES
    assert valid.dtype == np.bool_
    assert valid.tolist() == [True] * len(_VALUES)
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
    if not nullable:
        assert ts.to_numpy_and_mask()[0].tolist() == _VALUES


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


@pytest.mark.parametrize("nullable", [False, True])
def test_to_numpy_points_null_values_to_the_mask(nullable):
    ts = _series(
        spec=_spec().model_copy(update={"nullable": nullable}),
        loader=lambda: pa.array([1.0, None, 3.0], type=pa.float32()),
    )
    with pytest.raises(TimeFNullValuesError, match=r"to_numpy_and_mask\(\) or to_arrow\(\)"):
        ts.to_numpy()


def test_to_numpy_and_mask_rejects_null_values_under_a_spec_without_nulls():
    ts = _series(loader=lambda: pa.array([1.0, None, 3.0], type=pa.float32()))
    with pytest.raises(TimeFNullValuesError, match="has null values but nullable=False"):
        ts.to_numpy_and_mask()


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
    np.testing.assert_array_equal(ts.to_numpy_and_mask()[0], expected)


@pytest.mark.parametrize("nullable", [False, True])
def test_boolean_tensors_read_as_numpy_with_their_value_shape(nullable):
    frames = np.array([[True, False], [False, False], [True, True]])
    storage = pa.FixedSizeListArray.from_arrays(pa.array(frames.ravel()), 2)
    tensor = pa.ExtensionArray.from_storage(pa.fixed_shape_tensor(pa.bool_(), (2,)), storage)
    spec = _spec().model_copy(update={"dtype": "bool", "value_shape": (2,), "nullable": nullable})
    ts = Signal(name="c", data=tensor, spec=spec, time_axis=OrdinalAxis())
    np.testing.assert_array_equal(ts.to_numpy(), frames)
    dense, valid = ts.to_numpy_and_mask()
    np.testing.assert_array_equal(dense, frames)
    assert valid.tolist() == [True, True, True]


@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize("source", ["numpy", "list"])
def test_a_signal_built_in_memory_survives_pickling(source, irregular):
    ts = _signal_from(source, irregular=irregular)
    back = pickle.loads(pickle.dumps(ts))
    assert (back.id, back.name, back.n_values, back.time_axis) == (ts.id, ts.name, ts.n_values, ts.time_axis)
    assert back.to_arrow().equals(ts.to_arrow())
    np.testing.assert_array_equal(back.to_numpy(), ts.to_numpy())
    if irregular:
        np.testing.assert_array_equal(back.time_offsets_us(), ts.time_offsets_us())
