from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from timenet.dataset import Signal
from timenet.dataset.axis import IrregularAxis, OrdinalAxis, RegularAxis
from timenet.dataset.materialized_values import NumpyMaterializedValues, ValuesLoader
from timenet.errors import TimeFValidationError
from timenet.types import StepInterval, StepPoint, TimeInterval, TimeSeriesSpec, ureg


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


def test_signal_delegates_to_explicit_values_methods():
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    offsets = np.array([0, 2, 9], dtype=np.int64)
    calls = []

    class ExplicitLoader(ValuesLoader):
        def __call__(self):
            raise AssertionError("Signal must use the explicit methods")

        def to_arrow(self):
            calls.append("arrow")
            return pa.array(values)

        def to_numpy(self):
            calls.append("numpy")
            return values

        def read_steps(self, start, stop):
            calls.append((start, stop))
            return pa.array(values[start:stop])

    loader = ExplicitLoader()
    signal = Signal.from_loader(
        spec=_spec(),
        name="events",
        time_axis=IrregularAxis(first_us=0, last_us=9),
        n_values=len(values),
        loader=loader,
        time_offsets_loader=NumpyMaterializedValues(offsets),
    )
    assert calls == []
    assert signal.loader is loader
    assert signal.to_arrow().to_pylist() == values.tolist()
    assert signal.to_numpy() is values
    assert signal.read_steps(1, 3).to_pylist() == [2.0, 3.0]
    assert calls == ["arrow", "numpy", (1, 3)]
    assert np.shares_memory(signal.time_offsets_us(), offsets)


def test_callable_adapter_keeps_range_reads_lazy():
    values = pa.array([1.0, 2.0, 3.0], type=pa.float32())
    calls = []

    class RangeLoader:
        def __call__(self):
            calls.append("full")
            return values

        def read_steps(self, start, stop):
            calls.append((start, stop))
            return values.slice(start, stop - start)

    signal = _series(loader=RangeLoader())
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
