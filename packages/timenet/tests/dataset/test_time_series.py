from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest

from timenet.dataset import TimeSeries
from timenet.dataset.axis import OrdinalAxis, RegularAxis
from timenet.errors import TimeFValidationError
from timenet.types import StepInterval, StepPoint, TimeInterval, TimeSeriesSpec, ureg


def _spec():
    return TimeSeriesSpec(
        spec_type="ecg_lead",
        name="ECG Lead",
        unit_value=ureg.millivolt,
    )


def _series(**overrides):
    base = TimeSeries(
        spec=_spec(),
        channel="II",
        time_axis=RegularAxis.from_rate_hz(500),
        n_values=3,
        loader=lambda: pa.array([1.0, 2.0, 3.0], type=pa.float32()),
    )
    return replace(base, **overrides) if overrides else base


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
        ts.channel = "V1"


def test_default_id_unique_explicit_id_kept():
    assert _series().time_series_id != _series().time_series_id
    assert _series(time_series_id="fixed").time_series_id == "fixed"


@pytest.mark.parametrize(
    "overrides",
    [
        {"channel": ""},
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
    ts = TimeSeries.from_values([1.0, 2.0, 3.0, 4.0], spec=_spec(), channel="II", time_axis=RegularAxis.from_rate_hz(2))
    values = ts.to_numpy()
    assert values.dtype == np.float32
    assert values.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert ts.span_us == (0, 2_000_000)  # 4 values at 2 Hz


def test_from_values_counts_the_array_it_was_given():
    # The window is derived, so a caller cannot hand it one that disagrees with the values.
    ts = TimeSeries.from_values(
        np.array([1.0, 2.0]), spec=_spec(), channel="II", time_axis=RegularAxis.from_rate_hz(4).at_index(4)
    )
    assert ts.n_values == 2
    assert ts.span_us == (1_000_000, 1_500_000)  # starts 4 values into a 4 Hz axis


def test_from_values_generates_unique_id_unless_given():
    a = TimeSeries.from_values([1.0], spec=_spec(), channel="II", time_axis=RegularAxis.from_rate_hz(1))
    b = TimeSeries.from_values([1.0], spec=_spec(), channel="II", time_axis=RegularAxis.from_rate_hz(1))
    assert a.time_series_id != b.time_series_id
    fixed = TimeSeries.from_values(
        [1.0], spec=_spec(), channel="II", time_axis=RegularAxis.from_rate_hz(1), time_series_id="x"
    )
    assert fixed.time_series_id == "x"


def test_from_values_casts_to_the_specified_numpy_dtype():
    spec = replace(_spec(), dtype="int16")
    arr = TimeSeries.from_values([1, 2, 3], spec=spec, channel="II", time_axis=RegularAxis.from_rate_hz(2)).to_arrow()
    assert arr.type == pa.int16()
    assert arr.to_pylist() == [1, 2, 3]


def test_from_values_str_spec_keeps_string_labels():
    spec = replace(_spec(), dtype="str")
    arr = TimeSeries.from_values(
        ["normal", "afib"], spec=spec, channel="II", time_axis=RegularAxis.from_rate_hz(2)
    ).to_arrow()
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
    ts = TimeSeries.from_irregular(
        [0.0, 1.0, 2.0, 3.0], time_offsets_us=[0, 1_000_000, 2_000_000, 5_000_000], spec=_spec(), channel="c"
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
    ts = TimeSeries.from_values([1.0, 2.0, 3.0], spec=_spec(), channel="c", time_axis=OrdinalAxis())
    with pytest.raises(TimeFValidationError, match="no timeline"):
        ts.step_range(TimeInterval.seconds(0.0, 1.0))
