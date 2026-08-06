from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest

from timenet.dataset import TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.types import TimeSeriesSpec, ureg


def _spec():
    return TimeSeriesSpec(
        spec_type="ecg_lead",
        name="ECG Lead",
        unit_sampling_rate=ureg.hertz,
        unit_timestamp=ureg.second,
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
