from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest

from timenet.dataset import TimeSeries
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
        sampling_rate_hz=500.0,
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
        {"sampling_rate_hz": 0.0},
        {"sampling_rate_hz": -1.0},
        {"sampling_rate_hz": float("inf")},
        {"t_start_s": -1.0},
        {"t_start_s": 5.0, "t_end_s": 5.0},
        {"t_start_s": 5.0, "t_end_s": 4.0},
    ],
)
def test_validation_rejects(overrides):
    with pytest.raises(ValueError):
        _series(**overrides)


def test_window_ok():
    ts = _series(t_start_s=0.0, t_end_s=10.0)
    assert ts.t_end_s == pytest.approx(10.0)
