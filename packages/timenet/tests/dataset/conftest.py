from dataclasses import replace

import pyarrow as pa
import pytest

from timenet.dataset import TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.types import TimeSeriesSpec, ureg


@pytest.fixture
def spec():
    return TimeSeriesSpec(
        spec_type="ecg_lead",
        name="ECG Lead",
        unit_value=ureg.millivolt,
    )


@pytest.fixture
def make_series(spec):
    def _make(channel="II", values=(1.0, 2.0, 3.0), **overrides):
        base = TimeSeries(
            spec=spec,
            channel=channel,
            time_axis=RegularAxis.from_rate_hz(500),
            n_values=len(values),
            loader=lambda v=tuple(values): pa.array(list(v), type=pa.float32()),
        )
        return replace(base, **overrides) if overrides else base

    return _make
