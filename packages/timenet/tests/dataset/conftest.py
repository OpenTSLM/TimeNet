from typing import Any

import pyarrow as pa
import pytest

from timenet.dataset import Signal
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
    def _make(signal="II", values=(1.0, 2.0, 3.0), **overrides):
        fields: dict[str, Any] = {
            "spec": spec,
            "name": signal,
            "time_axis": RegularAxis.from_rate_hz(500),
            "n_values": len(values),
            "loader": lambda v=tuple(values): pa.array(list(v), type=pa.float32()),
            **overrides,
        }
        if "time_series_id" in fields:
            fields["id"] = fields.pop("time_series_id")
        if "signal" in fields:
            fields["name"] = fields.pop("signal")
        return Signal.from_loader(**fields)

    return _make
