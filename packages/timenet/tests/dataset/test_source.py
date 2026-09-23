from fractions import Fraction

import pyarrow as pa
import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source
from timenet.errors import TimeFValidationError
from timenet.types import TimeSeriesSpec, ureg


SPEC = TimeSeriesSpec(spec_type="acceleration", name="Acceleration", unit_value=ureg.Unit("m/s^2"))


def _signal(signal_id: str, name: str) -> Signal:
    return Signal.from_loader(
        spec=SPEC,
        name=name,
        time_axis=RegularAxis(period_us=Fraction(1_000)),
        loader=lambda: pa.array([1.0], type=pa.float32()),
        id=signal_id,
        n_values=1,
    )


def test_record_walks_recursive_sources_and_signals():
    accelerometer = Source(
        id="accelerometer",
        name="Accelerometer",
        signals=(_signal("x", "X"), _signal("y", "Y"), _signal("z", "Z")),
    )
    gyroscope = Source(id="gyroscope", name="Gyroscope", signals=(_signal("gx", "GX"),))
    imu = Source(id="imu", name="IMU", sources=(gyroscope, accelerometer))
    record = Record(record_id="record", sources=(imu,))

    assert [source.id for source in record.walk_sources()] == ["imu", "accelerometer", "gyroscope"]
    assert [signal.id for signal in record.walk_signals()] == ["x", "y", "z", "gx"]


def test_record_rejects_source_reused_under_two_parents():
    shared = Source(id="shared", name="Shared")
    left = Source(id="left", name="Left", sources=(shared,))
    right = Source(id="right", name="Right", sources=(shared,))

    with pytest.raises(TimeFValidationError, match="attached more than once"):
        Record(record_id="record", sources=(Source(id="root", name="Root", sources=(left, right)),))


def test_record_rejects_source_cycle():
    root = Source(id="root", name="Root")
    child = Source(id="child", name="Child", sources=(root,))
    root.sources = (child,)

    with pytest.raises(TimeFValidationError, match="cycle"):
        Record(record_id="record", sources=(root,))
