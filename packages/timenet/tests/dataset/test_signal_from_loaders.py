import pyarrow as pa
import pytest

from timenet.dataset import IrregularAxis, RegularAxis, Signal, SignalDefinition
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, TimeSeriesSpec


def test_from_loaders_preserves_lazy_values_and_signal_metadata():
    calls = []
    spec = TimeSeriesSpec(spec_type="other", name="test", unit_value=None, dtype="int32")
    axis = IrregularAxis(first_us=0, last_us=10)
    annotation = Annotation(key="sensor", value="wrist")

    def read_values():
        calls.append("values")
        return pa.array([1, 2], type=pa.int32())

    def read_offsets():
        calls.append("offsets")
        return pa.array([0, 10], type=pa.int64())

    def read_second_values():
        calls.append("second")
        return pa.array([3, 4], type=pa.int32())

    signals = Signal.from_loaders(
        channels=(
            SignalDefinition(
                name="first",
                spec=spec,
                time_axis=axis,
                n_values=2,
                loader=read_values,
                time_offsets_loader=read_offsets,
                null_count=0,
                annotations=(annotation,),
                metadata={"unit": "a"},
            ),
            SignalDefinition(
                name="second",
                spec=spec,
                time_axis=RegularAxis.from_rate_hz(1),
                n_values=2,
                loader=read_second_values,
            ),
        )
    )

    assert calls == []
    first, second = signals
    assert (first.name, second.name) == ("first", "second")
    assert first.spec == spec and first.time_axis == axis
    assert first.n_values == 2 and first.null_count == 0
    assert first.annotations == (annotation,) and first.metadata == {"unit": "a"}
    values = first.to_arrow()
    assert values.to_pylist() == [1, 2]
    assert values.type == pa.int32()
    assert second.to_arrow().to_pylist() == [3, 4]
    assert first.time_offsets_us().tolist() == [0, 10]
    assert calls == ["values", "second", "offsets"]


def test_from_loaders_uses_single_signal_validation():
    spec = TimeSeriesSpec(spec_type="other", name="test", unit_value=None, dtype="int32")
    channel = SignalDefinition(
        name="invalid",
        spec=spec,
        time_axis=RegularAxis.from_rate_hz(1),
        n_values=0,
        loader=lambda: pa.array([1], type=pa.int32()),
    )
    with pytest.raises(TimeFValidationError):
        Signal.from_loaders(channels=(channel,))
