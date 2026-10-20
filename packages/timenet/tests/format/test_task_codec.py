from timenet.dataset import OrdinalAxis, RegularAxis, Signal
from timenet.format.task_codec import decode_span, encode_span, encode_target, encode_task_payload
from timenet.types import (
    AnswerTask,
    StepInterval,
    TemporalLocalizationTask,
    TimePoint,
    TimeSeriesSpec,
    ureg,
)


def test_task_payload_keeps_targets_out_of_configuration_columns():
    task = AnswerTask(prompt="Alive?", targets=("Yes", 1))

    assert encode_task_payload(task) == {"target_schema": None, "unit": None, "target_name": None, "mode": None}


def _signals():
    spec = TimeSeriesSpec(spec_type="s", name="S", unit_value=ureg.dimensionless)
    timed = Signal.from_values([0.0] * 4, spec=spec, name="timed", time_axis=RegularAxis.from_rate_hz(1))
    ordinal = Signal.from_values([0.0] * 8, spec=spec, name="ordinal", time_axis=OrdinalAxis())
    return timed, ordinal


def test_task_span_codec_preserves_frame_shape_and_scope():
    timed, ordinal = _signals()
    spans = (
        TimePoint.seconds(2, time_series=(timed,)),
        StepInterval(time_series=ordinal, start=3, stop=8),
    )
    task = TemporalLocalizationTask(targets=(spans[0],))

    encoded = [encode_span(span) for span in spans]
    assert [item["span_type"] for item in encoded if item] == ["time_point", "step_interval"]
    assert [item["signal_ids"] for item in encoded if item] == [(timed.id,), (ordinal.id,)]
    signals = {timed.id: timed, ordinal.id: ordinal}
    assert tuple(decode_span(*_columns(item), signals=signals) for item in encoded) == spans
    assert encode_task_payload(task)["mode"] == "sparse"


def _columns(encoded):
    return encoded["span_type"], encoded["span_start"], encoded["span_end"], encoded["signal_ids"]


def test_target_codec_keeps_boolean_and_integer_rows_distinct():
    assert encode_target(True)["target_kind"] == "boolean"
    assert encode_target(1)["target_kind"] == "integer"
