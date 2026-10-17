from timenet.format.task_codec import decode_span, encode_span, encode_target, encode_task_payload
from timenet.types import AnswerTask, StepInterval, TemporalLocalizationTask, TimePoint, new_id


def test_task_payload_keeps_targets_out_of_configuration_columns():
    task = AnswerTask(prompt="Alive?", targets=("Yes", 1))

    assert encode_task_payload(task) == {"target_schema": None, "unit": None, "target_name": None, "mode": None}


def test_task_span_codec_preserves_frame_shape_and_scope():
    spans = (
        TimePoint.seconds(2, time_series_ids=(101,)),
        StepInterval(time_series_id=102, start=3, stop=8),
    )
    task = TemporalLocalizationTask(targets=(spans[0],))

    encoded = [encode_span(span) for span in spans]
    assert [item["span_type"] for item in encoded if item] == ["time_point", "step_interval"]
    assert [item["signal_ids"] for item in encoded if item] == [(101,), (102,)]
    assert tuple(decode_span(*_columns(item)) for item in encoded) == spans
    assert encode_task_payload(task)["mode"] == "sparse"


def test_span_codec_round_trips_full_range_int_ids():
    ids = (new_id(), 2**63 - 1)
    span = TimePoint.seconds(1, time_series_ids=ids)

    encoded = encode_span(span)
    assert encoded is not None
    decoded = decode_span(*_columns(encoded))

    assert encoded["signal_ids"] == ids
    assert isinstance(decoded, TimePoint)
    assert decoded == span
    assert decoded.time_series_ids is not None
    assert all(type(value) is int for value in decoded.time_series_ids)


def _columns(encoded):
    return encoded["span_type"], encoded["span_start"], encoded["span_end"], encoded["signal_ids"]


def test_target_codec_keeps_boolean_and_integer_rows_distinct():
    assert encode_target(True)["target_kind"] == "boolean"
    assert encode_target(1)["target_kind"] == "integer"
