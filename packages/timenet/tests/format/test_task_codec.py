from timenet.format.task_codec import decode_span, encode_span, encode_target, encode_task_payload
from timenet.types import AnswerTask, StepInterval, TemporalLocalizationTask, TimePoint


def test_task_payload_keeps_targets_out_of_configuration_json():
    task = AnswerTask(prompt="Alive?", targets=("Yes", 1))

    assert encode_task_payload(task) == {}


def test_task_span_codec_preserves_frame_shape_and_scope():
    spans = (
        TimePoint.seconds(2, time_series_ids=("ecg",)),
        StepInterval(time_series_id="tokens", start=3, stop=8),
    )
    task = TemporalLocalizationTask(targets=(spans[0],))

    assert tuple(decode_span(encode_span(span)) for span in spans) == spans
    assert encode_task_payload(task) == {"mode": "sparse"}


def test_target_codec_keeps_boolean_and_integer_rows_distinct():
    assert encode_target(True)["target_kind"] == "boolean"
    assert encode_target(1)["target_kind"] == "integer"
