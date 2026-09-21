from timenet.format.task_codec import decode_span, encode_span, encode_task_payload
from timenet.types import AnswerTask, StepInterval, TemporalLocalizationTask, TimePoint


def test_task_payload_keeps_relationships_out_of_json():
    task = AnswerTask(record_ids=("record-1",), prompt="Alive?", target="Yes")

    assert encode_task_payload(task) == {"target": "Yes"}


def test_task_span_codec_preserves_frame_shape_and_scope():
    spans = (
        TimePoint.seconds(2, time_series_ids=("ecg",)),
        StepInterval(time_series_id="tokens", start=3, stop=8),
    )
    task = TemporalLocalizationTask(target=(spans[0],))

    assert tuple(decode_span(encode_span(span)) for span in spans) == spans
    assert encode_task_payload(task)["target"] == [encode_span(spans[0])]
