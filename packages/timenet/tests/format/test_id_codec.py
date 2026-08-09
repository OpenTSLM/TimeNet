import pytest

from timenet.errors import TimeFValidationError
from timenet.format.schemas import IdCodec
from timenet.types import IntervalSpan, SpanFrame
from timenet.types.ids import new_id


def test_string_columns_pass_through():
    codec = IdCodec.from_uuid16(())
    assert codec.encode("sample_id", "anything") == "anything"
    assert codec.decode("sample_id", "anything") == "anything"


def test_uuid16_round_trips():
    codec = IdCodec.from_uuid16({"sample_id"})
    sid = new_id()
    encoded = codec.encode("sample_id", sid)
    assert isinstance(encoded, bytes) and len(encoded) == 16
    assert codec.decode("sample_id", encoded) == sid


def test_from_encoding_matches_from_uuid16():
    # the reader builds from the manifest map, the writer from the set; they must agree
    assert IdCodec.from_encoding({"sample_id": "uuid16", "task_id": "string"}) == IdCodec.from_uuid16({"sample_id"})


def test_non_uuid_in_a_uuid16_column_raises_with_context():
    # a dangling reference used to surface as a bare ValueError from inside uuid
    codec = IdCodec.from_uuid16({"sample_id"})
    with pytest.raises(TimeFValidationError, match="not a canonical UUID"):
        codec.encode("sample_id", "rec_001::future")


def test_none_passes_through_on_both_sides():
    codec = IdCodec.from_uuid16({"source_id"})
    assert codec.encode("source_id", None) is None
    assert codec.decode_opt("source_id", None) is None


def test_a_steps_span_round_trips_through_the_codec():
    codec = IdCodec.from_uuid16(())
    span = IntervalSpan.steps(0, 12, time_series_ids=("s",))
    assert codec.decode_span(codec.encode_span(span)) == span


def test_a_seconds_span_keeps_its_frame_through_the_codec():
    codec = IdCodec.from_uuid16(())
    span = IntervalSpan.seconds(1.0, 3.0, time_series_ids=("s",))
    restored = codec.decode_span(codec.encode_span(span))
    assert restored == span
    assert restored is not None and restored.frame is SpanFrame.SECONDS


def test_a_span_row_written_before_the_frame_column_reads_as_seconds():
    # A partition written before the frame column existed carries no "frame" key; it predates steps.
    codec = IdCodec.from_uuid16(())
    row = {"start_us": 1, "end_us": 3, "time_series_ids": None}
    restored = codec.decode_span(row)
    assert restored is not None and restored.frame is SpanFrame.SECONDS
