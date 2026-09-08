import pytest

from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.schemas import IdCodec
from timenet.types import StepInterval, TimeInterval, TimeSpan
from timenet.types.ids import new_id


def test_string_columns_pass_through():
    codec = IdCodec.from_uuid16(())
    assert codec.encode("record_id", "anything") == "anything"
    assert codec.decode("record_id", "anything") == "anything"


def test_uuid16_round_trips():
    codec = IdCodec.from_uuid16({"record_id"})
    sid = new_id()
    encoded = codec.encode("record_id", sid)
    assert isinstance(encoded, bytes) and len(encoded) == 16
    assert codec.decode("record_id", encoded) == sid


def test_from_id_types_matches_from_uuid16():
    import pyarrow as pa  # noqa: PLC0415

    from timenet.format.schemas import UUID16  # noqa: PLC0415

    id_types = {"record_id": UUID16, "task_id": pa.string()}
    assert IdCodec.from_id_types(id_types) == IdCodec.from_uuid16({"record_id"})


def test_non_uuid_in_a_uuid16_column_raises_with_context():
    # a dangling reference used to surface as a bare ValueError from inside uuid
    codec = IdCodec.from_uuid16({"record_id"})
    with pytest.raises(TimeFValidationError, match="not a canonical UUID"):
        codec.encode("record_id", "rec_001::future")


def test_none_passes_through_on_both_sides():
    codec = IdCodec.from_uuid16({"source_id"})
    assert codec.encode("source_id", None) is None
    assert codec.decode_opt("source_id", None) is None


def test_a_steps_span_round_trips_through_the_codec():
    codec = IdCodec.from_uuid16(())
    span = StepInterval(time_series_id="s", start=0, stop=12)
    assert codec.decode_span(codec.encode_span(span)) == span


def test_a_seconds_span_keeps_its_frame_through_the_codec():
    codec = IdCodec.from_uuid16(())
    span = TimeInterval.seconds(1.0, 3.0, time_series_ids=("s",))
    restored = codec.decode_span(codec.encode_span(span))
    assert restored == span
    assert isinstance(restored, TimeSpan)


def test_a_span_row_written_before_the_frame_column_reads_as_seconds():
    # A partition written before the frame column existed carries no "frame" key; it predates steps.
    codec = IdCodec.from_uuid16(())
    row = {"start_us": 1, "end_us": 3, "time_series_ids": None}
    restored = codec.decode_span(row)
    assert isinstance(restored, TimeSpan)


def test_an_unknown_span_frame_is_rejected():
    codec = IdCodec.from_uuid16(())
    row = {"start_us": 1, "end_us": 3, "time_series_ids": None, "frame": "furlongs"}
    with pytest.raises(TimeFFormatError, match="unknown span frame"):
        codec.decode_span(row)


def test_a_step_span_row_with_no_id_is_rejected():
    # A step span stores the one series it counts on; an empty id list cannot be rebuilt.
    codec = IdCodec.from_uuid16(())
    row = {"start_us": 0, "end_us": 12, "time_series_ids": [], "frame": "steps"}
    with pytest.raises(TimeFFormatError, match="id list is empty"):
        codec.decode_span(row)
