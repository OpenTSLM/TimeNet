import pytest

from timenet.errors import TimeFValidationError
from timenet.format.schemas import IdCodec
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
