import pytest

from timenet.errors import TimeFValidationError
from timenet.format.schemas import IdCodec
from timenet.types.ids import new_id


def test_string_columns_pass_through():
    codec = IdCodec.from_uuid16(())
    assert codec.encode("time_series_id", "anything") == "anything"


def test_uuid16_encodes_to_sixteen_bytes():
    codec = IdCodec.from_uuid16({"time_series_id"})
    sid = new_id()
    encoded = codec.encode("time_series_id", sid)
    assert isinstance(encoded, bytes) and len(encoded) == 16


def test_from_id_types_matches_from_uuid16():
    import pyarrow as pa  # noqa: PLC0415

    from timenet.format.schemas import UUID16  # noqa: PLC0415

    id_types = {"time_series_id": UUID16, "task_id": pa.string()}
    assert IdCodec.from_id_types(id_types) == IdCodec.from_uuid16({"time_series_id"})


def test_non_uuid_in_a_uuid16_column_raises_with_context():
    # a dangling reference used to surface as a bare ValueError from inside uuid
    codec = IdCodec.from_uuid16({"time_series_id"})
    with pytest.raises(TimeFValidationError, match="not a canonical UUID"):
        codec.encode("time_series_id", "ts_001::future")


def test_none_passes_through():
    codec = IdCodec.from_uuid16({"time_series_id"})
    assert codec.encode("time_series_id", None) is None


def test_encode_list_encodes_element_wise():
    codec = IdCodec.from_uuid16({"time_series_id"})
    ids = [new_id(), new_id()]
    assert [len(value) for value in codec.encode_list("time_series_id", ids)] == [16, 16]
