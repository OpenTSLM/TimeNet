import pyarrow as pa

from timenet.format.schemas import shard_schema


def test_shard_values_default_to_list_of_float32():
    schema = shard_schema()
    assert schema.field("values").type == pa.list_(pa.float32())


def test_shard_values_take_the_value_type():
    schema = shard_schema(pa.int16())
    assert schema.field("values").type == pa.list_(pa.int16())
    assert schema.field("time_offsets_us").type == pa.list_(pa.int64())


def test_index_columns_do_not_depend_on_value_type():
    schema = shard_schema(pa.int16())
    assert "time_series_id" not in schema.names
    assert schema.field("chunk_idx").type == pa.int32()
    assert schema.field("n_values").type == pa.int32()
