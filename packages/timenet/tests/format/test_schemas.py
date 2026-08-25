import pyarrow as pa

from timenet.format.schemas import default_id_types, shard_schema


def test_shard_values_default_to_list_of_float32():
    schema = shard_schema(default_id_types())
    assert schema.field("values").type == pa.list_(pa.float32())


def test_shard_values_take_the_value_type():
    schema = shard_schema(default_id_types(), pa.int16())
    assert schema.field("values").type == pa.list_(pa.int16())
    assert schema.field("time_offsets_us").type == pa.list_(pa.int64())


def test_id_and_time_offsets_columns_do_not_depend_on_value_type():
    schema = shard_schema(default_id_types(), pa.int16())
    assert schema.field("time_series_id").type == pa.string()
    assert schema.field("chunk_idx").type == pa.int32()
    assert schema.field("n_values").type == pa.int32()
