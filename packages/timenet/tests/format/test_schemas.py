import pyarrow as pa

from timenet.format.schemas import shard_schema


def test_value_type_changes_only_the_values_column():
    default, int16 = shard_schema(), shard_schema(pa.int16())
    assert int16.field("values").type == pa.list_(pa.int16())
    values_at = int16.get_field_index("values")
    assert int16.remove(values_at) == default.remove(values_at)
