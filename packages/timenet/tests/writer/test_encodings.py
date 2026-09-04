from timenet.format.schemas import task_schema
from timenet.types import TaskType
from timenet.writer.encodings import byte_stream_split_supported, parquet_kwargs, task_dictionary


def test_byte_stream_split_supported_is_float_only():
    assert byte_stream_split_supported("float32")
    assert byte_stream_split_supported("float64")
    for dtype in ("int8", "int16", "int32", "int64", "uint8", "uint32", "bool", "str", "string"):
        assert not byte_stream_split_supported(dtype)


def test_parquet_kwargs_passes_data_page_size():
    assert (
        parquet_kwargs(dictionary_columns=[], column_encoding=None, compression="zstd", compression_level=19)[
            "data_page_size"
        ]
        is None
    )
    assert (
        parquet_kwargs(
            dictionary_columns=[],
            column_encoding=None,
            compression="zstd",
            compression_level=19,
            data_page_size=8 * 2**20,
        )["data_page_size"]
        == 8 * 2**20
    )


def test_target_columns_are_dictionary_encoded():
    # Guards the target rename: the categorical list and the schema columns must stay in sync, else
    # the target columns silently lose dictionary+RLE encoding with no test or error to catch it.
    dict_cols = task_dictionary(task_schema(TaskType.CLASSIFICATION))
    assert "target" in dict_cols
    assert "target_schema" in dict_cols
