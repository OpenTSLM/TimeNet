from timenet.writer.encodings import byte_stream_split_supported, parquet_kwargs


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
