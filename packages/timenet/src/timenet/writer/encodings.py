"""Role-based Parquet encodings, pinned in code rather than left to writer heuristics.

Float waveform values get BYTE_STREAM_SPLIT + zstd (measured ~30% smaller than plain+zstd on quantized
biosignals); monotonic ints get DELTA_BINARY_PACKED; bounded categoricals get dictionary+RLE; id-like
columns stay plain. Encodings are addressed by the explicit ``use_dictionary`` list plus a
``column_encoding`` map (nested elements as ``values.list.element``), the only combination pyarrow
applies reliably.
"""

from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


SHARD_DICTIONARY = ["spec_type", "channel"]
SHARD_ENCODING = {
    "values.list.element": "BYTE_STREAM_SPLIT",
    "time_offsets_us.list.element": "DELTA_BINARY_PACKED",
    "chunk_idx": "DELTA_BINARY_PACKED",
}

INDEX_DICTIONARY = ["spec_type", "channel", "chunk_file"]
INDEX_ENCODING = {
    "chunk_idx": "DELTA_BINARY_PACKED",
    "chunk_major_idx": "DELTA_BINARY_PACKED",
    "chunk_minor_idx": "DELTA_BINARY_PACKED",
}

SAMPLES_DICTIONARY = [
    "view",
    "time_series.list.element.spec_type",
    "time_series.list.element.channel",
    "time_series.list.element.axis_type",
]

ANNOTATIONS_DICTIONARY = ["key"]

_TASK_CATEGORICAL = ("target", "target_schema", "target_name", "unit", "mode")


def task_dictionary(schema: pa.Schema) -> list[str]:
    """Return the categorical columns to dictionary-encode for a task partition.

    Only string columns qualify: ``target`` is a float for a scalar prediction and a list of span structs
    for a localization, and dictionary-encoding either buys nothing (or fails outright for the nested one).

    Args:
        schema: The task partition's Arrow schema.

    Returns:
        The subset of categorical task columns present in the schema as strings.
    """
    return [name for name in _TASK_CATEGORICAL if name in schema.names and pa.types.is_string(schema.field(name).type)]


def parquet_kwargs(
    *,
    dictionary_columns: list[str],
    column_encoding: dict[str, str] | None,
    compression: str,
    compression_level: int,
) -> dict[str, Any]:
    """Build the shared Parquet write options for one file.

    Args:
        dictionary_columns: Columns to dictionary-encode (must be disjoint from ``column_encoding``).
        column_encoding: Explicit per-column encodings (e.g. BYTE_STREAM_SPLIT on ``values.list.element``).
        compression: Codec name (``"zstd"``, ``"snappy"``, ``"none"``).
        compression_level: Pinned level, applied only for zstd.

    Returns:
        Keyword arguments for :class:`pyarrow.parquet.ParquetWriter` or ``write_table``.
    """
    return {
        "use_dictionary": list(dictionary_columns) if dictionary_columns else False,
        "column_encoding": dict(column_encoding) if column_encoding else None,
        "compression": compression,
        "compression_level": compression_level if compression == "zstd" else None,
        "write_statistics": True,
        "write_page_index": True,
        "write_page_checksum": True,
        # Content-defined chunking aligns data pages to content, so a re-curated or copy-on-write-edited
        # version re-stores only the chunks that changed on a dedup backend (e.g. Xet). Requires pyarrow>=21.
        "use_content_defined_chunking": True,
    }


def values_encoding_of(path: str, column_path: str = "values.list.element") -> set[str]:
    """Return the encodings applied to one shard column.

    Used as a writer self-check that the configured encoding was actually applied; pyarrow drops it
    silently when the column path is wrong, which costs the compression with no error to notice.

    Args:
        path: Path to a shard parquet file.
        column_path: The column's path in the parquet schema.

    Returns:
        The set of encoding names found on ``column_path`` across all row groups.
    """
    parquet_file = pq.ParquetFile(path)
    meta = parquet_file.metadata
    encodings: set[str] = set()
    for row_group in range(meta.num_row_groups):
        group = meta.row_group(row_group)
        for column in range(meta.num_columns):
            chunk = group.column(column)
            if chunk.path_in_schema == column_path:
                encodings.update(chunk.encodings)
    return encodings
