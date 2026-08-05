"""Role-based Parquet encodings, pinned in code rather than left to writer heuristics.

Monotonic ints get DELTA_BINARY_PACKED, bounded categoricals get dictionary+RLE, and id-like columns
stay plain. Float waveform values are the one column no fixed rule fits, so the writer measures the
data and picks per modality (see :mod:`timenet.writer.value_encoding`); this module translates that
choice into pyarrow write options and back out of a finished file's footer. Encodings are addressed
by the explicit ``use_dictionary`` list plus a ``column_encoding`` map (nested elements as
``values.list.element``), the only combination pyarrow applies reliably.
"""

from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.writer.value_encoding import ValueEncoding

VALUES_COLUMN = "values.list.element"
"""Parquet path of the shard values column; the nested element, not the list."""

TIME_OFFSETS_COLUMN = "time_offsets_us.list.element"
"""Parquet path of the shard time-offsets column; the nested element of the irregular-series list."""

SHARD_CATEGORICAL = ["spec_type", "channel"]
"""Shard columns dictionary-encoded regardless of the values encoding."""

_PARQUET_COLUMN_ENCODING = {
    ValueEncoding.BYTE_STREAM_SPLIT: "BYTE_STREAM_SPLIT",
    ValueEncoding.PLAIN: "PLAIN",
}
"""Parquet ``column_encoding`` name per value encoding. Dictionary is absent: pyarrow requests it
through ``use_dictionary``, and passing a column in both lists is rejected."""

_DICTIONARY_MARKERS = frozenset({"RLE_DICTIONARY", "PLAIN_DICTIONARY"})
"""Footer names for dictionary-encoded indices; PLAIN_DICTIONARY is the pre-2.4 spelling."""

INDEX_DICTIONARY = ["spec_type", "channel", "chunk_file"]
INDEX_ENCODING = {
    "chunk_idx": "DELTA_BINARY_PACKED",
    "chunk_major_idx": "DELTA_BINARY_PACKED",
    "chunk_minor_idx": "DELTA_BINARY_PACKED",
}

SAMPLES_DICTIONARY = [
    "time_series.list.element.spec_type",
    "time_series.list.element.channel",
    "time_series.list.element.axis_type",
]

ANNOTATIONS_DICTIONARY = ["key"]

_TASK_CATEGORICAL = ("target", "target_schema", "target_name", "unit", "mode")


def shard_dictionary(value_encoding: ValueEncoding) -> list[str]:
    """Return the shard columns to dictionary-encode.

    Args:
        value_encoding: The encoding selected for this shard's values column.

    Returns:
        The categorical columns, plus the values column when it was selected for dictionary encoding.
    """
    columns = list(SHARD_CATEGORICAL)
    if value_encoding == ValueEncoding.DICTIONARY:
        columns.append(VALUES_COLUMN)
    return columns


def shard_encoding(value_encoding: ValueEncoding) -> dict[str, str]:
    """Return the shard's explicit per-column encodings.

    Args:
        value_encoding: The encoding selected for this shard's values column.

    Returns:
        The ``column_encoding`` map, always pinning the monotonic index and the irregular series'
        time offsets, and without a values entry when the values column goes through
        ``use_dictionary`` instead.
    """
    column_encoding = {"chunk_idx": "DELTA_BINARY_PACKED", TIME_OFFSETS_COLUMN: "DELTA_BINARY_PACKED"}
    parquet_name = _PARQUET_COLUMN_ENCODING.get(value_encoding)
    if parquet_name is not None:
        column_encoding[VALUES_COLUMN] = parquet_name
    return column_encoding


def applied_matches(value_encoding: ValueEncoding, applied: set[str]) -> bool:
    """Report whether a footer's values-column encodings are the ones that were asked for.

    A dictionary column chunk lists a dictionary marker for its indices *and* PLAIN for the
    dictionary page itself, so PLAIN alone is not evidence of a plain column. It is, however, all
    that is left when Parquet abandons a dictionary that outgrew its page limit and finishes the
    chunk plain, which is lossless and not the writer's choice, so the dictionary case accepts it.

    Args:
        value_encoding: The encoding the writer selected.
        applied: Encoding names read from the finished file, via :func:`values_encoding_of`.

    Returns:
        ``True`` if the selection landed (or fell back in a way Parquet permits).
    """
    if value_encoding is ValueEncoding.DICTIONARY:
        return bool(applied & (_DICTIONARY_MARKERS | {"PLAIN"}))
    if value_encoding is ValueEncoding.BYTE_STREAM_SPLIT:
        return "BYTE_STREAM_SPLIT" in applied
    return "PLAIN" in applied and not (applied & _DICTIONARY_MARKERS)


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
        column_encoding: Explicit per-column encodings (e.g. BYTE_STREAM_SPLIT on the values column).
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


def values_column_bytes(path: str) -> int:
    """Return the compressed on-disk size of a shard's ``values.list.element`` column.

    The values plane is what an encoding choice moves; whole-file size also carries the id, index,
    and statistics columns, which the choice does not touch.

    Args:
        path: Path to a shard parquet file.

    Returns:
        Summed ``total_compressed_size`` of the values column across all row groups.
    """
    meta = pq.ParquetFile(path).metadata
    total = 0
    for row_group in range(meta.num_row_groups):
        group = meta.row_group(row_group)
        for column in range(meta.num_columns):
            chunk = group.column(column)
            if chunk.path_in_schema == VALUES_COLUMN:
                total += chunk.total_compressed_size
    return total
