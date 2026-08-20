"""This module pins Parquet encodings by column role, instead of the writer choosing them by heuristics.

Monotonic integers use DELTA_BINARY_PACKED. Bounded categorical columns use dictionary encoding with
RLE. ID-like columns stay in plain encoding. Float waveform values are the one column type with no
fixed rule. For these columns, the writer measures the data and selects an encoding for each modality
(see :mod:`timenet.writer.value_encoding`). This module converts that choice into pyarrow write
options. It also reads the choice back from the footer of a finished file. Encodings use the explicit
``use_dictionary`` list and a ``column_encoding`` map. Nested elements use the path
``values.list.element``. This is the only combination that pyarrow applies reliably.
"""

from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.writer.value_encoding import ValueEncoding


VALUES_COLUMN = "values.list.element"
"""Parquet path of the shard values column. This is the nested element, not the list."""

TIME_OFFSETS_COLUMN = "time_offsets_us.list.element"
"""Parquet path of the shard time-offsets column. This is the nested element of the irregular-series
list."""

SHARD_CATEGORICAL = ["spec_type", "channel"]
"""These shard columns always use dictionary encoding, independent of the values encoding choice."""

_PARQUET_COLUMN_ENCODING = {
    ValueEncoding.BYTE_STREAM_SPLIT: "BYTE_STREAM_SPLIT",
    ValueEncoding.PLAIN: "PLAIN",
}
"""The Parquet ``column_encoding`` name for each value encoding. Dictionary encoding is not in this
map. pyarrow requests dictionary encoding through ``use_dictionary`` instead. If a column appears in
both lists, pyarrow rejects it."""

_DICTIONARY_MARKERS = frozenset({"RLE_DICTIONARY", "PLAIN_DICTIONARY"})
"""Footer names for dictionary-encoded indices. PLAIN_DICTIONARY is the name used before Parquet
version 2.4."""

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
        value_encoding: The encoding the writer selected for this shard's values column.

    Returns:
        The categorical columns, and the values column if the writer chose dictionary encoding
        for it.
    """
    columns = list(SHARD_CATEGORICAL)
    if value_encoding == ValueEncoding.DICTIONARY:
        columns.append(VALUES_COLUMN)
    return columns


def shard_encoding(value_encoding: ValueEncoding) -> dict[str, str]:
    """Return the shard's explicit per-column encodings.

    Args:
        value_encoding: The encoding the writer selected for this shard's values column.

    Returns:
        The ``column_encoding`` map. This map always pins the monotonic index and the irregular
        series' time offsets. The map has no entry for the values column if the values column
        uses ``use_dictionary`` instead.
    """
    column_encoding = {"chunk_idx": "DELTA_BINARY_PACKED", TIME_OFFSETS_COLUMN: "DELTA_BINARY_PACKED"}
    parquet_name = _PARQUET_COLUMN_ENCODING.get(value_encoding)
    if parquet_name is not None:
        column_encoding[VALUES_COLUMN] = parquet_name
    return column_encoding


def applied_matches(value_encoding: ValueEncoding, applied: set[str]) -> bool:
    """Report whether a footer's encodings for the values column match the requested encoding.

    A dictionary-encoded column chunk lists two encodings: a dictionary marker for its indices,
    and PLAIN for the dictionary page itself. For this reason, PLAIN alone does not prove that a
    column is plain-encoded. PLAIN alone can also mean this: Parquet abandoned a dictionary that
    grew past its page limit. Parquet then finished the chunk in plain encoding. This fallback is
    lossless. The writer does not control this fallback. For this reason, the check accepts
    PLAIN for the dictionary case too.

    Args:
        value_encoding: The encoding the writer selected.
        applied: Encoding names that :func:`values_encoding_of` reads from the finished file.

    Returns:
        ``True`` if the file's encoding matches the request, or Parquet applied an accepted
        fallback.
    """
    if value_encoding is ValueEncoding.DICTIONARY:
        return bool(applied & (_DICTIONARY_MARKERS | {"PLAIN"}))
    if value_encoding is ValueEncoding.BYTE_STREAM_SPLIT:
        return "BYTE_STREAM_SPLIT" in applied
    return "PLAIN" in applied and not (applied & _DICTIONARY_MARKERS)


def task_dictionary(schema: pa.Schema) -> list[str]:
    """Return the categorical columns to dictionary-encode for a task partition.

    Only string columns qualify for dictionary encoding. The ``target`` column has two possible
    types. It is a float for a scalar prediction. It is a list of span structs for a localization.
    For the float type, dictionary encoding gives no benefit. For the list-of-structs type,
    dictionary encoding fails.

    Args:
        schema: The task partition's Arrow schema.

    Returns:
        The subset of categorical task columns present in the schema as strings.
    """
    return [name for name in _TASK_CATEGORICAL if name in schema.names and pa.types.is_string(schema.field(name).type)]


@dataclass(frozen=True)
class ParquetEncoding:
    """Encoding and compression options for one sharded parquet table.

    This class bundles the :func:`parquet_kwargs` inputs into one config object. A sharded-table
    writer can then take this one object, instead of four separate arguments.
    """

    dictionary_columns: list[str]
    compression: str
    compression_level: int
    column_encoding: dict[str, str] | None = None


def parquet_kwargs(
    *,
    dictionary_columns: list[str],
    column_encoding: dict[str, str] | None,
    compression: str,
    compression_level: int,
) -> dict[str, Any]:
    """Build the shared Parquet write options for one file.

    Args:
        dictionary_columns: Columns to dictionary-encode (must not overlap with ``column_encoding``).
        column_encoding: Explicit per-column encodings, for example BYTE_STREAM_SPLIT on the
            values column.
        compression: Codec name (``"zstd"``, ``"snappy"``, ``"none"``).
        compression_level: Compression level, used only for the zstd codec.

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
        # Content-defined chunking aligns data pages to the data's content. For this reason, a
        # dedup backend (for example, Xet) needs to re-store only the chunks that changed. This
        # happens when the writer stores a re-curated or copy-on-write-edited version. This
        # feature needs pyarrow>=21.
        "use_content_defined_chunking": True,
    }


def values_encoding_of(path: str, column_path: str = "values.list.element") -> set[str]:
    """Return the encodings applied to one shard column.

    The writer uses this function as a self-check. The check confirms that pyarrow actually
    applied the configured encoding. If the column path is wrong, pyarrow silently drops the
    encoding request. This silent failure costs the compression benefit, with no error to warn
    the writer.

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

    An encoding choice changes the size of the values column. Whole-file size also includes the
    id, index, and statistics columns. The encoding choice does not change the size of these
    other columns.

    Args:
        path: Path to a shard parquet file.

    Returns:
        The sum of ``total_compressed_size`` for the values column, across all row groups.
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
