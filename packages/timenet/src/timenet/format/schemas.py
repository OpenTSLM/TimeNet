"""Pinned Arrow schema of the Parquet value shards.

Shards hold no public ids. The DuckDB control plane maps each signal to its chunk placements.
"""

import pyarrow as pa


_DEFAULT_VALUES_TYPE = pa.float32()


def shard_schema(value_type: pa.DataType = _DEFAULT_VALUES_TYPE) -> pa.Schema:
    """Return the waveform shard schema.

    Args:
        value_type: The element type of the ``values`` list (defaults to ``float32``).

    Returns:
        The Arrow schema.
    """
    return pa.schema(
        [
            ("spec_type", pa.string()),
            ("signal", pa.string()),
            ("chunk_idx", pa.int32()),
            ("n_values", pa.int32()),
            ("values", pa.list_(value_type)),
            ("time_offsets_us", pa.list_(pa.int64())),
        ]
    )
