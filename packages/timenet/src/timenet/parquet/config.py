"""Tuning constants for the Parquet values plane."""

DEFAULT_PARQUET_COMPRESSION_LEVEL = 19
"""Default zstd level for Parquet shards. Write-once data reads faster than it writes, so pay the
higher level. Levels above 19 add nothing because pyarrow's zstd binding does not enable
long-distance matching."""
