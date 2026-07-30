"""Configuration for the Parquet values writer."""

from dataclasses import dataclass
from pathlib import Path

from timenet.format.schemas import IdCodec, IdTypes


@dataclass(frozen=True)
class ParquetValuesConfig:
    """Typed construction options for the Parquet values backend."""

    staging_dir: Path
    """Version staging directory; shards are written beneath it."""
    id_types: IdTypes
    """Resolved logical-id storage types."""
    codec: IdCodec
    """Shared logical-id codec used by the rest of the TimeF writer."""
    shard_target_bytes: int
    """Target size for rotating shards."""
    row_group_target_bytes: int
    """Target size for flushing row groups."""
    chunk_max_bytes: int
    """Maximum uncompressed values size of one logical chunk."""
    compression: str
    """Parquet compression codec."""
    compression_level: int
    """Parquet compression level."""
