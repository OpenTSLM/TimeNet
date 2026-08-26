"""Configuration for the Parquet values writer."""

from dataclasses import dataclass
from pathlib import Path

from timenet.format.schemas import IdCodec, IdTypes
from timenet.writer.value_encoding import ValueEncoding


@dataclass(frozen=True)
class ParquetValuesConfig:
    """Typed construction options for the Parquet values backend."""

    staging_dir: Path
    """Version staging directory. The writer puts shards beneath it."""
    id_types: IdTypes
    """Resolved logical-id storage types."""
    codec: IdCodec
    """Shared logical-id codec that the rest of the TimeF writer uses."""
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
    data_page_size: int | None = None
    """Target uncompressed bytes per data page, or ``None`` for pyarrow's default."""
    value_encoding: ValueEncoding | None = None
    """Forced values-column encoding for every modality, or ``None`` to select one per modality."""
