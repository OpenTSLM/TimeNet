"""Configuration for the Zarr values writer."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ZarrValuesConfig:
    """Typed construction options for the Zarr values backend."""

    staging_dir: Path
    """Version staging directory. The backend writes the Zarr store beneath it."""
    shard_target_bytes: int
    """Target size of one Zarr shard."""
    chunk_max_bytes: int
    """Target size of one Zarr storage chunk."""
    compression: str
    """Blosc inner compression codec."""
    compression_level: int
    """Blosc compression level."""
