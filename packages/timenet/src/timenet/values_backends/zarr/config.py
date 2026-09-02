"""Configuration for the Zarr values writer."""

from dataclasses import dataclass
from pathlib import Path


DEFAULT_ZARR_COMPRESSION_LEVEL = 9
"""Default Blosc compression level. Blosc accepts clevel 0-9."""


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
    compression_level: int = DEFAULT_ZARR_COMPRESSION_LEVEL
    """Blosc compression level."""
