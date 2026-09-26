"""Shared source pin and value contract for SLIP training and evaluation."""

from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.types import TimeSeriesSpec, ureg
from timenet_connectors.sources.hub import hub_snapshot


REPO = "LeoChen085/SlipDataset"
REVISION = "e5e4871a9f376ff5377ed16c598e7a84a160664a"

SERIES = TimeSeriesSpec(
    spec_type="slip_series",
    name="SLIP sensor series",
    unit_value=ureg.dimensionless,
)


def download_slip(cache_dir: Path, *patterns: str) -> Path:
    """Fetch selected files from the pinned SLIP release.

    Returns:
        The local snapshot root.
    """
    return hub_snapshot(REPO, REVISION, cache_dir, patterns)


@lru_cache(maxsize=2)
def nested_row_group(shard: Path, group: int, column: str) -> pa.ChunkedArray:
    """Read one nested sensor column for lazy channel loaders.

    Returns:
        The requested Parquet row group and column.
    """
    return pq.ParquetFile(shard).read_row_groups([group], columns=[column]).column(column)
