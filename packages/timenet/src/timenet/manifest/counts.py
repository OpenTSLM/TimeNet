"""The :class:`ManifestCounts` block: summary statistics computed at write time."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ManifestCounts:
    """Row/entity counts recorded in the manifest for quick inspection without opening the parquet."""

    samples: int = 0
    annotations: int = 0
    tasks: dict[str, int] = field(default_factory=dict)
    time_series_chunks: int = 0
    time_series_index_rows: int = 0
    time_series_specs: dict[str, int] = field(default_factory=dict)
