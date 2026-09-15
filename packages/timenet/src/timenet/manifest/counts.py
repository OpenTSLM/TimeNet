"""The :class:`ManifestCounts` block: summary statistics computed at write time."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ManifestCounts:
    """Row counts recorded in the manifest, so a quick inspection does not open the version."""

    records: int = 0
    """Total number of records in the dataset."""
    annotations: int = 0
    """Number of unique annotation ids across all records."""
    registered_annotations: int = 0
    """Number of task-referenced annotations that no record carries. When there are none, the reader
    skips the query that recovers them."""
    tasks: dict[str, int] = field(default_factory=dict)
    """Count of tasks keyed by task type."""
    time_series_chunks: int = 0
    """Number of time series chunk placements the values plane wrote."""
    time_series_index_rows: int = 0
    """Chunk placements counted once per referencing record. A series shared by two records counts
    its chunks twice, so this is more than ``time_series_chunks`` whenever a version shares a series."""
    time_series_specs: dict[str, int] = field(default_factory=dict)
    """Count of unique time series keyed by spec type."""
