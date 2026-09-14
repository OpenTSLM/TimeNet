"""The :class:`ManifestCounts` block: summary statistics computed at write time."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ManifestCounts:
    """Row/entity counts recorded in the manifest for quick inspection without opening the parquet."""

    records: int = 0
    """Total number of records in the dataset."""
    annotations: int = 0
    """Number of distinct annotation payloads, deduplicated on ``(name, value, unit)``."""
    registered_annotations: int = 0
    """Number of attachments of those payloads, summed over the five things an annotation can hang
    off: the dataset, a record, a source, a signal, or a task. One payload attached twice counts
    twice."""
    tasks: dict[str, int] = field(default_factory=dict)
    """Count of tasks keyed by task type. The writer leaves this empty: a task's type is a property of
    its payload, which the control database holds, not of the manifest."""
    time_series_chunks: int = 0
    """Number of time series chunk placements written to parquet."""
    time_series_index_rows: int = 0
    """Number of rows in the time series index."""
    time_series_specs: dict[str, int] = field(default_factory=dict)
    """Count of unique time series keyed by spec type. The writer leaves this empty for the same
    reason as ``tasks``."""
