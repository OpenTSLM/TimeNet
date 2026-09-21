"""The :class:`ManifestCounts` block: summary statistics computed at write time."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ManifestCounts:
    """TimeF entity counts recorded for inspection without opening DuckDB."""

    records: int = 0
    """Total number of records in the dataset."""
    sources: int = 0
    """Total number of Sources at every recursion depth."""
    signals: int = 0
    """Total number of Signals."""
    axes: int = 0
    """Number of distinct shared TimeAxes."""
    annotation_contents: int = 0
    """Number of reusable annotation content items."""
    annotation_occurrences: int = 0
    """Number of annotation attachments across all object types."""
    tasks: dict[str, int] = field(default_factory=dict)
    """Count of tasks keyed by task type."""
    signal_chunks: int = 0
    """Number of Signal chunk placements in the values plane."""
    signals_by_spec: dict[str, int] = field(default_factory=dict)
    """Signal count keyed by specification type."""
