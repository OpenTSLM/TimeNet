"""The :class:`ManifestCounts` block: summary statistics computed at write time."""

from pydantic import Field, StrictStr

from timenet.types._model import TimeFModel
from timenet.types._wire import NonNegativeInt


class ManifestCounts(TimeFModel):
    """TimeF entity counts recorded for inspection without opening DuckDB."""

    records: NonNegativeInt = 0
    """Total number of records in the dataset."""
    sources: NonNegativeInt = 0
    """Total number of Sources at every recursion depth."""
    signals: NonNegativeInt = 0
    """Total number of Signals."""
    axes: NonNegativeInt = 0
    """Number of distinct shared TimeAxes."""
    annotation_contents: NonNegativeInt = 0
    """Number of reusable annotation content items."""
    annotation_occurrences: NonNegativeInt = 0
    """Number of annotation attachments across all object types."""
    tasks: dict[StrictStr, NonNegativeInt] = Field(default_factory=dict)
    """Count of tasks keyed by task type."""
    signal_chunks: NonNegativeInt = 0
    """Number of Signal chunk placements in the values plane."""
    signals_by_spec: dict[StrictStr, NonNegativeInt] = Field(default_factory=dict)
    """Signal count keyed by specification type."""
