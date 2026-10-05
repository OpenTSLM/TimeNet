"""The :class:`ManifestCounts` block: summary statistics computed at write time."""

from pydantic import Field, StrictStr

from timenet.types._model import TimeFModel
from timenet.types.wire import StrictNonNegativeInt


class ManifestCounts(TimeFModel):
    """TimeF entity counts recorded for inspection without opening DuckDB."""

    records: StrictNonNegativeInt = 0
    """Total number of records in the dataset."""
    sources: StrictNonNegativeInt = 0
    """Total number of Sources at every recursion depth."""
    signals: StrictNonNegativeInt = 0
    """Total number of Signals."""
    axes: StrictNonNegativeInt = 0
    """Number of distinct shared TimeAxes."""
    annotation_contents: StrictNonNegativeInt = 0
    """Number of reusable annotation content items."""
    annotation_occurrences: StrictNonNegativeInt = 0
    """Number of annotation attachments across all object types."""
    tasks: dict[StrictStr, StrictNonNegativeInt] = Field(default_factory=dict)
    """Count of tasks keyed by task type."""
    signal_chunks: StrictNonNegativeInt = 0
    """Number of Signal chunk placements in the values plane."""
    signals_by_spec: dict[StrictStr, StrictNonNegativeInt] = Field(default_factory=dict)
    """Signal count keyed by specification type."""
