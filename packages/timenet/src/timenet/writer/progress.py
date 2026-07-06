"""Writer progress events."""

from dataclasses import dataclass
from enum import StrEnum


class ProgressStage(StrEnum):
    """The writer's incremental stages. Only stages that carry information are emitted."""

    TIME_SERIES = "time_series"
    SHARD_FINALIZED = "shard_finalized"
    COMMIT = "commit"


@dataclass(frozen=True)
class WriteProgressEvent:
    """A single progress event passed to the writer's ``progress_cb``."""

    stage: ProgressStage
    """The writer stage this event reports."""
    completed: int
    """Number of items finished in this stage so far."""
    total: int | None
    """Total number of items to process, or ``None`` if unknown."""
    message: str | None = None
    """Optional human-readable detail for the event."""
