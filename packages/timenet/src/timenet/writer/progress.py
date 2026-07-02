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
    completed: int
    total: int | None
    message: str | None = None
