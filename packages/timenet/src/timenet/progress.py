"""Progress events for copying and building releases.

Each release starts with a :class:`StepStarted` event.
A :class:`StepFinished` event follows only if the copy or build succeeds.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from timenet.types.references import DatasetRef


@dataclass(frozen=True)
class StepStarted:
    """The start of a copy or build for one release.

    Attributes:
        release: The release to copy or build.
        action: The operation, either ``"copy"`` or ``"build"``.
    """

    release: DatasetRef
    action: Literal["copy", "build"]


@dataclass(frozen=True)
class StepFinished:
    """The successful completion of a copy or build for one release.

    Attributes:
        release: The copied or built release.
        action: The completed operation, either ``"copy"`` or ``"build"``.
    """

    release: DatasetRef
    action: Literal["copy", "build"]


@dataclass(frozen=True)
class BytesMoved:
    """The byte count of a transfer in progress.

    Attributes:
        url: The source or target of the bytes, or the release of a copy (``"org/name@1.0.0"``).
        done: The number of bytes moved so far.
        total: The transfer size in bytes, or ``None`` if unknown.
    """

    url: str
    done: int
    total: int | None


@dataclass(frozen=True)
class ShardWritten:
    """The number of completed shards in a release build.

    Attributes:
        release: The release being built.
        count: The number of shards written so far.
    """

    release: DatasetRef
    count: int


ProgressEvent = StepStarted | StepFinished | BytesMoved | ShardWritten
"""A step, transfer, or shard progress event."""

ProgressCallback = Callable[[ProgressEvent], None]
"""A callback that receives a progress event and returns nothing."""
