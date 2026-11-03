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
    """Bytes of a transfer moved.

    Attributes:
        url: Where the bytes come from or go to.
        done: The number of bytes moved so far.
        total: The number of bytes of the whole transfer, or ``None`` if it is not known.
    """

    url: str
    done: int
    total: int | None


@dataclass(frozen=True)
class ShardWritten:
    """A build wrote shards of a release.

    Attributes:
        release: The release that the build writes.
        count: The number of shards written so far.
    """

    release: DatasetRef
    count: int


ProgressEvent = StepStarted | StepFinished | BytesMoved | ShardWritten
"""One progress event."""

ProgressCallback = Callable[[ProgressEvent], None]
"""A callback that receives a progress event and returns nothing."""
