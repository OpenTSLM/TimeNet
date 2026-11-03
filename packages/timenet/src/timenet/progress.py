"""Typed progress events for copies and builds.

A function that copies or builds releases takes an optional :data:`ProgressCallback` and sends it
these events. It never prints. A front end, such as a CLI, decides how to show them.

A step is the copy or the build of one release. Each step sends :class:`StepStarted` first and
:class:`StepFinished` when it is done. A step that fails sends no :class:`StepFinished`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from timenet.types.references import DatasetRef


@dataclass(frozen=True)
class StepStarted:
    """A step started to copy or build one release.

    Attributes:
        release: The release that the step copies or builds.
        action: ``"copy"`` or ``"build"``.
    """

    release: DatasetRef
    action: Literal["copy", "build"]


@dataclass(frozen=True)
class StepFinished:
    """A step finished the copy or the build of one release.

    Attributes:
        release: The release that the step copied or built.
        action: ``"copy"`` or ``"build"``.
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
"""A function that receives each progress event."""
