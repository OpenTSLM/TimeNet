"""Typed progress events for copies and builds.

A function that copies or builds releases sends these events to an optional
:data:`ProgressCallback` and prints nothing. Each step, the copy or the build of one release, sends
:class:`StepStarted` first and :class:`StepFinished` only if it succeeds.
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


ProgressEvent = StepStarted | StepFinished
"""One progress event."""

ProgressCallback = Callable[[ProgressEvent], None]
"""A function that receives each progress event."""
