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


ProgressEvent = StepStarted | StepFinished
"""An event that marks the start or successful end of a copy or build."""

ProgressCallback = Callable[[ProgressEvent], None]
"""A callback that receives a progress event and returns nothing."""
