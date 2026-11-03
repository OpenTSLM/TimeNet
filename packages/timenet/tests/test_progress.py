from dataclasses import FrozenInstanceError

import pytest

from timenet.progress import StepStarted
from timenet.types.references import DatasetRef


def test_progress_events_are_frozen():
    event = StepStarted(DatasetRef.parse("a/b@1.0.0"), "copy")
    with pytest.raises(FrozenInstanceError):
        event.action = "build"  # ty: ignore[invalid-assignment]
