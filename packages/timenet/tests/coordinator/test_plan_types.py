from dataclasses import FrozenInstanceError

import pytest

from timenet.coordinator.plan import Plan, Step
from timenet.types.references import DatasetRef


def test_a_step_is_frozen_and_names_its_dependencies():
    step = Step(DatasetRef.parse("a/b@1.0.0"), "copy", depends_on=(DatasetRef.parse("a/c@1.0.0"),))
    with pytest.raises(FrozenInstanceError):
        step.action = "build"  # ty: ignore[invalid-assignment]
    assert step.depends_on == (DatasetRef.parse("a/c@1.0.0"),)


def test_a_plan_holds_its_steps_in_order():
    steps = (Step(DatasetRef.parse("a/b@1.0.0"), "copy"), Step(DatasetRef.parse("a/c@1.0.0"), "build"))
    assert Plan(steps).steps == steps
