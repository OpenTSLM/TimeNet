from pydantic import ValidationError
import pytest

from timenet.errors import TimeFValidationError
from timenet.types import ClassificationTask, Label, LabelMode, LabelSet


STAGES = LabelSet(name="stage", labels=(Label(name="W", description="Awake."), "N1", "R"))
FINDINGS = LabelSet(name="findings", labels=("AF", "PVC", "LBBB"), mode=LabelMode.MULTI)


def test_a_plain_string_is_a_label_without_a_description():
    assert STAGES.labels == (Label(name="W", description="Awake."), Label(name="N1"), Label(name="R"))
    assert STAGES.names == ("W", "N1", "R")
    assert (STAGES.description("W"), STAGES.description("R")) == ("Awake.", None)
    with pytest.raises(KeyError, match="does not list 'X'"):
        STAGES.description("X")


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ((), "at least one label"),
        (("W", "W"), "more than once"),
        (("W", Label(name="W", description="Awake.")), "more than once"),
    ],
)
def test_a_label_set_needs_at_least_one_label_and_distinct_names(labels, message):
    with pytest.raises(ValidationError, match=message):
        LabelSet(name="stage", labels=labels)


@pytest.mark.parametrize("targets", [("W",), ("R",)])
def test_a_single_label_target_accepts_one_listed_label(targets):
    STAGES.check_targets(targets)


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        (("X",), "labels not in label set 'stage': \\['X'\\]"),
        ((), "0 labels, but label set 'stage' takes exactly one"),
        (("W", "R"), "2 labels"),
    ],
)
def test_a_single_label_target_rejects_unlisted_labels_and_other_counts(targets, message):
    with pytest.raises(TimeFValidationError, match=message):
        STAGES.check_targets(targets)


@pytest.mark.parametrize("targets", [(), ("PVC",), ("LBBB", "AF")])
def test_a_multi_label_target_accepts_any_distinct_listed_labels(targets):
    FINDINGS.check_targets(targets)


def test_a_multi_label_target_rejects_a_repeated_label():
    with pytest.raises(TimeFValidationError, match="repeated labels of label set 'findings': \\['AF'\\]"):
        FINDINGS.check_targets(("AF", "PVC", "AF"))


def test_a_classification_task_rejects_a_label_set_that_is_not_a_label_set_object():
    with pytest.raises(TimeFValidationError, match="must be a LabelSet or None, got str"):
        ClassificationTask(targets=("W",), label_set="stage")  # ty: ignore[invalid-argument-type]


def test_a_classification_task_rejects_both_a_label_set_and_a_target_schema():
    with pytest.raises(TimeFValidationError, match="both label_set and the deprecated target_schema"):
        ClassificationTask(targets=("W",), label_set=STAGES, target_schema="stage")
