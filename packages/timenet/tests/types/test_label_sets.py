from pydantic import ValidationError
import pytest

from timenet.errors import TimeFNotFoundError, TimeFValidationError
from timenet.types import ClassificationTask, Label, LabelSet, TextHandle


STAGES = LabelSet(name="stage", labels=(Label(name="W", description="Awake."), Label(name="N1"), Label(name="R")))
FINDINGS = LabelSet(name="findings", labels=(Label(name="AF"), Label(name="PVC"), Label(name="LBBB")), multi_label=True)


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ((), "at least one label"),
        ((Label(name="W"), Label(name="W", description="Awake.")), "more than once"),
    ],
)
def test_a_label_set_needs_at_least_one_label_and_distinct_names(labels, message):
    with pytest.raises(ValidationError, match=message):
        LabelSet(name="stage", labels=labels)


@pytest.mark.parametrize(("label_set", "type_name"), [(None, "NoneType"), ("stage", "str")])
def test_a_classification_task_requires_a_label_set(label_set, type_name):
    with pytest.raises(TimeFValidationError, match=f"label_set must be a LabelSet, got {type_name}"):
        ClassificationTask(targets=("W",), label_set=label_set)


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


@pytest.mark.parametrize("targets", [(), ("LBBB", "AF")])
def test_a_multi_label_target_accepts_any_distinct_listed_labels(targets):
    FINDINGS.check_targets(targets)


def test_a_multi_label_target_rejects_a_repeated_label():
    with pytest.raises(TimeFValidationError, match="repeated labels of label set 'findings': \\['AF'\\]"):
        FINDINGS.check_targets(("AF", "PVC", "AF"))


def test_a_label_set_looks_up_a_label_by_name():
    assert STAGES["W"] == Label(name="W", description="Awake.")
    with pytest.raises(TimeFNotFoundError, match="label set 'stage' has no label 'X'"):
        STAGES["X"]


def test_a_label_target_is_stored_as_its_name():
    task = ClassificationTask(targets=(STAGES["W"],), label_set=STAGES)

    assert task.targets is not None
    (target,) = task.targets
    assert isinstance(target, TextHandle)
    assert target.read() == "W"


def test_task_labels_are_the_set_labels_in_target_order():
    task = ClassificationTask(targets=("LBBB", "AF"), label_set=FINDINGS)

    assert task.labels == (FINDINGS["LBBB"], FINDINGS["AF"])


def test_task_labels_reject_a_target_that_the_set_does_not_list():
    task = ClassificationTask(targets=("X",), label_set=STAGES)
    with pytest.raises(TimeFNotFoundError, match="has no label 'X'"):
        task.labels  # ruff: ignore[useless-expression]
