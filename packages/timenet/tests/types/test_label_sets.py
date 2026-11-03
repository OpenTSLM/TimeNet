from pydantic import ValidationError
import pytest

from timenet.errors import TimeFValidationError
from timenet.types import ClassificationTask, LabelMode, LabelSet


STAGES = LabelSet(name="stage", labels=("W", "N1", "R"), label_descriptions={"W": "Awake."})
FINDINGS = LabelSet(name="findings", labels=("AF", "PVC", "LBBB"), mode=LabelMode.MULTI)


def test_a_label_set_is_single_label_by_default_and_equal_by_content():
    assert STAGES.mode is LabelMode.SINGLE
    assert LabelSet(name="stage", labels=("W", "N1", "R"), label_descriptions=("Awake.", None, None)) == STAGES
    assert LabelSet(name="stage", labels=("W", "N1", "R"), mode=LabelMode.MULTI) != STAGES


def test_descriptions_are_stored_in_label_order():
    assert STAGES.label_descriptions == ("Awake.", None, None)
    assert FINDINGS.label_descriptions == (None, None, None)
    assert (STAGES.description("W"), STAGES.description("R")) == ("Awake.", None)
    with pytest.raises(KeyError, match="does not list 'X'"):
        STAGES.description("X")


def test_a_shared_label_set_cannot_change_and_can_be_hashed():
    assert hash(STAGES) == hash(STAGES.model_copy())
    assert len({STAGES, STAGES.model_copy(), FINDINGS}) == 2
    with pytest.raises(AttributeError):
        STAGES.label_descriptions.append("x")  # ty: ignore[unresolved-attribute]


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"labels": ()}, "at least one label"),
        ({"labels": ("W", "W")}, "more than once"),
        ({"labels": ("W", "")}, "at least 1 character"),
        ({"labels": ("W",), "label_descriptions": {"R": "REM."}}, "does not list: \\['R'\\]"),
        ({"labels": ("W",), "label_descriptions": {"W": ""}}, "at least 1 character"),
        ({"labels": ("W", "R"), "label_descriptions": ("Awake.",)}, "1 descriptions for 2 labels"),
        ({"labels": ("W",), "mode": "several"}, "mode"),
    ],
)
def test_a_label_set_rejects_an_invalid_definition(fields, message):
    with pytest.raises(ValidationError, match=message):
        LabelSet(name="stage", **fields)


def test_a_label_set_cannot_change_after_construction():
    with pytest.raises(ValidationError, match="frozen"):
        STAGES.labels = ("W",)  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("targets", [("W",), ("R",)])
def test_a_single_label_target_holds_one_label_of_the_set(targets):
    STAGES.check_targets(targets, "task 1")


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        (("X",), "does not list: \\['X'\\]"),
        ((), "has 0 labels, but label set 'stage' takes exactly one"),
        (("W", "R"), "has 2 labels"),
    ],
)
def test_a_single_label_target_rejects_other_labels_and_counts(targets, message):
    with pytest.raises(TimeFValidationError, match=message):
        STAGES.check_targets(targets, "task 1")


@pytest.mark.parametrize("targets", [(), ("PVC",), ("LBBB", "AF")])
def test_a_multi_label_target_holds_any_distinct_labels_of_the_set(targets):
    FINDINGS.check_targets(targets, "task 1")


def test_a_multi_label_target_rejects_a_repeated_label():
    with pytest.raises(TimeFValidationError, match="repeats a label"):
        FINDINGS.check_targets(("AF", "AF"), "task 1")


def test_a_classification_task_takes_only_a_label_set():
    assert ClassificationTask(targets=("W",), label_set=STAGES).label_set is STAGES
    assert ClassificationTask(targets=("W",)).label_set is None
    with pytest.raises(TimeFValidationError, match="must be a LabelSet or None, got str"):
        ClassificationTask(targets=("W",), label_set="stage")  # ty: ignore[invalid-argument-type]
