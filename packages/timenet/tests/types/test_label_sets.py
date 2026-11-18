from pydantic import ValidationError
import pytest

from timenet.errors import TimeFValidationError
from timenet.types import ClassificationTask, Label, LabelSet, LabelSetRegistry


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


def test_a_set_without_a_name_gets_one_from_its_content():
    yes_no = LabelSet(labels=(Label(name="Yes"), Label(name="No")))

    assert yes_no.name.startswith("labels_")
    assert yes_no.name == LabelSet(labels=(Label(name="Yes"), Label(name="No"))).name
    assert LabelSet.model_validate(yes_no.model_dump()) == yes_no


def test_the_derived_name_ignores_the_order_of_the_labels():
    yes_no = LabelSet(labels=(Label(name="Yes"), Label(name="No")))
    no_yes = LabelSet(labels=(Label(name="No"), Label(name="Yes")))

    assert yes_no.name == no_yes.name
    assert yes_no.names == ("Yes", "No")
    assert no_yes.names == ("No", "Yes")


@pytest.mark.parametrize(
    "other",
    [
        LabelSet(labels=(Label(name="Yes"), Label(name="No")), multi_label=True),
        LabelSet(labels=(Label(name="Yes", description="Agreed."), Label(name="No"))),
        LabelSet(labels=(Label(name="Yes"), Label(name="Maybe"))),
    ],
)
def test_the_derived_name_changes_with_the_content(other):
    assert other.name != LabelSet(labels=(Label(name="Yes"), Label(name="No"))).name


def test_a_given_name_is_kept():
    assert LabelSet(name="answer", labels=(Label(name="Yes"),)).name == "answer"


def test_a_registry_reuses_the_first_set_that_has_the_same_content():
    registry = LabelSetRegistry()
    yes_no = LabelSet(labels=(Label(name="Yes"), Label(name="No")))
    no_yes = LabelSet(labels=(Label(name="No"), Label(name="Yes")))

    assert registry.register_or_reuse_matching_label_set(yes_no) is yes_no
    assert registry.register_or_reuse_matching_label_set(no_yes) is yes_no
    assert list(registry) == [yes_no]


def test_a_registry_rejects_a_set_with_a_used_name_and_other_content():
    registry = LabelSetRegistry()
    registry.register_or_reuse_matching_label_set(LabelSet(name="stage", labels=(Label(name="W"), Label(name="R"))))

    with pytest.raises(TimeFValidationError, match="'stage' differs from another set"):
        registry.register_or_reuse_matching_label_set(
            LabelSet(name="stage", labels=(Label(name="W"), Label(name="N1")))
        )
