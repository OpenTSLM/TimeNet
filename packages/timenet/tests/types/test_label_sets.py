from enum import StrEnum

from pydantic import ValidationError
import pytest

from timenet.errors import TimeFValidationError
from timenet.types import ClassificationTask, Label, LabelSet


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


class _Sex(StrEnum):
    FEMALE = "F"
    MALE = "M"


class _Other(StrEnum):
    FEMALE = "F"


def test_a_label_set_from_an_enum_has_one_label_per_member_in_order():
    sexes = LabelSet.from_enum(_Sex, name="sex", descriptions={_Sex.FEMALE: "Female."}, multi_label=True)

    assert sexes == LabelSet(
        name="sex",
        labels=(Label(name="F", description="Female."), Label(name="M")),
        multi_label=True,
    )


@pytest.mark.parametrize("key", [_Other.FEMALE, "F"])
def test_a_label_set_from_an_enum_rejects_a_description_for_another_value(key):
    # Both keys equal _Sex.FEMALE as strings, so without the check they would describe it silently.
    with pytest.raises(TimeFValidationError, match="not members of _Sex"):
        LabelSet.from_enum(_Sex, name="sex", descriptions={key: "Female."})
