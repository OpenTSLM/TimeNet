from pydantic import ValidationError
import pytest

from timenet.errors import TimeFValidationError
from timenet.types import Label, LabelSet


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
