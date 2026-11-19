import pickle

import pytest

from timenet.errors import TimeFValidationError
from timenet.types import Annotation, AnswerTask, ClassificationTask, Label, LabelSet, TimePoint


STAGES = LabelSet(labels=(Label(name="W", description="Awake."), Label(name="N1"), Label(name="R")))
FINDINGS = LabelSet(labels=(Label(name="AF"), Label(name="PVC"), Label(name="LBBB")), multi_label=True)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"labels": ()}, "at least one label"),
        ({"labels": None}, "takes a list of Label items, got NoneType"),
        ({"labels": 3}, "takes a list of Label items, got int"),
        ({"labels": "WR"}, "takes a list of Label items, got str 'WR'"),
        ({"labels": {"W": "Wake."}}, "takes a list of Label items, got dict"),
        ({"labels": Label(name="W")}, "takes a list of Label items, got Label"),
        ({"labels": ("W", "")}, "cannot have an empty label"),
        ({"labels": (Label(name="W"),), "annotations": None}, "takes a list of Annotation items, got NoneType"),
        ({"labels": (Label(name="W"), Label(name="W", description="Awake."))}, "more than once"),
        ({"labels": (3,)}, "takes Label objects, got int"),
        ({"labels": ("W", "W")}, "more than once"),
        ({"labels": (Label(name="W"),), "multi_label": 1}, "multi_label must be a bool, got int"),
        ({"labels": (Label(name="W"),), "annotations": ("name",)}, "takes Annotation objects, got str"),
    ],
)
def test_a_label_set_rejects_invalid_fields(kwargs, message):
    with pytest.raises(TimeFValidationError, match=message):
        LabelSet(**kwargs)


@pytest.mark.parametrize(("label_set", "type_name"), [(None, "NoneType"), ("stage", "str")])
def test_a_classification_task_requires_a_label_set(label_set, type_name):
    with pytest.raises(TimeFValidationError, match=f"label_set must be a LabelSet, got {type_name}"):
        ClassificationTask(targets=("W",), label_set=label_set)


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        (("X",), "labels not in label set \\['W', 'N1', 'R'\\]: \\['X'\\]"),
        ((), "0 labels, but label set \\['W', 'N1', 'R'\\] takes exactly one"),
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
    with pytest.raises(TimeFValidationError, match=r"repeated labels of label set \['AF', 'PVC', 'LBBB'\]: \['AF'\]"):
        FINDINGS.check_targets(("AF", "PVC", "AF"))


def test_equality_ignores_label_order_and_annotations_but_not_descriptions_or_multi_label():
    first = LabelSet(labels=(Label(name="F"), Label(name="M")), annotations=(Annotation(key="name", value="sex"),))
    second = LabelSet(labels=(Label(name="M"), Label(name="F")))
    described = LabelSet(labels=(Label(name="F", description="Female."), Label(name="M")))
    multi = LabelSet(labels=(Label(name="F"), Label(name="M")), multi_label=True)
    assert first._content_key() == second._content_key()
    assert len({first._content_key(), described._content_key(), multi._content_key()}) == 3
    assert first == LabelSet(labels=(Label(name="F"), Label(name="M")))
    assert first == second
    assert first != described
    assert first != multi


def test_annotate_attaches_an_occurrence_and_rejects_a_span():
    annotation = Annotation(key="name", value="sex")
    built = LabelSet(labels=(Label(name="F"),), annotations=(annotation,))
    assert built.annotations == (annotation,)
    assert built.annotations[0] is not annotation
    label_set = LabelSet(labels=(Label(name="F"),))
    attached = label_set.annotate(annotation)
    assert label_set.annotations == (attached,)
    assert attached is not annotation
    with pytest.raises(TimeFValidationError, match="has a span"):
        label_set.annotate(Annotation(key="x", span=TimePoint.seconds(1.0)))
    with pytest.raises(TimeFValidationError, match="has a span"):
        LabelSet(labels=(Label(name="F"),), annotations=(Annotation(key="x", span=TimePoint.seconds(1.0)),))


def test_a_classification_task_walks_its_annotations_then_those_of_its_label_set():
    task = ClassificationTask(
        targets=("W",), label_set=LabelSet(["W", "R"], annotations=(Annotation(key="set", value="s"),))
    )
    task.annotate(Annotation(key="task", value="t"))
    assert [annotation.key for annotation in task.walk_annotations()] == ["task", "set"]
    answer = AnswerTask(targets=("x",))
    answer.annotate(Annotation(key="answer", value="a"))
    assert [annotation.key for annotation in answer.walk_annotations()] == ["answer"]


def test_a_copied_task_gets_its_own_label_set():
    label_set = LabelSet(labels=(Label(name="W"),), annotations=(Annotation(key="name", value="stage"),))
    task = ClassificationTask(targets=("W",), label_set=label_set)
    copy = task.with_references({}, {})
    assert copy.label_set == label_set
    assert copy.label_set is not label_set
    assert copy.label_set.annotations == label_set.annotations
    assert copy.label_set.annotations[0] is not label_set.annotations[0]
    copy.label_set.annotate(Annotation(key="note", value="child"))
    assert [a.key for a in label_set.annotations] == ["name"]


def test_a_task_with_an_annotated_set_survives_a_pickle_round_trip():
    label_set = LabelSet(
        labels=(Label(name="W", description="Awake."),),
        multi_label=True,
        annotations=(Annotation(key="name", value="stage"),),
    )
    task = ClassificationTask(targets=("W",), label_set=label_set)
    restored = pickle.loads(pickle.dumps(task)).label_set
    assert restored == label_set
    assert [(a.key, a.value) for a in restored.annotations] == [("name", "stage")]


def test_a_set_takes_its_labels_as_plain_strings_or_label_objects():
    label_set = LabelSet(["W", Label(name="R", description="REM sleep.")], multi_label=True)
    assert label_set.labels == (Label(name="W"), Label(name="R", description="REM sleep."))
    assert label_set.multi_label is True
    assert LabelSet(labels=("W", "R")) == LabelSet([Label(name="W"), Label(name="R")])
