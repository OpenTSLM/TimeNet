from dataclasses import dataclass, replace
import pickle
from typing import Any, ClassVar, cast

import pytest

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.types import (
    TASKS,
    AnswerTask,
    ClassificationTask,
    ForecastingTask,
    InputModality,
    LocalizationMode,
    ScalarPredictionTask,
    Task,
    TaskType,
    TemporalLocalizationTask,
    TimeInterval,
    TimePoint,
    TSCorrespondenceTask,
    TSEditingTask,
    TSGenerationTask,
    ureg,
)
from timenet.types.tasks import _build_task_registry
from timenet.types.text import InlineText


NO_INPUT = frozenset({InputModality.NO_INPUT})


def test_task_is_an_abstract_base_class():
    with pytest.raises(TypeError):
        cast("Any", Task)()


def test_registry_contains_every_builtin_task_type():
    assert set(TASKS) == set(TaskType)


@pytest.mark.parametrize(
    "task_type",
    (
        AnswerTask,
        ClassificationTask,
        ForecastingTask,
        ScalarPredictionTask,
        TemporalLocalizationTask,
        TSCorrespondenceTask,
        TSEditingTask,
        TSGenerationTask,
    ),
)
def test_every_task_type_accepts_ordered_mixed_targets(task_type):
    record = Record()
    point = TimePoint.seconds(5)

    task = task_type(targets=["answer", record, 0.98, point])

    assert task.targets == ("answer", record, 0.98, point)


def test_none_and_an_explicit_empty_target_remain_distinct():
    assert AnswerTask().targets is None
    assert AnswerTask(targets=cast("Any", [])).targets == ()


def test_task_rejects_an_unsupported_target_item():
    with pytest.raises(TimeFValidationError, match="unsupported type"):
        AnswerTask(targets=cast("Any", ({"not": "a target"},)))


def test_spans_collects_scope_and_span_targets_in_order():
    scope = TimeInterval.seconds(0, 10)
    point = TimePoint.seconds(5)
    task = AnswerTask(scope=scope, targets=("event", point))

    assert task.spans() == (scope, point)


def test_scalar_prediction_normalizes_a_typed_unit():
    task = ScalarPredictionTask(targets=(62.0,), unit=ureg.bpm, target_name="mean_rate")

    assert task.unit == "bpm"


def test_scalar_prediction_rejects_an_unknown_unit():
    with pytest.raises(TimeFValidationError, match="unknown unit"):
        ScalarPredictionTask(targets=(1.0,), unit="not_a_unit")


def test_localization_mode_is_normalized():
    task = TemporalLocalizationTask(
        targets=(TimePoint.seconds(1),),
        mode="exhaustive",  # ty: ignore[invalid-argument-type]
    )

    assert task.mode is LocalizationMode.EXHAUSTIVE


def test_correspondence_record_targets_must_come_from_a_bounded_pool():
    candidate = Record()
    outside = Record()

    with pytest.raises(TimeFValidationError, match="not in candidate_records"):
        TSCorrespondenceTask(candidate_records=(candidate,), targets=(outside,))


def test_registry_rejects_a_task_type_collision():
    # Task refuses subclasses from other modules, so a stand-in class claims the tag instead.
    duplicate = cast("type[Task]", type("DuplicateAnswer", (), {"task_type": TaskType.ANSWER}))

    with pytest.raises(TimeFValidationError, match="both AnswerTask and DuplicateAnswer claim task_type"):
        _build_task_registry([AnswerTask, duplicate])


CONCRETE_TASK_CLASSES = sorted(TASKS.values(), key=lambda cls: cls.__name__)


@pytest.mark.parametrize("base", (Task, *CONCRETE_TASK_CLASSES), ids=lambda cls: cls.__name__)
def test_task_classes_refuse_subclasses_outside_their_module(base):
    # A preset like this one would replace the prompt field that wraps text in a handle.
    with pytest.raises(TypeError, match="AgeQuestion can't subclass a task class"):
        type("AgeQuestion", (base,), {"prompt": "How old is the subject?"})


def test_a_new_task_type_is_refused():
    # The reader maps each stored task_type back to a built-in class, so the set stays closed.
    with pytest.raises(TypeError, match="SummaryTask can't subclass a task class"):

        @dataclass(kw_only=True)
        class SummaryTask(Task):
            task_type: ClassVar[TaskType] = TaskType.ANSWER


@pytest.mark.parametrize("task_class", CONCRETE_TASK_CLASSES, ids=lambda cls: cls.__name__)
def test_every_concrete_task_class_is_final_for_type_checkers(task_class):
    assert getattr(task_class, "__final__", False) is True


def age_question(record: Record, age: int) -> AnswerTask:
    return AnswerTask(inputs=(record,), prompt="How old is the subject?", targets=(str(age),))


def test_a_builder_function_reuses_a_prompt():
    record = Record()
    first, second = age_question(record, 64), age_question(record, 71)

    assert type(first) is AnswerTask
    assert isinstance(first.prompt, InlineText)
    assert first.prompt == second.prompt == "How old is the subject?"
    assert (first.targets, second.targets) == (("64",), ("71",))


def test_replace_and_pickle_keep_a_task_and_its_text_handles():
    task = age_question(Record(), 64)

    renamed = replace(task, prompt="What is the subject's age?")
    assert type(renamed) is AnswerTask
    assert isinstance(renamed.prompt, InlineText)
    assert renamed.prompt.read() == "What is the subject's age?"
    restored = pickle.loads(pickle.dumps(task))
    assert type(restored) is AnswerTask
    assert isinstance(restored.prompt, InlineText)
    assert restored == task
