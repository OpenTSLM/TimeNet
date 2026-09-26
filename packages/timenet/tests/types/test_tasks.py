from dataclasses import dataclass
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


NO_INPUT = frozenset({InputModality.NO_INPUT})


def test_task_is_an_abstract_base_class():
    with pytest.raises(TypeError):
        cast("Any", Task)(input_modalities=NO_INPUT)


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
    record = Record(record_id="target-record")
    point = TimePoint.seconds(5)

    task = task_type(targets=["answer", record, 0.98, point], input_modalities=NO_INPUT)

    assert task.targets == ("answer", record, 0.98, point)


def test_none_and_an_explicit_empty_target_remain_distinct():
    assert AnswerTask(input_modalities=NO_INPUT).targets is None
    assert AnswerTask(targets=cast("Any", []), input_modalities=NO_INPUT).targets == ()


def test_task_rejects_an_unsupported_target_item():
    with pytest.raises(TimeFValidationError, match="unsupported type"):
        AnswerTask(targets=cast("Any", ({"not": "a target"},)), input_modalities=NO_INPUT)


def test_spans_collects_scope_and_span_targets_in_order():
    scope = TimeInterval.seconds(0, 10)
    point = TimePoint.seconds(5)
    task = AnswerTask(scope=scope, targets=("event", point), input_modalities=NO_INPUT)

    assert task.spans() == (scope, point)


def test_scalar_prediction_normalizes_a_typed_unit():
    task = ScalarPredictionTask(targets=(62.0,), unit=ureg.bpm, target_name="mean_rate", input_modalities=NO_INPUT)

    assert task.unit == "bpm"


def test_scalar_prediction_rejects_an_unknown_unit():
    with pytest.raises(TimeFValidationError, match="unknown unit"):
        ScalarPredictionTask(targets=(1.0,), unit="not_a_unit", input_modalities=NO_INPUT)


def test_localization_mode_is_normalized():
    task = TemporalLocalizationTask(
        targets=(TimePoint.seconds(1),),
        mode="exhaustive",  # ty: ignore[invalid-argument-type]
        input_modalities=NO_INPUT,
    )

    assert task.mode is LocalizationMode.EXHAUSTIVE


def test_correspondence_record_targets_must_come_from_a_bounded_pool():
    candidate = Record(record_id="candidate")
    outside = Record(record_id="outside")

    with pytest.raises(TimeFValidationError, match="not in candidate_records"):
        TSCorrespondenceTask(candidate_records=(candidate,), targets=(outside,), input_modalities=NO_INPUT)


def test_task_ids_are_unique_by_default():
    assert (
        AnswerTask(targets=("a",), input_modalities=NO_INPUT).id
        != AnswerTask(targets=("a",), input_modalities=NO_INPUT).id
    )


def test_task_input_modalities_require_a_declaration():
    with pytest.raises(TimeFValidationError, match="must declare"):
        AnswerTask(input_modalities=frozenset())
    with pytest.raises(TimeFValidationError, match="non-empty collection"):
        AnswerTask(input_modalities=cast("Any", None))
    task = AnswerTask(
        inputs=(Record(record_id="input"),),
        prompt="What happened?",
        input_modalities=frozenset({InputModality.TEXT, InputModality.TIME_SERIES}),
    )
    assert task.input_modalities == frozenset({InputModality.TEXT, InputModality.TIME_SERIES})

    with pytest.raises(TimeFValidationError, match="does not declare text"):
        AnswerTask(
            inputs=(Record(record_id="input"),),
            prompt="What happened?",
            input_modalities=frozenset({InputModality.TIME_SERIES}),
        )

    prompt_only = AnswerTask(
        prompt="Describe a series", input_modalities=frozenset({InputModality.TEXT, InputModality.NO_INPUT})
    )
    assert prompt_only.input_modalities == frozenset({InputModality.TEXT, InputModality.NO_INPUT})


def test_registry_rejects_a_task_type_collision():
    @dataclass(kw_only=True)
    class DuplicateAnswer(Task):
        task_type: ClassVar[TaskType] = TaskType.ANSWER

    with pytest.raises(TimeFValidationError, match="claim task_type"):
        _build_task_registry([AnswerTask, DuplicateAnswer])
