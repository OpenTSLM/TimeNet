from dataclasses import dataclass, replace
import pickle
from typing import Any, ClassVar, cast

import pytest

from timenet.dataset import OrdinalAxis, Record, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.testing import make_dataset
from timenet.types import (
    TASKS,
    Annotation,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    ForecastingTask,
    InputModality,
    License,
    LocalizationMode,
    ScalarPredictionTask,
    StepInterval,
    StepPoint,
    Task,
    TaskType,
    TemporalLocalizationTask,
    TimeInterval,
    TimePoint,
    TimeSeriesSpec,
    TSCorrespondenceTask,
    TSEditingTask,
    TSGenerationTask,
    Version,
    ureg,
)
from timenet.types.tasks import _build_task_registry
from timenet.types.text import InlineText


NO_INPUT = frozenset({InputModality.NO_INPUT})


def test_only_a_task_class_that_sets_task_type_can_be_built():
    with pytest.raises(TimeFValidationError, match="Task sets no task_type"):
        Task(inputs=())
    assert AnswerTask().task_type is TaskType.ANSWER


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


def _identities(objects):
    """Return the ``id()`` of each object, so that a comparison tells equal objects apart."""
    return [id(item) for item in objects]


def _signal(name):
    spec = TimeSeriesSpec(spec_type="x", name="X", unit_value=ureg.dimensionless)
    return Signal.from_values([0.0, 1.0, 2.0], spec=spec, name=name, time_axis=OrdinalAxis())


def _record(name):
    return Record(sources=(Source(name="Source", signals=(_signal(name),)),))


def _registered(*names):
    """Return a Record with one Signal for each name, registered in a dataset, so they have IDs."""
    metadata = DatasetMetadata(
        dataset_id="test/tasks",
        dataset_version=Version(1, 0, 0),
        name="Tasks",
        description="test",
        license=License.CC_BY_4_0,
    )
    dataset = TimeFDataset(metadata=metadata)
    return tuple(dataset.add_record(record=_record(name)) for name in names)


def _replacements(records):
    """Return a new Record for each Record and its new Signal for each Signal, by the ID of the original."""
    new = {record.id: _record(record.signals[0].name) for record in records}
    return new, {record.signals[0].id: new[record.id].signals[0] for record in records}


def test_a_task_names_its_inputs_then_its_record_targets_each_once():
    first, second, third = Record(), Record(), Record()
    task = TSEditingTask(inputs=(first, second), targets=(third, "denoised", second, _signal("s"), first))

    assert _identities(task.named_records()) == _identities((first, second, third))
    assert AnswerTask(targets=("no records",)).named_records() == ()


def test_a_correspondence_task_names_its_candidates_after_its_inputs():
    record, first, second = Record(), Record(), Record()
    bounded = TSCorrespondenceTask(inputs=(record,), candidate_records=(first, second, record), targets=(second,))
    # Without candidate Records, the pool is the whole dataset, and a Record target comes after the inputs.
    unbounded = TSCorrespondenceTask(inputs=(record,), targets=(second,))

    assert _identities(bounded.named_records()) == _identities((record, first, second))
    assert _identities(unbounded.named_records()) == _identities((record, second))


def test_a_task_references_its_signal_targets_then_the_signals_of_its_spans_each_once():
    first, second, third, fourth = (_signal(name) for name in "abcd")
    task = TemporalLocalizationTask(
        scope=TimeInterval.seconds(0.0, 1.0, time_series=(third, first)),
        targets=(
            first,
            StepInterval(time_series=second, start=0, stop=2),
            "event",
            TimePoint.seconds(0.5),
            StepPoint(time_series=fourth, start=1),
            second,
        ),
    )

    assert _identities(task.referenced_signals()) == _identities((first, second, third, fourth))
    assert AnswerTask(targets=("a",), scope=TimeInterval.seconds(0.0, 1.0)).referenced_signals() == ()


def test_a_copy_with_references_points_at_the_records_and_signals_of_the_same_ids():
    record, target, holder = _registered("a", "b", "c")
    records, signals = _replacements((record, target, holder))
    context = record.annotate(Annotation(key="context", value="ctx"))
    base = AnswerTask(targets=("base",))
    scope = TimeInterval.seconds(0.0, 0.002, time_series=record.signals)
    task = TSEditingTask(
        inputs=(record,),
        prompt="Denoise the input.",
        targets=(target, holder.signals[0], "denoised", 0.5),
        scope=scope,
        input_annotations=(context,),
        from_tasks=(base,),
        metadata={"origin": "test"},
    )
    own = task.annotate(Annotation(key="quality", value="good"))
    task._id = 7

    copy = task.with_references(records, signals)

    assert type(copy) is TSEditingTask
    assert copy._id is None
    assert _identities(copy.inputs) == _identities((records[record.id],))
    assert copy.targets is not None
    new_target, new_signal, *values = copy.targets
    assert _identities((new_target, new_signal)) == _identities((records[target.id], signals[holder.signals[0].id]))
    assert values == ["denoised", 0.5]
    assert type(copy.scope) is TimeInterval
    assert isinstance(copy.scope, TimeInterval)
    assert (copy.scope.start_us, copy.scope.end_us) == (scope.start_us, scope.end_us)
    assert _identities(copy.scope.time_series or ()) == _identities((signals[record.signals[0].id],))
    # The other fields keep their values. The caller changes them when it needs to.
    assert _identities((*copy.annotations, *copy.input_annotations, *copy.from_tasks)) == _identities(
        (own, context, base)
    )
    assert (copy.prompt, copy.metadata) == ("Denoise the input.", {"origin": "test"})
    # The task itself does not change.
    assert _identities((*task.inputs, *(task.targets or ())[:2], *(scope.time_series or ()))) == _identities(
        (record, target, holder.signals[0], *record.signals)
    )


def test_a_copy_with_references_of_a_correspondence_task_points_its_candidates_at_the_same_ids():
    record, first, second = _registered("a", "b", "c")
    records, signals = _replacements((record, first, second))
    task = TSCorrespondenceTask(inputs=(record,), candidate_records=(first, second), targets=(second,))

    copy = task.with_references(records, signals)

    assert type(copy) is TSCorrespondenceTask
    assert _identities(copy.candidate_records) == _identities((records[first.id], records[second.id]))
    assert _identities(copy.targets or ()) == _identities((records[second.id],))


def test_a_copy_with_references_keeps_the_class_of_each_span():
    first, second = _registered("a", "b")
    records, signals = _replacements((first, second))
    a, b = first.signals[0], second.signals[0]
    whole = TimeInterval.seconds(0.0, 0.002)
    task = TemporalLocalizationTask(
        inputs=(first, second),
        scope=StepInterval(time_series=a, start=0, stop=3),
        targets=(
            StepPoint(time_series=a, start=1),
            StepInterval(time_series=b, start=0, stop=2),
            TimePoint.micros(1_000, time_series=(a, b)),
            whole,
        ),
    )

    copy = task.with_references(records, signals)

    assert copy.targets is not None
    scope, (point, steps, instant, same) = copy.scope, copy.targets
    assert [type(span) for span in (scope, point, steps, instant)] == [StepInterval, StepPoint, StepInterval, TimePoint]
    assert isinstance(scope, StepInterval)
    assert isinstance(point, StepPoint)
    assert isinstance(steps, StepInterval)
    assert isinstance(instant, TimePoint)
    assert (scope.start, scope.stop, point.start, steps.start, steps.stop, instant.start_us) == (0, 3, 1, 0, 2, 1_000)
    named = (scope.time_series, point.time_series, steps.time_series, *(instant.time_series or ()))
    assert _identities(named) == _identities(
        (signals[a.id], signals[a.id], signals[b.id], signals[a.id], signals[b.id])
    )
    # A span that names no Signal stays as it is.
    assert same is whole


def test_a_copy_with_references_needs_an_object_for_each_referenced_id():
    record, other = _registered("a", "b")
    records, signals = _replacements((record,))

    with pytest.raises(TimeFValidationError, match=rf"got no Record for ID {other.id}, which the task references"):
        TSEditingTask(inputs=(record,), targets=(other,)).with_references(records, signals)
    with pytest.raises(TimeFValidationError, match=rf"got no Signal for ID {other.signals[0].id}"):
        TSEditingTask(inputs=(record,), targets=(other.signals[0],)).with_references(records, signals)


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


_TIME_SERIES = frozenset({InputModality.TIME_SERIES})


@pytest.mark.parametrize(
    ("with_record", "prompt", "modalities", "message"),
    [
        (True, None, frozenset(), "must declare an input modality"),
        (True, "Question?", _TIME_SERIES, "has a prompt but does not declare text input"),
        (True, None, NO_INPUT, "cannot combine no_input with record or media inputs"),
        (False, "Question?", frozenset({InputModality.TEXT}), "needs no_input when it has only a text prompt"),
    ],
)
def test_an_inconsistent_input_declaration_names_the_task(with_record, prompt, modalities, message):
    inputs = (Record(),) if with_record else ()
    # A task outside a dataset has no id yet.
    with pytest.raises(TimeFValidationError, match=f"^task {message}$"):
        AnswerTask(inputs=inputs, prompt=prompt, input_modalities=modalities)


def test_an_inconsistent_input_declaration_names_the_task_id():
    task = make_dataset().tasks[0]
    task.input_modalities = _TIME_SERIES
    task.prompt = "Question?"
    with pytest.raises(TimeFValidationError, match=rf"^task {task.id} has a prompt but does not declare text input$"):
        _ = task.resolved_input_modalities


def test_a_cached_valid_declaration_does_not_hide_a_later_error():
    record = Record()
    AnswerTask(inputs=(record,), input_modalities=_TIME_SERIES)
    # Same modalities, now with a prompt: the cache key includes the prompt, so this still fails.
    with pytest.raises(TimeFValidationError, match=r"^task has a prompt but does not declare text input$"):
        AnswerTask(inputs=(record,), prompt="Question?", input_modalities=_TIME_SERIES)
