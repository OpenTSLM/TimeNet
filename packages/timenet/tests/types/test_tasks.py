import pytest

from timenet.errors import TimeFValidationError
from timenet.types import (
    AnswerTask,
    ClassificationTask,
    ForecastingTask,
    LocalizationMode,
    ScalarPredictionTask,
    Span,
    Task,
    TaskType,
    TemporalLocalizationTask,
    TSCorrespondenceTask,
    TSEditingTask,
    TSGenerationTask,
    ureg,
)
from timenet.types.tasks import _build_task_registry


def test_task_types():
    assert ClassificationTask.task_type is TaskType.CLASSIFICATION
    assert AnswerTask.task_type is TaskType.ANSWER
    assert ScalarPredictionTask.task_type is TaskType.SCALAR_PREDICTION
    assert TemporalLocalizationTask.task_type is TaskType.TEMPORAL_LOCALIZATION
    assert ForecastingTask.task_type is TaskType.FORECASTING
    assert TSEditingTask.task_type is TaskType.TS_EDITING
    assert TSGenerationTask.task_type is TaskType.TS_GENERATION
    assert TSCorrespondenceTask.task_type is TaskType.TS_CORRESPONDENCE


def test_series_output_tasks_answer_with_a_sample():
    for cls in (ForecastingTask, TSEditingTask, TSGenerationTask):
        assert cls.answer_is_sample
    for cls in (ClassificationTask, AnswerTask, ScalarPredictionTask, TemporalLocalizationTask):
        assert not cls.answer_is_sample


def test_classification_labels_the_whole_sample_or_a_scope():
    whole = ClassificationTask(target="afib", target_schema="rhythm")
    assert whole.scope is None
    scoped = ClassificationTask(target="N2", scope=Span(start_s=30.0, end_s=60.0))
    assert scoped.scope is not None and not scoped.scope.is_point


def test_answer_task_is_a_caption_without_a_prompt_and_reasoning_with_a_rationale():
    caption = AnswerTask(target="A 10 s trace with rising amplitude.")
    assert caption.prompt is None and caption.rationale is None
    reasoned = AnswerTask(prompt="Bearing fault?", target="Yes.", rationale="The impact repeats.")
    assert reasoned.rationale == "The impact repeats."


def test_scalar_prediction_keeps_the_number_typed_and_normalizes_its_unit():
    task = ScalarPredictionTask(target=62.0, unit=ureg.bpm, target_name="mean_heart_rate")
    assert task.target == pytest.approx(62.0)
    assert task.unit == "bpm"  # a pint unit is stored as its canonical name


def test_scalar_prediction_rejects_an_unknown_unit():
    with pytest.raises(ValueError, match="unknown unit"):
        ScalarPredictionTask(target=1.0, unit="not_a_unit")


def test_localization_target_holds_points_and_intervals():
    task = TemporalLocalizationTask(
        prompt="Locate all R-peaks.",
        target=(Span(start_s=1.2, time_series_ids=("II",)), Span(start_s=2.0, end_s=2.5)),
    )
    assert task.mode is LocalizationMode.SPARSE  # sparse by default: unmarked time is unlabeled
    assert task.target is not None
    assert [span.is_point for span in task.target] == [True, False]


def test_localization_mode_is_coerced_so_a_read_back_task_compares_equal():
    # The reader passes the raw string read from the partition, so the enum is restored on construction.
    task = TemporalLocalizationTask(target=(Span(start_s=0.0),), mode="exhaustive")  # ty: ignore[invalid-argument-type]
    assert task.mode is LocalizationMode.EXHAUSTIVE


def test_localization_rejects_an_explicitly_empty_target():
    with pytest.raises(TimeFValidationError, match="non-empty"):
        TemporalLocalizationTask(prompt="Locate all R-peaks.", target=())


def test_series_output_payloads():
    assert ForecastingTask(context_sample_ids=("c1",), target_sample_id="t1").target_sample_id == "t1"
    edit = TSEditingTask(prompt="Denoise it.", source_sample_id="s1", target_sample_id="t1")
    assert edit.source_sample_id == "s1"
    assert TSGenerationTask(prompt="10 s of sinus rhythm.", target_sample_id="t1").target is None


def test_correspondence_answer_must_come_from_the_candidate_pool():
    task = TSCorrespondenceTask(
        prompt="Which trace is most similar?", candidate_sample_ids=("s1", "s2"), target=("s2",)
    )
    assert task.target == ("s2",)
    with pytest.raises(TimeFValidationError, match="must be one of the candidates"):
        TSCorrespondenceTask(candidate_sample_ids=("s1",), target=("s9",))


def test_correspondence_allows_an_unconstrained_pool():
    assert TSCorrespondenceTask(target=("s9",)).candidate_sample_ids == ()


def test_spans_collects_scope_and_span_valued_payload():
    scope = Span(start_s=0.0, end_s=1.0)
    assert ClassificationTask(target="a", scope=scope).spans() == (scope,)
    assert ClassificationTask(target="a").spans() == ()
    target = (Span(start_s=1.0), Span(start_s=2.0))
    localization = TemporalLocalizationTask(target=target, scope=scope)
    assert localization.spans() == (scope, *target)


def test_auto_id_unique():
    assert ClassificationTask(target="a").id != ClassificationTask(target="a").id


def test_from_task_ids_property():
    base = ClassificationTask(target="a")
    derived = AnswerTask(prompt="q", target="a", from_tasks=(base,))
    assert derived.from_task_ids == (base.id,)


def test_task_is_mutable_for_post_construction_linking():
    # add_tasks() populates sample_ids after construction, so Task must be mutable.
    t = ClassificationTask(target="a")
    t.sample_ids = ("sample-0",)
    assert t.sample_ids == ("sample-0",)


def test_base_task_has_no_task_type():
    with pytest.raises(AttributeError):
        _ = Task.task_type


def test_registry_rejects_task_type_collision():
    # Two classes claiming the same task_type would otherwise silently drop one from the registry.
    with pytest.raises(ValueError, match="task_type"):
        _build_task_registry([ClassificationTask, ClassificationTask])
