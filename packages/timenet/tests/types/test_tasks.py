import pytest

from timenet.errors import TimeFValidationError
from timenet.types import (
    AnswerTask,
    ClassificationTask,
    ForecastingTask,
    LocalizationMode,
    ScalarPredictionTask,
    StepInterval,
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
    scoped = ClassificationTask(target="N2", scope=TimeInterval.seconds(30.0, 60.0))
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
        target=(TimePoint.seconds(1.2, time_series_ids=("II",)), TimeInterval.seconds(2.0, 2.5)),
    )
    assert task.mode is LocalizationMode.SPARSE  # sparse by default: unmarked time is unlabeled
    assert task.target is not None
    assert [span.is_point for span in task.target] == [True, False]


def test_localization_mode_is_coerced_so_a_read_back_task_compares_equal():
    # The reader passes the raw string read from the partition, so the enum is restored on construction.
    task = TemporalLocalizationTask(target=(TimePoint.seconds(0.0),), mode="exhaustive")  # ty: ignore[invalid-argument-type]
    assert task.mode is LocalizationMode.EXHAUSTIVE


def test_localization_accepts_an_explicitly_empty_target():
    # An empty target is a positive "searched, found nothing here", distinct from None (the answer is
    # stored by reference in target_annotation_ids).
    task = TemporalLocalizationTask(prompt="Locate all R-peaks.", target=())
    assert task.target == ()


def test_localization_rejects_a_step_framed_target():
    # Localization reports on the recording timeline, so a step-framed target has no place there.
    with pytest.raises(TimeFValidationError, match="step span"):
        TemporalLocalizationTask(
            prompt="Locate the events.",
            target=(StepInterval(time_series_id="s", start=0, stop=4),),  # ty: ignore[invalid-argument-type]
        )


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
    scope = TimeInterval.seconds(0.0, 1.0)
    assert ClassificationTask(target="a", scope=scope).spans() == (scope,)
    assert ClassificationTask(target="a").spans() == ()
    target = (TimePoint.seconds(1.0), TimePoint.seconds(2.0))
    localization = TemporalLocalizationTask(target=target, scope=scope)
    assert localization.spans() == (scope, *target)


def test_auto_id_unique():
    assert ClassificationTask(target="a").id != ClassificationTask(target="a").id


def test_from_task_ids_property():
    base = ClassificationTask(target="a")
    derived = AnswerTask(prompt="q", target="a", from_tasks=(base,))
    assert derived.from_task_ids == (base.id,)


def test_task_is_mutable_for_post_construction_linking():
    # add_task()/add_tasks() populate sample_ids after construction, so Task must be mutable.
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


def test_forecasting_target_span_names_a_region_of_the_attached_sample():
    task = ForecastingTask(target_span=TimeInterval.seconds(132.0, 144.0), scope=TimeInterval.seconds(0.0, 132.0))
    assert task.target_span == TimeInterval.seconds(132.0, 144.0)
    assert task.target_sample_id is None
    assert task.context_sample_ids == ()


def test_forecasting_still_accepts_a_whole_target_sample():
    task = ForecastingTask(context_sample_ids=("c1",), target_sample_id="t1")
    assert task.target_sample_id == "t1"
    assert task.target_span is None


def test_forecasting_predicts_a_single_step_as_a_one_step_interval():
    # One-step-ahead is the most common forecasting protocol, and rejecting point spans must not stand
    # in its way. A point has no duration; the single step covering that point is the interval, the
    # only form that stays unambiguous on a sample holding several rates.
    task = ForecastingTask(
        scope=TimeInterval.seconds(0.0, 143.0),
        target_span=TimeInterval.seconds(143.0, 144.0),
    )
    assert task.target_span == TimeInterval.seconds(143.0, 144.0)  # an interval, not a point


def test_forecasting_target_span_is_exclusive_with_target_sample_id():
    with pytest.raises(TimeFValidationError, match="cannot be combined with target_sample_id"):
        ForecastingTask(target_sample_id="t1", target_span=TimeInterval.seconds(132.0, 144.0))


def test_forecasting_target_span_is_declared_as_a_span_field():
    scope = TimeInterval.seconds(0.0, 132.0)
    task = ForecastingTask(target_span=TimeInterval.seconds(132.0, 144.0), scope=scope)
    assert ForecastingTask.refs.span_fields == ("target_span",)
    assert task.spans() == (scope, TimeInterval.seconds(132.0, 144.0))


def test_forecasting_requires_a_target_sample_id_or_a_target_span():
    with pytest.raises(TimeFValidationError, match="requires either target_sample_id or target_span"):
        ForecastingTask()


def test_forecasting_target_span_requires_a_scope_when_checked():
    # The scope may be supplied at add_task, so construction defers the requirement; the check runs
    # once the scope is final (add_task calls check_against_scope after stamping any scope=).
    task = ForecastingTask(target_span=TimeInterval.seconds(132.0, 144.0))  # constructs, no scope yet
    with pytest.raises(TimeFValidationError, match="needs an explicit scope"):
        task.check_against_scope()


def test_forecasting_target_span_rejects_a_point():
    # A point spans no values, so it names no region to predict. The type says interval, and a point
    # slipped past the annotation is rejected at construction.
    with pytest.raises(TimeFValidationError, match="must be an interval"):
        ForecastingTask(
            scope=TimeInterval.seconds(0.0, 132.0),
            target_span=TimePoint.seconds(132.0),  # ty: ignore[invalid-argument-type]
        )


def test_forecasting_rejects_a_target_sample_that_is_also_its_own_context():
    # The forecast would read its own answer as input.
    with pytest.raises(TimeFValidationError, match="read its own answer as input"):
        ForecastingTask(context_sample_ids=("t1", "c2"), target_sample_id="t1")


def test_forecasting_target_span_rejects_context_sample_ids():
    # The target_span form draws context from scope, so a context sample (which would include the
    # target region) has no place here.
    with pytest.raises(TimeFValidationError, match="context_sample_ids must be empty"):
        ForecastingTask(
            scope=TimeInterval.seconds(0.0, 132.0),
            target_span=TimeInterval.seconds(132.0, 144.0),
            context_sample_ids=("c1",),
        )


def test_forecasting_rejects_an_empty_context_with_a_target_sample():
    # The separate-target-sample form draws its input from context_sample_ids. An empty tuple there
    # is a forecast with no input. The target_span form takes its context from scope, so an empty
    # context_sample_ids is correct in that form.
    with pytest.raises(TimeFValidationError, match="empty context_sample_ids"):
        ForecastingTask(target_sample_id="t1")


def test_forecasting_rejects_a_context_that_overlaps_the_target():
    # scope [0, 140) covers the first 8 s of target [132, 144) on the shared whole-sample series.
    with pytest.raises(TimeFValidationError, match="end at or before the target"):
        ForecastingTask(
            scope=TimeInterval.seconds(0.0, 140.0),
            target_span=TimeInterval.seconds(132.0, 144.0),
        )


def test_forecasting_rejects_a_context_that_covers_the_whole_target():
    with pytest.raises(TimeFValidationError, match="end at or before the target"):
        ForecastingTask(
            scope=TimeInterval.seconds(0.0, 144.0),
            target_span=TimeInterval.seconds(132.0, 144.0),
        )


def test_forecasting_rejects_a_context_that_follows_the_target():
    # This predicts the past from the future. The context sits entirely after the region to predict.
    with pytest.raises(TimeFValidationError, match="end at or before the target"):
        ForecastingTask(
            scope=TimeInterval.seconds(132.0, 144.0),
            target_span=TimeInterval.seconds(0.0, 12.0),
        )


def test_forecasting_allows_a_context_ending_exactly_at_the_target_start():
    # The context's exclusive end can touch the target's inclusive start. [0, 132) leaves 132 to predict.
    task = ForecastingTask(
        scope=TimeInterval.seconds(0.0, 132.0),
        target_span=TimeInterval.seconds(132.0, 144.0),
    )
    assert task.scope == TimeInterval.seconds(0.0, 132.0)


def test_forecasting_allows_context_and_target_on_disjoint_series():
    # A future-known covariate: the context reaches past the target start, but on a different series,
    # so there is nothing to leak.
    task = ForecastingTask(
        scope=TimeInterval.seconds(0.0, 144.0, time_series_ids=("cov",)),
        target_span=TimeInterval.seconds(132.0, 144.0, time_series_ids=("target",)),
    )
    assert task.target_span == TimeInterval.seconds(132.0, 144.0, time_series_ids=("target",))


def test_forecasting_checks_leakage_only_on_series_the_spans_share():
    # scope and target name series sets that overlap. The shared series ("y") must not leak.
    with pytest.raises(TimeFValidationError, match="end at or before the target"):
        ForecastingTask(
            scope=TimeInterval.seconds(0.0, 140.0, time_series_ids=("x", "y")),
            target_span=TimeInterval.seconds(132.0, 144.0, time_series_ids=("y", "z")),
        )


def test_forecasting_names_a_step_horizon_on_a_series_that_has_no_seconds():
    # AirPassengers is 144 monthly points with no fixed second per step; the horizon is the last 12
    # steps, and steps are the only frame a series with no timeline can carry.
    horizon = StepInterval(time_series_id="passengers", start=132, stop=144)
    task = ForecastingTask(
        scope=StepInterval(time_series_id="passengers", start=0, stop=132),
        target_span=horizon,
    )
    assert task.target_span == horizon
    assert horizon.n_steps == 12


def test_forecasting_scope_and_target_span_must_share_a_frame():
    # A context read in seconds and a horizon counted in steps do not lie on one axis.
    with pytest.raises(TimeFValidationError, match="share a frame"):
        ForecastingTask(
            scope=TimeInterval.seconds(0.0, 132.0),
            target_span=StepInterval(time_series_id="passengers", start=132, stop=144),
        )
