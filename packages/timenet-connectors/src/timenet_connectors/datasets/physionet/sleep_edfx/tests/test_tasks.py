import pytest

from timenet.errors import TimeFFormatError
from timenet.types import (
    US_PER_S,
    Annotation,
    ClassificationTask,
    LocalizationMode,
    ScalarPredictionTask,
    TemporalLocalizationTask,
    TimeInterval,
    TimePoint,
)
from timenet_connectors.datasets.physionet.sleep_edfx import tasks
from timenet_connectors.datasets.physionet.sleep_edfx.keys import AnnotationKey, Question


_PREFIX = "sleep-edfx"
_SAMPLE_ID = "sleep-edfx-SC4901E0"


def test_the_stage_set_holds_the_eight_labels_and_no_others():
    assert set(tasks.STAGE_LABELS) == {
        "Sleep stage W",
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage 4",
        "Sleep stage R",
        "Movement time",
        "Sleep stage ?",
    }


def test_wake_movement_and_unscored_are_not_asleep():
    assert tasks.ASLEEP_LABELS.isdisjoint({"Sleep stage W", "Movement time", "Sleep stage ?"})


def test_the_asleep_set_holds_the_five_stages_of_sleep():
    assert {
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage 4",
        "Sleep stage R",
    } == tasks.ASLEEP_LABELS


def test_a_vocabulary_is_one_annotation_for_each_set():
    built = tasks.build_vocabularies(_PREFIX)
    assert [one.key for one in built] == ["sleep_stage_vocabulary", "sex_vocabulary", "condition_vocabulary"]
    assert [one.id for one in built] == [
        "sleep-edfx-vocabulary-sleep_stage",
        "sleep-edfx-vocabulary-sex",
        "sleep-edfx-vocabulary-condition",
    ]


def test_a_vocabulary_holds_the_whole_set_and_not_one_member():
    stages, sexes, conditions = tasks.build_vocabularies(_PREFIX)
    assert stages.value == list(tasks.STAGE_LABELS)
    assert sexes.value == ["F", "M"]
    assert conditions.value == ["placebo", "temazepam"]


def test_a_vocabulary_carries_no_span():
    assert all(one.span is None for one in tasks.build_vocabularies(_PREFIX))


def test_the_vocabulary_id_names_the_key_and_not_a_sample():
    assert tasks.name_vocabulary(_PREFIX, AnnotationKey.SEX) == "sleep-edfx-vocabulary-sex"


def test_two_builds_give_one_set_of_vocabularies():
    first = tasks.build_vocabularies(_PREFIX)
    second = tasks.build_vocabularies(_PREFIX)
    assert [one.id for one in first] == [one.id for one in second]
    assert [one.value for one in first] == [one.value for one in second]


def test_an_epoch_task_id_names_its_epoch():
    assert tasks.name_epoch_task(_SAMPLE_ID, 1742) == "sleep-edfx-SC4901E0-epoch-1742"


def test_an_epoch_id_is_built_from_the_index_it_is_given():
    # The id carries the index and nothing else about the scoring, which is what lets an epoch
    # keep its id when an entry ahead of it changes. The builder is covered where it lands.
    assert [tasks.name_epoch_task(_SAMPLE_ID, i) for i in (0, 1, 1742)] == [
        "sleep-edfx-SC4901E0-epoch-0",
        "sleep-edfx-SC4901E0-epoch-1",
        "sleep-edfx-SC4901E0-epoch-1742",
    ]


def test_a_whole_sample_task_id_names_its_question():
    assert tasks.name_sample_task(_SAMPLE_ID, Question.AGE) == "sleep-edfx-SC4901E0-age"


def _scope(task) -> TimeInterval:
    # Task.scope is Span | None, so narrow it before reading its bounds.
    assert isinstance(task.scope, TimeInterval)
    return task.scope


def _stage(label, onset_s, end_s):
    return Annotation(
        key=AnnotationKey.SLEEP_STAGE,
        value=label,
        span=TimeInterval.micros(onset_s * US_PER_S, end_s * US_PER_S, time_series_ids=("a-channel",)),
        id=f"{_SAMPLE_ID}-stage-{onset_s}",
    )


def test_a_run_expands_into_one_task_for_each_epoch():
    built = list(tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, [_stage("Sleep stage W", 0, 900)]))
    assert len(built) == 30
    assert {one.target for one in built} == {"Sleep stage W"}


def test_an_entry_of_one_epoch_gives_one_task():
    built = list(tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, [_stage("Sleep stage 2", 30, 60)]))
    assert len(built) == 1


def test_the_scope_covers_the_epoch_and_names_no_series():
    first, second = tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, [_stage("Sleep stage 1", 60, 120)])
    assert (_scope(first).start_us, _scope(first).end_us) == (60 * US_PER_S, 90 * US_PER_S)
    assert (_scope(second).start_us, _scope(second).end_us) == (90 * US_PER_S, 120 * US_PER_S)
    assert _scope(first).time_series_ids is None


def test_the_id_names_the_epoch_from_the_start_of_the_recording():
    built = list(tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, [_stage("Sleep stage R", 300, 360)]))
    assert [one.id for one in built] == [f"{_SAMPLE_ID}-epoch-10", f"{_SAMPLE_ID}-epoch-11"]


def test_the_label_is_kept_as_the_scorer_wrote_it():
    built = list(tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, [_stage("Sleep stage 4", 0, 30)]))
    assert built[0].target == "Sleep stage 4"


def test_a_label_that_names_no_sleep_stage_still_becomes_a_task():
    built = list(tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, [_stage("Movement time", 0, 30)]))
    assert built[0].target == "Movement time"


def test_every_task_names_the_stage_vocabulary():
    built = list(tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, [_stage("Sleep stage 2", 0, 60)]))
    assert {one.target_schema for one in built} == {"sleep-edfx-vocabulary-sleep_stage"}


def test_no_task_carries_a_prompt():
    built = list(tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, [_stage("Sleep stage 2", 0, 60)]))
    assert all(one.prompt is None for one in built)


def test_a_scoring_that_starts_late_gets_no_task_for_the_time_before_it():
    built = list(tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, [_stage("Sleep stage W", 750, 780)]))
    assert [_scope(one).start_us for one in built] == [750 * US_PER_S]


def test_a_hole_between_two_entries_gets_no_task():
    stages = [_stage("Sleep stage W", 0, 30), _stage("Sleep stage 2", 90, 120)]
    built = list(tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, stages))
    assert [_scope(one).start_us // US_PER_S for one in built] == [0, 90]


def test_an_entry_that_does_not_divide_into_epochs_names_its_duration():
    with pytest.raises(TimeFFormatError, match="lasts 45000000 us"):
        list(tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, [_stage("Sleep stage W", 0, 45)]))


def test_an_entry_off_the_epoch_boundary_names_its_onset_and_not_its_duration():
    # 15 s to 45 s lasts exactly one epoch. The onset is what is wrong, and a message about the
    # duration would send a reader the wrong way.
    with pytest.raises(TimeFFormatError, match="starts at 15000000 us"):
        list(tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, [_stage("Sleep stage W", 15, 45)]))


def test_an_entry_with_no_span_raises_rather_than_being_dropped():
    entry = Annotation(key="sleep_stage", value="Sleep stage W", id=f"{_SAMPLE_ID}-stage-0")
    with pytest.raises(TimeFFormatError, match="states no span"):
        list(tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, [entry]))


def test_a_label_the_release_does_not_write_raises():
    with pytest.raises(TimeFFormatError, match="not a label this release writes"):
        list(tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, [_stage("N3", 0, 30)]))


def test_no_epoch_task_answers_by_reference():
    # target and target_annotation_ids are exclusive, and the reference would give up the
    # scalar-target path that ClassificationTask declares.
    built = list(tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, [_stage("Sleep stage 2", 0, 90)]))
    assert all(one.target_annotation_ids == () for one in built)


def test_a_second_build_of_one_scoring_gives_the_same_ids():
    stages = [_stage("Sleep stage W", 0, 60), _stage("Sleep stage 2", 60, 120)]
    first = [one.id for one in tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, stages)]
    second = [one.id for one in tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, stages)]
    assert first == second == [f"{_SAMPLE_ID}-epoch-{i}" for i in range(4)]


def test_an_id_survives_a_change_to_an_entry_before_it():
    # The index counts from the start of the recording, so re-labelling the first entry leaves
    # every later id alone. A running counter would renumber the rest of the night.
    before = [_stage("Sleep stage W", 0, 60), _stage("Sleep stage 2", 60, 120)]
    after = [_stage("Sleep stage 1", 0, 60), _stage("Sleep stage 2", 60, 120)]
    assert [one.id for one in tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, before)] == [
        one.id for one in tasks.build_epoch_tasks(_SAMPLE_ID, _PREFIX, after)
    ]


def _fact(key, value):
    return Annotation(key=key, value=value, id=f"{_SAMPLE_ID}-{key}")


_CASSETTE_FACTS = [
    _fact(AnnotationKey.STUDY, "sleep-cassette"),
    _fact(AnnotationKey.NIGHT, 1),
    _fact(AnnotationKey.SEX, "F"),
    _fact(AnnotationKey.AGE, 33),
]


def _pick(built, kind, suffix):
    # The builder yields Task, so narrow before reading a subclass field.
    task = next(one for one in built if one.id.endswith(suffix))
    assert isinstance(task, kind)
    return task


def test_age_is_a_scalar_with_a_unit():
    built = list(tasks.build_sample_tasks(_SAMPLE_ID, _PREFIX, _CASSETTE_FACTS, []))
    age = _pick(built, ScalarPredictionTask, "-age")
    assert age.target == pytest.approx(33.0)
    assert age.unit == "year"
    assert age.target_name == "age"
    assert age.scope is None


def test_sex_carries_the_decoded_letter():
    built = list(tasks.build_sample_tasks(_SAMPLE_ID, _PREFIX, _CASSETTE_FACTS, []))
    sex = _pick(built, ClassificationTask, "-sex")
    assert sex.target == "F"
    assert sex.target_schema == "sleep-edfx-vocabulary-sex"
    assert sex.scope is None


def test_the_same_sheet_code_decodes_to_the_other_letter_elsewhere():
    # The sheets code sex with opposite meanings, so the task takes what the decoder gave.
    telemetry = [_fact("sex", "M"), _fact("age", 40)]
    built = list(tasks.build_sample_tasks(_SAMPLE_ID, _PREFIX, telemetry, []))
    assert _pick(built, ClassificationTask, "-sex").target == "M"


def test_a_telemetry_night_carries_its_condition():
    facts = [*_CASSETTE_FACTS, _fact("condition", "placebo")]
    built = list(tasks.build_sample_tasks(_SAMPLE_ID, _PREFIX, facts, []))
    condition = _pick(built, ClassificationTask, "-condition")
    assert condition.target == "placebo"
    assert condition.scope is None


def test_a_cassette_recording_carries_no_condition_task():
    built = list(tasks.build_sample_tasks(_SAMPLE_ID, _PREFIX, _CASSETTE_FACTS, []))
    assert not any(t.id.endswith("-condition") for t in built)


def test_provenance_becomes_no_task():
    facts = [
        *_CASSETTE_FACTS,
        _fact("recording_start_local", "1989-04-24T16:13:00"),
        _fact("demographics_note", "header says Male_31yr"),
    ]
    built = list(tasks.build_sample_tasks(_SAMPLE_ID, _PREFIX, facts, []))
    # Exactly the two questions this recording can answer, and nothing drawn from the four
    # provenance keys the input carries.
    assert {one.id for one in built} == {f"{_SAMPLE_ID}-age", f"{_SAMPLE_ID}-sex"}


def test_a_missing_age_raises_rather_than_dropping_the_question():
    with pytest.raises(TimeFFormatError, match="states no age"):
        list(tasks.build_sample_tasks(_SAMPLE_ID, _PREFIX, [_fact("sex", "F")], []))


def test_a_missing_sex_raises_rather_than_dropping_the_question():
    with pytest.raises(TimeFFormatError, match="states no sex"):
        list(tasks.build_sample_tasks(_SAMPLE_ID, _PREFIX, [_fact("age", 33)], []))


def test_an_age_that_is_not_a_whole_number_raises():
    facts = [_fact("age", "thirty-three"), _fact("sex", "F")]
    with pytest.raises(TimeFFormatError, match="not a whole number of years"):
        list(tasks.build_sample_tasks(_SAMPLE_ID, _PREFIX, facts, []))


def test_a_sex_the_release_does_not_write_raises():
    facts = [_fact("age", 33), _fact("sex", 1)]
    with pytest.raises(TimeFFormatError, match="does not write"):
        list(tasks.build_sample_tasks(_SAMPLE_ID, _PREFIX, facts, []))


def test_a_key_stated_twice_raises():
    facts = [*_CASSETTE_FACTS, _fact("age", 40)]
    with pytest.raises(TimeFFormatError, match="twice"):
        list(tasks.build_sample_tasks(_SAMPLE_ID, _PREFIX, facts, []))


def test_the_night_runs_from_the_first_sleep_to_the_last():
    stages = [
        _stage("Sleep stage W", 0, 3600),
        _stage("Sleep stage 2", 3600, 12000),
        _stage("Sleep stage R", 12000, 25200),
        _stage("Sleep stage W", 25200, 32400),
    ]
    assert tasks.find_sleep_period(stages) == TimeInterval.micros(3600 * US_PER_S, 25200 * US_PER_S)


def test_a_trailing_unscored_entry_does_not_bound_the_night():
    stages = [
        _stage("Sleep stage 2", 3600, 25200),
        _stage("Movement time", 25200, 25230),
        _stage("Sleep stage ?", 32400, 36000),
    ]
    assert tasks.find_sleep_period(stages) == TimeInterval.micros(3600 * US_PER_S, 25200 * US_PER_S)


def test_a_scoring_with_no_sleep_gives_no_night():
    assert tasks.find_sleep_period([_stage("Sleep stage W", 0, 100 * 30)]) is None


def test_the_localization_task_is_sparse_and_names_no_series():
    stages = [_stage("Sleep stage 2", 3600, 25200)]
    built = list(tasks.build_sample_tasks(_SAMPLE_ID, _PREFIX, _CASSETTE_FACTS, stages))
    night = _pick(built, TemporalLocalizationTask, "-sleep_period")
    assert night.mode == LocalizationMode.SPARSE
    assert night.target == (TimeInterval.micros(3600 * US_PER_S, 25200 * US_PER_S),)
    assert night.scope is None


def test_a_recording_with_no_scored_sleep_carries_no_night_task():
    built = list(tasks.build_sample_tasks(_SAMPLE_ID, _PREFIX, _CASSETTE_FACTS, [_stage("Sleep stage W", 0, 30)]))
    assert not any(t.id.endswith("-sleep_period") for t in built)


def _moment(key, at_s, series=None):
    return Annotation(key=key, value="22:30:00", span=TimePoint.micros(at_s * US_PER_S), id=f"{_SAMPLE_ID}-{key}")


_SESSION = TimeInterval.micros(0, 86400 * US_PER_S)


def test_the_lights_off_moment_becomes_a_localization_task():
    built = tasks.build_lights_off_task(_SAMPLE_ID, _SESSION, [_moment(AnnotationKey.LIGHTS_OFF, 1800)])
    assert built is not None
    assert built.target == (TimePoint.micros(1800 * US_PER_S),)
    assert built.mode == LocalizationMode.SPARSE
    assert built.scope is None
    assert built.id == f"{_SAMPLE_ID}-lights_off"


def test_a_moment_the_session_does_not_hold_gets_no_task():
    # The sheets state a clock time and no date, so a lights-off time seconds before the
    # recorder started wraps forward about a day, past the end of the session.
    session = TimeInterval.micros(0, 30600 * US_PER_S)
    assert tasks.build_lights_off_task(_SAMPLE_ID, session, [_moment(AnnotationKey.LIGHTS_OFF, 86370)]) is None


def test_a_moment_at_the_last_instant_of_the_session_is_held():
    session = TimeInterval.micros(0, 1801 * US_PER_S)
    assert tasks.build_lights_off_task(_SAMPLE_ID, session, [_moment(AnnotationKey.LIGHTS_OFF, 1800)]) is not None


def test_a_recording_that_states_no_lights_off_gets_no_task():
    assert tasks.build_lights_off_task(_SAMPLE_ID, _SESSION, []) is None


def test_a_sample_with_no_declared_session_gets_no_lights_off_task():
    assert tasks.build_lights_off_task(_SAMPLE_ID, None, [_moment(AnnotationKey.LIGHTS_OFF, 1800)]) is None


def test_the_lights_off_task_names_no_vocabulary():
    # Its answer is a region and not a label, so there is no closed set to name.
    built = tasks.build_lights_off_task(_SAMPLE_ID, _SESSION, [_moment(AnnotationKey.LIGHTS_OFF, 1800)])
    assert built is not None
    assert built.target_annotation_ids == ()
    assert built.prompt is None
