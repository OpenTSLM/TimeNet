import json

import numpy as np
import pytest

from timenet.errors import TimeFFormatError
from timenet.types import (
    AnswerTask,
    ClassificationTask,
    ScalarPredictionTask,
    TemporalLocalizationTask,
    TimePoint,
)
from timenet_connectors.datasets.yang_ai_lab.hearts.tasks import (
    EXCLUDED,
    OPTIONS_KEY,
    TASK_DEFINITIONS,
    TaskDefinition,
    _answer_text,
    _label,
    build_task,
    name_vocabulary,
    option_annotations,
    plain,
)


# This module takes values and gives tasks. Nothing here reads a file.

_PREFIX = "hearts"
_RECORD = "hearts-coswara-audio_classification-30"


def _definition(task: str) -> TaskDefinition:
    return next(definition for definition in TASK_DEFINITIONS if definition.task == task)


def _an_answer_the_directory_scores(definition: TaskDefinition):
    # Each directory states its answer in its own shape: a direction index, a window-to-status map,
    # a boolean, or the label itself.
    if definition.task == "waveform_temporal_direction_detection":
        return 0
    if definition.task == "meal_react_comparison":
        return {"A": "Normal", "B": "Diabetes"}
    if definition.options == ("true", "false"):
        return True
    return definition.options[0]


def test_the_table_covers_every_directory_in_scope_once():
    directories = [definition.directory for definition in TASK_DEFINITIONS]
    assert len(directories) == len(set(directories)) == 21
    assert not [directory for directory in directories if directory in EXCLUDED]
    assert sum(definition.n_items for definition in TASK_DEFINITIONS) == 1005


def test_every_definition_carries_what_its_task_type_needs():
    for definition in TASK_DEFINITIONS:
        if definition.task_type is ClassificationTask:
            assert definition.options, definition.directory
        else:
            assert not definition.options, definition.directory
        if definition.task_type is ScalarPredictionTask:
            assert definition.unit and definition.target_name, definition.directory


def test_no_prompt_still_points_at_the_release_sandbox():
    # The release writes its agent's inputs into an input/ directory and asks about the files
    # there. A TimeF consumer has no such directory, so a surviving reference is a half-finished
    # transcription.
    assert not [definition.directory for definition in TASK_DEFINITIONS if "input/" in definition.prompt]


def test_no_prompt_carries_a_placeholder_the_release_would_have_filled_in():
    # The released prompts are f-strings. A surviving `{self.` or `{data[` would ask the agent
    # about a name nothing holds.
    left_over = [
        definition.directory
        for definition in TASK_DEFINITIONS
        if "{self." in definition.prompt or "{data[" in definition.prompt
    ]
    assert not left_over


def test_a_vocabulary_id_is_built_from_the_prefix():
    assert name_vocabulary(_PREFIX, "audio_classification") == "hearts-options-audio_classification"
    assert name_vocabulary("other", "audio_classification") == "other-options-audio_classification"


def test_one_vocabulary_annotation_per_closed_answer_directory():
    annotations = option_annotations(_PREFIX)
    assert len(annotations) == 12
    assert {annotation.key for annotation in annotations} == {OPTIONS_KEY}
    with_options = [definition for definition in TASK_DEFINITIONS if definition.options]
    assert [annotation.id for annotation in annotations] == [
        name_vocabulary(_PREFIX, definition.task) for definition in with_options
    ]
    assert [annotation.value for annotation in annotations] == [list(definition.options) for definition in with_options]


def test_a_classification_target_schema_is_the_id_of_its_registered_vocabulary():
    # target_schema is not the task name. It has to resolve to the annotation that holds the set,
    # or it names nothing.
    registered = {annotation.id for annotation in option_annotations(_PREFIX)}
    for definition in TASK_DEFINITIONS:
        if definition.task_type is not ClassificationTask:
            continue
        task = build_task(definition, _RECORD, _an_answer_the_directory_scores(definition), (), _PREFIX)
        assert isinstance(task, ClassificationTask)
        assert task.target_schema in registered, definition.directory


def test_a_direction_index_becomes_the_label_the_harness_scores():
    definition = _definition("waveform_temporal_direction_detection")
    assert _label(definition, 0) == "forward"
    assert _label(definition, np.int64(1)) == "reversed"
    with pytest.raises(TimeFFormatError, match="not a direction index"):
        _label(definition, 2)


def test_the_meal_comparison_label_is_the_window_whose_subject_is_normal():
    definition = _definition("meal_react_comparison")
    assert _label(definition, {"A": "Normal", "B": "Prediabetes"}) == "A"
    assert _label(definition, {"A": "Diabetes", "B": "Normal"}) == "B"
    with pytest.raises(TimeFFormatError, match="exactly one 'Normal' window"):
        _label(definition, {"A": "Normal", "B": "Normal"})


def test_a_boolean_answer_becomes_the_text_the_vocabulary_holds():
    definition = _definition("cough_detection_good_qual")
    assert _label(definition, True) == "true"
    assert _label(definition, np.bool_(False)) == "false"


def test_a_label_outside_its_vocabulary_stops_the_build():
    definition = _definition("audio_classification")
    assert _label(definition, "cough") == "cough"
    with pytest.raises(TimeFFormatError, match="which is not one of"):
        _label(definition, "wheeze")


def test_every_option_of_every_vocabulary_maps_to_itself():
    special = {"waveform_temporal_direction_detection", "meal_react_comparison"}
    for definition in TASK_DEFINITIONS:
        if not definition.options or definition.task in special:
            continue
        for option in definition.options:
            assert _label(definition, option) == option, definition.directory


def test_a_number_with_a_unit_becomes_a_scalar_prediction():
    task = build_task(_definition("iauc_calculation"), _RECORD, 859.8333333333333, (), _PREFIX)
    assert isinstance(task, ScalarPredictionTask)
    assert task.target == pytest.approx(859.8333333333333)
    assert task.unit == "mg*min/dL"
    assert task.target_name == "postprandial_iauc"


def test_a_structured_answer_becomes_canonical_json():
    task = build_task(_definition("hr_resp_pairing"), _RECORD, {"B": "2", "A": "1"}, (), _PREFIX)
    assert isinstance(task, AnswerTask)
    assert task.target == '{"A": "1", "B": "2"}'


def test_a_whole_meal_minute_lands_on_the_minute():
    task = build_task(_definition("meal_time_localization"), _RECORD, np.int64(4), (), _PREFIX)
    assert isinstance(task, TemporalLocalizationTask)
    assert task.target == (TimePoint.micros(4 * 60_000_000),)


def test_a_fractional_meal_minute_keeps_its_fraction():
    # The release's own prompt declares the answer a float. Reading it as an int would drop half a
    # minute without a word, so the fraction reaches the stored microsecond.
    task = build_task(_definition("meal_time_localization"), _RECORD, 4.5, (), _PREFIX)
    assert isinstance(task, TemporalLocalizationTask)
    assert task.target == (TimePoint.micros(4 * 60_000_000 + 30_000_000),)


def test_a_negative_meal_minute_stops_the_build():
    with pytest.raises(TimeFFormatError, match="negative minute offset"):
        build_task(_definition("meal_time_localization"), _RECORD, -1.0, (), _PREFIX)


def test_the_task_carries_the_prompt_and_the_inputs_it_was_given():
    definition = _definition("cough_covid_status_classification_with_symptoms")
    ids = ("hearts-options-cough_covid_status_classification_with_symptoms", f"{_RECORD}-symptoms")
    task = build_task(definition, _RECORD, "healthy", ids, _PREFIX)
    assert task.id == f"{_RECORD}-qa"
    assert task.prompt == definition.prompt
    assert task.input_annotation_ids == ids
    # The task is streamed, so nothing attaches it to its record afterwards.
    assert task.record_ids == (_RECORD,)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ([np.float32(0.1), np.float32(-247.3856), np.float32(1.25)], "[0.1, -247.3856, 1.25]"),
        ([np.float64(0.1), np.float64(1e-7)], "[0.1, 1e-07]"),
        ([np.int64(4), np.int32(-7), np.uint8(255)], "[4, -7, 255]"),
        ([0.1, 2, True, "x"], '[0.1, 2, true, "x"]'),
        ({"b": np.float32(0.1), "a": [np.int64(3)]}, '{"a": [3], "b": 0.1}'),
    ],
)
def test_a_number_is_written_at_its_shortest_round_tripping_text(answer, expected):
    assert _answer_text(answer) == expected


def test_plain_rebuilds_a_nested_node_out_of_python_types():
    node = {"a": np.bool_(True), "b": np.array([1.5, 2.5]), "c": (np.int64(3), "x")}
    rebuilt = plain(node)
    assert rebuilt == {"a": True, "b": [1.5, 2.5], "c": [3, "x"]}
    assert json.loads(json.dumps(rebuilt)) == rebuilt
    assert type(rebuilt["a"]) is bool
