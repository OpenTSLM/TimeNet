"""Check that the task table and the prompts describe every task directory coherently."""

from timenet.types import ClassificationTask, ScalarPredictionTask, TSCorrespondenceTask
from timenet_connectors.datasets.yang_ai_lab.hearts.release import (
    COLUMN_SPECS,
    LABEL_SETS,
    PROMPTS,
    REVISION,
    TASKS,
    Answer,
    Waveform,
)


PLACEHOLDERS = {"meal_time", "meal_info", "mask_start", "mask_end", "symptoms", "speaker"}


def test_thirty_task_directories_each_with_a_prompt():
    assert len(TASKS) == 30
    assert set(PROMPTS) == set(TASKS)
    assert all(prompt.template for prompt in PROMPTS.values())
    assert len(REVISION) == 40
    assert {directory.partition("/")[0] for directory in TASKS} == {
        "cgmacros",
        "harespod",
        "coswara",
        "coughvid",
        "vctk",
    }


def test_prompts_name_only_the_values_the_connector_fills_in():
    for directory, definition in TASKS.items():
        identifiers = set(PROMPTS[directory].get_identifiers())
        assert identifiers <= PLACEHOLDERS, directory
        assert ("meal_info" in identifiers) == definition.meal_info, directory
        assert ("symptoms" in identifiers) == definition.symptoms, directory


def test_answers_and_inputs_fit_their_task_types():
    for directory, definition in TASKS.items():
        assert bool(definition.options) == (definition.task_type is ClassificationTask), directory
        assert (directory in LABEL_SETS) == bool(definition.options), directory
        if definition.task_type is ScalarPredictionTask:
            assert definition.target_name, directory
        assert bool(definition.candidates) == (definition.task_type is TSCorrespondenceTask), directory
        if definition.task_type is not ClassificationTask:
            assert definition.answer is Answer.LABEL, directory
        waveforms = [entry for entry in definition.inputs if isinstance(entry, Waveform)]
        assert len(waveforms) <= 1 and (not waveforms or len(definition.inputs) == 1), directory


def test_label_set_names_are_distinct():
    # A name stands for one label set in a dataset, so two directories cannot share one.
    names = [label_set.name for label_set in LABEL_SETS.values()]
    assert len(names) == len(set(names)) == 14
    assert LABEL_SETS["cgmacros/meal_img_classification"].names == ("a.jpg", "b.jpg", "c.jpg", "d.jpg")


def test_every_value_column_has_a_spec_of_its_own_family():
    assert {spec.spec_type for spec in COLUMN_SPECS.values()} == {
        "cgm",
        "heart_rate",
        "activity_calories",
        "respiration_norm",
        "spo2_norm",
        "heart_rate_norm",
    }
