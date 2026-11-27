import pytest

from timenet.errors import TimeNetConnectorError
from timenet.types import ClassificationTask, LabelSet
from timenet_connectors.templates import (
    DEFAULT_PROMPT_TEMPLATE,
    PROMPT_THEN_OPTIONS_TEMPLATE,
    get_default_template_of_label_set,
    get_template_that_lists_options_after_prompt,
)


def _task(**fields) -> ClassificationTask:
    task = ClassificationTask(**fields)
    task._id = 3
    task.check_targets()
    return task


def test_the_default_template_asks_for_one_label_of_a_task_without_a_prompt():
    activities = LabelSet(["biking", "lying", "running"])
    task = _task(targets="lying", label_set=activities, template=get_default_template_of_label_set(activities))

    published = task.get_option_presentation(use_shuffled_label_key=False)
    shown = [task.get_option_presentation(seed=seed) for seed in range(10)]

    assert published.prompt == "Which label matches the input? Options: biking, lying, running. Answer with one label."
    assert published.answers == ("lying",)
    # A shuffle changes the order of the labels. The answer is the label itself, so it stays the same.
    assert len({presentation.prompt for presentation in shown}) > 1
    assert {presentation.answers for presentation in shown} == {("lying",)}


def test_the_default_template_asks_for_every_label_of_a_multi_label_task():
    findings = LabelSet(["AF", "PVC", "LBBB"], multi_label=True)
    task = _task(targets=("PVC", "AF"), label_set=findings, template=get_default_template_of_label_set(findings))

    published = task.get_option_presentation(use_shuffled_label_key=False)

    assert published.prompt == (
        "Which label matches the input? Options: AF, PVC, LBBB. Answer with every label that applies, one per line."
    )
    assert published.answers == ("AF", "PVC")


def test_the_options_of_a_prompt_that_does_not_print_them_are_listed_after_it():
    options = ["Yes, they are correlated", "No"]
    task = _task(
        prompt="Is there an anomaly in both series?",
        targets="No",
        label_set=LabelSet(options),
        template=get_template_that_lists_options_after_prompt(options),
    )

    published = task.get_option_presentation(use_shuffled_label_key=False)

    assert published.prompt == (
        "Is there an anomaly in both series?\n\nOptions:\n- Yes, they are correlated\n- No\n\nAnswer with one option."
    )
    assert published.answers == ("No",)
    assert {task.get_option_presentation(seed=seed).option_labels for seed in range(10)} == {
        ("Yes, they are correlated", "No"),
        ("No", "Yes, they are correlated"),
    }


def test_each_shared_template_names_its_sections():
    sections = ("question", "options", "answer_instruction")

    assert DEFAULT_PROMPT_TEMPLATE.sections == sections
    assert PROMPT_THEN_OPTIONS_TEMPLATE.sections == sections


def test_an_option_that_would_read_as_several_options_is_rejected():
    with pytest.raises(TimeNetConnectorError, match=r"\['walking, slow'\] hold a comma or a line break"):
        get_default_template_of_label_set(LabelSet(["walking, slow", "running"]))
    with pytest.raises(TimeNetConnectorError, match="hold a line break"):
        get_template_that_lists_options_after_prompt(["up\ndown", "flat"])
