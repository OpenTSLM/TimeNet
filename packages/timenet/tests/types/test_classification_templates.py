from typing import Any

import pytest

from timenet.errors import TimeFValidationError
from timenet.types import (
    ClassificationEnumerator,
    ClassificationTask,
    ClassificationTemplate,
    LabelSet,
    OptionKeyStyle,
    PromptTemplate,
    TemplateOption,
)


LETTERS = frozenset({ClassificationEnumerator.LATIN_UPPERCASE, ClassificationEnumerator.LATIN_LOWERCASE})
COLON = OptionKeyStyle(enumerator=ClassificationEnumerator.LATIN_UPPERCASE, marker_suffix=":")
PROMPT = (
    "Classify the series.\nRespond ONLY with the letter of the correct choice (A, B, C).\n\n"
    "Choices:\nA: up\nB: down\nC: None of the above\n"
)
# The template of PROMPT: a loop over the options, the keys in a list, and the word for a key as a fact.
CHOICES = PromptTemplate(
    source=(
        "{% block prompt %}Classify the series.\nRespond ONLY with the {{ key_noun }} of the correct choice "
        '({{ keys | join(", ") }}).\n\nChoices:\n'
        "{% for option in options %}{{ option.marker }} {{ option.text }}\n{% endfor %}{% endblock %}"
        "{% block answer %}{{ option.key }}{% endblock %}"
    )
)
TEXTS = {"A": "up", "B": "down", "C": "None of the above"}


def _keyed_template(**changes: Any) -> ClassificationTemplate:
    fields: dict[str, Any] = {
        "prompt_template": CHOICES,
        "options": [TemplateOption(label=label, text=text, pinned=label == "C") for label, text in TEXTS.items()],
        "key_style": COLON,
        "reproduces_published_prompt": True,
    }
    return ClassificationTemplate(**(fields | changes))


def _task(template: ClassificationTemplate, **changes: Any) -> ClassificationTask:
    fields: dict[str, Any] = {
        "prompt": PROMPT,
        "targets": "B",
        "label_set": LabelSet(list(TEXTS)),
        "template": template,
    }
    task = ClassificationTask(**(fields | changes))
    task._id = 7
    return task


def test_the_published_form_of_a_template_is_the_stored_prompt_and_target():
    task = _task(_keyed_template())

    task.check_targets()
    published = task.get_option_presentation(use_shuffled_label_key=False)

    assert published.prompt == PROMPT
    assert published.answers == ("B",)


def test_a_shuffle_keeps_the_answer_correct_and_follows_the_limits_of_the_template():
    task = _task(_keyed_template(allowable_enumerators=LETTERS))

    presentations = [task.get_option_presentation(seed=seed) for seed in range(40)]

    assert presentations[3] == task.get_option_presentation(seed=3)
    assert len({presentation.prompt for presentation in presentations}) > 10
    assert {presentation.key_style.enumerator for presentation in presentations if presentation.key_style} == LETTERS
    for presentation in presentations:
        (answer,) = presentation.answers
        style = presentation.key_style
        assert style is not None
        place = style.get_position_of_option_with_reference(answer)
        # The answer names the place that shows the text of the published answer, and the pinned option stays last.
        assert f"{style.get_marker_of_option_at(place)} down\n" in presentation.prompt
        assert presentation.get_label_of_answer(answer) == "B"
        assert presentation.option_labels[2] == "C"


def test_the_word_for_a_key_follows_the_enumerator():
    numbers = OptionKeyStyle(enumerator=ClassificationEnumerator.NUMERIC, marker_suffix=".")

    numbered = _keyed_template(key_style=numbers, reproduces_published_prompt=False).get_presentation_in_published_form(
        ["B"], multi_label=False, prompt=None
    )

    assert numbered.prompt == PROMPT.replace("letter", "number").replace("(A, B, C)", "(1, 2, 3)").replace(
        "A: up\nB: down\nC:", "1. up\n2. down\n3."
    )
    assert numbered.answers == ("2",)


def test_a_template_can_show_keys_and_answer_with_the_text_of_the_option():
    template = ClassificationTemplate(
        prompt_template=PromptTemplate(
            source=(
                "{% block prompt %}Choose from: {% for option in options %}{{ option.marker }}{{ option.text }}"
                '{{ ", " if not loop.last }}{% endfor %}.{% endblock %}{% block answer %}{{ option.text }}{% endblock %}'
            )
        ),
        options=[TemplateOption(label=label) for label in ("noise", "normal rhythm", "other rhythm")],
        key_style=OptionKeyStyle(enumerator=ClassificationEnumerator.LATIN_LOWERCASE, marker_suffix=")"),
        reproduces_published_prompt=True,
    )
    task = _task(
        template,
        prompt="Choose from: a)noise, b)normal rhythm, c)other rhythm.",
        targets="other rhythm",
        label_set=LabelSet(template.labels),
    )

    task.check_targets()

    assert {task.get_option_presentation(seed=seed).answers for seed in range(10)} == {("other rhythm",)}


def test_a_task_without_a_prompt_is_shown_through_a_template_that_uses_only_its_labels():
    labels = ("biking", "lying", "running", "sitting")
    template = ClassificationTemplate(
        prompt_template=PromptTemplate(
            source=(
                '{% block prompt %}Which labels match? Options: {{ options | map(attribute="text") | join(", ") }}.'
                "{% if multi_label %} Answer with every label that applies.{% endif %}{% endblock %}"
                "{% block answer %}{{ option.text }}{% endblock %}"
            )
        ),
        options=[TemplateOption(label=label) for label in labels],
    )
    task = _task(template, prompt=None, targets=("sitting", "lying"), label_set=LabelSet(labels, multi_label=True))

    task.check_targets()
    published = task.get_option_presentation(use_shuffled_label_key=False)
    shown = [task.get_option_presentation(seed=seed) for seed in range(10)]

    assert published.prompt == (
        "Which labels match? Options: biking, lying, running, sitting. Answer with every label that applies."
    )
    # The items of the answer are in the order in which the options are shown.
    assert published.answers == ("lying", "sitting")
    assert len({presentation.option_labels for presentation in shown}) > 1
    assert all(sorted(presentation.answers) == ["lying", "sitting"] for presentation in shown)


def test_a_template_can_add_the_options_to_the_stored_prompt():
    template = ClassificationTemplate(
        prompt_template=PromptTemplate(
            source=(
                "{% block prompt %}{{ prompt }}\nOptions:\n{% for option in options %}- {{ option.text }}\n"
                "{% endfor %}{% endblock %}{% block answer %}{{ option.text }}{% endblock %}"
            )
        ),
        options=[TemplateOption(label="Yes"), TemplateOption(label="No")],
    )
    task = _task(template, prompt="Is there an anomaly?", targets="No", label_set=LabelSet(["Yes", "No"]))

    task.check_targets()

    assert task.get_option_presentation(use_shuffled_label_key=False).prompt == (
        "Is there an anomaly?\nOptions:\n- Yes\n- No\n"
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"options": []}, "at least one option"),
        ({"options": [TemplateOption(label="A"), TemplateOption(label="A")]}, "names a label more than once"),
        (
            {"allowable_enumerators": {ClassificationEnumerator.NUMERIC}},
            "'latin_uppercase' of the key style is not one of the allowable enumerators",
        ),
    ],
)
def test_a_classification_template_rejects_fields_that_do_not_agree(changes, message):
    with pytest.raises(ValueError, match=message):
        _keyed_template(**changes)


@pytest.mark.parametrize(
    ("template", "changes", "message"),
    [
        (_keyed_template(), {"label_set": LabelSet(["A", "B"])}, r"labels that are not in .*\['C'\]"),
        (
            _keyed_template(options=[TemplateOption(label="A", text="up"), TemplateOption(label="C", text="down")]),
            {},
            r"target labels \['B'\] are not options",
        ),
        (
            _keyed_template(),
            {"prompt": PROMPT.replace("A: up", "A: UP")},
            "it reproduces the published prompt, but it writes",
        ),
        (_keyed_template(), {"prompt": None}, "it reproduces the published prompt, but it writes"),
        (
            _keyed_template(prompt_template=PromptTemplate(source=CHOICES.source.replace("option.key", "option.text"))),
            {},
            r"writes the answer \['down'\] and the stored targets are \['B'\]",
        ),
        (_keyed_template(), {"rationale": "The series falls, so B is right."}, "has a rationale"),
        (_keyed_template(key_style=None), {}, "cannot be written: .* has no attribute 'key'"),
    ],
)
def test_a_classification_task_rejects_a_template_that_does_not_fit_it(template, changes, message):
    with pytest.raises(TimeFValidationError, match=message):
        _task(template, **changes).check_targets()


def test_an_answer_that_names_no_option_or_several_options_is_rejected():
    published = _task(_keyed_template()).get_option_presentation(use_shuffled_label_key=False)
    same_answer_for_each_option = _keyed_template(
        prompt_template=PromptTemplate(source=CHOICES.source.replace("{{ option.key }}", "yes")),
        reproduces_published_prompt=False,
    ).get_presentation_in_published_form(["B"], multi_label=False, prompt=None)

    with pytest.raises(TimeFValidationError, match="names 0 options"):
        published.get_label_of_answer("D")
    with pytest.raises(TimeFValidationError, match="names 3 options"):
        same_answer_for_each_option.get_label_of_answer("yes")


def test_a_task_needs_a_template_to_show_its_options_and_an_id_to_shuffle_them():
    without_template = ClassificationTask(targets="A", label_set=LabelSet(["A", "B"]))
    not_in_a_dataset = _task(_keyed_template())
    not_in_a_dataset._id = None
    not_a_template: Any = {"options": []}

    with pytest.raises(TimeFValidationError, match="has no template"):
        without_template.get_option_presentation()
    assert not_in_a_dataset.get_option_presentation(use_shuffled_label_key=False).answers == ("B",)
    with pytest.raises(TimeFValidationError, match="has no id yet"):
        not_in_a_dataset.get_option_presentation()
    with pytest.raises(TimeFValidationError, match="template must be a ClassificationTemplate, got dict"):
        ClassificationTask(targets="A", label_set=LabelSet(["A", "B"]), template=not_a_template)
