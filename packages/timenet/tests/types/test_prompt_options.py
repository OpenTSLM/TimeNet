from typing import Any

import pytest

from timenet.errors import TimeFValidationError
from timenet.types import (
    ClassificationEnumerator,
    ClassificationTask,
    LabelSet,
    OptionKeyReference,
    OptionKeyStyle,
    PromptOption,
    PromptOptions,
    PromptSpan,
)


PROMPT = "Which trend? A: up B: down. Answer with A or B."
STYLE = OptionKeyStyle(enumerator=ClassificationEnumerator.LATIN_UPPERCASE, marker_suffix=":")
LETTERS = LabelSet(["A", "B"])


def _span(start: int, end: int) -> PromptSpan:
    return PromptSpan(start=start, end=end)


def _keyed_options(**changes: Any) -> PromptOptions:
    """Return the prompt options of ``PROMPT``, with some fields replaced."""
    fields: dict[str, Any] = {
        "options": [
            PromptOption(label="A", marker=_span(13, 15), text=_span(16, 18)),
            PromptOption(label="B", marker=_span(19, 21), text=_span(22, 26)),
        ],
        "key_style": STYLE,
        "key_references": [
            OptionKeyReference(option_position=0, span=_span(40, 41)),
            OptionKeyReference(option_position=1, span=_span(45, 46)),
        ],
    }
    return PromptOptions(**(fields | changes))


def _task(prompt_options: PromptOptions, **changes: Any) -> ClassificationTask:
    fields: dict[str, Any] = {"prompt": PROMPT, "targets": "B", "label_set": LETTERS, "prompt_options": prompt_options}
    task = ClassificationTask(**(fields | changes))
    task._id = 1
    return task


def test_prompt_options_with_keys_that_fit_the_prompt_pass_the_task_checks():
    options = _keyed_options()

    _task(options).check_targets()

    assert [option.text.read_from(PROMPT) for option in options.options if option.text] == ["up", "down"]
    assert [reference.span.read_from(PROMPT) for reference in options.key_references] == ["A", "B"]


def test_prompt_options_without_spans_fit_a_task_whose_prompt_does_not_print_them():
    trends = LabelSet(["up", "down"])
    options = PromptOptions(options=[PromptOption(label="down"), PromptOption(label="up")])

    _task(options, prompt="Which trend?", targets="up", label_set=trends, rationale="It rises.").check_targets()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"options": []}, "at least one option"),
        (
            {"options": [PromptOption(label="A", marker=_span(13, 15), text=_span(16, 18)), PromptOption(label="B")]},
            "either each prompt option has a text span, or none has one",
        ),
        (
            {"options": [PromptOption(label="A", marker=_span(13, 15)), PromptOption(label="B", marker=_span(19, 21))]},
            "a prompt option with a marker span must have a text span",
        ),
        ({"key_style": None, "key_references": []}, "a key style if, and only if, each option has a marker span"),
        (
            {"options": [PromptOption(label="A", text=_span(16, 18)), PromptOption(label="A", text=_span(22, 26))]},
            "name a label more than once",
        ),
        (
            {
                "options": [
                    PromptOption(label="A", marker=_span(19, 21), text=_span(22, 26)),
                    PromptOption(label="B", marker=_span(13, 15), text=_span(16, 18)),
                ]
            },
            "must follow the order of the options",
        ),
        ({"key_references": [OptionKeyReference(option_position=0, span=_span(17, 20))]}, "overlap"),
        ({"key_references": [OptionKeyReference(option_position=2, span=_span(40, 41))]}, "option position 2"),
        (
            {"allowable_enumerators": {ClassificationEnumerator.NUMERIC}},
            "'latin_uppercase' of the key style is not one of the allowable enumerators",
        ),
    ],
)
def test_prompt_options_reject_fields_that_do_not_agree(changes, message):
    with pytest.raises(ValueError, match=message):
        _keyed_options(**changes)


@pytest.mark.parametrize(
    ("options", "changes", "message"),
    [
        (_keyed_options(), {"label_set": LabelSet(["A", "C"]), "targets": "A"}, r"labels that are not in .*\['B'\]"),
        (_keyed_options(), {"prompt": None}, "have spans, but the task has no prompt"),
        (_keyed_options(), {"prompt": PROMPT[:30]}, "ends after the prompt, which has 30 characters"),
        (
            _keyed_options(key_style=STYLE.model_copy(update={"marker_suffix": ")"})),
            {},
            r"has 'A:' at the marker of the option at position 0, but the key style gives 'A\)'",
        ),
        (
            _keyed_options(key_references=[OptionKeyReference(option_position=1, span=_span(40, 41))]),
            {},
            "has 'A' at a key reference to the option at position 1, but the key style gives 'B'",
        ),
        (
            _keyed_options(
                options=[
                    PromptOption(label="B", marker=_span(13, 15), text=_span(16, 18)),
                    PromptOption(label="A", marker=_span(19, 21), text=_span(22, 26)),
                ]
            ),
            {},
            "has the label 'B' for the prompt option at position 0, but the key style gives the reference 'A'",
        ),
        (_keyed_options(), {"rationale": "The series rises, so B is wrong."}, "has a rationale"),
        (
            _keyed_options(),
            {"label_set": LabelSet(["A", "B", "C"]), "targets": "C"},
            r"target labels without a prompt option: \['C'\]",
        ),
    ],
)
def test_a_classification_task_rejects_prompt_options_that_do_not_fit_it(options, changes, message):
    with pytest.raises(TimeFValidationError, match=message):
        _task(options, **changes).check_targets()


def test_a_classification_task_rejects_prompt_options_of_another_type():
    not_prompt_options: Any = {"options": []}

    with pytest.raises(TimeFValidationError, match="prompt_options must be PromptOptions, got dict"):
        ClassificationTask(targets="A", label_set=LETTERS, prompt_options=not_prompt_options)
