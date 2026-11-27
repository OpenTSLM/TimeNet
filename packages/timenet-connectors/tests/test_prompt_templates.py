import pytest

from timenet.errors import TimeNetConnectorError
from timenet.types import (
    ClassificationEnumerator,
    ClassificationTask,
    ClassificationTemplate,
    LabelSet,
    OptionKeyStyle,
)
from timenet_connectors.sources.prompt_templates import RepeatedOptionListError, get_template_from_published_prompt


UPPERCASE_WITH_COLON = OptionKeyStyle(enumerator=ClassificationEnumerator.LATIN_UPPERCASE, marker_suffix=":")
LOWERCASE_WITH_COLON = OptionKeyStyle(enumerator=ClassificationEnumerator.LATIN_LOWERCASE, marker_suffix=":")
UPPERCASE_WITH_PARENTHESIS = OptionKeyStyle(enumerator=ClassificationEnumerator.LATIN_UPPERCASE, marker_suffix=")")
LOWERCASE_IN_PARENTHESES = OptionKeyStyle(
    enumerator=ClassificationEnumerator.LATIN_LOWERCASE,
    marker_prefix="(",
    marker_suffix=")",
    reference_prefix="(",
    reference_suffix=")",
)

# Each prompt has the form of one published dataset.
KEYS_NAMED_BEFORE_THE_OPTIONS = (
    "Classify the given time series into one of the categories below.\n"
    "Respond ONLY with the letter of the correct choice (A, B, C, or D).\n\n"
    "Choices:\nA: down-down (1306 cases)\nB: up-down (1248 cases)\nC: <series 2>\nD: up-up\n"
)
FIRST_OPTION_ON_THE_QUESTION_LINE = (
    "Given the time series signal <ts>, what is the health state? a:  Operating Normally\n"
    " b: Abnormal Degradation\n c: Imminent Failure\n"
)
OPTIONS_OF_SEVERAL_LINES = (
    "What description best relates to this time series?\n{'units': 'riders'}\nA) \nA city hosts dance classes.\n"
    "Ridership rises on Saturdays.\n\nB)  A store tracks potato sales. \nC) A meteor shower occurs.\nD) none"
)
INSTRUCTION_AFTER_THE_OPTIONS = (
    "Select one of the following answers:\n(a) This time series has a constant trend.\n"
    "(b) This time series has an upward trend.\n(c) This time series has a downward trend.\n"
    "Only answer (a), (b), or (c)."
)


def _texts(template: ClassificationTemplate) -> list[str | None]:
    return [option.text for option in template.options]


def _task(prompt: str, template: ClassificationTemplate, answer: str) -> ClassificationTask:
    """Build the task of a connector from a template and run the checks of a dataset on it.

    The checks make sure that the template writes the published prompt and answer exactly.
    """
    task = ClassificationTask(prompt=prompt, targets=answer, label_set=LabelSet(template.labels), template=template)
    task._id = 1
    task.check_targets()
    return task


def test_a_template_of_options_on_their_own_lines_writes_the_published_prompt():
    letters = frozenset({ClassificationEnumerator.LATIN_UPPERCASE, ClassificationEnumerator.LATIN_LOWERCASE})

    template = get_template_from_published_prompt(
        KEYS_NAMED_BEFORE_THE_OPTIONS,
        published_key_style=UPPERCASE_WITH_COLON,
        one_option_per_line=True,
        expected_key_list_count=1,
        allowable_enumerators=letters,
        pinned_option_texts=["up-up"],
    )

    assert template.labels == ("A", "B", "C", "D")
    assert _texts(template) == ["down-down (1306 cases)", "up-down (1248 cases)", "<series 2>", "up-up"]
    assert [option.pinned for option in template.options] == [False, False, False, True]
    assert template.allowable_enumerators == letters
    assert template.reproduces_published_prompt
    _task(KEYS_NAMED_BEFORE_THE_OPTIONS, template, "C")


def test_a_shuffled_template_names_the_new_keys_in_its_list_of_keys():
    template = get_template_from_published_prompt(
        INSTRUCTION_AFTER_THE_OPTIONS,
        published_key_style=LOWERCASE_IN_PARENTHESES,
        one_option_per_line=True,
        expected_key_list_count=1,
    )
    task = _task(INSTRUCTION_AFTER_THE_OPTIONS, template, "(b)")

    for seed in range(8):
        shown = task.get_option_presentation(seed=seed)
        style = shown.key_style
        assert style is not None
        (answer,) = shown.answers
        place = style.get_position_of_option_with_reference(answer)
        keys = [style.get_reference_to_option_at(position) for position in range(3)]
        assert f"{style.get_marker_of_option_at(place)} This time series has an upward trend.\n" in shown.prompt
        assert shown.prompt.endswith(f"Only answer {keys[0]}, {keys[1]}, or {keys[2]}.")


@pytest.mark.parametrize(
    ("prompt", "key_style", "one_option_per_line", "texts"),
    [
        (
            FIRST_OPTION_ON_THE_QUESTION_LINE,
            LOWERCASE_WITH_COLON,
            True,
            ["Operating Normally", "Abnormal Degradation", "Imminent Failure"],
        ),
        (
            OPTIONS_OF_SEVERAL_LINES,
            UPPERCASE_WITH_PARENTHESIS,
            False,
            [
                "A city hosts dance classes.\nRidership rises on Saturdays.",
                "A store tracks potato sales.",
                "A meteor shower occurs.",
                "none",
            ],
        ),
        # Text that is Jinja syntax stays text of the prompt.
        ("Fill {{ this }} and {% that %}:\nA: up\nB: down", UPPERCASE_WITH_COLON, True, ["up", "down"]),
        # Line ends with a carriage return stay as they are. Jinja reads a carriage return as a line feed.
        ("About the series\r\nWebsite traffic\r\n\nA) up\nB) down", UPPERCASE_WITH_PARENTHESIS, True, ["up", "down"]),
    ],
)
def test_a_template_writes_the_published_prompt_where_its_text_is_not_regular(
    prompt, key_style, one_option_per_line, texts
):
    template = get_template_from_published_prompt(
        prompt, published_key_style=key_style, one_option_per_line=one_option_per_line
    )

    assert _texts(template) == texts
    _task(prompt, template, template.labels[-1])


@pytest.mark.parametrize(
    ("prompt", "expected_key_list_count", "message"),
    [
        ("Which trend? A: up", 0, "has 1 of the markers"),
        ("Which trend?\nA: up\nB: down\nA: flat", 0, "the marker 'A:' occurs 2 times"),
        ("Which trend?\nA: up\nB: down\nD: flat", 0, "has the marker 'D:', but not the marker 'C:'"),
        ("Which trend?\nA: up\nB: \nC: flat", 0, "the option with the marker 'B:' has no text"),
        ("Which trend?\nA: up\nB: down\nAnswer with A or B.", 0, "has 1 lists of all keys, but 0 are expected"),
        ("Which trend?\nA: up\nB: down", 1, "has 0 lists of all keys, but 1 are expected"),
    ],
)
def test_a_prompt_that_does_not_have_the_declared_form_is_rejected(prompt, expected_key_list_count, message):
    with pytest.raises(TimeNetConnectorError, match=message):
        get_template_from_published_prompt(
            prompt,
            published_key_style=UPPERCASE_WITH_COLON,
            one_option_per_line=True,
            expected_key_list_count=expected_key_list_count,
        )


def test_a_prompt_that_prints_its_options_twice_raises_its_own_error():
    prompt = "Which trend?\nA) up\nB) down\n\nChoices:\nA: up\nB: down"

    with pytest.raises(RepeatedOptionListError, match=r"with the markers 'A\)' and 'B\)'"):
        get_template_from_published_prompt(prompt, published_key_style=UPPERCASE_WITH_COLON, one_option_per_line=True)
