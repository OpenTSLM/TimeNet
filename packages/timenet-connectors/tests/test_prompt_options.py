import pytest

from timenet.errors import TimeNetConnectorError
from timenet.types import ClassificationEnumerator, ClassificationTask, LabelSet, OptionKeyStyle, PromptOptions
from timenet_connectors.sources.prompt_options import RepeatedOptionListError, locate_prompt_options


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
    "What description best relates to this time series?\nA) \nA city hosts dance classes.\n"
    "Ridership rises on Saturdays.\n\nB)  A store tracks potato sales. \nC) A meteor shower occurs.\nD) none"
)
INSTRUCTION_AFTER_THE_OPTIONS = (
    "Select one of the following answers:\n(a) This time series has a constant trend.\n"
    "(b) This time series has an upward trend.\n(c) This time series has a downward trend.\n"
    "Only answer (a), (b), or (c)."
)


def _texts(prompt: str, options: PromptOptions) -> list[str]:
    return [option.text.read_from(prompt) for option in options.options if option.text is not None]


def _check_task(prompt: str, options: PromptOptions, answer: str) -> ClassificationTask:
    """Build the task of a connector from located options and run the checks of a dataset on it."""
    task = ClassificationTask(prompt=prompt, targets=answer, label_set=LabelSet(options.labels), prompt_options=options)
    task._id = 1
    task.check_targets()
    return task


def test_options_on_their_own_lines_are_located_with_the_list_of_their_keys():
    letters = frozenset({ClassificationEnumerator.LATIN_UPPERCASE, ClassificationEnumerator.LATIN_LOWERCASE})

    options = locate_prompt_options(
        KEYS_NAMED_BEFORE_THE_OPTIONS,
        published_key_style=UPPERCASE_WITH_COLON,
        one_option_per_line=True,
        expected_key_list_count=1,
        allowable_enumerators=letters,
        pinned_option_texts=["up-up"],
    )

    assert options.labels == ("A", "B", "C", "D")
    assert _texts(KEYS_NAMED_BEFORE_THE_OPTIONS, options) == [
        "down-down (1306 cases)",
        "up-down (1248 cases)",
        "<series 2>",
        "up-up",
    ]
    assert [reference.option_position for reference in options.key_references] == [0, 1, 2, 3]
    assert [option.pinned for option in options.options] == [False, False, False, True]
    assert options.allowable_enumerators == letters
    _check_task(KEYS_NAMED_BEFORE_THE_OPTIONS, options, "C")


def test_located_options_give_a_shuffled_prompt_that_names_the_new_keys():
    options = locate_prompt_options(
        INSTRUCTION_AFTER_THE_OPTIONS,
        published_key_style=LOWERCASE_IN_PARENTHESES,
        one_option_per_line=True,
        expected_key_list_count=1,
    )
    published = _check_task(INSTRUCTION_AFTER_THE_OPTIONS, options, "(b)").get_option_presentation(
        use_shuffled_label_key=False
    )

    numbered = published.with_options_in_order([2, 0, 1]).with_key_style(
        LOWERCASE_IN_PARENTHESES.model_copy(
            update={"enumerator": ClassificationEnumerator.NUMERIC, "marker_prefix": "", "marker_suffix": "."}
        )
    )

    assert numbered.prompt == (
        "Select one of the following answers:\n1. This time series has a downward trend.\n"
        "2. This time series has a constant trend.\n3. This time series has an upward trend.\n"
        "Only answer (1), (2), or (3)."
    )
    assert numbered.get_answer() == ("(3)",)


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
    ],
)
def test_options_are_located_where_the_whitespace_around_them_is_not_regular(
    prompt, key_style, one_option_per_line, texts
):
    options = locate_prompt_options(prompt, published_key_style=key_style, one_option_per_line=one_option_per_line)

    assert _texts(prompt, options) == texts
    _check_task(prompt, options, options.labels[-1])


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
        locate_prompt_options(
            prompt,
            published_key_style=UPPERCASE_WITH_COLON,
            one_option_per_line=True,
            expected_key_list_count=expected_key_list_count,
        )


def test_a_prompt_that_prints_its_options_twice_raises_its_own_error():
    prompt = "Which trend?\nA) up\nB) down\n\nChoices:\nA: up\nB: down"

    with pytest.raises(RepeatedOptionListError, match=r"with the markers 'A\)' and 'B\)'"):
        locate_prompt_options(prompt, published_key_style=UPPERCASE_WITH_COLON, one_option_per_line=True)
