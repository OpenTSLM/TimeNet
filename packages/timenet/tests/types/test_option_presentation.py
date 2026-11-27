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


PROMPT = "Pick one.\n(a) up\n(b) down\n(c) none of the above\nOnly answer (a), (b), or (c)."
PARENTHESES = OptionKeyStyle(
    enumerator=ClassificationEnumerator.LATIN_LOWERCASE,
    marker_prefix="(",
    marker_suffix=")",
    reference_prefix="(",
    reference_suffix=")",
)
NUMBERS_WITH_DOT = PARENTHESES.model_copy(
    update={"enumerator": ClassificationEnumerator.NUMERIC, "marker_prefix": "", "marker_suffix": "."}
)


def _span_of(text: str, start: int = 0) -> PromptSpan:
    """Return the span of the first occurrence of ``text`` in ``PROMPT`` at or after ``start``."""
    found = PROMPT.index(text, start)
    return PromptSpan(start=found, end=found + len(text))


def _task(**changes) -> ClassificationTask:
    """Return a task with three keyed options, of which the last is pinned, and the answer ``(b)``."""
    answer_line = PROMPT.index("Only answer")
    options = PromptOptions(
        options=[
            PromptOption(label="(a)", marker=_span_of("(a)"), text=_span_of("up")),
            PromptOption(label="(b)", marker=_span_of("(b)"), text=_span_of("down")),
            PromptOption(label="(c)", marker=_span_of("(c)"), text=_span_of("none of the above"), pinned=True),
        ],
        key_style=PARENTHESES,
        key_references=[
            OptionKeyReference(option_position=position, span=_span_of(key, answer_line))
            for position, key in enumerate(["(a)", "(b)", "(c)"])
        ],
        **changes,
    )
    task = ClassificationTask(
        prompt=PROMPT, targets="(b)", label_set=LabelSet(["(a)", "(b)", "(c)"]), prompt_options=options
    )
    task._id = 7
    task.check_targets()
    return task


def test_the_published_presentation_is_the_stored_prompt_and_answer():
    published = _task().get_option_presentation(use_shuffled_label_key=False)

    assert published.prompt == PROMPT
    assert published.get_answer() == ("(b)",)


def test_another_order_moves_the_texts_and_the_answer_but_not_the_keys():
    published = _task().get_option_presentation(use_shuffled_label_key=False)

    swapped = published.with_options_in_order([1, 0, 2])

    assert swapped.prompt == "Pick one.\n(a) down\n(b) up\n(c) none of the above\nOnly answer (a), (b), or (c)."
    assert swapped.get_answer() == ("(a)",)
    assert swapped.get_label_of_answer("(a)") == "(b)"
    assert swapped.with_options_in_order([1, 0, 2]) == published


def test_another_key_style_changes_the_markers_the_references_and_the_answer():
    published = _task().get_option_presentation(use_shuffled_label_key=False)

    numbered = published.with_key_style(NUMBERS_WITH_DOT)

    assert numbered.prompt == "Pick one.\n1. up\n2. down\n3. none of the above\nOnly answer (1), (2), or (3)."
    assert numbered.get_answer() == ("(2)",)
    assert numbered.get_label_of_answer("(2)") == "(b)"
    assert numbered.with_key_style(PARENTHESES) == published


def test_a_shuffle_gives_the_same_presentation_for_the_same_seed_and_keeps_the_answer_correct():
    task = _task()

    presentations = [task.get_option_presentation(seed=seed) for seed in range(40)]

    assert presentations[3] == task.get_option_presentation(seed=3)
    assert len({presentation.prompt for presentation in presentations}) > 10
    for presentation in presentations:
        (answer,) = presentation.get_answer()
        # The answer names the option that shows the text of the published answer, and the pinned option stays last.
        position = presentation.prompt_options.labels.index("(b)")
        text = presentation.prompt_options.options[position].text
        assert text is not None
        assert text.read_from(presentation.prompt or "") == "down"
        assert presentation.get_label_of_answer(answer) == "(b)"
        assert presentation.prompt_options.labels[2] == "(c)"


def test_a_shuffle_draws_only_allowable_enumerators():
    letters = {ClassificationEnumerator.LATIN_LOWERCASE, ClassificationEnumerator.LATIN_UPPERCASE}
    task = _task(allowable_enumerators=letters)

    styles = [task.get_option_presentation(seed=seed).prompt_options.key_style for seed in range(40)]

    assert {style.enumerator for style in styles if style is not None} == letters


def test_options_that_the_prompt_does_not_print_change_their_order_only():
    labels = ("up", "down", "flat")
    task = ClassificationTask(
        prompt="Which trend?",
        targets="flat",
        label_set=LabelSet(labels),
        prompt_options=PromptOptions(options=[PromptOption(label=label) for label in labels]),
    )
    task._id = 7

    presentations = [task.get_option_presentation(seed=seed) for seed in range(10)]

    assert {presentation.prompt for presentation in presentations} == {"Which trend?"}
    assert len({presentation.prompt_options.labels for presentation in presentations}) > 1
    for presentation in presentations:
        assert sorted(presentation.prompt_options.labels) == ["down", "flat", "up"]
        assert presentation.get_answer() == ("flat",)
        assert presentation.get_label_of_answer("flat") == "flat"


def test_a_presentation_rejects_changes_that_the_options_do_not_allow():
    published = _task(allowable_enumerators={ClassificationEnumerator.LATIN_LOWERCASE}).get_option_presentation(
        use_shuffled_label_key=False
    )

    with pytest.raises(TimeFValidationError, match="moves the pinned options"):
        published.with_options_in_order([0, 2, 1])
    with pytest.raises(TimeFValidationError, match="must hold each of the 3 positions once"):
        published.with_options_in_order([0, 0, 2])
    with pytest.raises(TimeFValidationError, match="'numeric' is not one of the allowable enumerators"):
        published.with_key_style(NUMBERS_WITH_DOT)
    with pytest.raises(TimeFValidationError, match="names no option"):
        published.get_label_of_answer("(d)")


def test_a_task_that_is_not_in_a_dataset_has_only_its_published_presentation():
    task = _task()
    task._id = None

    assert task.get_option_presentation(use_shuffled_label_key=False).get_answer() == ("(b)",)
    with pytest.raises(TimeFValidationError, match="has no id yet"):
        task.get_option_presentation()


def test_a_task_without_prompt_options_has_no_presentation():
    task = ClassificationTask(targets="up", label_set=LabelSet(["up", "down"]))

    with pytest.raises(TimeFValidationError, match="has no prompt_options"):
        task.get_option_presentation()
