"""Build the template of a published multiple-choice prompt.

A source often publishes a question as one text with its options, for example::

    Which trend does the series show?
    A: up
    B: down
    Answer with A or B.

:func:`get_template_from_published_prompt` finds the options and their keys in such a text. It returns a
template that writes the same text, and that can also write it with the options in another order and with
other keys. A connector gives the template to its :class:`~timenet.types.ClassificationTask`. The prompt of
the task stays as the source publishes it.
"""

from collections.abc import Collection
import re
from typing import NamedTuple

from pydantic import ValidationError

from timenet.errors import TimeNetConnectorError
from timenet.types import (
    ALL_CLASSIFICATION_ENUMERATORS,
    ClassificationEnumerator,
    ClassificationTemplate,
    OptionKeyStyle,
    PromptTemplate,
    TemplateOption,
)


class RepeatedOptionListError(TimeNetConnectorError):
    """A prompt prints its options a second time, with other markers.

    Only one list of options can be found. A connector can catch this error to keep such a row without a
    template.
    """


_MINIMUM_OPTION_COUNT = 2
"""The smallest number of options that a list of options has."""
_KEY_LIST_SEPARATOR = r"(?:\s*,\s*(?:or\s+|and\s+)?|\s+or\s+|\s+and\s+|\s*/\s*)"
"""What stands between two keys in a list of keys: a comma, ``or``, ``and``, or a slash."""


class _Span(NamedTuple):
    """A part of a prompt: the characters from ``start`` up to, but not including, ``end``."""

    start: int
    end: int


class _Slot(NamedTuple):
    """A part of a prompt that the template writes from a fact."""

    span: _Span
    expression: str
    """The Jinja expression that writes the part, for example ``options[0].text``."""


def _find_marker_spans(prompt: str, key_style: OptionKeyStyle) -> list[_Span]:
    """Find the marker of each option, from the first option on.

    A marker stands at the start of the prompt or after whitespace, and whitespace or the end of the prompt
    follows it.

    Returns:
        The span of each marker, in the order of the options.

    Raises:
        TimeNetConnectorError: If a marker occurs more than once, or the markers skip a key.
    """
    spans: list[_Span] = []
    position = 0
    while key_style.enumerator.has_symbols_for_option_count(position + 1):
        marker = key_style.get_marker_of_option_at(position)
        found = [match.span() for match in re.finditer(rf"(?<!\S){re.escape(marker)}(?=\s|$)", prompt)]
        after_previous_marker = [span for span in found if not spans or span[0] >= spans[-1].end]
        if not after_previous_marker:
            break
        if len(found) > 1:
            raise TimeNetConnectorError(f"the marker {marker!r} occurs {len(found)} times in the prompt")
        spans.append(_Span(*found[0]))
        position += 1
    if spans and key_style.enumerator.has_symbols_for_option_count(position + 2):
        skipped = key_style.get_marker_of_option_at(position)
        following = key_style.get_marker_of_option_at(position + 1)
        if re.search(rf"(?<!\S){re.escape(following)}(?=\s|$)", prompt[spans[-1].end :]):
            raise TimeNetConnectorError(f"the prompt has the marker {following!r}, but not the marker {skipped!r}")
    return spans


def _check_no_second_list_of_options(prompt: str, key_style: OptionKeyStyle) -> None:
    """Make sure that the prompt does not print its options a second time with other markers.

    Some sources repeat the options in the question, for example as ``A) up`` before the list ``A: up``.
    Such a second list is not found, so it must not be there.

    Raises:
        RepeatedOptionListError: If the markers of the first two options occur in another common form.
    """
    published_marker = key_style.get_marker_of_option_at(0)
    for variant in key_style.get_variants_with_common_markers(frozenset({key_style.enumerator})):
        first, second = variant.get_marker_of_option_at(0), variant.get_marker_of_option_at(1)
        if first == published_marker:
            continue
        found = re.search(rf"(?<!\S){re.escape(first)}(?=\s|$)", prompt)
        if found and re.search(rf"(?<!\S){re.escape(second)}(?=\s|$)", prompt[found.end() :]):
            raise RepeatedOptionListError(
                f"the prompt holds a second list of options, with the markers {first!r} and {second!r}"
            )


def _find_text_span(prompt: str, marker: _Span, end_of_search: int, *, one_option_per_line: bool) -> _Span:
    """Find the text of the option that follows a marker.

    Args:
        prompt: The published prompt.
        marker: The span of the marker of the option.
        end_of_search: The start of the next marker, or the length of the prompt for the last option.
        one_option_per_line: Whether the text of an option ends at the end of its line.

    Returns:
        The span of the text, without the whitespace around it.

    Raises:
        TimeNetConnectorError: If the option has no text.
    """
    start = marker.end
    while start < end_of_search and prompt[start].isspace():
        start += 1
    end = end_of_search
    if one_option_per_line:
        end_of_line = prompt.find("\n", start, end_of_search)
        end = end if end_of_line == -1 else end_of_line
    while end > start and prompt[end - 1].isspace():
        end -= 1
    if end <= start:
        raise TimeNetConnectorError(f"the option with the marker {prompt[marker.start : marker.end]!r} has no text")
    return _Span(start, end)


def _find_key_lists(prompt: str, key_style: OptionKeyStyle, option_count: int) -> list[_Slot]:
    """Find each list that names every key once, in the order of the options.

    For example, ``A, B, C, or D`` and ``(a) or (b)`` are such lists. A letter or a digit does not stand
    directly before or after a list.

    Returns:
        One slot per key of each list, in the order of the prompt.
    """
    references = [key_style.get_reference_to_option_at(position) for position in range(option_count)]
    pattern = _KEY_LIST_SEPARATOR.join(f"({re.escape(reference)})" for reference in references)
    return [
        _Slot(_Span(match.start(position + 1), match.end(position + 1)), f"keys[{position}]")
        for match in re.finditer(rf"(?<![^\W_]){pattern}(?![^\W_])", prompt)
        for position in range(option_count)
    ]


def _write_template_source(prompt: str, slots: list[_Slot]) -> str:
    """Write the Jinja source that gives the prompt, with an expression in place of each slot.

    The answer block writes the key of the correct option.

    Returns:
        The source of the template.

    Raises:
        TimeNetConnectorError: If two slots overlap.
    """
    pieces: list[str] = []
    end_of_previous = 0
    for slot in sorted(slots, key=lambda slot: slot.span.start):
        if slot.span.start < end_of_previous:
            raise TimeNetConnectorError(
                f"the parts {slot.span.start} to {slot.span.end} of the prompt overlap with the part before them"
            )
        pieces.append(PromptTemplate.escape_text(prompt[end_of_previous : slot.span.start]))
        pieces.append(f"{{{{ {slot.expression} }}}}")
        end_of_previous = slot.span.end
    pieces.append(PromptTemplate.escape_text(prompt[end_of_previous:]))
    return "{% block prompt %}" + "".join(pieces) + "{% endblock %}{% block answer %}{{ option.key }}{% endblock %}"


def get_template_from_published_prompt(  # ruff: ignore[too-many-arguments] - one argument per fact that the connector declares
    prompt: str,
    *,
    published_key_style: OptionKeyStyle,
    one_option_per_line: bool,
    expected_key_list_count: int = 0,
    allowable_enumerators: frozenset[ClassificationEnumerator] = ALL_CLASSIFICATION_ENUMERATORS,
    pinned_option_texts: Collection[str] = (),
) -> ClassificationTemplate:
    """Build the template of a published prompt that prints its options with keys and is answered with a key.

    The connector declares how the source writes the keys. The function does not guess. It raises an error
    for a prompt that does not have the declared form, and the connector then decides what to do with that
    row.

    The label of each option is the reference to its key, for example ``B``. A source that gives its answer
    as a key thus needs no change of its answer::

        template = get_template_from_published_prompt(
            row.question,
            published_key_style=OptionKeyStyle(
                enumerator=ClassificationEnumerator.LATIN_UPPERCASE, marker_suffix=":"
            ),
            one_option_per_line=True,
            expected_key_list_count=1,
        )
        ClassificationTask(
            prompt=row.question,
            targets=row.answer,
            label_set=LabelSet(template.labels),
            template=template,
            ...
        )

    With the options and the keys of the source, the template writes the published prompt exactly. The
    dataset checks this when it adds the task.

    Args:
        prompt: The published prompt.
        published_key_style: How the prompt writes the keys of its options.
        one_option_per_line: Whether the text of an option ends at the end of its line. If ``False``, the
            text ends before the next marker, and the text of the last option ends with the prompt.
        expected_key_list_count: The number of lists in the prompt that name every key once in order,
            outside the options. For example, ``Answer with A, B, C, or D.`` holds one list.
        allowable_enumerators: The enumerators that can replace the published one. Restrict them if the
            prompt fits only some of them, for example if it says ``Answer with the letter``.
        pinned_option_texts: The texts of the options that must keep their place, for example
            ``None of the above``.

    Returns:
        The template, with the options in the order of the prompt.

    Raises:
        RepeatedOptionListError: If the prompt prints its options a second time with other markers.
        TimeNetConnectorError: If the prompt has fewer than two markers, a marker occurs more than once,
            the markers skip a key, an option has no text, or the number of key lists is not the expected
            one.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by _check_no_second_list_of_options)
    markers = _find_marker_spans(prompt, published_key_style)
    if len(markers) < _MINIMUM_OPTION_COUNT:
        first_markers = [
            published_key_style.get_marker_of_option_at(position) for position in range(_MINIMUM_OPTION_COUNT)
        ]
        raise TimeNetConnectorError(
            f"the prompt has {len(markers)} of the markers {first_markers}, but a list of options has at least two"
        )
    _check_no_second_list_of_options(prompt, published_key_style)
    ends_of_search = [*(marker.start for marker in markers[1:]), len(prompt)]
    options: list[TemplateOption] = []
    slots: list[_Slot] = []
    for position, (marker, end_of_search) in enumerate(zip(markers, ends_of_search, strict=True)):
        text = _find_text_span(prompt, marker, end_of_search, one_option_per_line=one_option_per_line)
        option_text = prompt[text.start : text.end]
        slots.append(_Slot(marker, f"options[{position}].marker"))
        slots.append(_Slot(text, f"options[{position}].text"))
        options.append(
            TemplateOption(
                label=published_key_style.get_reference_to_option_at(position),
                text=option_text,
                pinned=option_text in pinned_option_texts,
            )
        )
    key_references = _find_key_lists(prompt, published_key_style, len(options))
    key_list_count = len(key_references) // len(options)
    if key_list_count != expected_key_list_count:
        raise TimeNetConnectorError(
            f"the prompt has {key_list_count} lists of all keys, but {expected_key_list_count} are expected"
        )
    try:
        return ClassificationTemplate(
            prompt_template=PromptTemplate(source=_write_template_source(prompt, [*slots, *key_references])),
            options=options,
            key_style=published_key_style,
            allowable_enumerators=allowable_enumerators,
            reproduces_published_prompt=True,
        )
    except ValidationError as error:
        raise TimeNetConnectorError(f"the template of the prompt is not valid: {error}") from error
