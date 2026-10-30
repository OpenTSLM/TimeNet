"""Read a published question: its format, its choices, and the series it prints, against the authors' templates.

Every question ends with an instruction line and a block of choices (paper Appendix B)::

    <stem>
    Respond ONLY with the letter of the correct choice (A, B, C, or D).

    Choices:
    A: <choice>
    ...

A true-or-false question offers ``T: True.`` and ``F: False.``. A puzzle has no instruction line and
lists ``Patches:`` instead. Data transformation and temporal relationship questions print series as
JSON lists: one sequence after the stem of a true-or-false question, or one per choice or patch. Those
series become Records, so :class:`Question` keeps the text around them, and the values separately.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
import json
import math
import re
from typing import Any

from timenet.errors import TimeFValidationError
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.vocabulary import (
    PATCH_LETTERS,
    TRUE_OR_FALSE_CHOICES,
    QuestionFormat,
    SourceTask,
    Transform,
)


_CHOICES = "\n\nChoices:\n"
_PATCHES = "\n\nPatches:\n"
_RESPOND = re.compile(
    # The letters read "T or F", "A, B", "A, B, C, D" or "A, B, C, or D".
    r"\nRespond ONLY with the letter of the correct choice \((?P<letters>[A-Z](?:(?:, | or |, or )[A-Z])*)\)\.$"
)
_CHOICE_LINE = re.compile(r"(?P<letter>[A-Z]): (?P<text>.*)")
_TRUE_OR_FALSE_LINES = ("True.", "False.")

# The fixed stems of the templated tasks (paper Appendix B.5 to B.8), and the puzzle's whole head.
_ANOMALY_STEM = "Determine whether the given time series contains anomalies."
_CLASSIFICATION_STEM = "Classify the given time series into one of the categories below."
_PUZZLE_HEAD = (
    "The given time series is the first patch of the sequence.\n"
    "Below are the remaining patches, labeled as A, B, C, and D.\n"
    "Arrange A, B, C, D in the correct order to reconstruct the original sequence."
)
_TRANSFORM_TRUE_OR_FALSE = re.compile(r"Is the following sequence the (?P<transform>.+) of the given time series\?")
_TRANSFORM_CHOICE = re.compile(
    r"Which of the following choices is most likely the (?P<transform>.+) of the given time series\?"
)
_CONTINUATION_TRUE_OR_FALSE = "Is the following patch the future continuation of the given time series?"
_CONTINUATION_CHOICE = "Which of the following choices is most likely the future continuation of the given time series?"

PLACEHOLDER_MARK = "<series"
"""Text no published question contains, so a placeholder in a prompt is never the authors' own text."""


@dataclass(frozen=True)
class Question:
    """A published question, read against its template, without the values of the series it prints."""

    format: QuestionFormat
    letters: tuple[str, ...]
    """The letters a model may answer with, in the order the question lists them."""
    choices: tuple[tuple[str, str], ...]
    """Each choice's letter and text, for choices that are text rather than series. A puzzle has none."""
    segments: tuple[str, ...]
    """The question's text around the series it prints: one more segment than series."""
    series_labels: tuple[str | None, ...]
    """The label of each printed series, in text order: a choice or patch letter, or ``None`` for the
    sequence after a true-or-false stem."""
    series_lengths: tuple[int, ...]
    """The number of values of each printed series, in text order."""
    transform: Transform | None = None
    """The transform a data transformation question names."""

    def text(self, series_texts: Sequence[str]) -> str:
        """Return the question with a text in place of each printed series.

        Args:
            series_texts: One text per printed series, in text order.

        Returns:
            The segments joined around the texts.

        Raises:
            TimeFValidationError: If the number of texts differs from the number of printed series.
        """
        if len(series_texts) != len(self.series_labels):
            raise TimeFValidationError(f"the question prints {len(self.series_labels)} series, got {len(series_texts)}")
        parts = [self.segments[0]]
        for text, segment in zip(series_texts, self.segments[1:], strict=True):
            parts += [text, segment]
        return "".join(parts)


def series_text(values: Sequence[float]) -> str:
    """Spell a series as the authors print one in a question.

    The release prints a zero as ``0`` and every other value as Python's ``repr`` of the float, such as
    ``2.0`` or ``-0.0588``, inside ``[...]`` with ``", "`` between values. :func:`parse_question`
    requires every printed series to read back this way, so the rule holds for the whole release.

    Returns:
        The series' text.
    """
    return (
        "["
        + ", ".join("0" if value == 0 and not math.copysign(1, value) < 0 else repr(value) for value in values)
        + "]"
    )


def parse_question(text: str, task: SourceTask) -> tuple[Question, tuple[list[float], ...]]:
    """Read a published question against its task's templates.

    Args:
        text: The published ``question``.
        task: The row's published ``task``, which decides which templates apply.

    Returns:
        The question, and the values of each series it prints, in text order.

    Raises:
        TimeFValidationError: If the question doesn't follow its task's template, offers malformed
            choices, or prints a series that isn't a nonempty list of finite numbers spelled as
            :func:`series_text` spells it.
    """
    if PLACEHOLDER_MARK in text:
        raise TimeFValidationError(f"a question contains {PLACEHOLDER_MARK!r}")
    if not text.endswith("\n"):
        raise TimeFValidationError("a question ends with a newline")
    if _PATCHES in text:
        return _parse_puzzle(text, task)
    head, separator, block = text.partition(_CHOICES)
    if not separator or _CHOICES in block:
        raise TimeFValidationError("a question lists its choices once, after 'Choices:'")
    respond = _RESPOND.search(head)
    if respond is None:
        raise TimeFValidationError("a question's stem ends with the 'Respond ONLY with the letter ...' line")
    choices = _choice_lines(block)
    letters = tuple(letter for letter, _ in choices)
    if tuple(re.findall(r"[A-Z]", respond["letters"])) != letters:
        raise TimeFValidationError(f"the instruction offers ({respond['letters']}), the choices {letters}")
    stem = head[: respond.start()]
    if letters == TRUE_OR_FALSE_CHOICES:
        if tuple(choice for _, choice in choices) != _TRUE_OR_FALSE_LINES:
            raise TimeFValidationError("a true-or-false question offers 'T: True.' and 'F: False.'")
        return _true_or_false(text, stem, task)
    if letters != PATCH_LETTERS[: len(letters)] or not 2 <= len(letters) <= len(PATCH_LETTERS):  # noqa: PLR2004
        raise TimeFValidationError(f"a multiple-choice question offers A to D in order, got {letters}")
    return _multiple_choice(text, stem, choices, task)


def _choice_lines(block: str) -> list[tuple[str, str]]:
    """Split a choices block into letters and texts, one per line.

    Returns:
        Each choice's letter and text, in order.

    Raises:
        TimeFValidationError: If a line isn't ``<letter>: <text>``.
    """
    choices = []
    for line in block.removesuffix("\n").split("\n"):
        match = _CHOICE_LINE.fullmatch(line)
        if match is None:
            raise TimeFValidationError(f"a choice line reads '<letter>: <text>', got {line[:60]!r}")
        choices.append((match["letter"], match["text"]))
    return choices


def _true_or_false(text: str, stem: str, task: SourceTask) -> tuple[Question, tuple[list[float], ...]]:
    """Read a true-or-false question: a templated stem, then a printed sequence for two of the tasks.

    Returns:
        The question and the values of the sequence it prints, if any.

    Raises:
        TimeFValidationError: If the stem doesn't follow its task's template.
    """
    transform = None
    if task.shows_series_in_question:
        first, newline, sequence = stem.partition("\n")
        if not newline:
            raise TimeFValidationError("a true-or-false question prints its sequence on the line after the stem")
        transform = _stem_transform(first, task, choice=False)
        start = len(first) + 1
        values = _series(sequence)
        question = _question(text, QuestionFormat.TRUE_OR_FALSE, TRUE_OR_FALSE_CHOICES, (), [(None, start, values)])
        return _with_transform(question, transform), (values,)
    if task is SourceTask.ANOMALY_DETECTION and stem != _ANOMALY_STEM:
        raise TimeFValidationError("an anomaly detection question follows the paper's template (Appendix B.7)")
    if task is SourceTask.CLASSIFICATION:
        raise TimeFValidationError("a classification question is multiple-choice")
    return _question(text, QuestionFormat.TRUE_OR_FALSE, TRUE_OR_FALSE_CHOICES, (), []), ()


def _multiple_choice(
    text: str, stem: str, choices: list[tuple[str, str]], task: SourceTask
) -> tuple[Question, tuple[list[float], ...]]:
    """Read a multiple-choice question, whose choices are series for two of the tasks and text otherwise.

    Returns:
        The question and the values of its series choices, if any.

    Raises:
        TimeFValidationError: If the stem doesn't follow its task's template.
    """
    letters = tuple(letter for letter, _ in choices)
    if task.shows_series_in_question:
        transform = _stem_transform(stem, task, choice=True)
        if letters != PATCH_LETTERS:
            raise TimeFValidationError(f"a {task} question offers four series, got {letters}")
        printed = _printed_choices(text, _CHOICES, choices)
        question = _question(text, QuestionFormat.MULTIPLE_CHOICE, letters, (), printed)
        return _with_transform(question, transform), tuple(values for _, _, values in printed)
    if task is SourceTask.CLASSIFICATION and stem != _CLASSIFICATION_STEM:
        raise TimeFValidationError("a classification question follows the paper's template (Appendix B.8)")
    if task is SourceTask.ANOMALY_DETECTION:
        raise TimeFValidationError("an anomaly detection question is true-or-false")
    return _question(text, QuestionFormat.MULTIPLE_CHOICE, letters, tuple(choices), []), ()


def _parse_puzzle(text: str, task: SourceTask) -> tuple[Question, tuple[list[float], ...]]:
    """Read a puzzle: the paper's fixed head, then four patches A to D.

    Returns:
        The question and the values of its patches, in letter order.

    Raises:
        TimeFValidationError: If the puzzle isn't a temporal relationship question in the paper's template.
    """
    head, _, block = text.partition(_PATCHES)
    if _PATCHES in block or _CHOICES in text:
        raise TimeFValidationError("a puzzle lists its patches once, after 'Patches:', and offers no choices")
    if task is not SourceTask.TEMPORAL_RELATIONSHIP or head != _PUZZLE_HEAD:
        raise TimeFValidationError("a puzzle is a temporal relationship question in the paper's template (B.6)")
    choices = _choice_lines(block)
    if tuple(letter for letter, _ in choices) != PATCH_LETTERS:
        raise TimeFValidationError("a puzzle lists patches A, B, C and D")
    printed = _printed_choices(text, _PATCHES, choices)
    question = _question(text, QuestionFormat.PUZZLING, PATCH_LETTERS, (), printed)
    return question, tuple(values for _, _, values in printed)


def _stem_transform(stem: str, task: SourceTask, *, choice: bool) -> Transform | None:
    """Check a data transformation or temporal relationship stem against its template.

    Returns:
        The transform a data transformation stem names, ``None`` for a temporal relationship stem.

    Raises:
        TimeFValidationError: If the stem isn't its task's template, or names an unknown transform.
    """
    if task is SourceTask.TEMPORAL_RELATIONSHIP:
        if stem != (_CONTINUATION_CHOICE if choice else _CONTINUATION_TRUE_OR_FALSE):
            raise TimeFValidationError("a temporal relationship question follows the paper's template (B.6)")
        return None
    match = (_TRANSFORM_CHOICE if choice else _TRANSFORM_TRUE_OR_FALSE).fullmatch(stem)
    if match is None or match["transform"] not in set(Transform):
        raise TimeFValidationError("a data transformation question names a known transform in its template (B.5)")
    return Transform(match["transform"])


def _printed_choices(
    text: str, separator: str, choices: list[tuple[str, str]]
) -> list[tuple[str | None, int, list[float]]]:
    """Locate each choice's series in the question's text.

    Returns:
        Each choice's letter, the offset of its series in ``text``, and its values.
    """
    printed: list[tuple[str | None, int, list[float]]] = []
    offset = text.index(separator) + len(separator)
    for letter, series in choices:
        start = offset + len(letter) + 2
        printed.append((letter, start, _series(series)))
        offset = start + len(series) + 1
    return printed


def _question(
    text: str,
    fmt: QuestionFormat,
    letters: tuple[str, ...],
    choices: tuple[tuple[str, str], ...],
    printed: list[tuple[str | None, int, list[float]]],
) -> Question:
    """Build a question from its text and the offsets of the series it prints.

    Returns:
        The question, its text cut around each series.
    """
    segments: list[str] = []
    position = 0
    for _, start, values in printed:
        segments.append(text[position:start])
        position = start + len(series_text(values))
    segments.append(text[position:])
    return Question(
        format=fmt,
        letters=letters,
        choices=choices,
        segments=tuple(segments),
        series_labels=tuple(label for label, _, _ in printed),
        series_lengths=tuple(len(values) for _, _, values in printed),
    )


def _with_transform(question: Question, transform: Transform | None) -> Question:
    return question if transform is None else replace(question, transform=transform)


def _reject_constant(name: str) -> Any:
    raise TimeFValidationError(f"JSON holds {name}")


def decode_json(text: str, what: str) -> Any:
    """Decode published JSON, rejecting the ``NaN`` and ``Infinity`` that Python's decoder accepts.

    Args:
        text: The JSON text.
        what: What the text is, for the error message.

    Returns:
        The decoded value.

    Raises:
        TimeFValidationError: If the text isn't strict JSON.
    """
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise TimeFValidationError(f"{what} is JSON: {exc}") from exc


def _series(text: str) -> list[float]:
    """Decode a printed series, and require that :func:`series_text` spells its values back the same.

    Returns:
        The values, as floats.
    """
    return checked_series(decode_json(text, "a printed series"), text)


def checked_series(values: Any, text: str | None = None) -> list[float]:
    """Require a decoded series to be a nonempty list of finite numbers, and return it as floats.

    Args:
        values: The decoded JSON value.
        text: The printed text it was decoded from, which must equal :func:`series_text` of the values;
            ``None`` for a series the question doesn't print, such as ``input_ts``.

    Returns:
        The values, as floats.

    Raises:
        TimeFValidationError: If the value isn't such a list, or isn't spelled as the authors print one.
    """
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)
    ):
        raise TimeFValidationError("a series is a nonempty JSON list of numbers")
    floats = [float(value) for value in values]
    if not all(math.isfinite(value) for value in floats):
        raise TimeFValidationError("a series holds a value that is not finite")
    if text is not None and series_text(floats) != text:
        raise TimeFValidationError(f"a printed series is not spelled as series_text spells it: {text[:60]!r}")
    return floats
