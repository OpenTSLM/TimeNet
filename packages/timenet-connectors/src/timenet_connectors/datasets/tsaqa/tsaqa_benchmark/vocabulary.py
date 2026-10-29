"""TSAQA's closed sets: the paper's tasks and question formats, the published question types, and the transforms.

The paper (arXiv 2601.23204v2) defines six tasks (Table 2) and three question formats (Section 3.1):
true-or-false (TF), multiple-choice (MC) and puzzling (PZ). The release names the task in its ``task``
column and the format in its ``question_type`` column. A question's own text says which format it has,
and for 10,000 rows that disagrees with ``question_type``, so both are kept.
"""

from enum import StrEnum, unique


@unique
class SourceTask(StrEnum):
    """One of the paper's six tasks, as the release's ``task`` column names it."""

    ANOMALY_DETECTION = "anomaly_detection"
    CLASSIFICATION = "classification"
    CHARACTERIZATION = "characterization"
    COMPARISON = "comparison"
    DATA_TRANSFORMATION = "data_transformation"
    TEMPORAL_RELATIONSHIP = "temporal_relationship"

    @property
    def input_series(self) -> int:
        """Return how many series a row's ``input_ts`` holds: two for comparison, one otherwise."""
        return 2 if self is SourceTask.COMPARISON else 1

    @property
    def shows_series_in_question(self) -> bool:
        """Return whether the question prints series of its own: the candidates of these two tasks."""
        return self in {SourceTask.DATA_TRANSFORMATION, SourceTask.TEMPORAL_RELATIONSHIP}


@unique
class QuestionType(StrEnum):
    """A row's ``question_type``, as published.

    It doesn't always name the question's format: 5,049 ``ordering`` rows ask a true-or-false question,
    and 4,951 a multiple-choice one. :class:`QuestionFormat` is the format the question has.
    """

    TRUE_OR_FALSE = "true_or_false"
    MULTIPLE_CHOICES = "multiple_choices"
    ORDERING = "ordering"


@unique
class QuestionFormat(StrEnum):
    """The format a question has, read from its text: the paper's TF, MC or PZ."""

    TRUE_OR_FALSE = "true_or_false"
    """Answer ``T`` or ``F``: whether a claim about the series holds."""
    MULTIPLE_CHOICE = "multiple_choice"
    """Answer one letter of two to four choices."""
    PUZZLING = "puzzling"
    """Answer the four patch letters in chronological order, such as ``C,B,D,A``."""

    @property
    def question_type(self) -> QuestionType:
        """Return the ``question_type`` the release gives this format when it names it right."""
        return _QUESTION_TYPES[self]


_QUESTION_TYPES: dict[QuestionFormat, QuestionType] = {
    QuestionFormat.TRUE_OR_FALSE: QuestionType.TRUE_OR_FALSE,
    QuestionFormat.MULTIPLE_CHOICE: QuestionType.MULTIPLE_CHOICES,
    QuestionFormat.PUZZLING: QuestionType.ORDERING,
}


@unique
class Transform(StrEnum):
    """The transform a data transformation question asks about (paper Appendix B.5), as the question names it."""

    FOURIER = "Fourier Transform"
    WAVELET = "Wavelet Transform"
    FIRST_ORDER_DIFFERENCE = "First Order Difference"


TRUE_OR_FALSE_CHOICES = ("T", "F")
"""A true-or-false question's choices, as every one lists them: ``T: True.`` and ``F: False.``."""
PATCH_LETTERS = ("A", "B", "C", "D")
"""A puzzle's patch letters, and a multiple-choice question's letters when it has four choices."""


def context_entries(task: SourceTask, fmt: QuestionFormat) -> int | None:
    """Return how many entries a row's ``meta_info``, ``domain`` and ``dataset`` hold.

    A single-series task publishes plain text. The other tasks publish a JSON list: one entry per
    series for comparison; the input's sequence then the distractors' for a data transformation or
    temporal relationship question; and one entry, the sequence the patches come from, for a puzzle.

    Returns:
        The number of list entries, or ``None`` when the fields are plain text.
    """
    if task in {SourceTask.ANOMALY_DETECTION, SourceTask.CLASSIFICATION, SourceTask.CHARACTERIZATION}:
        return None
    if task is SourceTask.COMPARISON:
        return 2
    return {QuestionFormat.TRUE_OR_FALSE: 2, QuestionFormat.MULTIPLE_CHOICE: 4, QuestionFormat.PUZZLING: 1}[fmt]
