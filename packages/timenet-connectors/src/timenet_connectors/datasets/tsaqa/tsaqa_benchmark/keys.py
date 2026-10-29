"""The names TSAQA writes into a dataset, so writing and reading them share one spelling."""

from enum import StrEnum, unique


@unique
class AnnotationKey(StrEnum):
    """The key of an annotation the connector builds. A row's context keeps its published column name."""

    SOURCE_REPOSITORY = "source_repository"
    """The dataset's Hugging Face repository."""
    SOURCE_REVISION = "source_revision"
    """The dataset's pinned commit of that repository."""
    TRUE_OR_FALSE_VOCABULARY = "true_or_false_vocabulary"
    """The dataset's true-or-false choices, which every true-or-false task names as its ``target_schema``."""
    META_INFO = "meta_info"
    """An input Record's published ``meta_info``, which the paper calls its description: text, or a list."""
    DOMAIN = "domain"
    """An input Record's published ``domain``: text, or a list."""
    DATASET = "dataset"
    """An input Record's published ``dataset``, the source the series was sampled from: text, or a list."""


CONTEXT_KEYS = (AnnotationKey.META_INFO, AnnotationKey.DOMAIN, AnnotationKey.DATASET)
"""The annotations of an input Record that its task gives the model as context."""


@unique
class TaskKey(StrEnum):
    """A key of the task metadata the connector writes. Metadata is provenance, never model input."""

    TASK = "task"
    """The row's published ``task``: one of the paper's six tasks."""
    QUESTION_TYPE = "question_type"
    """The row's published ``question_type``, which 10,000 ``ordering`` rows give wrongly."""
    QUESTION_FORMAT = "question_format"
    """The format the question has, read from its text: ``true_or_false``, ``multiple_choice`` or ``puzzling``."""
    FILE = "file"
    """The row's source file, as named in the Hub repository."""
    ROW = "row"
    """The row's position in its file, counted from 0."""
    TRANSFORM = "transform"
    """A data transformation question's transform, as the question names it."""
    ANSWER_TEXT = "answer_text"
    """A multiple-choice question's right choice, when its choices are text rather than series."""
    SOURCE_ISSUES = "source_issues"
    """The row's source issues, the names of ``issues.SourceIssue``: an empty list when it has none."""


SERIES = "series"
"""The name of every Record's one Source and its one Signal."""
