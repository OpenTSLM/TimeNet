"""The names the connector writes into a dataset, so writing and reading them share one spelling."""

from enum import StrEnum, unique


@unique
class AnnotationKey(StrEnum):
    """The key of an annotation the connector builds. A Record's text keeps its published field name."""

    DESCRIPTION = "description"
    """A series Record's description, the description task's answer."""
    DESCRIPTION_SHORT = "description_short"
    DESCRIPTION_TINY = "description_tiny"
    """A series Record's few-word title, which the authors' question prompts include."""
    CHARACTERISTICS = "characteristics"
    GENERATOR = "generator"
    """The authors' Python code that generates the Record's series: ``generator`` for a series,
    ``new_generator`` for a changed series."""
    METADATA = "metadata"
    """A series Record's published metadata as the authors' question prompts print it: ``str()`` of the
    dict, in published key order."""
    COUNTERFACTUAL = "counterfactual"
    """A changed series Record's change, as a "What if ...?" question."""
    SOURCE_REPOSITORY = "source_repository"
    """The dataset's Hugging Face repository."""
    SOURCE_REVISION = "source_revision"
    """The dataset's pinned commit of that repository."""


@unique
class TaskKey(StrEnum):
    """A key of the task metadata the connector writes. Metadata is provenance, never model input."""

    FILE = "file"
    """The task's source file, as named in the Hub repository."""
    LINE = "line"
    """The task's line in its file, counted from 1."""
    SCENARIO_CATEGORY = "scenario_category"
    """A description task's domain, by which the authors split the description rows."""
    ANSWER_INDEX = "answer_index"
    """A question task's published ``answer_index``: the position of the right option."""
    ANSWER_TEXT = "answer_text"
    """A question task's right option, the text its target letter stands for in the prompt."""
    CATEGORY = "category"
    """A two-series question's published ``category``: ``difference`` or ``similarity``."""
    SOURCE_ISSUES = "source_issues"
    """A question task's source issues, the names of ``issues.SourceIssue``: an empty list when its row
    has none."""


SERIES = "series"
"""The name of every Record's one Source and its one Signal."""
