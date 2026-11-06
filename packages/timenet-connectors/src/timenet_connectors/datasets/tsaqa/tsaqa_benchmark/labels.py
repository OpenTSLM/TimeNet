"""The fixed values of TSAQA: its split files, tasks, roles, question formats, label sets, and specs."""

from enum import StrEnum, unique

from timenet.types import Label, LabelSet, Split, TimeSeriesSpec


@unique
class SplitFile(StrEnum):
    """A split file of the release, named as in the Hub repository without ``.parquet``."""

    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"

    @property
    def filename(self) -> str:
        """The parquet filename for this split."""
        return f"{self.value}.parquet"


SPLITS = {SplitFile.TRAIN: Split.TRAIN, SplitFile.VALIDATION: Split.VALIDATION, SplitFile.TEST: Split.TEST}
"""The TimeNet split of each split file."""


@unique
class AnalysisTask(StrEnum):
    """The six source tasks of the release, as its ``task`` column names them."""

    ANOMALY_DETECTION = "anomaly_detection"
    CLASSIFICATION = "classification"
    CHARACTERIZATION = "characterization"
    COMPARISON = "comparison"
    DATA_TRANSFORMATION = "data_transformation"
    TEMPORAL_RELATIONSHIP = "temporal_relationship"


@unique
class Role(StrEnum):
    """The role of a Record that is not a printed candidate series, which has its letter as its role."""

    INPUT = "input"
    SECOND_INPUT = "second_input"
    SEQUENCE = "sequence"
    """The series a true-or-false question prints, which has no letter."""


TRUE_OR_FALSE_LABEL_SET = LabelSet(
    name="true_or_false",
    labels=(Label(name="T", description="True."), Label(name="F", description="False.")),
)
"""The label set that every true-or-false task shares."""

CONTEXT_KEYS = ("meta_info", "domain", "dataset")
"""The published columns an input Record is annotated with, and its task gives the model as context."""

PUBLISHED_LISTS = {key: f"published_{key}" for key in CONTEXT_KEYS}
"""The annotation key of each context column for a row that publishes the column as a list."""

LETTERS = ("A", "B", "C", "D")
"""A puzzle's patch letters, and the most letters a multiple-choice question offers."""

TRUE_OR_FALSE_LETTERS = TRUE_OR_FALSE_LABEL_SET.names
"""The letters of a true-or-false question: T stands for 'True.' and F for 'False.'"""

INPUT_ROLES = (Role.INPUT, Role.SECOND_INPUT)
"""The roles of the input series, in order."""

PRINTING_TASKS = {AnalysisTask.DATA_TRANSFORMATION, AnalysisTask.TEMPORAL_RELATIONSHIP}
"""The tasks whose questions print candidate series of their own."""


@unique
class QuestionFormat(StrEnum):
    """The format a question has, read from its text: the paper's TF, MC or PZ."""

    TRUE_OR_FALSE = "true_or_false"
    MULTIPLE_CHOICE = "multiple_choice"
    PUZZLING = "puzzling"


QUESTION_TYPES = {
    QuestionFormat.TRUE_OR_FALSE: "true_or_false",
    QuestionFormat.MULTIPLE_CHOICE: "multiple_choices",
    QuestionFormat.PUZZLING: "ordering",
}
"""The ``question_type`` the release gives each question format when it names it right."""

TRANSFORMS = {
    "Fourier Transform": ("tsaqa_fourier_transform", "Fourier transform"),
    "Wavelet Transform": ("tsaqa_wavelet_transform", "Wavelet transform"),
    "First Order Difference": ("tsaqa_first_order_difference", "First-order difference"),
}

SERIES_SPEC = TimeSeriesSpec(spec_type="tsaqa_series", name="TSAQA series", unit_value=None, dtype="float64")

TRANSFORM_SPECS = {
    transform: TimeSeriesSpec(spec_type=spec_type, name=f"{label} of a TSAQA series", unit_value=None, dtype="float64")
    for transform, (spec_type, label) in TRANSFORMS.items()
}


@unique
class SourceIssue(StrEnum):
    """A problem in a published row, which the connector keeps as published and names in the task's metadata."""

    QUESTION_TYPE_MISMATCH = "question_type_mismatch"
    """The published ``question_type`` names another format than the question has."""
    REPEATED_QUESTION = "repeated_question"
    """Another row asks the same question about the same series, and expects the same answer."""
    REPEATED_ACROSS_SPLITS = "repeated_across_splits"
    """A copy of the row's question and series is in another split."""
    CONFLICTING_ANSWER = "conflicting_answer"
    """Another row asks the same question about the same series, and expects another answer."""
    AMBIGUOUS_ANSWER = "ambiguous_answer"
    """Another answer is as right as the published one: two patches, or a choice and the right one, are equal."""
    WRONG_TRANSFORM_TARGET = "wrong_transform_target"
    """The input of a data transformation question is all zeros, which every transform maps to zeros,
    and the published answer says otherwise."""


SOURCE_ISSUES = "source_issues"
"""The task metadata key that lists the row's source issues."""
