"""Validate and parse TSAQA source rows and metadata."""

from dataclasses import dataclass
from enum import Enum
import re
from typing import Literal, Self

import pyarrow as pa
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    Json,
    RootModel,
    StrictFloat,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from timenet.errors import TimeFValidationError
from timenet.json import JsonMapping, validate_json_mapping
from timenet.types.wire import NonEmptyString
from timenet_connectors.sources.readers import DataRow


class AnalysisTask(Enum):
    """The six analysis tasks published by TSAQA."""

    ANOMALY_DETECTION = "anomaly_detection"
    CLASSIFICATION = "classification"
    CHARACTERIZATION = "characterization"
    COMPARISON = "comparison"
    DATA_TRANSFORMATION = "data_transformation"
    TEMPORAL_RELATIONSHIP = "temporal_relationship"


class QuestionFormat(Enum):
    """The answer format, distinct from the sometimes incorrect published question type."""

    TRUE_OR_FALSE = "true_or_false"
    MULTIPLE_CHOICE = "multiple_choices"
    ORDERING = "ordering"

    def accepts_choices(self, letters: tuple[str, ...]) -> bool:
        """Whether the question offers the choices required by this format, in order."""
        if self is QuestionFormat.TRUE_OR_FALSE:
            return letters == ("T", "F")
        if self is QuestionFormat.ORDERING:
            return letters == ("A", "B", "C", "D")
        return letters in {
            ("A", "B"),
            ("A", "B", "C"),
            ("A", "B", "C", "D"),
        }

    def num_answer_options(self) -> int:
        """Number of candidate series in a transformation or temporal question."""
        if self is QuestionFormat.TRUE_OR_FALSE:
            return 1
        return 4


class QuestionRepresentation(Enum):
    """Whether the prompt contains published arrays or refers to labelled raw series."""

    PUBLISHED = "published"
    RAW_CANDIDATE_REFERENCES = "raw_candidate_references"


class TSAQAMetadata(BaseModel):
    """Published source context and provenance, kept separate from model inputs."""

    model_config = ConfigDict(extra="forbid")

    file: NonEmptyString
    row: int = Field(strict=True, ge=0)
    task: AnalysisTask
    question_type: QuestionFormat
    question_format: QuestionFormat
    meta_info: NonEmptyString
    domain: NonEmptyString
    dataset: NonEmptyString
    original_question: NonEmptyString
    question_representation: QuestionRepresentation = QuestionRepresentation.PUBLISHED
    series_representation: Literal["raw_ts"] = "raw_ts"
    source_issues: list[Literal["question_type_mismatch", "ambiguous_series_context"]] = Field(default_factory=list)

    def to_timenet_metadata(self) -> JsonMapping:
        """Serialize to TimeNet's JSON metadata contract without changing source strings."""
        return validate_json_mapping(self.model_dump(mode="json"))


class TSAQARawSeries(RootModel[list[StrictFloat]]):
    """One series from TSAQA's ``raw_ts`` JSON field.

    Require at least one finite number. Reject strings, booleans, and missing values instead
    of changing the published observations. Preserve their order when converting to float64.
    """

    model_config = ConfigDict(allow_inf_nan=False)

    root: list[StrictFloat] = Field(min_length=1)

    def to_arrow(self) -> pa.Array:
        """Copy the source values into a float64 Arrow array."""
        return pa.array(self.root, type=pa.float64())


@dataclass(frozen=True)
class SeriesDescription:
    """A time series together with its description extracted from the meta_info column.

    Indices refer to positions in raw_ts, starting at zero. Several indices mean the
    description applies to all of those series, as with patches of one original series.
    None means we do not know which series the description belongs to.
    This is source data, not a TimeNet annotation.
    """

    text: str
    series_indices: tuple[int, ...] | None


class TSAQARow(DataRow):
    """One TSAQA row with its raw series, question, answer, and source metadata."""

    task: AnalysisTask
    question_type: QuestionFormat
    raw_ts: Json[TSAQARawSeries | list[TSAQARawSeries]]
    question: NonEmptyString
    answer: NonEmptyString
    meta_info: NonEmptyString
    domain: NonEmptyString
    dataset: NonEmptyString

    @property
    def series_descriptions(self) -> tuple[SeriesDescription, ...]:
        """Read the descriptions in meta_info and identify which series each describes.

        Returns:
            Each description and the indices of the series it describes. If we cannot tell
            which series it describes, its series_indices is None.

        Raises:
            TimeFValidationError: If meta_info is malformed or has an unsupported description count.
        """
        descriptions = self.parse_meta_info()

        if self.has_one_description_per_series(descriptions):
            return tuple(
                SeriesDescription(text=text, series_indices=(index,)) for index, text in enumerate(descriptions)
            )

        if self.ordering_patches_share_one_description(descriptions):
            patch_indices = tuple(range(len(self.raw_series)))
            shared_description = SeriesDescription(text=descriptions[0], series_indices=patch_indices)
            return (shared_description,)

        if self.candidate_descriptions_have_unresolved_mapping(descriptions):
            return tuple(SeriesDescription(text=text, series_indices=None) for text in descriptions)

        raise TimeFValidationError("meta_info description count does not match the raw series")

    def parse_meta_info(self) -> list[str]:
        """Read meta_info, which contains either one description or a JSON list of descriptions.

        Returns:
            Descriptions in their published order.

        Raises:
            TimeFValidationError: If a JSON list contains invalid descriptions or is malformed.
        """
        if not self.meta_info.lstrip().startswith("["):
            return [self.meta_info]
        try:
            return TypeAdapter(list[NonEmptyString]).validate_json(self.meta_info)
        except ValidationError as exc:
            raise TimeFValidationError("meta_info must contain a JSON list of nonempty descriptions") from exc

    def has_one_description_per_series(self, descriptions: list[str]) -> bool:
        """Whether there is one description for each series, in the same order."""
        return len(descriptions) == len(self.raw_series)

    def ordering_patches_share_one_description(self, descriptions: list[str]) -> bool:
        """Whether one description applies to all patches in an ordering question."""
        return (
            self.task is AnalysisTask.TEMPORAL_RELATIONSHIP
            and self.question_format is QuestionFormat.ORDERING
            and len(descriptions) == 1
        )

    def candidate_descriptions_have_unresolved_mapping(self, descriptions: list[str]) -> bool:
        """Check for candidate questions with four descriptions but five series.

        We have not established which of the five series each description belongs to.
        We keep these descriptions without guessing. Other unexpected counts are rejected.

        Returns:
            Whether the row has this four-descriptions, five-series mismatch.
        """
        uses_candidate_series = self.task in {AnalysisTask.TEMPORAL_RELATIONSHIP, AnalysisTask.DATA_TRANSFORMATION}
        if not uses_candidate_series or self.question_format is not QuestionFormat.MULTIPLE_CHOICE:
            return False
        candidate_count = self.question_format.num_answer_options()
        has_given_series_and_candidates = len(self.raw_series) == 1 + candidate_count
        has_candidate_count_descriptions = len(descriptions) == candidate_count
        return has_given_series_and_candidates and has_candidate_count_descriptions

    @property
    def raw_series(self) -> list[TSAQARawSeries]:
        """All series from raw_ts, including any that are kept but not given to the model."""
        if isinstance(self.raw_ts, TSAQARawSeries):
            return [self.raw_ts]
        return self.raw_ts

    @property
    def choice_letters(self) -> tuple[str, ...]:
        """The choice labels printed at the start of question lines, in source order."""
        choices = re.findall(r"^([A-DFT]):\s*(.+)$", self.question, re.MULTILINE)
        return tuple(letter for letter, _ in choices)

    @property
    def question_format(self) -> QuestionFormat:
        """Resolve the answer format while preserving the published ``question_type``.

        Some questions in this version of TSAQA ask whether a series continues a given time
        series, but their ``question_type`` is incorrectly set to ``"ordering"``. They expect a
        single answer such as ``"T"`` or ``"A"``, rather than an ordering such as ``"B,D,A,C"``.
        Using the incorrect type would reject their published answers and choose the wrong
        TimeNet task type for a T/F question.

        This exception applies only to temporal relationship rows labeled ``ordering`` with a
        ``Choices:`` block and recognized choice labels. Every other row uses its declared type;
        incompatible choices or answers still fail validation. See training fixture rows 12 and
        13 and ``test_question_format_mismatch_keeps_original_label_prompt_and_answer``.

        This property does not modify the question, answer, or source label. ``to_task_metadata``
        stores both the original ``question_type`` and this resolved ``question_format``, adding
        ``question_type_mismatch`` to ``source_issues`` when they differ.
        """
        if self.question_type is not QuestionFormat.ORDERING:
            return self.question_type
        if self.task is not AnalysisTask.TEMPORAL_RELATIONSHIP:
            return self.question_type
        # Real ordering questions present a Patches: block. The affected continuation questions
        # present Choices: instead; inspect their labels only within this known upstream exception.
        if "\nChoices:\n" in self.question:
            if QuestionFormat.TRUE_OR_FALSE.accepts_choices(self.choice_letters):
                return QuestionFormat.TRUE_OR_FALSE
            if QuestionFormat.MULTIPLE_CHOICE.accepts_choices(self.choice_letters):
                return QuestionFormat.MULTIPLE_CHOICE
        return self.question_type

    @property
    def is_ordering_question(self) -> bool:
        """Whether the question asks the model to put the patches in order."""
        return self.question_format is QuestionFormat.ORDERING

    @model_validator(mode="after")
    def _validate_question(self) -> Self:
        self.validate_question_choices()
        self.validate_answer()
        return self

    def validate_question_choices(self) -> None:
        """Require the choice labels and task family specified by the question format."""
        if not self.question_format.accepts_choices(self.choice_letters):
            raise TimeFValidationError("question must offer T/F or consecutive choices starting with A")

    def validate_answer(self) -> None:
        """Require one offered choice, or every patch exactly once for an ordering answer."""
        if self.is_ordering_question:
            self.validate_patch_order()
        elif self.answer not in self.choice_letters:
            raise TimeFValidationError("answer must name one of the question's choices")

    def validate_patch_order(self) -> None:
        """Require the answer to name each offered patch exactly once."""
        ordered_patches = self.answer.split(",")
        if sorted(ordered_patches) != sorted(self.choice_letters):
            raise TimeFValidationError("ordering answer must be a permutation of A,B,C,D")

    def require_analysis_task(self, expected: AnalysisTask) -> None:
        """Require the row to belong to the requested task family."""
        if self.task is not expected:
            raise TimeFValidationError(f"expected {expected.value}, got {self.task.value}")

    def require_question_format(self, *formats: QuestionFormat) -> None:
        """Require a question format supported by the concrete task."""
        if self.question_format not in formats:
            raise TimeFValidationError(f"{self.task.value} does not support {self.question_format.value} questions")

    def require_num_raw_series(self, count: int) -> None:
        """Check that raw_ts contains the number of series this task expects."""
        if len(self.raw_series) != count:
            raise TimeFValidationError(f"{self.task.value} requires {count} raw series in raw_ts")

    def to_task_metadata(self, filename: str, index: int) -> TSAQAMetadata:
        """Keep the original source fields, filename, and row number in task metadata."""
        metadata = TSAQAMetadata(
            file=filename,
            row=index,
            task=self.task,
            question_type=self.question_type,
            question_format=self.question_format,
            meta_info=self.meta_info,
            domain=self.domain,
            dataset=self.dataset,
            original_question=self.question,
        )
        if self.question_type is not self.question_format:
            metadata.source_issues.append("question_type_mismatch")
        return metadata

    def to_arrow_series(self) -> list[pa.Array]:
        """Convert every raw series to float64, keeping their original order."""
        return [series.to_arrow() for series in self.raw_series]
