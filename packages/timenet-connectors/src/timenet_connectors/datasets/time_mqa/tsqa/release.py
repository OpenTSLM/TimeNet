"""Read TSQA release files, question-and-answer rows, and printed series.

A ``QA_list`` cell holds ``"question": "<text>", "answer": "<text>"`` with both texts as they are.
A quote or a line break in a text makes the cell invalid JSON, so this module reads it with its
own pattern. A question prints each series as a bracketed list of decimal numbers.
"""

from collections.abc import Iterator, Mapping, Sequence
import csv
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
import re
from typing import Protocol, Self

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeNetConnectorError
from timenet.json import JsonMapping, JsonValue
from timenet.types import LabelSet, Task, TimeSeriesSpec
from timenet_connectors.bases.huggingface import HubFileRef


HF_REPO = "Time-MQA/TSQA"
SOURCE_REVISION = "a7f9e95667b695d2647b9eff92bb6a38b5183a34"
"""The pinned Hub repository commit."""

# The release states no unit and no sampling rate for a printed series.
SERIES = TimeSeriesSpec(spec_type="series", name="Series", unit_value=None, dtype="float64")
# An imputation question prints a gap marker in place of each missing value.
SERIES_WITH_GAPS = TimeSeriesSpec(
    spec_type="series_with_gaps", name="Series with gaps", unit_value=None, dtype="float64", nullable=True
)
GAP_MARKER = "'X'"
"""The marker for a missing value in an imputation question."""


@unique
class TaskFamily(StrEnum):
    """The release task families, named by their ``task_type`` values."""

    ANOMALY_DETECTION = "anomaly detection"
    CLASSIFICATION = "classification"
    FORECASTING = "forecasting"
    IMPUTATION = "imputation"
    OPEN_ENDED_QA = "open-ended QA"
    """Open-ended questions use free-text task types, such as ``trend analysis``."""


@unique
class ReleasePart(StrEnum):
    """The CSV file parts of the release."""

    ANOMALY_DETECTION = "anomaly_detection"
    CLASSIFICATION = "classification"
    FORECASTING_IMPUTATION_1 = "forecasting_imputation_1"
    FORECASTING_IMPUTATION_2 = "forecasting_imputation_2"
    OPEN_ENDED_QA = "open_ended_qa"

    @property
    def filename(self) -> str:
        """The file path in the Hub repository."""
        return _FILENAMES[self]

    @property
    def columns(self) -> tuple[str, ...]:
        """The header of the file. Only the open-ended file has a ``question_format`` column."""
        return _COLUMNS_WITH_QUESTION_FORMAT if self is ReleasePart.OPEN_ENDED_QA else _COLUMNS

    @property
    def families(self) -> frozenset[TaskFamily]:
        """The task families in the file."""
        return _FAMILIES[self]

    def family_of_task_type(self, task_type: str) -> TaskFamily:
        """Return the task family for a ``task_type`` value in this file.

        Returns:
            The task family.

        Raises:
            TimeNetConnectorError: If the task type is invalid for this file.
        """
        if self is ReleasePart.OPEN_ENDED_QA:
            return TaskFamily.OPEN_ENDED_QA
        try:
            family = TaskFamily(task_type)
        except ValueError:
            family = None
        if family is None or family not in self.families:
            raise TimeNetConnectorError(f"task_type {task_type!r} is not one of {sorted(self.families)}")
        return family


_FILENAMES: Mapping[ReleasePart, str] = {
    ReleasePart.ANOMALY_DETECTION: "Anomaly_Detection/anomaly_detection.csv",
    ReleasePart.CLASSIFICATION: "Classification/classification.csv",
    ReleasePart.FORECASTING_IMPUTATION_1: "Forecasting+Imputation/Forecasting+Imputation_1/forecasting_imputation1.csv",
    ReleasePart.FORECASTING_IMPUTATION_2: "Forecasting+Imputation/Forecasting+Imputation_2/forecasting_imputation2.csv",
    ReleasePart.OPEN_ENDED_QA: "Open_Ended_QA/open_ended_QA.csv",
}
_FORECASTING_AND_IMPUTATION = frozenset({TaskFamily.FORECASTING, TaskFamily.IMPUTATION})
_FAMILIES: Mapping[ReleasePart, frozenset[TaskFamily]] = {
    ReleasePart.ANOMALY_DETECTION: frozenset({TaskFamily.ANOMALY_DETECTION}),
    ReleasePart.CLASSIFICATION: frozenset({TaskFamily.CLASSIFICATION}),
    ReleasePart.FORECASTING_IMPUTATION_1: _FORECASTING_AND_IMPUTATION,
    ReleasePart.FORECASTING_IMPUTATION_2: _FORECASTING_AND_IMPUTATION,
    ReleasePart.OPEN_ENDED_QA: frozenset({TaskFamily.OPEN_ENDED_QA}),
}
_COLUMNS = ("application_domain", "task_type", "QA_list")
_COLUMNS_WITH_QUESTION_FORMAT = ("application_domain", "task_type", "question_format", "QA_list")
SOURCE_FILES: Mapping[str, str] = {part.value: part.filename for part in ReleasePart}
"""The Hub file path for each release part."""

# The cell writes the two texts as they are, so the question runs up to the one separator.
_QUESTION_AND_ANSWER = re.compile(r'"question": "(?P<question>.*?)", "answer": "(?P<answer>.*)"', re.DOTALL)
_BRACKETED_LIST = re.compile(r"\[([^\[\]]*)\]")
_DECIMAL_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class PrintedSeries:
    """A series represented as a bracketed list of decimal numbers in text."""

    start: int
    end: int
    """The character span of the list in the text, including brackets."""
    values: tuple[float | None, ...]
    """The values, with None for each gap marker."""

    @property
    def has_gaps(self) -> bool:
        """Whether the series contains a gap marker."""
        return any(value is None for value in self.values)

    def create_record(self, metadata: JsonMapping) -> Record:
        """Create a record with one float64 signal on an ordinal axis, nullable if the series has gaps.

        Returns:
            The record, which is not yet in a dataset.
        """
        spec = SERIES_WITH_GAPS if self.has_gaps else SERIES
        signal = Signal.from_values(self.values, spec=spec, name="series", time_axis=OrdinalAxis())
        return Record(sources=(Source(name="TSQA", signals=(signal,)),), metadata=dict(metadata))


def find_printed_series(text: str, *, with_gaps: bool = False) -> tuple[PrintedSeries, ...]:
    """Find nonempty bracketed lists of decimal numbers in text order.

    When ``with_gaps`` is true, a list can also contain gap markers. Other bracketed text, such as
    ``[?, ?, ?]`` or a citation, stays text.

    Returns:
        The printed series.
    """
    found = []
    for match in _BRACKETED_LIST.finditer(text):
        items = [item.strip() for item in match[1].split(",")]
        if match[1].strip() and all(
            _DECIMAL_NUMBER.fullmatch(item) or (with_gaps and item == GAP_MARKER) for item in items
        ):
            values = tuple(None if item == GAP_MARKER else float(item) for item in items)
            found.append(PrintedSeries(match.start(), match.end(), values))
    return tuple(found)


def replace_printed_series_with_placeholders(text: str, series: Sequence[PrintedSeries]) -> str:
    """Replace each supplied series with ``<series N>``, numbered from 1 in sequence order.

    Returns:
        The text with placeholders.
    """
    pieces = []
    cursor = 0
    for number, printed in enumerate(series, start=1):
        pieces.append(text[cursor : printed.start])
        pieces.append(f"<series {number}>")
        cursor = printed.end
    pieces.append(text[cursor:])
    return "".join(pieces)


@dataclass(frozen=True)
class ReleaseRow:
    """A question-and-answer row from a release file."""

    part: ReleasePart
    number: int
    """The data row number in the file, starting at 1."""
    application_domain: str
    task_type: str
    question: str
    answer: str
    question_format: str | None = None
    """The question format for an open-ended row, or None for other rows."""

    @property
    def family(self) -> TaskFamily:
        """The task family of the row.

        Raises:
            TimeNetConnectorError: If the task type is invalid for the file.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by family_of_task_type
        try:
            return self.part.family_of_task_type(self.task_type)
        except TimeNetConnectorError as exc:
            raise self.malformed(str(exc)) from exc

    def malformed(self, message: str) -> TimeNetConnectorError:
        """Create an error with the file name and data row number.

        Returns:
            The error.
        """
        return TimeNetConnectorError(f"{Path(self.part.filename).name} data row {self.number}: {message}")

    def task_metadata(self) -> JsonMapping:
        """Return the file, row number, and release columns as task metadata.

        Returns:
            The metadata.
        """
        metadata: dict[str, JsonValue] = {
            "file": self.part.value,
            "row": self.number,
            "application_domain": self.application_domain,
            "task_type": self.task_type,
        }
        if self.question_format is not None:
            metadata["question_format"] = self.question_format
        return metadata

    def record_metadata(self, role: str) -> JsonMapping:
        """Return the file, row number, and record role as metadata.

        Returns:
            The metadata.
        """
        return {"file": self.part.value, "row": self.number, "role": role}

    def only_printed_series_of_question(self, *, with_gaps: bool = False) -> PrintedSeries:
        """Return the single series in the question.

        Returns:
            The printed series.

        Raises:
            TimeNetConnectorError: If the question does not contain exactly one series.
        """  # ruff: ignore[docstring-extraneous-exception] - raised through malformed()
        return self._only_printed_series("question", self.question, with_gaps=with_gaps)

    def only_printed_series_of_answer(self) -> PrintedSeries:
        """Return the single series in the answer.

        Returns:
            The printed series.

        Raises:
            TimeNetConnectorError: If the answer does not contain exactly one series.
        """  # ruff: ignore[docstring-extraneous-exception] - raised through malformed()
        return self._only_printed_series("answer", self.answer, with_gaps=False)

    def _only_printed_series(self, text_name: str, text: str, *, with_gaps: bool) -> PrintedSeries:
        series = find_printed_series(text, with_gaps=with_gaps)
        if len(series) != 1:
            raise self.malformed(f"the {text_name} prints {len(series)} series, not one")
        return series[0]

    def question_with_series_placeholders(self, series: Sequence[PrintedSeries]) -> str:
        """Replace each supplied series in the question with a ``<series N>`` placeholder.

        Returns:
            The prompt.
        """
        return replace_printed_series_with_placeholders(self.question, series)

    def match_of_answer_template(self, template: re.Pattern[str]) -> re.Match[str]:
        """Match the whole answer against a pattern.

        Returns:
            The match.

        Raises:
            TimeNetConnectorError: If the answer does not match the pattern.
        """  # ruff: ignore[docstring-extraneous-exception] - raised through malformed()
        match = template.fullmatch(self.answer)
        if match is None:
            raise self.malformed(f"answer {self.answer!r} does not follow the template {template.pattern!r}")
        return match

    def label_of_answer(self, template: re.Pattern[str], label_set: LabelSet) -> str:
        """Return the answer label from the pattern's ``label`` group.

        Returns:
            The label.

        Raises:
            TimeNetConnectorError: If the answer does not match the pattern or the label is not in the set.
        """  # ruff: ignore[docstring-extraneous-exception] - raised through malformed()
        label = self.match_of_answer_template(template)["label"]
        if label not in label_set.names:
            raise self.malformed(f"label {label!r} is not one of {label_set.names}")
        return label


class TaskRow(Protocol):
    """A parsed row that can add its records and task to a dataset."""

    def add_records_and_task_to(self, dataset: TimeFDataset) -> Task:
        """Add the row's records and task to the dataset.

        Returns:
            The added task.
        """
        ...


@dataclass(frozen=True)
class ReleaseFile:
    """A release CSV file downloaded from the pinned revision."""

    part: ReleasePart
    path: Path

    @classmethod
    def from_hub_file_ref(cls, ref: HubFileRef) -> Self:
        """Create a release file from a downloaded file reference.

        Returns:
            The release file.
        """
        return cls(part=ReleasePart(ref.part), path=ref.path)

    def read_rows(self) -> Iterator[ReleaseRow]:
        """Read question-and-answer rows in file order, skipping empty ``QA_list`` cells.

        Yields:
            Each row.

        Raises:
            TimeNetConnectorError: If the header or a nonempty ``QA_list`` cell has an invalid format.
        """
        with self.path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != self.part.columns:
                raise TimeNetConnectorError(
                    f"{self.path.name}: expected columns {self.part.columns}, got {reader.fieldnames}"
                )
            for number, cells in enumerate(reader, start=1):
                if not cells["QA_list"]:
                    continue
                match = _QUESTION_AND_ANSWER.fullmatch(cells["QA_list"])
                if match is None:
                    raise TimeNetConnectorError(
                        f'{self.path.name} data row {number}: QA_list does not have the form "question": ..., "answer": ...'
                    )
                yield ReleaseRow(
                    part=self.part,
                    number=number,
                    application_domain=cells["application_domain"],
                    task_type=cells["task_type"],
                    question=match["question"],
                    answer=match["answer"],
                    question_format=cells.get("question_format"),
                )
