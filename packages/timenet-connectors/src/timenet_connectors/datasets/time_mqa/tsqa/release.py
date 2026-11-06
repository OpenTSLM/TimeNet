"""The parts of the TSQA release that every task family shares: its files, rows, and printed series.

Each CSV row of the release is one question-and-answer pair. The ``QA_list`` cell writes the pair
as ``"question": "<text>", "answer": "<text>"``, with both texts as they are. A quote or a line
break in a text therefore does not make the cell valid JSON, and this module reads the cell with
its own pattern. A question prints each time series as a list of decimal numbers in square
brackets. This module finds the printed series, builds their Records, and writes the prompt with a
placeholder in place of each series.
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
from timenet.json import JsonMapping
from timenet.types import LabelSet, Task, TimeSeriesSpec
from timenet_connectors.bases.huggingface import HubFileRef


HF_REPO = "Time-MQA/TSQA"
SOURCE_REVISION = "a7f9e95667b695d2647b9eff92bb6a38b5183a34"
"""The commit of the Hub repo that the connector reads."""

# The release states no unit and no sampling rate for a printed series.
SERIES = TimeSeriesSpec(spec_type="series", name="Series", unit_value=None, dtype="float64")
# An imputation question prints a gap marker in place of each missing value.
SERIES_WITH_GAPS = TimeSeriesSpec(
    spec_type="series_with_gaps", name="Series with gaps", unit_value=None, dtype="float64", nullable=True
)
GAP_MARKER = "'X'"
"""How an imputation question prints a missing value."""


@unique
class TaskFamily(StrEnum):
    """The task families of the release, each named as the ``task_type`` column writes it."""

    ANOMALY_DETECTION = "anomaly detection"
    CLASSIFICATION = "classification"
    FORECASTING = "forecasting"
    IMPUTATION = "imputation"


@unique
class ReleasePart(StrEnum):
    """The CSV files of the release, named as the parts of the connector."""

    ANOMALY_DETECTION = "anomaly_detection"
    CLASSIFICATION = "classification"
    FORECASTING_IMPUTATION_1 = "forecasting_imputation_1"
    FORECASTING_IMPUTATION_2 = "forecasting_imputation_2"

    @property
    def filename(self) -> str:
        """The path of the file in the Hub repo."""
        return _FILENAMES[self]

    @property
    def columns(self) -> tuple[str, ...]:
        """The header of the file."""
        return _COLUMNS

    @property
    def families(self) -> frozenset[TaskFamily]:
        """The task families that the rows of the file belong to."""
        return _FAMILIES[self]

    def family_of_task_type(self, task_type: str) -> TaskFamily:
        """Return the family of a row of this file from its ``task_type`` cell.

        Returns:
            The family.

        Raises:
            TimeNetConnectorError: If the file holds no rows of the task type.
        """
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
}
_FORECASTING_AND_IMPUTATION = frozenset({TaskFamily.FORECASTING, TaskFamily.IMPUTATION})
_FAMILIES: Mapping[ReleasePart, frozenset[TaskFamily]] = {
    ReleasePart.ANOMALY_DETECTION: frozenset({TaskFamily.ANOMALY_DETECTION}),
    ReleasePart.CLASSIFICATION: frozenset({TaskFamily.CLASSIFICATION}),
    ReleasePart.FORECASTING_IMPUTATION_1: _FORECASTING_AND_IMPUTATION,
    ReleasePart.FORECASTING_IMPUTATION_2: _FORECASTING_AND_IMPUTATION,
}
_COLUMNS = ("application_domain", "task_type", "QA_list")
SOURCE_FILES: Mapping[str, str] = {part.value: part.filename for part in ReleasePart}
"""The file of each part, for the Hub download."""

# The cell writes the two texts as they are, so the question runs up to the one separator.
_QUESTION_AND_ANSWER = re.compile(r'"question": "(?P<question>.*?)", "answer": "(?P<answer>.*)"', re.DOTALL)
_BRACKETED_LIST = re.compile(r"\[([^\[\]]*)\]")
_DECIMAL_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class PrintedSeries:
    """A series that a text prints as a bracketed list of decimal numbers."""

    start: int
    end: int
    """The character span of the list in the text, brackets included."""
    values: tuple[float | None, ...]
    """The printed values; None where the text prints the gap marker."""

    @property
    def has_gaps(self) -> bool:
        """Whether the text prints the gap marker in place of at least one value."""
        return any(value is None for value in self.values)

    def create_record(self, metadata: JsonMapping) -> Record:
        """Create the Record of the series: one float64 Signal on an ordinal axis, nullable if the series has gaps.

        Returns:
            The Record, not yet added to a dataset.
        """
        spec = SERIES_WITH_GAPS if self.has_gaps else SERIES
        signal = Signal.from_values(self.values, spec=spec, name="series", time_axis=OrdinalAxis())
        return Record(sources=(Source(name="TSQA", signals=(signal,)),), metadata=dict(metadata))


def find_printed_series(text: str, *, with_gaps: bool = False) -> tuple[PrintedSeries, ...]:
    """Find each bracketed list of decimal numbers in a text, in text order.

    A bracketed list counts only if it is nonempty and each item is a decimal number or, with gaps,
    the gap marker. Any other bracketed text, such as ``[?, ?, ?]`` or a citation, stays text.

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
    """Return the text with ``<series N>`` in place of each of the series, numbered from 1 in text order.

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
    """One question-and-answer row of a release file, as the file writes it."""

    part: ReleasePart
    number: int
    """The data row number in the file, counted from 1."""
    application_domain: str
    task_type: str
    question: str
    answer: str

    @property
    def family(self) -> TaskFamily:
        """The task family of the row.

        Raises:
            TimeNetConnectorError: If the file holds no rows of the row's task type.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by family_of_task_type
        try:
            return self.part.family_of_task_type(self.task_type)
        except TimeNetConnectorError as exc:
            raise self.malformed(str(exc)) from exc

    def malformed(self, message: str) -> TimeNetConnectorError:
        """Return the error for a malformed row, with the file and the data row in its message.

        Returns:
            The error, for the caller to raise.
        """
        return TimeNetConnectorError(f"{Path(self.part.filename).name} data row {self.number}: {message}")

    def task_metadata(self) -> JsonMapping:
        """Return the provenance of the row for its task: the file, the row, and the release columns.

        Returns:
            The metadata.
        """
        return {
            "file": self.part.value,
            "row": self.number,
            "application_domain": self.application_domain,
            "task_type": self.task_type,
        }

    def record_metadata(self, role: str) -> JsonMapping:
        """Return the provenance of a Record of the row: the file, the row, and the role of the Record.

        Returns:
            The metadata.
        """
        return {"file": self.part.value, "row": self.number, "role": role}

    def only_printed_series_of_question(self, *, with_gaps: bool = False) -> PrintedSeries:
        """Return the one series that the question prints.

        Returns:
            The printed series.

        Raises:
            TimeNetConnectorError: If the question prints no series or more than one.
        """  # ruff: ignore[docstring-extraneous-exception] - raised through malformed()
        return self._only_printed_series("question", self.question, with_gaps=with_gaps)

    def only_printed_series_of_answer(self) -> PrintedSeries:
        """Return the one series that the answer prints.

        Returns:
            The printed series.

        Raises:
            TimeNetConnectorError: If the answer prints no series or more than one.
        """  # ruff: ignore[docstring-extraneous-exception] - raised through malformed()
        return self._only_printed_series("answer", self.answer, with_gaps=False)

    def _only_printed_series(self, text_name: str, text: str, *, with_gaps: bool) -> PrintedSeries:
        series = find_printed_series(text, with_gaps=with_gaps)
        if len(series) != 1:
            raise self.malformed(f"the {text_name} prints {len(series)} series, not one")
        return series[0]

    def question_with_series_placeholders(self, series: Sequence[PrintedSeries]) -> str:
        """Return the question with ``<series N>`` in place of each of the series.

        Returns:
            The prompt.
        """
        return replace_printed_series_with_placeholders(self.question, series)

    def match_of_answer_template(self, template: re.Pattern[str]) -> re.Match[str]:
        """Return the match of the whole answer against the template of its family.

        Returns:
            The match.

        Raises:
            TimeNetConnectorError: If the answer does not follow the template.
        """  # ruff: ignore[docstring-extraneous-exception] - raised through malformed()
        match = template.fullmatch(self.answer)
        if match is None:
            raise self.malformed(f"answer {self.answer!r} does not follow the template {template.pattern!r}")
        return match

    def label_of_answer(self, template: re.Pattern[str], label_set: LabelSet) -> str:
        """Return the label that the answer names in the ``label`` group of the template.

        Returns:
            The label.

        Raises:
            TimeNetConnectorError: If the answer does not follow the template or names no label of the set.
        """  # ruff: ignore[docstring-extraneous-exception] - raised through malformed()
        label = self.match_of_answer_template(template)["label"]
        if label not in label_set.names:
            raise self.malformed(f"label {label!r} is not one of {label_set.names}")
        return label


class TaskRow(Protocol):
    """A row of one task family, parsed and ready to add its Records and task to a dataset."""

    def add_records_and_task_to(self, dataset: TimeFDataset) -> Task:
        """Add the Records of the row and then its task to the dataset.

        Returns:
            The added task.
        """
        ...


@dataclass(frozen=True)
class ReleaseFile:
    """One CSV file of the release, downloaded from the pinned revision."""

    part: ReleasePart
    path: Path

    @classmethod
    def from_hub_file_ref(cls, ref: HubFileRef) -> Self:
        """Return the file of a downloaded part.

        Returns:
            The file.
        """
        return cls(part=ReleasePart(ref.part), path=ref.path)

    def read_rows(self) -> Iterator[ReleaseRow]:
        """Read the question-and-answer rows of the file, in file order.

        A row with an empty ``QA_list`` cell holds no question, so this method skips it. The
        open-ended file has 16 such rows.

        Yields:
            Each row.

        Raises:
            TimeNetConnectorError: If the header is not the expected one, or a ``QA_list`` cell does not
                have the question-and-answer form. The message names the file and the data row.
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
                )
