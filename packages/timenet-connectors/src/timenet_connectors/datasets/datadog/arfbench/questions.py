"""The ARFBench question table: 750 multiple-choice questions, each a series task or a chart task."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
import json
from typing import TYPE_CHECKING, Self, TypeVar

from timenet.errors import TimeNetConnectorError
from timenet.types import Annotation, AnswerTask, InputModality, Split


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from timenet.dataset import Record

ChoiceT = TypeVar("ChoiceT", bound=Enum)

QUESTION_COLUMNS = (
    "Unnamed: 0",
    "question",
    "task_category",
    "difficulty",
    "options_str",
    "correct_answer",
    "query_group",
    "options",
    "interpolate_1",
    "interpolate_2",
)
"""The columns of the question table. ``options`` repeats ``options_str`` as objects and is not read."""

MAX_CITED_METRICS = 2  # a question is about one metric or a pair


class TaskCategory(Enum):
    """The eight kinds of question in the release."""

    ANOMALY_PRESENCE = "Anomaly Presence"
    ANOMALY_START = "Anomaly Start"
    ANOMALY_END = "Anomaly End"
    ANOMALY_MAGNITUDE = "Anomaly Magnitude"
    ANOMALY_CATEGORIZATION = "Anomaly Categorization"
    ANOMALY_IDENTIFICATION = "Anomaly Identification"
    ANOMALY_INDICATOR = "Anomaly Indicator"
    ANOMALY_CORRELATION = "Anomaly Correlation"


class Difficulty(Enum):
    """The three difficulty tiers of the release, from easiest to hardest."""

    TIER_1 = "Tier 1"
    TIER_2 = "Tier 2"
    TIER_3 = "Tier 3"


def _parse_choice(row: Mapping[str, str], column: str, choices: type[ChoiceT]) -> ChoiceT:
    """Parse a cell that must hold one value of a closed set.

    Returns:
        The member of ``choices`` with the cell's value.

    Raises:
        TimeNetConnectorError: If the cell holds another value.
    """
    try:
        return choices(row[column])
    except ValueError as exc:
        raise TimeNetConnectorError(
            f"{column} must be one of {[choice.value for choice in choices]}, got {row[column]!r}"
        ) from exc


def _parse_flag(row: Mapping[str, str], column: str) -> bool:
    """Parse a ``0`` or ``1`` cell.

    Returns:
        Whether the cell is ``1``.

    Raises:
        TimeNetConnectorError: If the cell is neither ``0`` nor ``1``.
    """
    if row[column] not in {"0", "1"}:
        raise TimeNetConnectorError(f"{column} must be 0 or 1, got {row[column]!r}")
    return row[column] == "1"


def _parse_answer_options(cell: str) -> tuple[str, ...]:
    """Parse the ``options_str`` cell, a JSON list of answer texts.

    Returns:
        The answer options in release order.

    Raises:
        TimeNetConnectorError: If the cell is not a non-empty JSON list of strings.
    """
    try:
        options = json.loads(cell)
    except ValueError as exc:
        raise TimeNetConnectorError(f"options_str is not JSON: {cell!r}") from exc
    if not isinstance(options, list) or not options or not all(isinstance(option, str) for option in options):
        raise TimeNetConnectorError(f"options_str must be a non-empty list of strings, got {cell!r}")
    return tuple(options)


@dataclass(frozen=True)
class ARFBenchQuestion:
    """One row of the question table: a multiple-choice question about one or two metrics."""

    index: int
    """The row index of the question in the release table."""
    question: str
    """The question text, which includes the caption of each cited metric."""
    task_category: TaskCategory
    difficulty: Difficulty
    answer_options: tuple[str, ...]
    correct_answer: str
    cited_metrics: tuple[str, ...]
    """The metric IDs the question is about, in the order the question names them: one or two."""
    interpolate_1: bool
    """Whether the visualization that an engineer saw of the first metric was interpolated."""
    interpolate_2: bool
    """Whether the visualization that an engineer saw of the second metric was interpolated."""

    @classmethod
    def from_csv_row(cls, row: Mapping[str, str]) -> Self:
        """Parse one data row of the question table.

        Returns:
            The parsed question.

        Raises:
            TimeNetConnectorError: If the row cites no metric or more than two, if the options are
                malformed, if the correct answer is not an option, if the category or the
                difficulty is unknown, or if a flag is not 0 or 1.
        """
        cited_metrics = tuple(part.strip() for part in row["query_group"].split(","))
        if not all(cited_metrics) or len(cited_metrics) > MAX_CITED_METRICS:
            raise TimeNetConnectorError(f"query_group must name one or two metrics, got {row['query_group']!r}")
        answer_options = _parse_answer_options(row["options_str"])
        if row["correct_answer"] not in answer_options:
            raise TimeNetConnectorError(f"correct_answer {row['correct_answer']!r} is not one of {answer_options}")
        return cls(
            index=int(row["Unnamed: 0"]),
            question=row["question"],
            task_category=_parse_choice(row, "task_category", TaskCategory),
            difficulty=_parse_choice(row, "difficulty", Difficulty),
            answer_options=answer_options,
            correct_answer=row["correct_answer"],
            cited_metrics=cited_metrics,
            interpolate_1=_parse_flag(row, "interpolate_1"),
            interpolate_2=_parse_flag(row, "interpolate_2"),
        )

    def chart_stems(self) -> tuple[str, ...]:
        """Name the charts of the question in the order the release shows a vision model.

        Returns:
            For a pair, the combined chart first and then each metric's own chart. For one
            metric, its own chart.
        """
        if len(self.cited_metrics) == 1:
            return self.cited_metrics
        return ("-".join(self.cited_metrics), *self.cited_metrics)

    def create_series_task(self, metric_records: Mapping[str, Record]) -> AnswerTask:
        """Create the task that asks the question about the cited metric records.

        Args:
            metric_records: The metric Records by metric ID.

        Returns:
            The task, not yet added to a dataset.

        Raises:
            TimeNetConnectorError: If a cited metric has no record.
        """
        missing = [metric for metric in self.cited_metrics if metric not in metric_records]
        if missing:
            raise TimeNetConnectorError(f"question {self.index}: no series file for cited metric {', '.join(missing)}")
        inputs = tuple(metric_records[metric] for metric in self.cited_metrics)
        return self._create_task(inputs, InputModality.TIME_SERIES)

    def create_chart_task(self, chart_records: Mapping[str, Record]) -> AnswerTask:
        """Create the task that asks the question about the released charts.

        Args:
            chart_records: The chart Records by file stem.

        Returns:
            The task, not yet added to a dataset.

        Raises:
            TimeNetConnectorError: If a chart of the question has no record.
        """
        missing = [stem for stem in self.chart_stems() if stem not in chart_records]
        if missing:
            raise TimeNetConnectorError(f"question {self.index}: no chart for {', '.join(missing)}")
        inputs = tuple(chart_records[stem] for stem in self.chart_stems())
        return self._create_task(inputs, InputModality.IMAGE)

    def _create_task(self, inputs: tuple[Record, ...], modality: InputModality) -> AnswerTask:
        """Create a test task with the question as prompt and the correct answer as target.

        Both tasks of a question carry its row index in their metadata, so a consumer can pair
        them. The question's annotations describe it, and the interpolation flags describe what
        an engineer saw. Nothing here interpolates.

        Returns:
            The task, not yet added to a dataset.
        """
        task = AnswerTask(
            inputs=inputs,
            prompt=self.question,
            targets=(self.correct_answer,),
            input_modalities=frozenset({InputModality.TEXT, modality}),
            split=Split.TEST,
            metadata={"question_index": self.index},
        )
        task.annotate(Annotation(key="answer_options", value=list(self.answer_options)))
        task.annotate(Annotation(key="task_category", value=self.task_category.value))
        task.annotate(Annotation(key="difficulty", value=self.difficulty.value))
        task.annotate(Annotation(key="interpolate_1", value=self.interpolate_1))
        task.annotate(Annotation(key="interpolate_2", value=self.interpolate_2))
        return task


def read_questions(qa_csv: Path) -> Iterator[ARFBenchQuestion]:
    """Read the question table, in table order.

    Yields:
        Each parsed question.

    Raises:
        TimeNetConnectorError: If the header is not the expected one, or a row is malformed. The
            message names the file and the data row.
    """
    with qa_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != QUESTION_COLUMNS:
            raise TimeNetConnectorError(f"{qa_csv.name}: expected columns {QUESTION_COLUMNS}, got {reader.fieldnames}")
        for number, row in enumerate(reader, start=1):
            try:
                question = ARFBenchQuestion.from_csv_row(row)
            except TimeNetConnectorError as exc:
                raise TimeNetConnectorError(f"{qa_csv.name} data row {number}: {exc}") from exc
            yield question
