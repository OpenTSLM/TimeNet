"""Convert ARFBench questions, metric series, and charts to TimeF.

The release holds one question table, one series file per metric and rollup interval, and one
chart per metric or cited metric pair. Each metric becomes one Record, each chart one Record, and
each question a series task and a chart task.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
import json
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.connectors import BaseConnector
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeNetConnectorError, TimeNetDownloadError
from timenet.types import (
    Annotation,
    AnswerTask,
    InputModality,
    MissingReason,
    MissingValues,
    Split,
    TimeOrigin,
    TimeSeriesSpec,
)
from timenet_connectors.sources.huggingface_hub import hub_snapshot
from timenet_connectors.sources.images import image_signal


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

REPO = "Datadog/ARFBench"
REVISION = "1cc8ae54e9633f596b755f7c4bce54ccb0cb9f5a"
QA_CSV = "arfbench-qa.csv"
SERIES_DIR = "arfbench-ts-data"  # {metric}_{interval}.parquet: one metric at one rollup interval, in seconds
CHART_DIR = "arfbench-images"  # {metric}.png, and {metric}-{metric}.png for the pairs the questions cite
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

US_PER_S = 1_000_000
UNTAGGED = "untagged"  # the source name for the empty tag label of a metric with no grouping
MAX_CITED_METRICS = 2  # a question is about one metric or a pair

# The release states no unit for any metric, and it stores values as float64. A rollup file lists
# the tag groups of a metric on one time grid. A step at which the metric reported nothing for a
# group holds a null, or the file skips it. Inside a group's observed window both are a missing
# timestep. In the 1.0.0 release, 42 of 142 metrics have such gaps, and a sparse metric misses up
# to 99% of the steps of a fine grid. The release records each series' null count.
METRIC = TimeSeriesSpec(
    spec_type="metric",
    name="Observability metric",
    unit_value=None,
    dtype="float64",
    missing=MissingValues(
        reason=MissingReason.NOT_RECORDED,
        description=(
            "The metric reported no data point for this step of the rollup grid. A sparse metric can "
            "miss almost every step of a fine rollup."
        ),
    ),
)


@dataclass(frozen=True)
class TagGroupSeries:
    """One tag group's values on its rollup grid, from the group's first value to its last."""

    start_us: int
    """The timestamp of the first value, in microseconds since the epoch."""
    values: pa.Array
    """The value at each grid step, with a null at each step without a value."""

    @classmethod
    def from_rows(cls, epochs_us: np.ndarray, values: np.ndarray, missing: np.ndarray, *, period_us: int) -> Self:
        """Place the rows of one group on the grid of its rollup.

        The rows must be in time order and start and end with a value. A step that the rows skip
        is a missing timestep, like a row that holds a null.

        Args:
            epochs_us: The timestamp of each row, in microseconds since the epoch.
            values: The value of each row. The value of a row that holds a null is undefined.
            missing: Whether each row holds a null.
            period_us: The rollup interval in microseconds.

        Returns:
            The group's series on the grid.

        Raises:
            TimeNetConnectorError: If a row is off the grid, or two rows share a step.
        """
        steps = epochs_us - epochs_us[0]
        if np.any(steps % period_us) or np.any(np.diff(steps) <= 0):
            raise TimeNetConnectorError(f"rows are off the {period_us // US_PER_S} s grid or repeat a step")
        steps //= period_us
        grid_values = np.full(int(steps[-1]) + 1, np.nan)
        grid_values[steps] = values
        present = np.zeros(len(grid_values), dtype=bool)
        present[steps[~missing]] = True
        return cls(start_us=int(epochs_us[0]), values=pa.array(grid_values, mask=~present))

    def to_signal(self, *, period_us: int, origin_us: int, name: str) -> Signal:
        """Build the signal for this group at its rollup interval.

        Args:
            period_us: The rollup interval in microseconds.
            origin_us: The record's start, which the axis offset counts from.
            name: The signal name.

        Returns:
            The signal on a regular axis, with a missing timestep at each step without a value.
        """
        axis = RegularAxis(period_us=Fraction(period_us), offset_us=self.start_us - origin_us)
        return Signal.from_values(self.values, spec=METRIC, name=name, time_axis=axis)


@dataclass(frozen=True)
class RollupFile:
    """One series file: the values of one metric at one rollup interval."""

    metric: str
    """The metric ID, such as ``35928_0``."""
    interval_s: int
    """The rollup interval in seconds."""
    path: Path

    @classmethod
    def from_path(cls, path: Path) -> Self:
        """Parse the metric ID and the interval from a file name such as ``35928_0_10.parquet``.

        Returns:
            The rollup file.

        Raises:
            TimeNetConnectorError: If the file name does not end in an interval.
        """
        metric, _, interval_s = path.stem.rpartition("_")
        if not metric or not interval_s.isdigit():
            raise TimeNetConnectorError(f"{path.name} is not named {{metric}}_{{interval}}.parquet")
        return cls(metric=metric, interval_s=int(interval_s), path=path)

    @property
    def period_us(self) -> int:
        """The rollup interval in microseconds."""
        return self.interval_s * US_PER_S

    def read_tag_group_series(self) -> dict[str, TagGroupSeries]:
        """Read the file and cut each tag group to its observed window.

        The file lists the tag groups of the metric on one time grid, with a null where the metric
        reported nothing. Most files list every group at every step. Steps before a group's first
        value and after its last one are grid padding, so each group is cut to its observed window.
        A group with no value is left out. Rows are sorted by group and time, because the file
        order is arbitrary.

        Returns:
            Each tag label mapped to its series on the grid.

        Raises:
            TimeNetConnectorError: If a group has rows off the grid. The message names the file
                and the group.
        """
        table = pq.read_table(self.path, columns=["epoch", "group", "value"])
        table = table.sort_by([("group", "ascending"), ("epoch", "ascending")])
        labels = table.column("group").to_numpy(zero_copy_only=False)
        epochs = table.column("epoch").to_numpy().astype("datetime64[us]").astype("int64")
        values = table.column("value").to_numpy(zero_copy_only=False)
        missing = table.column("value").is_null().to_numpy(zero_copy_only=False)
        bounds = [0, *(np.flatnonzero(labels[1:] != labels[:-1]) + 1), len(labels)]
        groups = {}
        for start, stop in pairwise(bounds):
            observed = np.flatnonzero(~missing[start:stop]) + start
            if not len(observed):
                continue
            window = slice(int(observed[0]), int(observed[-1]) + 1)
            label = str(labels[start])
            try:
                groups[label] = TagGroupSeries.from_rows(
                    epochs[window], values[window], missing[window], period_us=self.period_us
                )
            except TimeNetConnectorError as exc:
                raise TimeNetConnectorError(f"{self.path.name} group {label!r}: {exc}") from exc
        return groups

    def read_query_name(self) -> str:
        """Read the ``query_name`` column, which holds one value per file.

        Returns:
            The identifier that the release gives the metric. It differs from the metric ID for
            most metrics, and the release does not explain it.
        """
        return pq.read_table(self.path, columns=["query_name"]).column("query_name")[0].as_py()


@dataclass(frozen=True)
class MetricSeriesFiles:
    """The series files of one metric, one per rollup interval."""

    metric: str
    """The metric ID, such as ``35928_0``."""
    rollups: tuple[RollupFile, ...]
    """The metric's rollup files, finest interval first."""

    @classmethod
    def from_series_dir(cls, series_dir: Path) -> list[Self]:
        """Group the series files of a directory by metric.

        Returns:
            The metrics in ID order, each with its rollups finest first.
        """
        by_metric: dict[str, list[RollupFile]] = {}
        for path in sorted(series_dir.glob("*.parquet")):
            rollup = RollupFile.from_path(path)
            by_metric.setdefault(rollup.metric, []).append(rollup)
        return [
            cls(metric=metric, rollups=tuple(sorted(rollups, key=lambda rollup: rollup.interval_s)))
            for metric, rollups in sorted(by_metric.items())
        ]

    @property
    def finest(self) -> RollupFile:
        """The rollup file with the smallest interval."""
        return self.rollups[0]

    def create_record(self) -> Record:
        """Create the metric's Record with a Source for each tag group.

        Each source holds the group's series at every rollup where the group has a value. A group
        with no value at any rollup gets no source. The record starts at the earliest observation.

        Returns:
            The Record, not yet added to a dataset, with the metric IDs and the finest interval as
            annotations.
        """
        series_by_rollup = {rollup: rollup.read_tag_group_series() for rollup in self.rollups}
        origin_us = min(group.start_us for groups in series_by_rollup.values() for group in groups.values())
        sources = []
        for label in sorted({label for groups in series_by_rollup.values() for label in groups}):
            signals = tuple(
                groups[label].to_signal(period_us=rollup.period_us, origin_us=origin_us, name=f"{rollup.interval_s}s")
                for rollup, groups in series_by_rollup.items()
                if label in groups
            )
            sources.append(Source(name=label or UNTAGGED, signals=signals))
        record = Record(
            metadata={"source_record_id": f"arfbench-{self.metric}"},
            start_time=TimeOrigin(origin_us),
            sources=tuple(sources),
        )
        record.add_annotations(
            [
                Annotation(key="metric_id", value=self.metric),
                Annotation(key="incident_id", value=self.metric.partition("_")[0]),
                Annotation(key="query_name", value=self.finest.read_query_name()),
                Annotation(key="finest_interval_s", value=self.finest.interval_s, unit="second"),
            ]
        )
        return record


def _create_chart_record(path: Path) -> Record:
    """Create the Record of a chart, with the image as its one signal.

    Returns:
        The Record, not yet added to a dataset.
    """
    return Record(
        metadata={"source_record_id": f"arfbench-chart-{path.stem}"},
        sources=(Source(name="ARFBench chart", signals=(image_signal(path, name="chart"),)),),
    )


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
    task_category: str
    difficulty: str
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
                malformed, if the correct answer is not an option, or if a flag is not 0 or 1.
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
            task_category=row["task_category"],
            difficulty=row["difficulty"],
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
        task.annotate(Annotation(key="task_category", value=self.task_category))
        task.annotate(Annotation(key="difficulty", value=self.difficulty))
        task.annotate(Annotation(key="interpolate_1", value=self.interpolate_1))
        task.annotate(Annotation(key="interpolate_2", value=self.interpolate_2))
        return task


@dataclass(frozen=True)
class ARFBenchSource:
    """Paths to the ARFBench question table, series files, and charts."""

    qa_csv: Path
    series_dir: Path
    chart_dir: Path

    def read_questions(self) -> Iterator[ARFBenchQuestion]:
        """Read the question table, in table order.

        Yields:
            Each parsed question.

        Raises:
            TimeNetConnectorError: If the header is not the expected one, or a row is malformed.
                The message names the file and the data row.
        """
        with self.qa_csv.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != QUESTION_COLUMNS:
                raise TimeNetConnectorError(
                    f"{self.qa_csv.name}: expected columns {QUESTION_COLUMNS}, got {reader.fieldnames}"
                )
            for number, row in enumerate(reader, start=1):
                try:
                    question = ARFBenchQuestion.from_csv_row(row)
                except TimeNetConnectorError as exc:
                    raise TimeNetConnectorError(f"{self.qa_csv.name} data row {number}: {exc}") from exc
                yield question

    def metric_series_files(self) -> list[MetricSeriesFiles]:
        """List the series files, grouped by metric.

        Returns:
            The metrics in ID order.
        """
        return MetricSeriesFiles.from_series_dir(self.series_dir)

    def chart_paths(self) -> list[Path]:
        """List the chart files.

        Returns:
            The PNG files in name order.
        """
        return sorted(self.chart_dir.glob("*.png"))


class ARFBenchConnector(BaseConnector[ARFBenchSource]):
    """Connector for ARFBench (Hub repo ``Datadog/ARFBench``)."""

    values_backend = "zarr"  # the chart tensors need it

    async def download_async(self, cache_dir: Path) -> list[ARFBenchSource]:  # ruff: ignore[no-self-use] - BaseConnector override
        """Download the question table, series files, and charts from the pinned release.

        Args:
            cache_dir: Directory for cached Hub files.

        Returns:
            A list with the paths to the table and the two directories.

        Raises:
            TimeNetDownloadError: If the download fails or the question table is missing.
        """
        root = await hub_snapshot(REPO, REVISION, cache_dir, (QA_CSV, f"{SERIES_DIR}/*.parquet", f"{CHART_DIR}/*.png"))
        if not (root / QA_CSV).is_file():
            raise TimeNetDownloadError(f"{REPO!r} at {REVISION!r} holds no {QA_CSV} under {root}")
        return [ARFBenchSource(qa_csv=root / QA_CSV, series_dir=root / SERIES_DIR, chart_dir=root / CHART_DIR)]

    def convert(self, raw_refs: list[ARFBenchSource]) -> TimeFDataset:
        """Create a Record for each metric and chart, and two tasks for each question.

        Args:
            raw_refs: The one source from :meth:`download_async`.

        Returns:
            The populated dataset.

        Raises:
            TimeNetConnectorError: If there is not exactly one source.
        """
        if len(raw_refs) != 1:
            raise TimeNetConnectorError(f"expected one ARFBench source, got {len(raw_refs)}")
        source = raw_refs[0]
        dataset = TimeFDataset(metadata=self.metadata())
        metric_records = {
            files.metric: dataset.add_record(record=files.create_record()) for files in source.metric_series_files()
        }
        chart_records = {
            path.stem: dataset.add_record(record=_create_chart_record(path)) for path in source.chart_paths()
        }
        for question in source.read_questions():
            dataset.add_task(task=question.create_series_task(metric_records))
            dataset.add_task(task=question.create_chart_task(chart_records))
        return dataset


CONNECTOR = ARFBenchConnector
