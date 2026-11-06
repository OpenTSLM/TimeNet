"""Convert ARFBench questions, metric series, and charts to TimeF."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import IrregularAxis
from timenet.errors import TimeNetDownloadError
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
from timenet_connectors.time_axes import axis_for_offsets, offsets_from_origin


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

REPO = "Datadog/ARFBench"
REVISION = "1cc8ae54e9633f596b755f7c4bce54ccb0cb9f5a"
QA_CSV = "arfbench-qa.csv"
SERIES_DIR = "arfbench-ts-data"  # {metric}_{interval}.parquet: one metric at one rollup interval, in seconds
CHART_DIR = "arfbench-images"  # {metric}.png, and {metric}-{metric}.png for the pairs the questions cite

US_PER_S = 1_000_000
UNTAGGED = "untagged"  # the source name for the empty tag label of a metric with no grouping
_FLAGS = ("interpolate_1", "interpolate_2")

# The release states no unit for any metric, and it stores values as float64. A rollup file lists
# every tag group at every step of one time grid, and a step at which the metric reported nothing
# for a group holds a null. Inside a group's observed window such a null stays a missing timestep.
# In the 1.0.0 release, 42 of 142 metrics have such gaps, and a sparse metric misses up to 99% of
# the steps of a fine grid, so the bound is the whole series.
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
        max_fraction=1.0,
    ),
)


@dataclass(frozen=True)
class ARFBenchSource:
    """Paths to the ARFBench question table, series files, and charts."""

    qa_csv: Path
    series_dir: Path
    chart_dir: Path


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
        """Build records for the metrics and charts, and two tasks per question.

        Args:
            raw_refs: A list containing the paths to the source files.

        Returns:
            The populated dataset.
        """
        source = raw_refs[0]
        dataset = TimeFDataset(metadata=self.metadata())
        metrics = {
            metric: dataset.add_record(record=_metric_record(metric, files))
            for metric, files in _series_files(source.series_dir).items()
        }
        charts = {
            path.stem: dataset.add_record(record=_chart_record(path)) for path in sorted(source.chart_dir.glob("*.png"))
        }
        dataset.add_tasks(tasks=_tasks(source.qa_csv, metrics, charts))
        return dataset


def _series_files(series_dir: Path) -> dict[str, dict[int, Path]]:
    """Group the series files by metric.

    Returns:
        Each metric's files keyed by rollup interval in seconds, finest first.
    """
    files: dict[str, dict[int, Path]] = {}
    for path in series_dir.glob("*.parquet"):
        metric, _, interval_s = path.stem.rpartition("_")
        files.setdefault(metric, {})[int(interval_s)] = path
    return {metric: dict(sorted(intervals.items())) for metric, intervals in sorted(files.items())}


@dataclass(frozen=True)
class TagGroupSeries:
    """One tag group's values at one rollup interval, cut to the window in which the group has a value."""

    epochs_us: np.ndarray
    """The timestamp of each grid step, in microseconds since the epoch."""
    values: np.ndarray
    """The metric value at each step. The value at a missing step is undefined."""
    missing: np.ndarray
    """Whether each step is missing: the metric reported nothing for the group at that step."""

    def to_signal(self, *, interval_s: int, origin_us: int, name: str) -> Signal:
        """Build the signal for this group at its rollup interval.

        A group that holds every step of its grid gets a regular axis. A group that skips a step
        keeps its own time offsets. A missing step stays an Arrow null, not a NaN.

        Args:
            interval_s: The rollup interval in seconds, which sets the grid period.
            origin_us: The record's start, which the time offsets count from.
            name: The signal name.

        Returns:
            The signal with missing values and a regular or irregular time axis.
        """
        offsets = offsets_from_origin(self.epochs_us, origin_us)
        present = pa.array(self.values, mask=self.missing)
        axis = axis_for_offsets(offsets, period_us=Fraction(interval_s * US_PER_S))
        if isinstance(axis, IrregularAxis):
            return Signal.from_irregular(present, time_offsets_us=offsets, spec=METRIC, name=name)
        return Signal.from_values(present, spec=METRIC, name=name, time_axis=axis)


def _read_rollup(path: Path) -> dict[str, TagGroupSeries]:
    """Read a series file and group its values by tag.

    The file lists every tag group at every step of one time grid, with a null where the metric
    reported nothing. Steps before a group's first value and after its last one are grid padding,
    so each group is cut to its observed window. A group with no value is left out. A null inside
    the window stays a missing value. Rows are sorted by group and time, because the file order is
    arbitrary.

    Returns:
        Each tag label mapped to its series, in time order.
    """
    table = pq.read_table(path, columns=["epoch", "group", "value"])
    table = table.sort_by([("group", "ascending"), ("epoch", "ascending")])
    labels = table.column("group").to_numpy(zero_copy_only=False)
    epochs = table.column("epoch").to_numpy().astype("datetime64[us]").astype("int64")
    values = table.column("value").to_numpy(zero_copy_only=False)
    missing = table.column("value").is_null().to_numpy(zero_copy_only=False)
    bounds = [0, *(np.flatnonzero(labels[1:] != labels[:-1]) + 1), len(labels)]
    groups = {}
    for start, stop in pairwise(bounds):
        observed = np.flatnonzero(~missing[start:stop])
        if not len(observed):
            continue
        first, last = start + int(observed[0]), start + int(observed[-1]) + 1
        groups[str(labels[start])] = TagGroupSeries(epochs[first:last], values[first:last], missing[first:last])
    return groups


def _metric_record(metric: str, files: Mapping[int, Path]) -> Record:
    """Build a metric record with a source for each tag group.

    Each source holds the group's series at every rollup where the group has a value. A group with
    no value at any rollup gets no source. The record starts at the earliest observation.

    Args:
        metric: The metric ID, such as ``35928_0``.
        files: The metric's series files by interval in seconds, finest first.

    Returns:
        The record with the metric IDs and finest interval as annotations.
    """
    record_id = f"arfbench-{metric}"
    rollups = {interval_s: _read_rollup(path) for interval_s, path in files.items()}
    origin_us = min(int(group.epochs_us[0]) for groups in rollups.values() for group in groups.values())
    sources = []
    for label in sorted({label for groups in rollups.values() for label in groups}):
        signals = tuple(
            groups[label].to_signal(interval_s=interval_s, origin_us=origin_us, name=f"{interval_s}s")
            for interval_s, groups in rollups.items()
            if label in groups
        )
        sources.append(Source(name=label or UNTAGGED, signals=signals))
    record = Record(metadata={"source_record_id": record_id}, start_time=TimeOrigin(origin_us), sources=tuple(sources))
    finest = min(files)
    query_name = pq.read_table(files[finest], columns=["query_name"]).column("query_name")[0].as_py()
    record.add_annotations(
        [
            Annotation(key="metric_id", value=metric),
            Annotation(key="incident_id", value=metric.partition("_")[0]),
            Annotation(key="query_name", value=query_name),
            Annotation(key="finest_interval_s", value=finest, unit="second"),
        ]
    )
    return record


def _chart_record(path: Path) -> Record:
    """Build a record for a chart.

    Returns:
        The record with the chart image as its signal.
    """
    record_id = f"arfbench-chart-{path.stem}"
    signal = image_signal(path, name="chart")
    return Record(
        metadata={"source_record_id": record_id},
        sources=(Source(name="ARFBench chart", signals=(signal,)),),
    )


def _tasks(qa_csv: Path, metrics: Mapping[str, Record], charts: Mapping[str, Record]) -> Iterator[AnswerTask]:
    """Yield a metric task and a chart task for each question.

    Each task uses the question as its prompt and the correct answer as its target. Every task is
    in the test split, because the release is an evaluation set. The chart task lists the combined
    chart of a pair first and then each metric's own chart, in the order the release shows a vision
    model.

    Args:
        qa_csv: The question and answer table.
        metrics: The metric records by metric ID.
        charts: The chart records by file stem.

    Yields:
        The metric task and chart task for each row, in table order.
    """
    with qa_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cited = [part.strip() for part in row["query_group"].split(",")]
            chart_names = [*(["-".join(cited)] if len(cited) > 1 else []), *cited]
            for inputs, modality in (
                ([metrics[metric] for metric in cited], InputModality.TIME_SERIES),
                ([charts[name] for name in chart_names], InputModality.IMAGE),
            ):
                task = AnswerTask(
                    inputs=tuple(inputs),
                    prompt=row["question"],
                    targets=(row["correct_answer"],),
                    input_modalities=frozenset({InputModality.TEXT, modality}),
                    split=Split.TEST,
                )
                for annotation in _question_annotations(row):
                    task.annotate(annotation)
                yield task


def _question_annotations(row: Mapping[str, str]) -> list[Annotation]:
    """Build annotations for a question.

    The interpolation flags say whether the visualization an engineer saw was interpolated. Nothing
    here interpolates.

    Returns:
        The answer options, category, difficulty, and interpolation flags.
    """
    return [
        Annotation(key="answer_options", value=json.loads(row["options_str"])),
        Annotation(key="task_category", value=row["task_category"]),
        Annotation(key="difficulty", value=row["difficulty"]),
        *(Annotation(key=flag, value=row[flag] == "1") for flag in _FLAGS),
    ]


CONNECTOR = ARFBenchConnector
