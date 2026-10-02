"""Convert ARFBench: 750 questions about Datadog incident metrics, with every published rollup and chart."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import IrregularAxis
from timenet.errors import TimeNetDownloadError
from timenet.types import Annotation, AnswerTask, InputModality, Split, TimeOrigin, TimeSeriesSpec
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

# The release states no unit for any metric, and it stores values as float64. A null value is a
# point the release publishes as missing, so a series keeps the timestep and marks it missing.
METRIC = TimeSeriesSpec(
    spec_type="metric", name="Observability metric", unit_value=None, dtype="float64", nullable=True
)


@dataclass(frozen=True)
class ARFBenchSource:
    """What ``download`` hands ``convert``: paths, and no rows."""

    qa_csv: Path
    series_dir: Path
    chart_dir: Path


class ARFBenchConnector(BaseConnector[ARFBenchSource]):
    """Connector for ARFBench (Hub repo ``Datadog/ARFBench``)."""

    values_backend = "zarr"  # the chart tensors need it

    def download(self, cache_dir: Path) -> list[ARFBenchSource]:  # noqa: PLR6301 - BaseConnector override
        """Fetch the QA table, every series file, and every chart of the pinned release.

        Args:
            cache_dir: Directory where the connector caches Hub files.

        Returns:
            One handle naming the table and the two folders.

        Raises:
            TimeNetDownloadError: If the fetch returned no QA table.
        """
        root = hub_snapshot(REPO, REVISION, cache_dir, (QA_CSV, f"{SERIES_DIR}/*.parquet", f"{CHART_DIR}/*.png"))
        if not (root / QA_CSV).is_file():
            raise TimeNetDownloadError(f"{REPO!r} at {REVISION!r} holds no {QA_CSV} under {root}")
        return [ARFBenchSource(qa_csv=root / QA_CSV, series_dir=root / SERIES_DIR, chart_dir=root / CHART_DIR)]

    def convert(self, raw_refs: list[ARFBenchSource]) -> TimeFDataset:
        """Build one record per metric and one per chart, then two tasks per question.

        Args:
            raw_refs: The single handle :meth:`download` gave back.

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


def _read_rollup(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read one series file into its tag groups.

    Row order in the release is arbitrary, so the rows are sorted by tag group and then by time. The
    file's other columns are not read: ``num_groups`` is the number of rows at that epoch,
    ``query_name`` is the alias the metric's record states once, and the pandas index restores no
    order.

    Returns:
        Each tag label's ``(epochs in microseconds, values, missing mask)``, in time order.
    """
    table = pq.read_table(path, columns=["epoch", "group", "value"])
    table = table.sort_by([("group", "ascending"), ("epoch", "ascending")])
    labels = table.column("group").to_numpy(zero_copy_only=False)
    epochs = table.column("epoch").to_numpy().astype("datetime64[us]").astype("int64")
    values = table.column("value").to_numpy(zero_copy_only=False)
    missing = table.column("value").is_null().to_numpy(zero_copy_only=False)
    bounds = [0, *(np.flatnonzero(labels[1:] != labels[:-1]) + 1), len(labels)]
    return {
        str(labels[start]): (epochs[start:stop], values[start:stop], missing[start:stop])
        for start, stop in pairwise(bounds)
    }


def _metric_record(metric: str, files: Mapping[int, Path]) -> Record:
    """Build one metric's record: a source per tag group, holding that group's series at every rollup it has.

    A metric's tag keys are the same at every rollup, so a tag label names one series that the
    release publishes at several intervals. The record's zero is the metric's earliest observation.

    Args:
        metric: The metric id, such as ``35928_0``.
        files: The metric's series files by interval in seconds, finest first.

    Returns:
        The record, annotated with the metric's ids and its finest interval.
    """
    record_id = f"arfbench-{metric}"
    rollups = {interval_s: _read_rollup(path) for interval_s, path in files.items()}
    origin_us = min(int(epochs[0]) for groups in rollups.values() for epochs, _, _ in groups.values())
    sources = []
    for index, label in enumerate(sorted({label for groups in rollups.values() for label in groups})):
        source_id = f"{record_id}-{index:05d}"
        signals = tuple(
            _signal(
                *groups[label],
                interval_s=interval_s,
                origin_us=origin_us,
                name=f"{interval_s}s",
                id=f"{source_id}-{interval_s}s",
                source_id=files[interval_s].stem,
            )
            for interval_s, groups in rollups.items()
            if label in groups
        )
        sources.append(Source(id=source_id, name=label or UNTAGGED, signals=signals))
    record = Record(record_id=record_id, start_time=TimeOrigin(origin_us), sources=tuple(sources))
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


def _signal(  # noqa: PLR0913 - the group's arrays, and what names the signal
    epochs: np.ndarray,
    values: np.ndarray,
    missing: np.ndarray,
    *,
    interval_s: int,
    origin_us: int,
    name: str,
    id: str,
    source_id: str,
) -> Signal:
    """Build one rollup of one tag group.

    A series that holds every step of its interval's grid gets a regular axis; one that skips a step
    keeps its own time offsets. A missing point stays a missing timestep rather than becoming a NaN.

    Returns:
        The signal.
    """
    offsets = offsets_from_origin(epochs, origin_us)
    present = values
    if missing.any():
        present = values.astype(object)
        present[missing] = None
    axis = axis_for_offsets(offsets, period_us=Fraction(interval_s * US_PER_S))
    if isinstance(axis, IrregularAxis):
        return Signal.from_irregular(
            present, time_offsets_us=offsets, spec=METRIC, name=name, source_id=source_id, id=id
        )
    return Signal.from_values(present, spec=METRIC, name=name, time_axis=axis, source_id=source_id, id=id)


def _chart_record(path: Path) -> Record:
    """Build the record of one released chart: a metric's finest rollup, or a cited pair, as the release plots it.

    Returns:
        The record, holding the image as its one signal.
    """
    record_id = f"arfbench-chart-{path.stem}"
    signal = image_signal(path, signal_id=f"{record_id}-image", name="chart")
    return Record(
        record_id=record_id,
        sources=(Source(id=f"{record_id}-source", name="ARFBench chart", signals=(signal,)),),
    )


def _tasks(qa_csv: Path, metrics: Mapping[str, Record], charts: Mapping[str, Record]) -> Iterator[AnswerTask]:
    """Yield two tasks per question: one over the cited metrics, one over the released charts.

    The question is the prompt and the correct option the target. Every question is a test item, as
    the release is an evaluation set. The chart task lists the combined chart of a pair first and then
    each metric's own chart, the order the release's evaluation shows a vision model.

    Args:
        qa_csv: The QA table.
        metrics: The metric records by metric id.
        charts: The chart records by file stem.

    Yields:
        The series task and the chart task of each row, in table order.
    """
    with qa_csv.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            cited = [part.strip() for part in row["query_group"].split(",")]
            chart_names = [*(["-".join(cited)] if len(cited) > 1 else []), *cited]
            for kind, inputs, modality in (
                ("series", [metrics[metric] for metric in cited], InputModality.TIME_SERIES),
                ("chart", [charts[name] for name in chart_names], InputModality.IMAGE),
            ):
                task = AnswerTask(
                    id=f"arfbench-{index:03d}-{kind}",
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
    """Build a question's annotations, fresh for each task that carries them.

    The interpolation flags say whether the visualization an engineer saw was interpolated; nothing
    here interpolates.

    Returns:
        The candidate answers, the category, the tier, and the two interpolation flags.
    """
    return [
        Annotation(key="answer_options", value=json.loads(row["options_str"])),
        Annotation(key="task_category", value=row["task_category"]),
        Annotation(key="difficulty", value=row["difficulty"]),
        *(Annotation(key=flag, value=row[flag] == "1") for flag in _FLAGS),
    ]


CONNECTOR = ARFBenchConnector
