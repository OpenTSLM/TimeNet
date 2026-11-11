"""Convert ARFBench questions, metric series, and charts to TimeF.

The release holds one question table, one series file per metric and rollup interval, and one
chart per metric or cited metric pair. Each metric becomes one Record, each chart one Record, and
each question a series task and a chart task. The loop that walks the release is in ``convert``.
:mod:`~timenet_connectors.datasets.datadog.arfbench.series` reads the series files, and
:mod:`~timenet_connectors.datasets.datadog.arfbench.questions` reads the question table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Source, TimeFDataset
from timenet.errors import TimeNetConnectorError, TimeNetDownloadError
from timenet.values_backends import ValuesBackend
from timenet_connectors.datasets.datadog.arfbench.questions import ARFBenchQuestion, read_questions
from timenet_connectors.datasets.datadog.arfbench.series import MetricSeriesFiles
from timenet_connectors.sources.huggingface_hub import hub_snapshot
from timenet_connectors.sources.images import image_signal


if TYPE_CHECKING:
    from collections.abc import Iterator

REPO = "Datadog/ARFBench"
REVISION = "1cc8ae54e9633f596b755f7c4bce54ccb0cb9f5a"
QA_CSV = "arfbench-qa.csv"
SERIES_DIR = "arfbench-ts-data"  # {metric}_{interval}.parquet: one metric at one rollup interval, in seconds
CHART_DIR = "arfbench-images"  # {metric}.png, and {metric}-{metric}.png for the pairs the questions cite


def _create_chart_record(path: Path) -> Record:
    """Create the Record of a chart, with the image as its one signal.

    Returns:
        The Record, not yet added to a dataset.
    """
    return Record(
        metadata={"source_record_id": f"arfbench-chart-{path.stem}"},
        sources=(Source(name="ARFBench chart", signals=(image_signal(path, name="chart"),)),),
    )


@dataclass(frozen=True)
class ARFBenchSource:
    """Paths to the ARFBench question table, series files, and charts."""

    qa_csv: Path
    series_dir: Path
    chart_dir: Path

    def read_questions(self) -> Iterator[ARFBenchQuestion]:
        """Read the question table, in table order.

        Returns:
            Each parsed question.
        """
        return read_questions(self.qa_csv)

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

    values_backend = ValuesBackend.ZARR  # the chart tensors need it

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
