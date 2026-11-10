"""Convert the ARFBench questions and metric series to TimeF.

The release holds one question table, one series file per metric and rollup interval, and one
chart per metric or cited metric pair. This connector reads the table and the series: each metric
becomes one Record, and each question a series task. The loop that walks the release is in
``convert``. :mod:`~timenet_connectors.datasets.datadog.arfbench.series` reads the series files,
and :mod:`~timenet_connectors.datasets.datadog.arfbench.questions` reads the question table. The
``datadog/arfbench-plots`` connector adds the charts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetConnectorError, TimeNetDownloadError
from timenet_connectors.datasets.datadog.arfbench.questions import ARFBenchQuestion, read_questions
from timenet_connectors.datasets.datadog.arfbench.series import MetricSeriesFiles
from timenet_connectors.sources.huggingface_hub import hub_snapshot


if TYPE_CHECKING:
    from collections.abc import Iterator

REPO = "Datadog/ARFBench"
REVISION = "1cc8ae54e9633f596b755f7c4bce54ccb0cb9f5a"
QA_CSV = "arfbench-qa.csv"
SERIES_DIR = "arfbench-ts-data"  # {metric}_{interval}.parquet: one metric at one rollup interval, in seconds


@dataclass(frozen=True)
class ARFBenchSource:
    """The local snapshot of the ARFBench release."""

    root: Path

    @property
    def qa_csv(self) -> Path:
        """The question table."""
        return self.root / QA_CSV

    @property
    def series_dir(self) -> Path:
        """The directory of the series files."""
        return self.root / SERIES_DIR

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


class ARFBenchConnector(BaseConnector[ARFBenchSource]):
    """Connector for the ARFBench questions over the metric series (Hub repo ``Datadog/ARFBench``).

    The values are plain series, so the dataset uses the default Parquet backend.
    """

    HUB_PATTERNS: ClassVar[tuple[str, ...]] = (QA_CSV, f"{SERIES_DIR}/*.parquet")
    """The release files to download."""

    async def download_async(self, cache_dir: Path) -> list[ARFBenchSource]:
        """Download the release files in :attr:`HUB_PATTERNS` from the pinned revision.

        Args:
            cache_dir: Directory for cached Hub files.

        Returns:
            A list with the snapshot.

        Raises:
            TimeNetDownloadError: If the download fails or the question table is missing.
        """
        root = await hub_snapshot(REPO, REVISION, cache_dir, self.HUB_PATTERNS)
        if not (root / QA_CSV).is_file():
            raise TimeNetDownloadError(f"{REPO!r} at {REVISION!r} holds no {QA_CSV} under {root}")
        return [ARFBenchSource(root=root)]

    def convert(self, raw_refs: list[ARFBenchSource]) -> TimeFDataset:
        """Create a Record for each metric and a series task for each question.

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
        for question in source.read_questions():
            dataset.add_task(task=question.create_series_task(metric_records))
        return dataset


CONNECTOR = ARFBenchConnector
