"""Convert the ARFBench questions and metric series to TimeF.

The release holds one question table, one series file per metric and rollup interval, and one
chart per metric or cited metric pair. This connector reads the table and the series: each metric
becomes one Record, and each question a series task. The loop that walks the release is in
``convert``. :mod:`~timenet_connectors.datasets.datadog.arfbench.series` reads the series files,
and :mod:`~timenet_connectors.datasets.datadog.arfbench.questions` reads the question table. The
``datadog/arfbench-plots`` dataset adds the charts on top of this one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector
from timenet_connectors.datasets.datadog.arfbench.questions import read_questions
from timenet_connectors.datasets.datadog.arfbench.series import MetricSeriesFiles


if TYPE_CHECKING:
    from pathlib import Path

    from timenet_connectors.bases.huggingface import HubFileRef

REPO = "Datadog/ARFBench"
REVISION = "1cc8ae54e9633f596b755f7c4bce54ccb0cb9f5a"
QA_CSV = "arfbench-qa.csv"
SERIES_DIR = "arfbench-ts-data"  # {metric}_{interval}.parquet: one metric at one rollup interval, in seconds


def release_paths(raw_refs: list[HubFileRef], *parts: str) -> dict[str, Path]:
    """Map each named part of the release to its local path.

    Returns:
        The path of each part: a file, or the directory of a pattern part.

    Raises:
        TimeNetConnectorError: If a part is missing.
    """
    paths = {ref.part: ref.path for ref in raw_refs}
    missing = [part for part in parts if part not in paths]
    if missing:
        raise TimeNetConnectorError(f"the ARFBench release lacks {', '.join(missing)}")
    return paths


class ARFBenchConnector(BaseHuggingFaceConnector):
    """Connector for the ARFBench questions over the metric series (Hub repo ``Datadog/ARFBench``).

    The values are plain series, so the dataset uses the default Parquet backend.
    """

    HF_REPO = REPO
    SOURCE_REVISION = REVISION
    SOURCE_FILES: ClassVar = {"questions": QA_CSV, "series": f"{SERIES_DIR}/*.parquet"}

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Create a Record for each metric and a series task for each question.

        Args:
            raw_refs: The question table and the series directory from :meth:`download_async`.

        Returns:
            The populated dataset.
        """
        paths = release_paths(raw_refs, "questions", "series")
        dataset = TimeFDataset(metadata=self.metadata())
        metric_records = {
            files.metric: dataset.add_record(record=files.create_record())
            for files in MetricSeriesFiles.from_series_dir(paths["series"])
        }
        for question in read_questions(paths["questions"]):
            dataset.add_task(task=question.create_series_task(metric_records))
        return dataset


CONNECTOR = ARFBenchConnector
