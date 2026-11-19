"""Convert the ARFBench charts and questions to TimeF.

Each chart becomes one Record and each question becomes one chart task. The charts are image
tensors, so the dataset uses the Zarr backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from timenet.dataset import Record, Source, TimeFDataset
from timenet.values_backends import ValuesBackend
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector
from timenet_connectors.datasets.datadog.arfbench.connector import QA_CSV, REPO, REVISION
from timenet_connectors.datasets.datadog.arfbench.questions import read_questions
from timenet_connectors.sources.images import image_signal


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

CHART_DIR = "arfbench-images"  # {metric}.png, and {metric}-{metric}.png for the pairs that the questions cite


def _create_chart_record(path: Path) -> Record:
    """Create the Record of a chart, with the image as its one signal.

    Returns:
        The Record, not yet added to a dataset.
    """
    return Record(
        metadata={"source_record_id": f"arfbench-chart-{path.stem}"},
        sources=[Source("ARFBench chart", signals=[image_signal(path, name="chart")])],
    )


class ARFBenchChartsConnector(
    BaseHuggingFaceConnector,
    hf_repo=REPO,
    source_revision=REVISION,
    source_files={"questions": QA_CSV, "charts": f"{CHART_DIR}/*.png"},
):
    """Connector for the ARFBench charts (Hub repo ``Datadog/ARFBench``)."""

    values_backend = ValuesBackend.ZARR  # the chart tensors need the Zarr backend

    def convert_files(self, files: Mapping[str, Path]) -> TimeFDataset:
        """Create a Record for each chart and a chart task for each question.

        Args:
            files: The question table and the chart directory.

        Returns:
            The populated dataset.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        chart_records = {
            path.stem: dataset.add_record(_create_chart_record(path)) for path in sorted(files["charts"].glob("*.png"))
        }
        for question in read_questions(files["questions"]):
            dataset.add_task(question.create_chart_task(chart_records))
        return dataset


CONNECTOR = ARFBenchChartsConnector
