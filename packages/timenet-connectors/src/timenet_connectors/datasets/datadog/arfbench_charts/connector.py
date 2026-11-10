"""Convert the ARFBench charts and questions to TimeF.

Each chart becomes one Record and each question becomes one chart task. The charts are image
tensors, so the dataset uses the Zarr backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from timenet.dataset import Record, Source, TimeFDataset
from timenet.values_backends import ValuesBackend
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector
from timenet_connectors.datasets.datadog.arfbench.connector import QA_CSV, REPO, REVISION
from timenet_connectors.datasets.datadog.arfbench.questions import read_questions
from timenet_connectors.sources.images import image_signal


if TYPE_CHECKING:
    from pathlib import Path

    from timenet_connectors.bases.huggingface import HubFileRef

CHART_DIR = "arfbench-images"  # {metric}.png, and {metric}-{metric}.png for the pairs that the questions cite


def _create_chart_record(path: Path) -> Record:
    """Create the Record of a chart, with the image as its one signal.

    Returns:
        The Record, not yet added to a dataset.
    """
    return Record(
        metadata={"source_record_id": f"arfbench-chart-{path.stem}"},
        sources=(Source(name="ARFBench chart", signals=(image_signal(path, name="chart"),)),),
    )


class ARFBenchChartsConnector(BaseHuggingFaceConnector):
    """Connector for the ARFBench charts (Hub repo ``Datadog/ARFBench``)."""

    values_backend = ValuesBackend.ZARR  # the chart tensors need the Zarr backend
    HF_REPO = REPO
    SOURCE_REVISION = REVISION
    SOURCE_FILES: ClassVar = {"questions": QA_CSV, "charts": f"{CHART_DIR}/*.png"}

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Create a Record for each chart and a chart task for each question.

        Args:
            raw_refs: The question table and the chart directory from :meth:`download_async`.

        Returns:
            The populated dataset.
        """
        paths = self.ref_paths(raw_refs, "questions", "charts")
        dataset = TimeFDataset(metadata=self.metadata())
        chart_records = {
            path.stem: dataset.add_record(record=_create_chart_record(path))
            for path in sorted(paths["charts"].glob("*.png"))
        }
        for question in read_questions(paths["questions"]):
            dataset.add_task(task=question.create_chart_task(chart_records))
        return dataset


CONNECTOR = ARFBenchChartsConnector
