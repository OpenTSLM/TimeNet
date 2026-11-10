"""Build the ARFBench charts as a child dataset of ``datadog/arfbench``.

This dataset is a child of ``datadog/arfbench``. Its card declares that parent with the alias
``series``. It imports every series task of the parent, with the metric Records that the tasks cite.
It then adds one Record per chart in the release and one chart task per question. The imported
series continue to refer to the stored values of the parent, so this release stores only the
charts. The charts are image tensors, and only the Zarr backend can store them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from timenet.dataset import Record, Source, TimeFDataset
from timenet.types.reference_keys import ParentAlias
from timenet.values_backends import ValuesBackend
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector
from timenet_connectors.datasets.datadog.arfbench.connector import QA_CSV, REPO, REVISION, release_paths
from timenet_connectors.datasets.datadog.arfbench.questions import read_questions
from timenet_connectors.sources.images import image_signal


if TYPE_CHECKING:
    from pathlib import Path

    from timenet.connectors import BuildContext
    from timenet_connectors.bases.huggingface import HubFileRef

CHART_DIR = "arfbench-images"  # {metric}.png, and {metric}-{metric}.png for the pairs that the questions cite
SERIES_PARENT = ParentAlias("series")


def _create_chart_record(path: Path) -> Record:
    """Create the Record of a chart, with the image as its one signal.

    Returns:
        The Record, not yet added to a dataset.
    """
    return Record(
        metadata={"source_record_id": f"arfbench-chart-{path.stem}"},
        sources=(Source(name="ARFBench chart", signals=(image_signal(path, name="chart"),)),),
    )


class ARFBenchPlotsConnector(BaseHuggingFaceConnector):
    """Connector for the ARFBench charts, as a child of the ``datadog/arfbench`` series."""

    values_backend = ValuesBackend.ZARR  # the chart tensors need the Zarr backend
    HF_REPO = REPO
    SOURCE_REVISION = REVISION
    SOURCE_FILES: ClassVar = {"questions": QA_CSV, "charts": f"{CHART_DIR}/*.png"}

    def compose(self, raw_refs: list[HubFileRef], context: BuildContext) -> TimeFDataset:
        """Import the parent's series tasks, then add a Record for each chart and a chart task for each question.

        Args:
            raw_refs: The question table and the chart directory from :meth:`download_async`.
            context: The opened ``datadog/arfbench`` parent, under the alias ``series``.

        Returns:
            The dataset: the series tasks first, then the chart tasks, paired by ``question_index``.
        """
        self.check_refs(raw_refs)
        paths = release_paths(raw_refs, "questions", "charts")
        series = context.parents[SERIES_PARENT]
        dataset = TimeFDataset(metadata=self.metadata())
        dataset.import_from(series, tasks=[task.id for task in series.iter_tasks()])
        chart_records = {
            path.stem: dataset.add_record(record=_create_chart_record(path))
            for path in sorted(paths["charts"].glob("*.png"))
        }
        for question in read_questions(paths["questions"]):
            dataset.add_task(task=question.create_chart_task(chart_records))
        return dataset


CONNECTOR = ARFBenchPlotsConnector
