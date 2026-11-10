"""Convert the ARFBench questions, metric series, and charts to TimeF.

This connector extends the ``datadog/arfbench`` connector with the released charts: each chart
becomes one Record, and each question gets a chart task next to its series task. The charts are
image tensors, which only the Zarr backend stores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from timenet.dataset import Record, Source
from timenet_connectors.datasets.datadog.arfbench.connector import ARFBenchConnector
from timenet_connectors.sources.images import image_signal


if TYPE_CHECKING:
    from pathlib import Path

    from timenet.dataset import TimeFDataset
    from timenet_connectors.datasets.datadog.arfbench.connector import ARFBenchSource

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


class ARFBenchPlotsConnector(ARFBenchConnector):
    """Connector for the ARFBench questions over the metric series and the charts."""

    values_backend = "zarr"  # the chart tensors need it
    HUB_PATTERNS: ClassVar[tuple[str, ...]] = (*ARFBenchConnector.HUB_PATTERNS, f"{CHART_DIR}/*.png")

    def convert(self, raw_refs: list[ARFBenchSource]) -> TimeFDataset:
        """Create the series dataset, then a Record for each chart and a chart task for each question.

        Args:
            raw_refs: The one source from :meth:`download_async`.

        Returns:
            The populated dataset: the series tasks first, then the chart tasks, paired by
            ``question_index``.
        """
        dataset = super().convert(raw_refs)
        source = raw_refs[0]
        chart_records = {
            path.stem: dataset.add_record(record=_create_chart_record(path))
            for path in sorted((source.root / CHART_DIR).glob("*.png"))
        }
        for question in source.read_questions():
            dataset.add_task(task=question.create_chart_task(chart_records))
        return dataset


CONNECTOR = ARFBenchPlotsConnector
