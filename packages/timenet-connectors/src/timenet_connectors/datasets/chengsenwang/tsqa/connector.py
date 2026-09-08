"""The TSQA connector: a time-series question-answering dataset from the HuggingFace Hub.

Source repo ``ChengsenWang/TSQA``. Each row has the columns ``Task, Size, Question, Answer, Label,
Series``. ``Series`` is a JSON float list. A univariate series is a flat list. A multivariate series
is a list of lists. Each row becomes one record that carries the series and an
:class:`~timenet.types.AnswerTask`.
"""

import json
from typing import Any

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import OrdinalAxis
from timenet.types import Annotation, AnswerTask, TimeSeriesSpec, ureg
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector


_SPEC = TimeSeriesSpec(
    spec_type="tsqa_series",
    name="TSQA Series",
    unit_value=ureg.dimensionless,
)


class TSQAConnector(BaseHuggingFaceConnector):
    """Connector for the TSQA time-series QA dataset (Hub repo ``ChengsenWang/TSQA``)."""

    HF_REPO = "ChengsenWang/TSQA"  # the external Hub repo id (keeps its own casing)

    def convert(self, raw_refs: list[dict[str, Any]]) -> TimeFDataset:
        """Build one record per row: the parsed series plus its question-and-answer task.

        Args:
            raw_refs: The rows from :meth:`download`.

        Returns:
            The populated :class:`~timenet.dataset.TimeFDataset`.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for index, row in enumerate(raw_refs):
            series = json.loads(row["Series"])
            signals = series if series and isinstance(series[0], list) else [series]
            time_series = tuple(
                TimeSeries.from_values(
                    values,
                    spec=_SPEC,
                    signal=f"c{signal}",
                    time_axis=OrdinalAxis(),
                    time_series_id=f"row-{index}-c{signal}",
                )
                for signal, values in enumerate(signals)
            )
            record = dataset.add_record(time_series=time_series, record_id=f"row-{index}")
            record.add_annotation(Annotation(key="task", value=row["Task"], id=f"task-{index}"))
            if row.get("Label"):
                record.add_annotation(Annotation(key="label", value=row["Label"], id=f"label-{index}"))
            dataset.add_task(record, AnswerTask(prompt=row["Question"], target=row["Answer"], id=f"qa-{index}"))
        return dataset


CONNECTOR = TSQAConnector
