"""The TSQA connector: a time-series question-answering dataset from the HuggingFace Hub.

Source: ``ChengsenWang/TSQA`` — each row has ``Task, Size, Question, Answer, Label, Series`` where
``Series`` is a JSON float list (univariate, or a list-of-lists for multivariate). Each row becomes one
sample carrying the series plus a :class:`~timenet.types.QATask`.
"""

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.types import (
    QATask,
    StaticAnnotation,
    TimeSeriesSpec,
    View,
    ureg,
)
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector


_SPEC = TimeSeriesSpec(
    spec_type="tsqa_series",
    name="TSQA Series",
    unit_sampling_rate=ureg.hertz,
    unit_timestamp=ureg.second,
    unit_value=ureg.dimensionless,
)


def _loader(values: list[float]) -> Callable[[], pa.Array]:
    """Build a loader returning the parsed values as a float32 Arrow array.

    Args:
        values: The parsed series values.

    Returns:
        A no-argument loader returning the values as a float32 Arrow array.
    """

    def load() -> pa.Array:
        return pa.array(np.asarray(values, dtype=np.float32))

    return load


class TSQAConnector(BaseHuggingFaceConnector):
    """Connector for the TSQA time-series QA dataset (Hub repo ``ChengsenWang/TSQA``)."""

    HF_REPO = "ChengsenWang/TSQA"  # the external Hub repo id (keeps its own casing)

    CARD = Path(__file__).with_name("tsqa.yaml")

    def convert(self, raw_refs: list[dict[str, Any]]) -> TimeFDataset:
        """Build one sample per row: the parsed series plus its QA task.

        Args:
            raw_refs: The rows from :meth:`download`.

        Returns:
            The populated :class:`~timenet.dataset.TimeFDataset`.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for index, row in enumerate(raw_refs):
            series = json.loads(row["Series"])
            channels = series if series and isinstance(series[0], list) else [series]
            time_series = tuple(
                TimeSeries(
                    spec=_SPEC,
                    channel=f"c{channel}",
                    sampling_rate_hz=1.0,
                    loader=_loader(values),
                    time_series_id=f"row-{index}-c{channel}",
                    t_start_s=0.0,
                    t_end_s=float(len(values)),
                )
                for channel, values in enumerate(channels)
            )
            sample = dataset.add_sample(time_series=time_series, view=View.FULL, sample_id=f"row-{index}")
            sample.add_annotation(StaticAnnotation(key="task", value=row["Task"], id=f"task-{index}"))
            if row.get("Label"):
                sample.add_annotation(StaticAnnotation(key="label", value=row["Label"], id=f"label-{index}"))
            dataset.add_task(sample, QATask(question=row["Question"], answer=row["Answer"], id=f"qa-{index}"))
        return dataset


CONNECTOR = TSQAConnector
