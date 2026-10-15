"""The TSQA connector: a time-series question-answering dataset from the HuggingFace Hub.

Source repo ``ChengsenWang/TSQA``. Each row has the columns ``Task, Size, Question, Answer, Label,
Series``. ``Series`` is a JSON float list. A univariate series is a flat list. A multivariate series
is a list of lists. Each row becomes one record that carries the series and an
:class:`~timenet.types.AnswerTask`.
"""

from typing import Any

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.types import Annotation, AnswerTask, TimeSeriesSpec, ureg
from timenet.types.splits import Split
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector
from timenet_connectors.datasets.chengsenwang.tsqa.sample import TSQASample, get_category, get_signal_id
from timenet_connectors.splitting import Splitter, StratifiedSplitter


_SPEC = TimeSeriesSpec(
    spec_type="tsqa_series",
    name="TSQA Series",
    unit_value=ureg.dimensionless,
)


class TSQAConnector(BaseHuggingFaceConnector):
    """Connector for the TSQA time-series QA dataset (Hub repo ``ChengsenWang/TSQA``)."""

    HF_REPO = "ChengsenWang/TSQA"  # the external Hub repo id (keeps its own casing)

    def __init__(self) -> None:
        super().__init__()
        self.splitter: Splitter[TSQASample] = StratifiedSplitter(
            train=0.8,
            validation=0.1,
            test=0.1,
            stratify_by=get_category,  # Balance question categories across partitions.
            # Keep all questions about an identical signal in the same partition.
            # Otherwise, evaluation could reuse a signal already seen during training.
            # Group isolation takes priority over exact split sizes or category balance.
            group_by=get_signal_id,
            seed=42,
        )

    def convert(self, raw_refs: list[dict[str, Any]]) -> TimeFDataset:
        """Build one record per row: the parsed series plus its question-and-answer task.

        Args:
            raw_refs: The rows from :meth:`download`.

        Returns:
            The populated :class:`~timenet.dataset.TimeFDataset`.
        """
        samples = [TSQASample.from_row(index, row) for index, row in enumerate(raw_refs)]
        splits = self.splitter.split(samples)
        dataset = TimeFDataset(metadata=self.metadata())
        for split_name, samples_in_split in splits.items():
            for sample in samples_in_split:
                record = self._make_record(sample)
                dataset.add_record(record=record)
                task = self._make_task(sample, record, split=split_name)
                dataset.add_task(task=task)
        return dataset

    @staticmethod
    def _make_record(sample: TSQASample) -> Record:
        index = sample.index
        signals: list[Signal] = []
        for channel, values in enumerate(sample.signals):
            signal = Signal.from_values(
                values,
                spec=_SPEC,
                name=f"c{channel}",
                time_axis=OrdinalAxis(),
                id=f"row-{index}-c{channel}",
            )
            signals.append(signal)
        record = Record(
            record_id=f"row-{index}",
            sources=(Source(id=f"row-{index}-source", name="TSQA series", signals=tuple(signals)),),
        )
        record.annotate(Annotation(key="task", value=sample.category, id=f"task-{index}"))
        if sample.label:
            record.annotate(Annotation(key="label", value=sample.label, id=f"label-{index}"))
        return record

    @staticmethod
    def _make_task(sample: TSQASample, record: Record, *, split: Split) -> AnswerTask:
        return AnswerTask(
            inputs=(record,),
            prompt=sample.question,
            targets=(sample.answer,),
            id=f"qa-{sample.index}",
            split=split,
        )


CONNECTOR = TSQAConnector
