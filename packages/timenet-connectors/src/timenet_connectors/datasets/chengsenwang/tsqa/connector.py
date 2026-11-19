"""The TSQA connector: a time-series question-answering dataset from the HuggingFace Hub.

Source repo ``ChengsenWang/TSQA``. The connector reads the parquet copy the Hub converted the source CSV
to, pinned to one commit. Each row has the columns ``Task, Size, Question, Answer, Label, Series``.
``Series`` is a JSON float list. A univariate series is a flat list. A multivariate series is a list of
lists. Each row becomes one record that carries the series and an :class:`~timenet.types.AnswerTask`.
"""

from collections.abc import Mapping
from pathlib import Path

from pydantic import ConfigDict, Field

from timenet.dataset import OrdinalAxis, Record, Signal, Source, TimeFDataset
from timenet.types import Annotation, AnswerTask, Split, TimeSeriesSpec, ureg
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector
from timenet_connectors.datasets.chengsenwang.tsqa.sample import TSQASample, get_category, get_signal_id
from timenet_connectors.sources.readers import DataRow, ParquetDataReader
from timenet_connectors.splitting import Splitter, StratifiedSplitter


_SPEC = TimeSeriesSpec(
    spec_type="tsqa_series",
    name="TSQA Series",
    unit_value=ureg.dimensionless,
)


class _SourceRow(DataRow):
    """Columns consumed from a TSQA Parquet row."""

    model_config = ConfigDict(extra="ignore")

    task: str = Field(alias="Task")
    question: str = Field(alias="Question")
    answer: str = Field(alias="Answer")
    label: str | None = Field(default=None, alias="Label")
    series: str = Field(alias="Series")


class TSQAConnector(
    BaseHuggingFaceConnector,
    hf_repo="ChengsenWang/TSQA",  # the external Hub repo id (keeps its own casing)
    # The commit refs/convert/parquet pointed to when this was pinned. The Hub rewrites that ref
    # whenever it reconverts the dataset, so a rebuild reads this commit instead of the ref.
    source_revision="22518209e3ee6302be4b4957f2692b3e8de849c3",
    source_files={"train": "default/train/0000.parquet"},
):
    """Connector for the TSQA time-series QA dataset (Hub repo ``ChengsenWang/TSQA``)."""

    def __init__(self) -> None:
        super().__init__()
        self.splitter: Splitter[TSQASample] = StratifiedSplitter(
            train_folds=8,
            validation_folds=1,
            test_folds=1,
            stratify_by=get_category,  # Balance question categories across partitions.
            # Keep all questions about an identical signal in the same partition.
            # Otherwise, evaluation could reuse a signal already seen during training.
            # Group isolation takes priority over exact split sizes or category balance.
            group_by=get_signal_id,
            seed=42,
        )

    def convert(self, files: Mapping[str, Path]) -> TimeFDataset:
        """Build one record per row: the parsed series plus its question-and-answer task.

        Args:
            files: The parquet file of each part.

        Returns:
            The populated :class:`~timenet.dataset.TimeFDataset`.
        """
        rows = (row for path in files.values() for row in ParquetDataReader(path, _SourceRow).iter_rows())
        samples = [TSQASample.from_row(row.model_dump(by_alias=True)) for row in rows]
        splits = self.splitter.split(samples)
        dataset = TimeFDataset(metadata=self.metadata())
        for split_name, samples_in_split in splits.items():
            for sample in samples_in_split:
                record = self._make_record(sample)
                dataset.add_record(record)
                task = self._make_task(sample, record, split=split_name)
                dataset.add_task(task)
        return dataset

    @staticmethod
    def _make_record(sample: TSQASample) -> Record:
        signals: list[Signal] = []
        for channel, values in enumerate(sample.signals):
            signal = Signal.from_values(
                values,
                spec=_SPEC,
                name=f"c{channel}",
                time_axis=OrdinalAxis(),
            )
            signals.append(signal)
        record = Record(sources=[Source("TSQA series", signals=signals)])
        record.annotate(Annotation("task", value=sample.category))
        if sample.label:
            record.annotate(Annotation("label", value=sample.label))
        return record

    @staticmethod
    def _make_task(sample: TSQASample, record: Record, *, split: Split) -> AnswerTask:
        return AnswerTask(
            inputs=(record,),
            prompt=sample.question,
            targets=sample.answer,
            split=split,
        )


CONNECTOR = TSQAConnector
