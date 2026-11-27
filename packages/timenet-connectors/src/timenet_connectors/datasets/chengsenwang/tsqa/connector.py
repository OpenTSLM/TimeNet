"""The TSQA connector: a time-series question-answering dataset from the HuggingFace Hub.

Source repo ``ChengsenWang/TSQA``. The connector reads the parquet copy the Hub converted the source CSV
to, pinned to one commit. Each row has the columns ``Task, Size, Question, Answer, Label, Series``.
``Series`` is a JSON float list. A univariate series is a flat list. A multivariate series is a list of
lists. Each row becomes one record that carries the series and a
:class:`~timenet.types.ClassificationTask`.

A question prints three answers with the keys ``(a)``, ``(b)`` and ``(c)``, and the published answer is a
key. The prompt and the target stay as published. The labels of the task are the keys, and its template
writes the prompt with the answers in another order and with other keys.
"""

from collections.abc import Mapping
from pathlib import Path

from pydantic import ConfigDict, Field

from timenet.dataset import OrdinalAxis, Record, Signal, Source, TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.types import (
    Annotation,
    ClassificationEnumerator,
    ClassificationTask,
    LabelSet,
    OptionKeyStyle,
    Split,
    TimeSeriesSpec,
    ureg,
)
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector
from timenet_connectors.datasets.chengsenwang.tsqa.sample import TSQASample, get_category, get_signal_id
from timenet_connectors.sources.prompt_templates import get_template_from_published_prompt
from timenet_connectors.sources.readers import DataRow, ParquetDataReader
from timenet_connectors.splitting import Splitter, StratifiedSplitter


_SPEC = TimeSeriesSpec(
    spec_type="tsqa_series",
    name="TSQA Series",
    unit_value=ureg.dimensionless,
)


_ANSWER_KEY_STYLE = OptionKeyStyle(
    enumerator=ClassificationEnumerator.LATIN_LOWERCASE,
    marker_prefix="(",
    marker_suffix=")",
    reference_prefix="(",
    reference_suffix=")",
)
"""How a question writes the key of an answer: ``(a)``, before the answer and in the published answer."""
_ANSWER_KEYS = LabelSet(["(a)", "(b)", "(c)"], annotations=[Annotation("name", value="answer_key")])
"""The label set that every task shares: the keys of the three answers of a question."""
_QUESTION_START_LENGTH = 60
"""How many characters of a question an error message shows."""


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
        annotations = [Annotation("task", value=sample.category)]
        if sample.label:
            annotations.append(Annotation("label", value=sample.label))
        return Record(sources=[Source("TSQA series", signals=signals)], annotations=annotations)

    @staticmethod
    def _make_task(sample: TSQASample, record: Record, *, split: Split) -> ClassificationTask:
        """Build the task of a row, with a template of its question and its three answers.

        A question prints one answer per line, then the sentence ``Only answer (a), (b), or (c).``

        Returns:
            The task, not yet added to a dataset.

        Raises:
            TimeNetConnectorError: If the question does not print three answers with the keys ``(a)``,
                ``(b)`` and ``(c)`` in this form.
        """
        question_start = sample.question[:_QUESTION_START_LENGTH]
        try:
            template = get_template_from_published_prompt(
                sample.question,
                published_key_style=_ANSWER_KEY_STYLE,
                one_option_per_line=True,
                expected_key_list_count=1,
            )
        except TimeNetConnectorError as exc:
            raise TimeNetConnectorError(f"the {sample.category} question {question_start!r}: {exc}") from exc
        if template.labels != _ANSWER_KEYS.names:
            raise TimeNetConnectorError(
                f"the {sample.category} question {question_start!r} has the answers {template.labels}, "
                f"not {_ANSWER_KEYS.names}"
            )
        return ClassificationTask(
            inputs=(record,),
            prompt=sample.question,
            targets=sample.answer,
            split=split,
            label_set=_ANSWER_KEYS,
            template=template,
        )


CONNECTOR = TSQAConnector
