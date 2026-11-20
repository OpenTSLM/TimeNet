"""The TimeSeriesExam connector: multiple-choice questions about time series, from the HuggingFace Hub.

The source repository is ``AutonLab/TimeSeriesExam1``. The connector pins one commit and reads only the published
test split. A row has a question, its options, the answer, and one series (``ts``) or two (``ts1`` and ``ts2``).
Each series becomes a Record.

Each row becomes one :class:`~timenet.types.ClassificationTask`. Its labels are the option texts, and its target is
the published answer text. The same options are in a different order in different rows, so the order has no meaning
and the label set is not ordered.

The task metadata holds the other published columns. It is provenance and never model input.
The prompt is the question alone. The options are the labels of the task, and ``format_hint`` names option
letters that the task does not use.
"""

from typing import Self

from pydantic import model_validator

from timenet.dataset import OrdinalAxis, Record, Signal, Source, TimeFDataset
from timenet.types import ClassificationTask, LabelSet, Split, TimeSeriesSpec, ureg
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.sources.readers import DataRow, ParquetDataReader


_SPEC = TimeSeriesSpec(
    spec_type="tse1_series", name="TimeSeriesExam series", unit_value=ureg.dimensionless, dtype="float64"
)
_AXIS = OrdinalAxis()
_SERIES_COLUMNS = ("ts", "ts1", "ts2")
_NOT_METADATA = {"question", "options", "answer", *_SERIES_COLUMNS}
"""The published columns that a task does not keep in its metadata. It keeps the others, as published."""


class _SourceRow(DataRow):
    """A published row. A series column is ``None`` where the row has no such series."""

    id: int
    question: str
    options: list[str]
    answer: str
    question_type: str
    tid: int
    difficulty: str
    format_hint: str
    relevant_concepts: list[str]
    question_hint: str
    category: str
    subcategory: str
    ts: list[float] | None
    ts1: list[float] | None
    ts2: list[float] | None

    @property
    def series(self) -> dict[str, list[float]]:
        """The series that the row has, by column name."""
        return {column: values for column in _SERIES_COLUMNS if (values := getattr(self, column)) is not None}

    @model_validator(mode="after")
    def _has_ts_or_ts1_and_ts2(self) -> Self:
        if list(self.series) not in (["ts"], ["ts1", "ts2"]):
            raise ValueError(f"row id {self.id}: expected ts, or ts1 and ts2, found {list(self.series)}")
        return self


def _record(row_id: int, role: str, values: list[float]) -> Record:
    signal = Signal.from_values(values, spec=_SPEC, name="series", time_axis=_AXIS)
    return Record(sources=[Source("series", signals=[signal])], metadata={"row_id": row_id, "role": role})


class TimeSeriesExam1Connector(
    BaseHuggingFaceConnector,
    hf_repo="AutonLab/TimeSeriesExam1",
    source_revision="628f686757bb484d9b608b6855fb41f6a302ecda",
    source_files={"test": "data/test-00000-of-00001.parquet"},
):
    """Build TimeSeriesExam from the pinned test parquet file of its Hugging Face release."""

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Build one Record per series and one classification task per question.

        Args:
            raw_refs: The downloaded test file.

        Returns:
            The populated :class:`~timenet.dataset.TimeFDataset`.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for row in ParquetDataReader(raw_refs[0].path, _SourceRow).iter_rows():
            inputs = [dataset.add_record(_record(row.id, role, values)) for role, values in row.series.items()]
            dataset.add_task(
                ClassificationTask(
                    inputs=inputs,
                    prompt=row.question,
                    label_set=LabelSet(row.options),
                    targets=row.answer,
                    split=Split.TEST,
                    metadata=row.model_dump(exclude=_NOT_METADATA),
                )
            )
        return dataset


CONNECTOR = TimeSeriesExam1Connector
