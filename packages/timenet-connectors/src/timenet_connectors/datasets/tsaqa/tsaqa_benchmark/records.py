"""The Records: one per series a row gives a model, its input series first, then each series its question prints.

Each Record has one Source and one lazy Signal on an ordinal axis. The release gives no timestamps or
units: every input series is z-score normalized and printed to 4 decimals (paper Section 3.3), and a
data transformation question's series are transforms of such series. A Signal rereads its row from the
split file when read.

The row's context, its published ``meta_info``, ``domain`` and ``dataset`` texts, is annotated on its
input Record, or split between the two input Records of a comparison, one entry each.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum, unique
from functools import lru_cache, partial
from pathlib import Path
from typing import Self

import pyarrow as pa

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeFFormatError
from timenet.types import Annotation, TimeSeriesSpec
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.index import ReleaseIndex
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.keys import SERIES, AnnotationKey
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.questions import parse_question
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.release import SourceSplit
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.rows import (
    Row,
    RowRef,
    context_list,
    parse_inputs,
    read_series_columns,
)
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.vocabulary import SourceTask, Transform
from timenet_connectors.metadata import RecordMetadata


_SERIES_SPEC = TimeSeriesSpec(
    spec_type="tsaqa_series", name="TSAQA z-score normalized series", unit_value=None, dtype="float64"
)
_TRANSFORM_SPECS = {
    Transform.FOURIER: TimeSeriesSpec(
        spec_type="tsaqa_fourier_transform",
        name="Fourier transform of a TSAQA series",
        unit_value=None,
        dtype="float64",
    ),
    Transform.WAVELET: TimeSeriesSpec(
        spec_type="tsaqa_wavelet_transform",
        name="Wavelet transform of a TSAQA series",
        unit_value=None,
        dtype="float64",
    ),
    Transform.FIRST_ORDER_DIFFERENCE: TimeSeriesSpec(
        spec_type="tsaqa_first_order_difference",
        name="First-order difference of a TSAQA series",
        unit_value=None,
        dtype="float64",
    ),
}
_AXIS = OrdinalAxis()


@unique
class SeriesRole(StrEnum):
    """Which series of its row a Record holds."""

    INPUT = "input"
    """The row's ``input_ts``, or its first series for a comparison: "time series 1"."""
    SECOND_INPUT = "second_input"
    """A comparison's second series in ``input_ts``: "time series 2"."""
    SEQUENCE = "sequence"
    """The sequence a true-or-false data transformation or temporal relationship question prints."""
    A = "A"
    """Choice or patch A, as the question prints it; likewise B, C and D."""
    B = "B"
    C = "C"
    D = "D"


class TSAQARecordMetadata(RecordMetadata):
    """A TSAQA Record's metadata: the row and the series of it the Record holds.

    The release names no upstream sequence or subject, so the shared ``subject_id`` stays unset.
    """

    file: str
    """The row's split file, as named in the Hub repository."""
    row: int
    """The row's position in its file, counted from 0."""
    role: SeriesRole


def record_metadata(record: Record) -> TSAQARecordMetadata:
    """Read a TSAQA Record's typed metadata back.

    Returns:
        The row and series the Record holds.

    Raises:
        TimeFValidationError: If the Record's metadata is not TSAQA metadata.
    """  # noqa: DOC502 - raised by from_json
    return TSAQARecordMetadata.from_json(record.metadata)


class SeriesLoader:
    """Load one series of a row lazily, holding one split file's series text and parsing each row once.

    The writer reads Records in order, so the file of one split is read once while its Records are
    written. A loader serves the Records of one load.
    """

    def __init__(self, paths: Mapping[SourceSplit, Path]) -> None:
        self._paths = dict(paths)
        self._columns = lru_cache(maxsize=1)(self._read_columns)
        # A row's Records load one after another, so one parsed row suffices.
        self._series = lru_cache(maxsize=1)(self._row_series)

    def _read_columns(self, split: SourceSplit) -> tuple[pa.StringArray, pa.StringArray]:
        return read_series_columns(self._paths[split])

    def _row_series(self, ref: RowRef, task: SourceTask) -> tuple[list[float], ...]:
        input_ts, question = self._columns(ref.split)
        try:
            inputs = parse_inputs(str(input_ts[ref.row].as_py()), task)
            _, printed = parse_question(str(question[ref.row].as_py()), task)
        except (ValueError, IndexError) as exc:
            raise TimeFFormatError(f"{ref.context}: {exc}; the file changed since conversion") from exc
        return (*inputs, *printed)

    def __call__(self, row: Row, position: int) -> pa.Array:
        """Return one series of a row, by its position among the row's Records, as float64 values.

        Returns:
            The series, checked against the converted row's lengths.

        Raises:
            TimeFFormatError: If the row's series differ in number or length from the converted row's.
        """
        series = self._series(row.ref, row.task)
        if tuple(len(values) for values in series) != series_lengths(row):
            raise TimeFFormatError(f"{row.ref.context}: the series differ from the converted row; the file changed")
        return pa.array(series[position], type=pa.float64())


def series_lengths(row: Row) -> tuple[int, ...]:
    """Return the length of each series a row gives a model, in Record order: inputs, then printed series."""
    return (*row.input_lengths, *row.question.series_lengths)


def _roles(row: Row) -> tuple[SeriesRole, ...]:
    inputs = (SeriesRole.INPUT, SeriesRole.SECOND_INPUT)[: len(row.input_lengths)]
    printed = tuple(SeriesRole.SEQUENCE if label is None else SeriesRole(label) for label in row.question.series_labels)
    return inputs + printed


def _context(row: Row, position: int) -> tuple[Annotation, ...]:
    """Return the context annotations of a row's Record at ``position``.

    Paper Appendix E.1 measures a model's input as ``ts + description + domain + dataset + task +
    question_type + question``, and Appendix E.2 says the domain names "are provided to the model as
    textual description". The authors publish no prompt code, so how they join these is unknown. The
    connector keeps ``meta_info`` (the paper's description), ``domain`` and ``dataset`` as published,
    on the input Record, and its tasks give them to the model as input annotations. ``task`` and
    ``question_type`` are task metadata: the question already states both, and 10,000 rows publish a
    wrong ``question_type``.

    Each value is the published text, the string E.1 measures: plain text, or the JSON list text of a
    data transformation or temporal relationship row, whose list names the input's sequence first,
    then distractors' in no letter order. A comparison's lists hold one entry per series, so each input
    Record gets its own entry instead. TimeF gives an annotation key one value type, so a value is
    always text.

    Returns:
        The annotations, none for a Record that isn't an input.
    """
    if position >= len(row.input_lengths):
        return ()
    texts = {
        AnnotationKey.META_INFO: row.meta_info,
        AnnotationKey.DOMAIN: row.domain,
        AnnotationKey.DATASET: row.dataset,
    }
    if row.task is SourceTask.COMPARISON:
        texts = {key: context_list(text, key)[position] for key, text in texts.items()}
    return tuple(Annotation(key=key, value=text) for key, text in texts.items())


def _spec(row: Row, position: int) -> TimeSeriesSpec:
    transform = row.question.transform
    if position < len(row.input_lengths) or transform is None:
        return _SERIES_SPEC
    return _TRANSFORM_SPECS[transform]


def build_row_records(row: Row, loader: SeriesLoader) -> tuple[Record, ...]:
    """Build a row's Records in the order its task gives them to a model: input series, then printed series.

    Args:
        row: The validated row.
        loader: The loader the Records' signals read their values through.

    Returns:
        The row's Records, whose values stay in the split file until read.
    """
    records: list[Record] = []
    for position, (role, length) in enumerate(zip(_roles(row), series_lengths(row), strict=True)):
        signal = Signal.from_loader(
            name=SERIES,
            spec=_spec(row, position),
            time_axis=_AXIS,
            n_values=length,
            loader=partial(loader, row, position),
        )
        metadata = TSAQARecordMetadata(file=row.ref.split.source_file, row=row.ref.row, role=role)
        # Provenance only. The context annotations are what a task gives the model.
        record = Record(sources=(Source(name=SERIES, signals=(signal,)),), metadata=metadata.to_json())
        for annotation in _context(row, position):
            record.annotate(annotation)
        records.append(record)
    return tuple(records)


@dataclass(frozen=True)
class ReleaseRecords:
    """Every row's Records, in row order, each row's in the order its task gives them."""

    of_rows: tuple[tuple[Record, ...], ...]
    """The Records of each row of the index, in index order."""

    @classmethod
    def build(cls, index: ReleaseIndex, loader: SeriesLoader) -> Self:
        """Build every row's Records, without reading any values.

        Returns:
            The Records.
        """
        return cls(tuple(build_row_records(row, loader) for row in index.rows))

    def __iter__(self) -> Iterator[Record]:
        """Return every Record, row by row."""
        return (record for records in self.of_rows for record in records)
