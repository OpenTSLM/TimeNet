"""The Records: one per series, per rounded copy of a series in the description files, and per changed series.

Each Record has one Source and one lazy Signal on an ordinal axis, whose values a loader rereads from
the row's line. The release states start, end, units and frequency only as free text, kept in the
``metadata`` annotation, so the Signal has no unit and no timestamps.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import ClassVar, Literal, Self

import numpy as np
import pyarrow as pa

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import OrdinalAxis
from timenet.types import Annotation, TimeSeriesSpec
from timenet_connectors.datasets.mikeam.time_series_reasoning.index import ReleaseIndex
from timenet_connectors.datasets.mikeam.time_series_reasoning.keys import SERIES, AnnotationKey
from timenet_connectors.datasets.mikeam.time_series_reasoning.release import SourceFile
from timenet_connectors.datasets.mikeam.time_series_reasoning.rows import (
    CounterfactualRow,
    DescriptionRow,
    Row,
    RowRef,
    SeriesShape,
    ValueType,
    reread_values,
)
from timenet_connectors.metadata import MetadataExtension, RecordMetadata


# One spec_type has one dtype, so a series written in integers has its own.
_SPECS: dict[ValueType, TimeSeriesSpec] = {
    ValueType.FLOAT64: TimeSeriesSpec(
        spec_type="tsr_series", name="Synthetic series", unit_value=None, dtype="float64"
    ),
    ValueType.INT32: TimeSeriesSpec(
        spec_type="tsr_integer_series", name="Synthetic integer series", unit_value=None, dtype="int32"
    ),
}


class TimeSeriesReasoningRecordMetadata(MetadataExtension):
    """The extension of a Record's metadata: the published series it holds, and the line it is read from.

    The release names no subject, so the core ``subject_id`` stays unset.
    """

    NAME: ClassVar[str] = "time_series_reasoning"

    uuid: str
    """The series' ``uuid``. A changed series and a rounded copy keep the uuid of the series they come from."""
    file: SourceFile
    line: int
    """The line, counted from 1, that holds the Record's values."""


def record_metadata(record: Record) -> TimeSeriesReasoningRecordMetadata:
    """Read a Record's typed metadata back.

    Returns:
        The published series the Record holds.

    Raises:
        TimeFValidationError: If the Record has no metadata of this dataset.
    """  # noqa: DOC502 - raised by from_json and extension
    extensions = [TimeSeriesReasoningRecordMetadata]
    return RecordMetadata.from_json(record.metadata, extensions).extension(TimeSeriesReasoningRecordMetadata)


def _load(ref: RowRef, field: Literal["series", "new_series"], shape: SeriesShape) -> pa.Array:
    # The Arrow type of the spec's dtype, as timenet's own Signal values derive it.
    arrow_type = pa.from_numpy_dtype(np.dtype(_SPECS[shape.dtype].dtype))
    return pa.array(reread_values(ref, field, shape), type=arrow_type)


def _record(
    uuid: str,
    ref: RowRef,
    field: Literal["series", "new_series"],
    shape: SeriesShape,
    annotations: tuple[Annotation, ...],
) -> Record:
    """Build a Record whose one Signal rereads one series from one line.

    Returns:
        The Record, whose values stay in the source file until read.
    """
    signal = Signal.from_loader(
        name=SERIES,
        spec=_SPECS[shape.dtype],
        time_axis=OrdinalAxis(),
        n_values=shape.length,
        loader=partial(_load, ref, field, shape),
    )
    metadata = RecordMetadata().extend(TimeSeriesReasoningRecordMetadata(uuid=uuid, file=ref.file, line=ref.line))
    # Provenance only, never model input: a task's prompt holds what the authors show a model.
    record = Record(sources=(Source(name=SERIES, signals=(signal,)),), metadata=metadata.to_json())
    for annotation in annotations:
        record.annotate(annotation)
    return record


def _series_annotations(row: Row) -> tuple[Annotation, ...]:
    """Return a series' texts, each under its published field name.

    Returns:
        The description, its short and tiny versions, the characteristics, the generator, and the
        metadata as the authors' prompt prints it.
    """
    return (
        Annotation(key=AnnotationKey.DESCRIPTION, value=row.description),
        Annotation(key=AnnotationKey.DESCRIPTION_SHORT, value=row.description_short),
        Annotation(key=AnnotationKey.DESCRIPTION_TINY, value=row.description_tiny),
        Annotation(key=AnnotationKey.CHARACTERISTICS, value=row.characteristics),
        Annotation(key=AnnotationKey.GENERATOR, value=row.generator),
        # The authors' loader writes str(data["metadata"]) into the prompt.
        Annotation(key=AnnotationKey.METADATA, value=str(row.metadata)),
    )


def _changed_annotations(row: CounterfactualRow) -> tuple[Annotation, ...]:
    """Return a changed series' change and the authors' code for it, published as ``new_generator``.

    Returns:
        The ``counterfactual`` and ``generator`` annotations.
    """
    return (
        Annotation(key=AnnotationKey.COUNTERFACTUAL, value=row.counterfactual),
        Annotation(key=AnnotationKey.GENERATOR, value=row.new_generator),
    )


@dataclass(frozen=True)
class ReleaseRecords:
    """Every Record, and the lookups the tasks use to find theirs."""

    series: Mapping[str, Record]
    """One Record per row of ``TS_Dataset.jsonl``, by uuid."""
    copies: Mapping[tuple[str, str], Record]
    """One Record per distinct series in the description files that differs from its series' row, by
    uuid and value digest."""
    changed: Mapping[tuple[str, str], Record]
    """One Record per changed series, by uuid and counterfactual."""

    @classmethod
    def build(cls, index: ReleaseIndex) -> Self:
        """Build every Record from the index, without reading any values.

        Returns:
            The Records.
        """
        series = {
            uuid: _record(uuid, row.ref, "series", row.series_shape, _series_annotations(row))
            for uuid, row in index.series.items()
        }
        copies: dict[tuple[str, str], Record] = {}
        for row in index.descriptions:
            key = (row.uuid, row.series_shape.digest)
            if row.series_shape != index.series[row.uuid].series_shape and key not in copies:
                # The authors' description task reads this file's values. Its texts equal the series' row.
                copies[key] = _record(row.uuid, row.ref, "series", row.series_shape, _series_annotations(row))
        changed = {
            key: _record(row.uuid, row.ref, "new_series", row.new_series_shape, _changed_annotations(row))
            for key, row in index.changed.items()
        }
        return cls(series, copies, changed)

    def __iter__(self) -> Iterator[Record]:
        """Return the series Records, then the copies, then the changed series, each in source order."""
        return chain(self.series.values(), self.copies.values(), self.changed.values())

    def __len__(self) -> int:
        return len(self.series) + len(self.copies) + len(self.changed)

    def of_description(self, row: DescriptionRow) -> Record:
        """Return the Record whose values a description row holds: its copy's, or else its series'."""
        copy = self.copies.get((row.uuid, row.series_shape.digest))
        return self.series[row.uuid] if copy is None else copy
