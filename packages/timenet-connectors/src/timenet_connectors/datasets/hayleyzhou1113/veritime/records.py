"""One Record per distinct series: its values as one lazy signal, and the rows that publish it in typed metadata."""

from functools import lru_cache, partial

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from timenet.dataset import Record, Signal, Source
from timenet_connectors.datasets.hayleyzhou1113.veritime.release import Part
from timenet_connectors.datasets.hayleyzhou1113.veritime.rows import Row, check_series, reread_series
from timenet_connectors.datasets.hayleyzhou1113.veritime.series import Series, SeriesKind
from timenet_connectors.metadata import RecordMetadata


class VeriTimeRowMetadata(BaseModel):
    """One published row, as its Record's VeriTime metadata lists it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    part: Part
    row: int
    """The row's position in its file, from 0."""


class VeriTimeRecordMetadata(RecordMetadata):
    """A VeriTime Record's metadata: what its series is, and the rows that publish it.

    The release names no upstream series, file or subject, so the shared ``subject_id`` stays unset.
    """

    kind: SeriesKind
    rows: list[VeriTimeRowMetadata]
    """Every row that publishes this series, in file order. Scenario rows can be in either split."""


def record_metadata(record: Record) -> VeriTimeRecordMetadata:
    """Read a VeriTime Record's typed metadata back.

    Returns:
        What the Record's series is, and the rows that publish it.

    Raises:
        TimeFValidationError: If the Record's metadata is not VeriTime metadata.
    """  # noqa: DOC502 - raised by from_json
    return VeriTimeRecordMetadata.from_json(record.metadata)


class SeriesLoader:
    """Load a series lazily, rereading its first row.

    A loader serves the Records of one load, like RATS40K's ``ChannelLoader``.
    """

    def __init__(self) -> None:
        # One signal per Record, and a Record's values load once, so one cached row suffices. A
        # RowRef is hashable, unlike a Row, so it keys the cache.
        self._reread = lru_cache(maxsize=1)(reread_series)

    def __call__(self, row: Row) -> pa.Array:
        """Return a row's series as float64 values.

        Returns:
            The values, checked against the converted row.
        """
        values = self._reread(row.ref)
        check_series(values, row)
        return pa.array(values, type=pa.float64())


def build_record(series: Series, loader: SeriesLoader) -> Record:
    """Build a Record with one lazy signal and the rows that publish its series.

    Args:
        series: The distinct series and its rows.
        loader: The loader the Record's signal reads its values through.

    Returns:
        The series' Record, whose values stay in the source file until read.
    """
    kind = series.kind
    signal = Signal.from_loader(
        name=kind.signal_name,
        spec=kind.spec,
        time_axis=kind.time_axis(),
        n_values=series.published.n_values,
        loader=partial(loader, series.value_source),
    )
    rows = [VeriTimeRowMetadata(part=row.ref.part, row=row.ref.row) for row in series.rows]
    metadata = VeriTimeRecordMetadata(kind=kind, rows=rows)
    # Provenance only: metadata is never model input, and every task states its question in its prompt.
    return Record(sources=(Source(name=kind.spec.name, signals=(signal,)),), metadata=metadata.to_json())
