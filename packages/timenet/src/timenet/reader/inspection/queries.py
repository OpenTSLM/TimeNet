"""Bounded query operations for the TimeNet viewer."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import pyarrow as pa

from timenet.dataset import IrregularAxis, Signal
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.query_helpers import like_literal
from timenet.reader.inspection.limits import validate_page
from timenet.reader.inspection.types import InspectionPage, RecordSummary, SignalSummary, SourceSummary


if TYPE_CHECKING:
    from timenet.reader.reader import TimeFReader


_DEFAULT_LIMIT = 50


class ReaderQueries:
    """Read bounded record, source, signal, and irregular-axis viewer projections.

    The object borrows its reader's lifetime.  It intentionally exposes only simple Python types,
    allowing the HTTP layer to choose its transport format.
    """

    def __init__(self, reader: TimeFReader) -> None:
        """Create an inspection facade for ``reader``."""
        self._reader = reader

    @property
    def dataset_id(self) -> str:
        """Return the inspected root's public dataset ID."""
        return self._reader.metadata.dataset_id

    def records_page(
        self, query: str | None = None, after: str | None = None, limit: int = _DEFAULT_LIMIT
    ) -> InspectionPage[RecordSummary]:
        """Return a literal case-insensitive record-ID search page in stable ID order."""
        size = validate_page(limit)
        escaped = like_literal(query or "")
        rows = (
            self._reader._control_reader()
            .connection.execute(
                """SELECT record_id, time_span_start_us, time_span_end_us
               FROM records
               WHERE record_id > ? AND lower(record_id) LIKE lower(?) ESCAPE '\\'
               ORDER BY record_id LIMIT ?""",
                [after or "", f"%{escaped}%", size + 1],
            )
            .fetchall()
        )
        items = tuple(RecordSummary(*row) for row in rows[:size])
        return InspectionPage(items, items[-1].record_id if len(rows) > size else None)

    def sources_page(
        self, record_id: str, parent_id: str | None, after: str | None = None, limit: int = _DEFAULT_LIMIT
    ) -> InspectionPage[SourceSummary]:
        """Return immediate Sources below ``parent_id`` for one Record."""
        self.require_record(record_id)
        size = validate_page(limit)
        rows = (
            self._reader._control_reader()
            .connection.execute(
                """SELECT sources.source_id, parents.source_id, sources.name
               FROM sources JOIN records USING (record_key)
               LEFT JOIN sources parents ON sources.parent_source_key = parents.source_key
               WHERE records.record_id = ? AND sources.source_id > ?
                 AND ((? IS NULL AND sources.parent_source_key IS NULL) OR parents.source_id = ?)
               ORDER BY sources.source_id LIMIT ?""",
                [record_id, after or "", parent_id, parent_id, size + 1],
            )
            .fetchall()
        )
        items = tuple(SourceSummary(*row) for row in rows[:size])
        return InspectionPage(items, items[-1].source_id if len(rows) > size else None)

    def signals_page(
        self, record_id: str, source_id: str, after: str | None = None, limit: int = _DEFAULT_LIMIT
    ) -> InspectionPage[SignalSummary]:
        """Return direct Signals for one Source, checking that it belongs to ``record_id``."""
        self.require_record(record_id)
        size = validate_page(limit)
        rows = (
            self._reader._control_reader()
            .connection.execute(
                """SELECT signals.signal_id, signals.name, axes.axis_type, signals.n_values,
                      signals.dtype, signals.unit, signals.value_shape
               FROM signals JOIN sources USING (source_key) JOIN records USING (record_key)
               JOIN axes USING (axis_key)
               WHERE records.record_id = ? AND sources.source_id = ? AND signals.signal_id > ?
               ORDER BY signals.signal_id LIMIT ?""",
                [record_id, source_id, after or "", size + 1],
            )
            .fetchall()
        )
        items = tuple(
            SignalSummary(
                signal_id=row[0],
                name=row[1],
                axis_kind=row[2],
                n_values=row[3],
                dtype=row[4],
                unit=row[5],
                value_shape=tuple(row[6]),
            )
            for row in rows[:size]
        )
        return InspectionPage(items, items[-1].signal_id if len(rows) > size else None)

    def signal(self, record_id: str, signal_id: str) -> Signal:
        """Return one lazy Signal owned by the selected Record."""
        self.require_record(record_id)
        return self._reader._control_reader().read_signal(record_id, signal_id)

    def require_record(self, record_id: str) -> None:
        """Validate that the record exists in this dataset.

        Raises:
            TimeFValidationError: If the record is not visible.
        """
        row = (
            self._reader._control_reader()
            .connection.execute(
                "SELECT 1 FROM records WHERE record_id = ?",
                [record_id],
            )
            .fetchone()
        )
        if row is None:
            raise TimeFValidationError(f"no such record {record_id!r}")

    def offsets(self, record_id: str, signal_id: str, start: int, stop: int) -> pa.Array:
        """Return a bounded half-open range of irregular offsets.

        Raises:
            TimeFValidationError: If the range or requested signal is invalid.
            TimeFFormatError: If persisted offsets violate their axis contract.
        """
        if start < 0 or stop < start:
            raise TimeFValidationError("offset range must satisfy 0 <= start <= stop")
        self.require_record(record_id)
        signal = self.signal(record_id, signal_id)
        if not isinstance(signal.time_axis, IrregularAxis):
            raise TimeFValidationError(f"signal {signal_id!r} does not have an irregular axis")
        if stop > signal.n_values:
            raise TimeFValidationError(f"offset range ends after signal {signal_id!r}")
        row = (
            self._reader._control_reader()
            .connection.execute(
                """SELECT axis_key FROM signals JOIN sources USING (source_key) JOIN records USING (record_key)
               WHERE records.record_id = ? AND signals.signal_id = ?""",
                [record_id, signal_id],
            )
            .fetchone()
        )
        if row is None:
            raise TimeFValidationError(f"record {record_id!r} has no signal {signal_id!r}")
        read_start = max(0, start - 1)
        table = (
            self._reader._control_reader()
            .connection.execute(
                """SELECT position, offset_us FROM axis_offsets WHERE axis_key = ? AND position >= ? AND position < ?
               ORDER BY position""",
                [row[0], read_start, stop],
            )
            .to_arrow_table()
        )
        positions, offsets = (table.column(i).combine_chunks() for i in range(2))
        values = offsets.to_pylist()
        if positions.to_pylist() != list(range(read_start, stop)) or any(value is None for value in values):
            raise TimeFFormatError("irregular axis has missing, duplicate, or null offsets")
        if any(a > b for a, b in pairwise(values)):
            raise TimeFFormatError("irregular axis offsets decrease")
        if values and (
            (read_start == 0 and values[0] != signal.time_axis.first_us)
            or (stop == signal.n_values and values[-1] != signal.time_axis.last_us)
        ):
            raise TimeFFormatError("irregular axis endpoints disagree with offsets")
        return offsets.slice(start - read_start)
