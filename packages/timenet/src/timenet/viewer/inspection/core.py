"""Bounded query operations for the TimeNet viewer."""

from __future__ import annotations

from functools import partial
from itertools import pairwise
from typing import TYPE_CHECKING

import pyarrow as pa

from timenet.dataset import IrregularAxis, Signal
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.control_reader import _SignalRow
from timenet.viewer.inspection.requests import (
    AnnotationQuery,
    AnnotationWindowQuery,
    OwnerRecordQuery,
    RecordDetailQuery,
    TaskDetailQuery,
    TaskQuery,
)
from timenet.viewer.inspection.types import InspectionPage, RecordSummary, SignalSummary, SourceSummary


if TYPE_CHECKING:
    from timenet.reader.reader import TimeFReader


_DEFAULT_LIMIT = 50
_MAX_LIMIT = 200


class ViewerInspection:
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

    def independent_reader(self) -> TimeFReader:
        """Return independent lazy connections, preserving shared pinned ancestors."""
        from timenet.reader.reader import TimeFReader  # noqa: PLC0415

        copies: dict[int, TimeFReader] = {}

        def copy(node: TimeFReader) -> TimeFReader:
            key = id(node)
            if key not in copies:
                copies[key] = TimeFReader(
                    node._version, parents={name: copy(parent) for name, parent in node._parents.items()}
                )
            return copies[key]

        return copy(self._reader)

    def dependencies(self) -> tuple[TimeFReader, ...]:
        """Return each pinned reader once, parents before their dependents."""
        result: list[TimeFReader] = []
        seen: set[tuple[str, str]] = set()

        def visit(node: TimeFReader) -> None:
            key = (node.metadata.dataset_id, str(node.metadata.dataset_version))
            if key in seen:
                return
            seen.add(key)
            for parent in node._parents.values():
                visit(parent)
            result.append(node)

        visit(self._reader)
        return tuple(result)

    def overview(self) -> dict[str, object]:
        """Return manifest-backed overview facts without reading signal values."""
        from timenet.viewer.inspection.details import overview  # noqa: PLC0415

        return overview(self._reader)

    def record_detail(self, body: RecordDetailQuery) -> dict[str, object]:
        """Return persisted clock, span provenance and bounded metadata."""
        from timenet.viewer.inspection.details import record_detail  # noqa: PLC0415

        return record_detail(self._reader, body)

    def owner_record(self, body: OwnerRecordQuery) -> dict[str, object] | None:
        """Return the containing Record for a Source or Signal owner."""
        from timenet.viewer.inspection.details import owner_record  # noqa: PLC0415

        return owner_record(self._reader, body)

    def tasks(self, body: TaskQuery) -> dict[str, object]:
        """Return a bounded, lightweight task page."""
        from timenet.viewer.inspection.tasks import tasks  # noqa: PLC0415

        return tasks(self._reader, body)

    def task_detail(self, body: TaskDetailQuery) -> dict[str, object] | None:
        """Return bounded task relationships for the selected task."""
        from timenet.viewer.inspection.tasks import task_detail  # noqa: PLC0415

        return task_detail(self._reader, body)

    def browse_annotations(self, body: AnnotationQuery) -> dict[str, object]:
        """Return a bounded annotation projection."""
        from timenet.viewer.inspection.annotations import browse_annotations  # noqa: PLC0415

        return browse_annotations(self._reader, body)

    def window_annotations(self, body: AnnotationWindowQuery) -> dict[str, object]:
        """Return a bounded annotation projection."""
        from timenet.viewer.inspection.annotations import window_annotations  # noqa: PLC0415

        return window_annotations(self._reader, body)

    def records_page(
        self, query: str | None = None, after: str | None = None, limit: int = _DEFAULT_LIMIT
    ) -> InspectionPage[RecordSummary]:
        """Return a literal case-insensitive record-ID search page in stable ID order."""
        size = _page_limit(limit)
        escaped = _like_literal(query or "")
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
        owner = self.record_owner(record_id)
        if owner is not self._reader:
            return ViewerInspection(owner).sources_page(record_id, parent_id, after, limit)
        size = _page_limit(limit)
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
        owner = self.record_owner(record_id)
        if owner is not self._reader:
            return ViewerInspection(owner).signals_page(record_id, source_id, after, limit)
        size = _page_limit(limit)
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
        """Return one lazy Signal, rejecting IDs that are not visible in the requested Record.

        Raises:
            TimeFValidationError: If the signal is not visible in the requested record.
        """
        owner = self.record_owner(record_id)
        control = owner._control_reader()
        row = control.connection.execute(
            """SELECT signals.* FROM signals JOIN sources USING (source_key)
               JOIN records USING (record_key) WHERE record_id = ? AND signal_id = ?""",
            [record_id, signal_id],
        ).fetchone()
        if row is None:
            raise TimeFValidationError(f"record {record_id!r} has no signal {signal_id!r}")
        stored = _SignalRow(*row)
        spec = control._spec_from_row(stored)
        axis = control._read_axes([stored.axis_key])[stored.axis_key].axis
        return Signal.from_loader(
            id=stored.signal_id,
            name=stored.name,
            spec=spec,
            time_axis=axis,
            n_values=stored.n_values,
            loader=owner._make_signal_loader(stored.signal_key, stored.signal_id, spec),
            time_offsets_loader=(
                partial(self.offsets, record_id, signal_id, 0, stored.n_values)
                if isinstance(axis, IrregularAxis)
                else None
            ),
        )

    def record_owner(self, record_id: str) -> TimeFReader:
        """Resolve an imported record through the pinned parent chain.

        Returns:
            The reader storing the record's hierarchy and values.

        Raises:
            TimeFValidationError: If the record is not visible.
            TimeFFormatError: If an import has no resolved parent.
        """
        row = (
            self._reader._control_reader()
            .connection.execute(
                """SELECT parent_dataset_id FROM records LEFT JOIN record_imports USING (record_key)
               WHERE record_id = ?""",
                [record_id],
            )
            .fetchone()
        )
        if row is None:
            raise TimeFValidationError(f"no such record {record_id!r}")
        if row[0] is None:
            return self._reader
        parent = self._reader._parents.get(row[0])
        if parent is None:
            raise TimeFFormatError("record import has no pinned parent")
        return ViewerInspection(parent).record_owner(record_id)

    def offsets(self, record_id: str, signal_id: str, start: int, stop: int) -> pa.Array:
        """Return a bounded half-open range of irregular offsets.

        Raises:
            TimeFValidationError: If the range or requested signal is invalid.
            TimeFFormatError: If persisted offsets violate their axis contract.
        """
        if start < 0 or stop < start:
            raise TimeFValidationError("offset range must satisfy 0 <= start <= stop")
        owner = self.record_owner(record_id)
        if owner is not self._reader:
            return ViewerInspection(owner).offsets(record_id, signal_id, start, stop)
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

    def read_steps(self, record_id: str, signal_id: str, start: int, stop: int) -> pa.Array:
        """Read a validated, half-open signal range without materializing other signals.

        Args:
            record_id: The record that owns the signal.
            signal_id: The signal's ID within that record.
            start: First step, inclusive.
            stop: Last step, exclusive.

        Returns:
            The requested Arrow values.

        Raises:
            TimeFValidationError: If the range does not fit the selected signal.
        """
        signal = self.signal(record_id, signal_id)
        if start < 0 or stop < start or stop > signal.n_values:
            raise TimeFValidationError(f"invalid range [{start}, {stop}) for signal {signal_id!r}")
        return signal.read_steps(start, stop)


def _page_limit(limit: int) -> int:
    """Validate a public page size.

    Returns:
        The validated size.

    Raises:
        TimeFValidationError: If the size is outside the supported range.
    """
    if not 1 <= limit <= _MAX_LIMIT:
        raise TimeFValidationError(f"limit must be between 1 and {_MAX_LIMIT}")
    return limit


def _like_literal(value: str) -> str:
    """Escape a user string for DuckDB's LIKE expression.

    Returns:
        A literal-safe LIKE fragment.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
