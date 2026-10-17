"""Hydrate the TimeF object hierarchy from ``control.duckdb``."""

from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator
from contextlib import contextmanager
from fractions import Fraction
from functools import partial
import json
from pathlib import Path
from typing import Any, NamedTuple, TypeVar, cast

import duckdb
import pyarrow as pa
import pyarrow.compute as pc

from timenet.dataset import IrregularAxis, OrdinalAxis, Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.annotation_codec import decode_annotation_value
from timenet.format.control_schema import ANNOTATED_OBJECT_TYPES
from timenet.format.duckdb import check_control_schema, connect_control
from timenet.format.task_codec import (
    decode_input_modalities,
    decode_span,
    decode_split,
    decode_target,
    decode_task_payload,
)
from timenet.types import (
    TASKS,
    Annotation,
    InputModality,
    Task,
    TaskType,
    TimeInterval,
    TimeOrigin,
    TimePoint,
    TimeSeriesSpec,
    check_id,
    ureg,
)
from timenet.types.splits import Split, parse_split
from timenet.types.wire import ValueDtype


ValueLoader = Callable[[int, TimeSeriesSpec], pa.Array]
ValueLoaderFactory = Callable[[int, TimeSeriesSpec], Callable[[], pa.Array]]
OffsetsLoader = Callable[[IrregularAxis, int], pa.Array]


class _LoadedAxis(NamedTuple):
    """One hydrated axis and the shared length of its stored offsets."""

    axis: RegularAxis | IrregularAxis | OrdinalAxis
    n_values: int | None


class _RecordRow(NamedTuple):
    """One ``records`` row in schema column order."""

    record_id: int
    clock_id: int
    time_span_start_us: int | None
    time_span_end_us: int | None
    metadata: str


class _SourceRow(NamedTuple):
    """One ``sources`` row in schema column order."""

    source_id: int
    record_id: int
    parent_source_id: int | None
    name: str
    metadata: str


class _SignalRow(NamedTuple):
    """One ``signals`` row in schema column order."""

    signal_id: int
    source_id: int
    name: str
    axis_id: int
    spec_type: str
    spec_name: str
    unit: str | None
    dtype: str
    categories: list[str]
    value_shape: list[int]
    dimension_names: list[str]
    nullable: bool
    n_values: int
    metadata: str
    modality: str

    @property
    def spec_fields(self) -> tuple[Any, ...]:
        """Return the columns that together identify one inline specification, as a hashable key."""
        return (
            self.spec_type,
            self.spec_name,
            self.unit,
            self.dtype,
            tuple(self.categories),
            tuple(self.value_shape),
            tuple(self.dimension_names),
            self.nullable,
            self.modality,
        )


class _OccurrenceRow(NamedTuple):
    """One annotation occurrence joined with its content."""

    occurrence_id: int
    object_type: str
    object_id: int | None
    span_type: str
    start_us: int | None
    end_us: int | None
    provenance: str | None
    confidence: float | None
    metadata: str
    content_id: int | None
    name: str | None
    value_kind: str | None
    text_value: str | None
    integer_value: int | None
    float_value: float | None
    boolean_value: bool | None
    text_list_value: list[str] | None
    unit: str | None
    description: str | None
    content_metadata: str | None


class _TaskRow(NamedTuple):
    """One ``tasks`` row in schema column order."""

    task_id: int
    task_type: str
    prompt: str | None
    input_modalities: list[str] | None
    split: str | None
    scope_type: str | None
    scope_start: int | None
    scope_end: int | None
    scope_signal_ids: list[int] | None
    has_inline_targets: bool
    target_schema: str | None
    unit: str | None
    target_name: str | None
    mode: str | None
    rationale: str | None
    metadata: str

    @property
    def payload_columns(self) -> dict[str, str | None]:
        """Return the subclass configuration columns for the task codec."""
        return {
            "target_schema": self.target_schema,
            "unit": self.unit,
            "target_name": self.target_name,
            "mode": self.mode,
        }


class _TargetRow(NamedTuple):
    """One ``task_targets`` row, without its position, which the query orders by."""

    task_id: int
    target_kind: str
    text_value: str | None
    integer_value: int | None
    float_value: float | None
    boolean_value: bool | None
    record_id: int | None
    signal_id: int | None
    span_start: int | None
    span_end: int | None
    span_signal_ids: list[int] | None


_TASK_BATCH_SIZE = 5_000
"""Tasks hydrated per round of keyed queries."""


def _annotation_query(object_type: str | None, *, keyed: bool) -> str:
    """Build the occurrence query, restricted to one object type and to requested IDs when asked.

    Args:
        object_type: Keep one object type, or ``None`` for all.
        keyed: Whether the caller passes owner IDs. The filter sits inside the occurrences
            subquery so it applies before the join. It matches IDs of every type, and the caller
            groups the rows by ``(object_type, object_id)``.

    Returns:
        SQL with a positional parameter for the requested IDs when keyed, then one for the object
        type when it is given.

    Raises:
        TimeFFormatError: If ``object_type`` is not an annotated object type.
    """
    if object_type is not None and object_type not in ANNOTATED_OBJECT_TYPES:
        raise TimeFFormatError(f"unknown annotated object type {object_type!r}")
    # A parameter inside an OR keeps DuckDB from pushing the filter below the join, which made a
    # one-Record read scan every occurrence, so each filter is one plain predicate.
    occurrences = (
        "(SELECT * FROM annotation_occurrences WHERE object_id IN (SELECT unnest(?)))"
        if keyed
        else "annotation_occurrences"
    )
    where = "" if object_type is None else "WHERE o.object_type = ?"
    return f"""SELECT o.occurrence_id, o.object_type, o.object_id, o.span_type,
               o.start_us, o.end_us, o.provenance, o.confidence,
               o.metadata, o.content_id, c.name, c.value_kind, c.text_value, c.integer_value,
               c.float_value, c.boolean_value, c.text_list_value, c.unit, c.description, c.metadata
        FROM {occurrences} o
        LEFT JOIN annotation_contents c USING (content_id)
        {where}
        ORDER BY o.object_type, o.object_id, o.content_id, o.span_type,
                 o.start_us NULLS FIRST, o.end_us NULLS FIRST, o.occurrence_id"""  # noqa: S608 - fixed fragments


def _check_owners(
    annotated: Iterable[tuple[str, int | None]],
    owners: dict[str, Collection[int]],
) -> None:
    """Reject occurrences whose owner a full read should have hydrated but did not.

    Args:
        annotated: The ``(object_type, object_id)`` owners that occurrences name.
        owners: The hydrated IDs of each object type to check. Other types are skipped.

    Raises:
        TimeFFormatError: If an occurrence of a checked type names an ID outside its set.
    """
    for object_type, object_id in annotated:
        ids = owners.get(object_type)
        if ids is not None and object_id not in ids:
            raise TimeFFormatError(f"annotation occurrence refers to missing {object_type} {object_id!r}")


class _TaskFacts(NamedTuple):
    """Facts about one control database that every task hydration reuses."""

    parent_task_ids: frozenset[int]
    tasks_annotated: bool


_HierarchyRow = TypeVar("_HierarchyRow", _RecordRow, _SourceRow, _SignalRow)
_Row = TypeVar("_Row", bound=tuple[Any, ...])
_Ref = TypeVar("_Ref", Record, Annotation)
_T = TypeVar("_T")

_HIERARCHY_TABLES: dict[type[tuple[Any, ...]], tuple[str, str, str]] = {
    _RecordRow: ("records", "record_id", "record_id"),
    _SourceRow: ("sources", "record_id", "source_id"),
    _SignalRow: ("signals", "source_id", "signal_id"),
}
"""Table name, relation column, and public ID column of each hierarchy row type."""


def _rows(cursor: duckdb.DuckDBPyConnection, make: Callable[[Iterable[Any]], _Row]) -> list[_Row]:
    """Fetch every row of a query as a named tuple.

    Args:
        cursor: The executed query.
        make: The row type's ``_make`` constructor.

    Returns:
        The rows in query order.

    Raises:
        TimeFFormatError: If the query returned a different number of columns than the row type has.
    """
    try:
        return [make(row) for row in cursor.fetchall()]
    except TypeError as exc:
        raise TimeFFormatError(f"control.duckdb returned an unexpected column layout: {exc}") from exc


def _required(value: _T | None, what: str) -> _T:
    """Return a column value that the schema allows to be NULL but the row's kind does not.

    Raises:
        TimeFFormatError: If the value is NULL.
    """
    if value is None:
        raise TimeFFormatError(f"control.duckdb is missing {what}")
    return value


def _decode_json(value: str | None, *, default: Any = None) -> Any:
    """Decode one DuckDB JSON value.

    Returns:
        The decoded value, or ``default`` for SQL ``NULL``.

    Raises:
        TimeFFormatError: If stored JSON is malformed.
    """
    if value is None:
        return default
    if value == "{}":
        return {}
    if value == "[]":
        return []
    if value == "null":
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise TimeFFormatError(f"control.duckdb contains invalid JSON: {value!r}") from exc


def _occurrence_span(row: _OccurrenceRow) -> TimePoint | TimeInterval | None:
    """Rebuild the span of one annotation occurrence.

    Returns:
        The point or interval span, or ``None`` for a static occurrence.

    Raises:
        TimeFFormatError: If the occurrence has an unknown span type or lacks a required bound.
    """
    if row.span_type == "static":
        return None
    start_us = _required(row.start_us, f"the start of annotation occurrence {row.occurrence_id!r}")
    if row.span_type == "point":
        return TimePoint(start_us=start_us)
    if row.span_type == "interval":
        end_us = _required(row.end_us, f"the end of annotation occurrence {row.occurrence_id!r}")
        return TimeInterval(start_us=start_us, end_us=end_us)
    raise TimeFFormatError(f"annotation occurrence {row.occurrence_id!r} has unknown span type {row.span_type!r}")


def _missing_values(signal_id: int, _spec: TimeSeriesSpec) -> pa.Array:
    """Fail when metadata-only hydration is asked to load values.

    Raises:
        TimeFFormatError: Always, because no values loader was configured.
    """
    raise TimeFFormatError(f"no values loader is configured for signal {signal_id!r}")


class DuckDBControlReader:
    """Keep one read-only DuckDB connection and hydrate requested records from it."""

    def __init__(
        self,
        path: Path,
        *,
        value_loader: ValueLoader | None = None,
        value_loader_factory: ValueLoaderFactory | None = None,
        offsets_loader: OffsetsLoader | None = None,
    ) -> None:
        """Open and validate an immutable control database.

        Args:
            path: Local path to ``control.duckdb``.
            value_loader: Lazy values-plane resolver keyed by signal ID.
            value_loader_factory: Factory for range-aware per-Signal loaders.
            offsets_loader: Lazy irregular-axis resolver keyed by axis ID.

        Raises:
            TimeFFormatError: If the database cannot be opened or has an unsupported schema.
        """
        self.path = Path(path)
        try:
            self.connection = connect_control(self.path, read_only=True)
            check_control_schema(self.connection)
        except duckdb.Error as exc:
            raise TimeFFormatError(f"could not open control database {self.path}: {exc}") from exc
        self._value_loader = value_loader or _missing_values
        self._value_loader_factory = value_loader_factory
        self._offsets_loader = offsets_loader
        self._clock_cache: dict[int, TimeOrigin] = {}
        self._task_facts: _TaskFacts | None = None

    def close(self) -> None:
        """Close the reader's DuckDB connection."""
        self.connection.close()
        self._clock_cache.clear()

    def __enter__(self) -> "DuckDBControlReader":
        """Return this open reader.

        Returns:
            This reader.
        """
        return self

    def __exit__(self, *_: object) -> None:
        """Close the connection when leaving a context manager."""
        self.close()

    def record_ids(self) -> tuple[int, ...]:
        """Return every stored Record ID in the order :meth:`read_records` yields them, without hydrating.

        Returns:
            The Record IDs sorted by ID.
        """
        return tuple(
            row[0] for row in self.connection.execute("SELECT record_id FROM records ORDER BY record_id").fetchall()
        )

    def read_records(  # noqa: PLR0912, PLR0914, PLR0915 - related row sets stay together
        self,
        record_ids: Iterable[int] | None = None,
        *,
        with_annotations: bool = True,
    ) -> tuple[Record, ...]:
        """Hydrate complete recursive records while leaving signal values lazy.

        Args:
            record_ids: Requested IDs in result order, or ``None`` for every record.
            with_annotations: Whether to hydrate annotations on records, sources, and signals.

        Returns:
            The hydrated records.

        Raises:
            TimeFValidationError: If a requested record is absent.
            TimeFFormatError: If the stored hierarchy is inconsistent.
        """
        requested = None if record_ids is None else tuple(check_id(item, "record id") for item in record_ids)
        read_all = requested is None
        rows = self._read_rows(_RecordRow, requested)
        by_id = self._index_rows(rows, "record", lambda row: row.record_id)
        order = tuple(by_id) if requested is None else requested
        missing = [record_id for record_id in order if record_id not in by_id]
        if missing:
            raise TimeFValidationError(f"no such record(s) in control.duckdb: {missing}")

        source_rows = self._read_rows(_SourceRow, None if read_all else by_id)
        source_data = self._index_rows(source_rows, "source", lambda row: row.source_id)
        signal_rows = self._read_rows(_SignalRow, None if read_all else source_data)
        signal_data = self._index_rows(signal_rows, "signal", lambda row: row.signal_id)
        for signal_row in signal_data.values():
            if signal_row.source_id not in source_data:
                raise TimeFFormatError(
                    f"signal {signal_row.signal_id!r} refers to missing source {signal_row.source_id!r}"
                )
        annotations: dict[tuple[str, int | None], tuple[Annotation, ...]] = {}
        if with_annotations:
            object_ids = [*by_id, *source_data, *signal_data]
            annotations = self._read_annotations(None if read_all else object_ids)
            if read_all:
                _check_owners(
                    annotations.keys(),
                    {
                        "Record": by_id.keys(),
                        "Source": source_data.keys(),
                        "Signal": signal_data.keys(),
                    },
                )
        axes = self._read_axes({row.axis_id for row in signal_rows})

        signals_by_source: dict[int, list[Signal]] = defaultdict(list)
        specs: dict[tuple[Any, ...], TimeSeriesSpec] = {}
        for signal_row in signal_rows:
            loaded_axis = axes.get(signal_row.axis_id)
            if loaded_axis is None:
                raise TimeFFormatError(f"signal {signal_row.signal_id!r} refers to missing axis {signal_row.axis_id!r}")
            axis = loaded_axis.axis
            spec = specs.get(signal_row.spec_fields)
            if spec is None:
                spec = self._spec_from_row(signal_row)
                specs[signal_row.spec_fields] = spec
            offsets_loader = None
            if isinstance(axis, IrregularAxis):
                if loaded_axis.n_values is None:
                    raise TimeFFormatError(f"irregular axis {axis.id!r} has no Signal length")
                offsets_loader = partial(self._offsets_loader or self.load_axis_offsets, axis, loaded_axis.n_values)
            signals_by_source[signal_row.source_id].append(
                Signal.from_loader(
                    id=signal_row.signal_id,
                    name=signal_row.name,
                    spec=spec,
                    time_axis=axis,
                    n_values=signal_row.n_values,
                    loader=(
                        self._value_loader_factory(signal_row.signal_id, spec)
                        if self._value_loader_factory is not None
                        else partial(self._value_loader, signal_row.signal_id, spec)
                    ),
                    time_offsets_loader=offsets_loader,
                    annotations=annotations.get(("Signal", signal_row.signal_id), ()),
                    metadata=_decode_json(signal_row.metadata, default={}),
                )
            )

        missing_clocks = {row.clock_id for row in rows} - self._clock_cache.keys()
        if missing_clocks:
            for clock_id, timestamp in self.connection.execute(
                "SELECT clock_id, start_time_us FROM clocks WHERE clock_id IN (SELECT unnest(?))",
                [list(missing_clocks)],
            ).fetchall():
                self._clock_cache[clock_id] = TimeOrigin(timestamp)
        if missing_clocks - self._clock_cache.keys():
            raise TimeFFormatError(
                f"records reference missing clocks {sorted(missing_clocks - self._clock_cache.keys())}"
            )

        children, roots_by_record = self._group_sources(source_rows)
        hydrated_sources: set[int] = set()

        def hydrate_source(source_id: int, record_id: int) -> Source:
            source_row = source_data[source_id]
            if source_row.record_id != record_id:
                raise TimeFFormatError(f"source {source_id!r} crosses record boundaries")
            hydrated_sources.add(source_id)
            return Source(
                id=source_id,
                name=source_row.name,
                sources=tuple(hydrate_source(child, record_id) for child in children[source_id]),
                signals=tuple(signals_by_source[source_id]),
                annotations=annotations.get(("Source", source_row.source_id), ()),
                metadata=_decode_json(source_row.metadata, default={}),
            )

        records: list[Record] = []
        for record_id in order:
            record_row = by_id[record_id]
            span = None
            if record_row.time_span_start_us is not None:
                span = TimeInterval(
                    start_us=record_row.time_span_start_us,
                    end_us=_required(record_row.time_span_end_us, f"the time span end of record {record_id!r}"),
                )
            root_sources = tuple(hydrate_source(source_id, record_id) for source_id in roots_by_record[record_id])
            record = Record(
                id=record_id,
                sources=root_sources,
                start_time=self._clock_cache[record_row.clock_id],
                annotations=annotations.get(("Record", record_id), ()),
                time_span=span,
                metadata=_decode_json(record_row.metadata, default={}),
            )
            records.append(record)
        unreachable = source_data.keys() - hydrated_sources
        if unreachable:
            raise TimeFFormatError(
                f"source hierarchy contains a missing record, missing parent, or cycle: {sorted(unreachable)}"
            )
        return tuple(records)

    def _read_rows(
        self,
        row_type: type[_HierarchyRow],
        related_ids: Iterable[object] | None,
    ) -> list[_HierarchyRow]:
        """Read all rows or only rows related to the supplied IDs.

        Args:
            row_type: The hierarchy row type, which names the table and its columns.
            related_ids: Values of the table's relation column to keep, or ``None`` for every row.

        Returns:
            Rows from the requested hierarchy table.
        """
        table, relation_column, id_column = _HIERARCHY_TABLES[row_type]
        columns = ", ".join(row_type._fields)
        query = (
            f"SELECT {columns} FROM {table} WHERE ? OR {relation_column} "  # noqa: S608 - fixed identifiers
            f"IN (SELECT unnest(?)) ORDER BY {id_column}"
        )
        cursor = self.connection.execute(query, [related_ids is None, list(related_ids or ())])
        return _rows(cursor, row_type._make)

    @staticmethod
    def _index_rows(
        rows: list[_HierarchyRow],
        kind: str,
        id_of: Callable[[_HierarchyRow], int],
    ) -> dict[int, _HierarchyRow]:
        """Return rows by public ID and reject duplicate IDs.

        Args:
            rows: The rows to index.
            kind: The object kind named in the error message.
            id_of: The public ID of each row.

        Raises:
            TimeFFormatError: If a public ID occurs more than once.
        """
        indexed = {id_of(row): row for row in rows}
        if len(indexed) != len(rows):
            raise TimeFFormatError(f"control.duckdb contains duplicate {kind} IDs")
        return indexed

    @staticmethod
    def _group_sources(
        rows: list[_SourceRow],
    ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        """Group child and root Source IDs.

        Returns:
            Child IDs by parent Source ID and root IDs by Record ID.
        """
        children: dict[int, list[int]] = defaultdict(list)
        roots: dict[int, list[int]] = defaultdict(list)
        for row in rows:
            if row.parent_source_id is None:
                roots[row.record_id].append(row.source_id)
            else:
                children[row.parent_source_id].append(row.source_id)
        return children, roots

    def read_dataset_annotations(self, dataset_id: str) -> tuple[Annotation, ...]:
        """Return annotations attached directly to a dataset.

        Returns:
            Dataset-level annotation occurrences in stable occurrence-ID order.
        """
        stored = self.connection.execute("SELECT 1 FROM datasets WHERE dataset_id = ?", [dataset_id]).fetchone()
        if stored is None:
            return ()
        return self._read_annotations(object_type="Dataset").get(("Dataset", None), ())

    def chunk_rows(self, signal_id: int) -> list[dict[str, Any]]:
        """Return one Signal's chunk locations in chunk order.

        Raises:
            TimeFFormatError: If the Signal has no stored chunks.
        """
        rows = self.chunk_rows_by_ids((signal_id,))[signal_id]
        if not rows:
            raise TimeFFormatError(f"signal {signal_id!r} has no stored value chunks")
        return rows

    def chunk_rows_by_ids(self, signal_ids: Iterable[int]) -> dict[int, list[dict[str, Any]]]:
        """Return chunk locators for a batch of Signals.

        Args:
            signal_ids: Signal IDs.

        Returns:
            Chunk rows keyed by Signal ID. Each Signal's rows are in chunk-index order.
        """
        ids = tuple(signal_ids)
        if not ids:
            return {}
        rows = self.connection.execute(
            """SELECT signal_id, chunk_index, value_path, chunk_major_index, chunk_minor_index, n_values
               FROM signal_chunks
               WHERE signal_id IN (SELECT unnest(?))
               ORDER BY signal_id, chunk_index""",
            [list(ids)],
        ).fetchall()
        grouped: dict[int, list[tuple[Any, ...]]] = defaultdict(list)
        for signal_id, *chunk_row in rows:
            grouped[signal_id].append(tuple(chunk_row))
        return {signal_id: self._chunk_dicts(grouped[signal_id]) for signal_id in ids}

    @staticmethod
    def _chunk_dicts(rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
        """Shape chunk rows as the dicts the values backends read.

        Returns:
            One dict per chunk, in the rows' order.
        """
        return [
            {
                "chunk_idx": row[0],
                "chunk_file": row[1],
                "chunk_major_idx": row[2],
                "chunk_minor_idx": row[3],
                "n_values": row[4],
            }
            for row in rows
        ]

    def read_tasks(
        self, records: Iterable[Record] | None = None, *, cached_records: Iterable[Record] = ()
    ) -> tuple[Task, ...]:
        """Hydrate every task attached to ``records`` and restore all in-memory object references.

        Args:
            records: Input Records to select, or ``None`` for every task. Tasks reuse these objects.
            cached_records: Hydrated Records to reuse without restricting the selection.

        Returns:
            Concrete tasks in stable ID order.

        Raises:
            TimeFFormatError: If a relationship refers to a missing object or has an invalid field.
        """  # noqa: DOC502 - raised by _TaskHydration
        hydration = _TaskHydration(self, None if records is None else tuple(records), cached_records=cached_records)
        return tuple(hydration.read_all())

    def iter_tasks(
        self,
        records: Iterable[Record] | None = None,
        *,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
        cached_records: Iterable[Record] = (),
    ) -> Iterator[Task]:
        """Yield the tasks whose inputs include one of ``records``, a bounded batch at a time.

        Each batch runs a handful of keyed queries and builds only that batch's tasks, so a caller
        that walks one Record at a time never holds the whole task table in memory. Only ``None``
        skips the input Record filter; an empty collection selects no tasks.

        Tasks a batch derives from through ``from_tasks`` are hydrated first when they have not
        been yielded yet, and kept until their dependants are built. Records a task refers to
        outside ``records`` are hydrated on demand.

        Args:
            records: Input Records to select, or ``None`` for every task. Tasks reuse these objects.
            split: Keep only tasks of this split, or ``None`` for every task.
            cached_records: Hydrated Records to reuse without restricting the selection.

        Yields:
            Concrete tasks in stable ID order.

        Raises:
            TimeFFormatError: If a relationship refers to a missing object or has an invalid field.
        """  # noqa: DOC502 - raised by _TaskHydration
        hydration = _TaskHydration(
            self,
            None if records is None else tuple(records),
            cached_records=cached_records,
            required_modalities=required_modalities,
            supported_modalities=supported_modalities,
            split=split,
        )
        yield from hydration.iter_tasks()

    def task_facts(self) -> _TaskFacts:
        """Return the per-database facts task hydration needs, read once per reader.

        Returns:
            The IDs of tasks other tasks derive from, and whether any task carries annotations.
        """
        if self._task_facts is None:
            parent_ids = {
                row[0]
                for row in self.connection.execute("SELECT DISTINCT parent_task_id FROM task_dependencies").fetchall()
            }
            (annotated,) = self.connection.execute(
                "SELECT count(*) FROM annotation_occurrences WHERE object_type = 'Task'"
            ).fetchone() or (0,)
            self._task_facts = _TaskFacts(frozenset(parent_ids), annotated > 0)
        return self._task_facts

    def task_table(
        self,
        record_ids: Iterable[int] | None = None,
        *,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
    ) -> pa.Table:
        """Return the task table as Arrow, with object references as public IDs, without building Tasks.

        One row per task with its type, prompt, rationale, the four scope columns, the configuration
        columns, ordered ``input_record_ids``, ordered
        ``from_task_ids``, and the metadata JSON string. Targets are in :meth:`target_table`.

        Args:
            record_ids: Keep only tasks whose inputs include one of these Records, or ``None`` for all.
            split: Keep only tasks of this split, or ``None`` for all.

        Returns:
            The tasks in task ID order.
        """
        where, parameters = self._selection_filter(record_ids, required_modalities, supported_modalities, split)
        return self.connection.execute(
            f"""WITH inputs AS (
                    SELECT task_id, list(record_id ORDER BY position) AS input_record_ids
                    FROM task_record_refs
                    WHERE field = 'inputs'
                    GROUP BY task_id
                ),
                parents AS (
                    SELECT task_id, list(parent_task_id ORDER BY position) AS from_task_ids
                    FROM task_dependencies
                    GROUP BY task_id
                )
                SELECT t.task_id, t.task_type, t.prompt, t.rationale,
                       t.scope_type, t.scope_start, t.scope_end, t.scope_signal_ids,
                       t.target_schema, t.unit, t.target_name, t.mode, t.has_inline_targets,
                       t.input_modalities, t.split,
                       coalesce(inputs.input_record_ids, []) AS input_record_ids,
                       coalesce(parents.from_task_ids, []) AS from_task_ids,
                       t.metadata
                FROM tasks t
                LEFT JOIN inputs USING (task_id)
                LEFT JOIN parents USING (task_id)
                {where}
                ORDER BY t.task_id""",  # noqa: S608 - fixed fragments
            parameters,
        ).to_arrow_table()

    def target_table(
        self,
        record_ids: Iterable[int] | None = None,
        *,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
    ) -> pa.Table:
        """Return every inline target as Arrow, one typed row each, with references as public IDs.

        The columns follow :func:`timenet.format.task_codec.encode_target`: a step span names its
        series in ``signal_id`` and leaves ``span_signal_ids`` NULL.

        Args:
            record_ids: Keep only targets of tasks whose inputs include one of these Records.
            split: Keep only targets of tasks of this split, or ``None`` for all.

        Returns:
            Targets in task ID and position order.
        """
        where, parameters = self._selection_filter(record_ids, required_modalities, supported_modalities, split)
        return self.connection.execute(
            f"""SELECT t.task_id, x.position, x.target_kind, x.text_value, x.integer_value, x.float_value,
                       x.boolean_value, x.record_id, x.signal_id, x.span_start, x.span_end, x.span_signal_ids
                FROM task_targets x
                JOIN tasks t USING (task_id)
                {where}
                ORDER BY t.task_id, x.position""",  # noqa: S608 - fixed fragments
            parameters,
        ).to_arrow_table()

    def annotation_table(self, object_type: str | None = None) -> pa.Table:
        """Return every annotation occurrence joined with its content as Arrow.

        Args:
            object_type: Keep only occurrences on this object type, or ``None`` for all.

        Returns:
            One row per occurrence with the annotated object's type and ID (NULL for the Dataset,
            which has a string ``dataset_id`` instead), the content's ID and key,
            typed value columns, unit, span columns, confidence, and JSON metadata.

        Raises:
            TimeFFormatError: If ``object_type`` is not an annotated object type.
        """
        if object_type is not None and object_type not in ANNOTATED_OBJECT_TYPES:
            raise TimeFFormatError(f"unknown annotated object type {object_type!r}")
        where = "" if object_type is None else "WHERE o.object_type = ?"
        return self.connection.execute(
            f"""SELECT o.occurrence_id, o.object_type, o.object_id, c.content_id, c.name AS key,
                       c.value_kind, c.text_value, c.integer_value, c.float_value, c.boolean_value,
                       c.text_list_value, c.unit, c.description, o.span_type, o.start_us, o.end_us,
                       o.confidence, o.provenance, c.metadata AS content_metadata, o.metadata
                FROM annotation_occurrences o
                JOIN annotation_contents c USING (content_id)
                {where}
                ORDER BY o.object_type, o.object_id, o.occurrence_id""",  # noqa: S608 - fixed fragments
            [] if object_type is None else [object_type],
        ).to_arrow_table()

    @staticmethod
    def _record_filter(record_ids: Iterable[int] | None) -> tuple[str, list[object]]:
        """Return the WHERE clause that keeps tasks whose inputs include one of ``record_ids``.

        Returns:
            The SQL fragment, empty for every task, and its parameters.
        """
        if record_ids is None:
            return "", []
        return (
            """WHERE t.task_id IN (
                   SELECT task_id FROM task_record_refs
                   WHERE field = 'inputs' AND record_id IN (SELECT unnest(?))
               )""",
            [[check_id(item, "record id") for item in record_ids]],
        )

    def _selection_filter(
        self,
        record_ids: Iterable[int] | None,
        required_modalities: Iterable[InputModality] | None,
        supported_modalities: Iterable[InputModality] | None,
        split: Split | str | None = None,
    ) -> tuple[str, list[object]]:
        """Build the task selection predicate for Arrow table reads.

        Returns:
            The SQL WHERE clause and its bound values.
        """
        record_where, record_params = self._record_filter(record_ids)
        modality_where, modality_params = self._modality_filter(required_modalities, supported_modalities)
        split_where, split_params = self._split_filter(split)
        predicates = [record_where.removeprefix("WHERE ")] if record_where else []
        if modality_where:
            predicates.append(modality_where)
        if split_where:
            predicates.append(split_where)
        parameters = [*record_params, *modality_params, *split_params]
        return ("WHERE " + " AND ".join(predicates) if predicates else "", parameters)

    @staticmethod
    def _split_filter(split: Split | str | None) -> tuple[str, list[object]]:
        """Build a SQL predicate that keeps the tasks of one split.

        Returns:
            The predicate and its bound value, both empty when no split is asked for.

        Raises:
            TimeFValidationError: If the split is unknown.
        """  # noqa: DOC502 - raised by parse_split
        if split is None:
            return "", []
        return "t.split = ?", [parse_split(split).value]

    @staticmethod
    def _modality_filter(
        required_modalities: Iterable[InputModality] | None,
        supported_modalities: Iterable[InputModality] | None,
    ) -> tuple[str, list[object]]:
        """Build a SQL predicate that keeps compatible declared inputs.

        Returns:
            The predicate and its bound modality lists.

        Raises:
            TimeFValidationError: If a requested modality is unknown.
        """
        try:
            required = (
                None
                if required_modalities is None
                else sorted({InputModality(item).value for item in required_modalities})
            )
            supported = (
                None
                if supported_modalities is None
                else sorted({InputModality(item).value for item in supported_modalities})
            )
        except ValueError as exc:
            raise TimeFValidationError("unknown input modality in task filter") from exc
        if not required and supported is None:
            return "", []
        predicates: list[str] = []
        parameters: list[object] = []
        if required:
            predicates.append("list_has_all(t.input_modalities, ?)")
            parameters.append(required)
        if supported is not None:
            predicates.append("list_has_all(?, t.input_modalities)")
            parameters.append(supported)
        return " AND ".join(predicates), parameters

    @contextmanager
    def _task_row_cursor(
        self,
        record_ids: list[int] | None,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
    ) -> Iterator[duckdb.DuckDBPyConnection]:
        """Stream selected task rows in task ID order through a separate query cursor.

        Args:
            record_ids: IDs of the Records whose input tasks to select, or ``None`` for every task.
            split: Keep only tasks of this split, or ``None`` for every task.

        Yields:
            A cursor for fetching selected rows in batches.
        """
        where, params = self._selection_filter(record_ids, required_modalities, supported_modalities, split)
        columns = ", ".join(f"t.{name}" for name in _TaskRow._fields)
        sql = f"SELECT {columns} FROM tasks t {where} ORDER BY t.task_id"  # noqa: S608 - fixed identifiers and predicates
        # Hydration runs other queries while this result is open, so it needs its own cursor.
        with self.connection.cursor() as selection:
            selection.execute(sql, params)
            yield selection

    @staticmethod
    def _task_filter(task_ids: list[int] | None) -> tuple[str, list[object]]:
        """Return the WHERE clause and parameters that select one batch, or nothing for every task.

        Returns:
            The SQL fragment, empty for the whole table, and its parameters.
        """
        if task_ids is None:
            return "", []
        return "WHERE task_id IN (SELECT unnest(?))", [task_ids]

    def _task_rows(self, task_ids: list[int] | None) -> list[_TaskRow]:
        """Read the task rows for one batch of IDs, or every task, in task ID order.

        Returns:
            The task rows.
        """
        columns = ", ".join(_TaskRow._fields)
        where, parameters = self._task_filter(task_ids)
        return _rows(
            self.connection.execute(
                f"SELECT {columns} FROM tasks {where} ORDER BY task_id",  # noqa: S608 - fixed identifiers
                parameters,
            ),
            _TaskRow._make,
        )

    def _target_rows(self, task_ids: list[int] | None) -> list[_TargetRow]:
        """Read the target rows for one batch of tasks, or every task, in task and position order.

        Returns:
            The target rows.
        """
        columns = ", ".join(_TargetRow._fields)
        where, parameters = self._task_filter(task_ids)
        return _rows(
            self.connection.execute(
                f"SELECT {columns} FROM task_targets {where} ORDER BY task_id, position",  # noqa: S608 - fixed identifiers
                parameters,
            ),
            _TargetRow._make,
        )

    def _check_signal_ids_stored(self, signal_ids: set[int]) -> None:
        """Check that span and step Signal IDs name stored Signals, without touching the hierarchy.

        Raises:
            TimeFFormatError: If an ID names no stored Signal.
        """
        if not signal_ids:
            return
        stored = {
            signal_id
            for (signal_id,) in self.connection.execute(
                "SELECT signal_id FROM signals WHERE signal_id IN (SELECT unnest(?))", [sorted(signal_ids)]
            ).fetchall()
        }
        missing = sorted(signal_ids - stored)
        if missing:
            raise TimeFFormatError(f"task refers to missing signal {missing[0]!r}")

    def _task_relationships(
        self, table: str, column: str, task_ids: list[int] | None
    ) -> dict[int, dict[str, tuple[int, ...]]]:
        """Read one ordered task relationship table for a batch, or for every task.

        Args:
            table: ``task_record_refs`` or ``task_annotation_refs``.
            column: The referenced ID column of that table.
            task_ids: The batch, or ``None`` for the whole table.

        Returns:
            ``task_id -> field -> ordered referenced IDs``.
        """
        where, parameters = self._task_filter(task_ids)
        rows = self.connection.execute(
            f"SELECT task_id, field, {column} FROM {table} {where} ORDER BY task_id, field, position",  # noqa: S608
            parameters,
        ).fetchall()
        return self._group_relationships(rows)

    def _task_dependencies(self, task_ids: list[int] | None) -> dict[int, tuple[int, ...]]:
        """Read the ordered parent IDs of each task in one batch, or of every task.

        Returns:
            ``task_id -> ordered parent task IDs`` for tasks that have parents.
        """
        where, parameters = self._task_filter(task_ids)
        rows = self.connection.execute(
            f"SELECT task_id, parent_task_id FROM task_dependencies {where} ORDER BY task_id, position",  # noqa: S608
            parameters,
        ).fetchall()
        parents: dict[int, list[int]] = defaultdict(list)
        for task_id, parent_id in rows:
            parents[task_id].append(parent_id)
        return {task_id: tuple(ids) for task_id, ids in parents.items()}

    @staticmethod
    def _group_relationships(rows: list[tuple[int, str, int]]) -> dict[int, dict[str, tuple[int, ...]]]:
        """Group ordered relationship query rows.

        Returns:
            Relationship values grouped by task ID and field.
        """
        grouped: dict[int, dict[str, tuple[int, ...]]] = {}
        current_task: int | None = None
        current_field: str | None = None
        current_values: list[int] = []
        for task_id, field, value in rows:
            if task_id != current_task or field != current_field:
                if current_task is not None and current_field is not None:
                    grouped.setdefault(current_task, {})[current_field] = tuple(current_values)
                current_task = task_id
                current_field = field
                current_values = [value]
            else:
                current_values.append(value)
        if current_task is not None and current_field is not None:
            grouped.setdefault(current_task, {})[current_field] = tuple(current_values)
        return grouped

    @staticmethod
    def _resolve(
        task_id: int,
        field: str,
        refs: dict[str, tuple[int, ...]],
        objects: dict[int, _Ref],
        label: str,
    ) -> tuple[_Ref, ...]:
        """Resolve one ordered reference field to hydrated objects.

        Args:
            task_id: The task the field belongs to, for the error message.
            field: The relationship field name.
            refs: The task's ordered IDs by field.
            objects: Hydrated objects by public ID.
            label: What the IDs name in the error message, such as ``"record"``.

        Returns:
            The referenced objects in stored order.

        Raises:
            TimeFFormatError: If an ID names no hydrated object.
        """
        ids = refs.get(field)
        if not ids:
            return ()
        try:
            return tuple(map(objects.__getitem__, ids))
        except KeyError as exc:
            raise TimeFFormatError(
                f"task {task_id!r} field {field!r} refers to missing {label} {exc.args[0]!r}"
            ) from exc

    def _read_axes(
        self,
        axis_ids: Iterable[int] | None = None,
    ) -> dict[int, _LoadedAxis]:
        """Hydrate each shared axis exactly once.

        Returns:
            Axes and irregular offset lengths keyed by axis ID.

        Raises:
            TimeFFormatError: If an axis has an unknown type.
        """
        requested = None if axis_ids is None else tuple(axis_ids)
        if requested == ():
            return {}
        axes: dict[int, _LoadedAxis] = {}
        query = """SELECT axis_id, axis_type, period_numerator_us, period_denominator,
                          start_index, offset_us, first_us, last_us,
                          lengths.min_n_values, lengths.max_n_values
                   FROM axes
                   LEFT JOIN (
                       SELECT axis_id, min(n_values) AS min_n_values, max(n_values) AS max_n_values
                       FROM signals GROUP BY axis_id
                   ) lengths USING (axis_id)"""
        rows = (
            self.connection.execute(query).fetchall()
            if requested is None
            else self.connection.execute(
                f"{query} WHERE axis_id IN (SELECT unnest(?))",
                [list(requested)],
            ).fetchall()
        )
        for (
            axis_id,
            axis_type,
            numerator,
            denominator,
            start_index,
            offset,
            first,
            last,
            min_n_values,
            max_n_values,
        ) in rows:
            if axis_type == "regular":
                axes[axis_id] = _LoadedAxis(
                    RegularAxis(
                        id=axis_id,
                        period_us=Fraction(numerator, denominator),
                        start_index=start_index,
                        offset_us=offset,
                    ),
                    None,
                )
            elif axis_type == "irregular":
                if min_n_values is None or max_n_values is None:
                    raise TimeFFormatError(f"irregular axis {axis_id!r} has no Signals")
                if min_n_values != max_n_values:
                    raise TimeFFormatError(
                        f"axis {axis_id!r} is shared by Signals with different lengths "
                        f"{sorted((min_n_values, max_n_values))}"
                    )
                axes[axis_id] = _LoadedAxis(
                    IrregularAxis(id=axis_id, first_us=first, last_us=last),
                    int(min_n_values),
                )
            elif axis_type == "ordinal":
                axes[axis_id] = _LoadedAxis(OrdinalAxis(id=axis_id), None)
            else:
                raise TimeFFormatError(f"axis {axis_id!r} has unknown type {axis_type!r}")
        return axes

    def load_axis_offsets(self, axis: IrregularAxis, n_values: int) -> pa.Array:
        """Load the offsets of one shared irregular axis.

        Returns:
            The axis offsets as Arrow int64 values.

        Raises:
            TimeFFormatError: If the offsets disagree with the axis or its Signals.
        """
        column = (
            self.connection.execute(
                """SELECT offsets.offset_us
               FROM axis_offsets offsets
               WHERE axis_id = ? ORDER BY offsets.position""",
                [axis.id],
            )
            .to_arrow_table()
            .column(0)
        )
        offsets = column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()
        if len(offsets) != n_values:
            raise TimeFFormatError(
                f"axis {axis.id!r} stores {len(offsets)} offsets but its Signals declare length {n_values}"
            )
        if offsets.null_count:
            raise TimeFFormatError(f"axis {axis.id!r} stores null offsets")
        endpoints = (axis.first_us, axis.last_us)
        if len(offsets) == 0 or (offsets[0].as_py(), offsets[-1].as_py()) != endpoints:
            raise TimeFFormatError(f"axis {axis.id!r} has offsets disagreeing with its axis endpoints {endpoints}")
        if len(offsets) > 1:
            decreasing = pc.less(  # ty: ignore[unresolved-attribute]
                offsets.slice(1), offsets.slice(0, len(offsets) - 1)
            )
            if pc.any(decreasing).as_py():  # ty: ignore[unresolved-attribute]
                raise TimeFFormatError(f"axis {axis.id!r} has decreasing offsets")
        return offsets

    @staticmethod
    def _spec_from_row(row: _SignalRow) -> TimeSeriesSpec:
        """Reconstruct an inline signal specification.

        Returns:
            The typed specification.

        Raises:
            TimeFFormatError: If the stored Signal modality is invalid.
        """
        try:
            modality = InputModality(row.modality)
        except ValueError as exc:
            raise TimeFFormatError(f"invalid modality {row.modality!r} for signal {row.signal_id!r}") from exc
        if modality is InputModality.NO_INPUT:
            raise TimeFFormatError(f"signal {row.signal_id!r} cannot have the no_input modality")
        return TimeSeriesSpec(
            spec_type=row.spec_type,
            name=row.spec_name,
            unit_value=None if row.unit is None else ureg.Unit(row.unit),
            dtype=cast("ValueDtype", row.dtype),
            categories=tuple(row.categories),
            value_shape=tuple(row.value_shape),
            dimension_names=tuple(row.dimension_names),
            nullable=row.nullable,
            modality=modality,
        )

    def _read_annotations(
        self,
        object_ids: Iterable[int] | None = None,
        *,
        object_type: str | None = None,
    ) -> dict[tuple[str, int | None], tuple[Annotation, ...]]:
        """Hydrate annotation content and occurrences.

        Args:
            object_ids: Public IDs of the annotated objects, of any type, or ``None`` for every
                occurrence. An ID that two kinds share returns the occurrences of both.
            object_type: Restrict the read to occurrences on one object type.

        Returns:
            Attached occurrences keyed by object type and public object ID, which is ``None`` for
            the Dataset.

        Raises:
            TimeFFormatError: If an occurrence has unknown content, an unknown object type, an owner
                ID that does not fit its type, or an unknown span type.
        """
        requested = None if object_ids is None else list(object_ids)
        if requested == []:
            return {}
        parameters: list[object] = [] if requested is None else [requested]
        if object_type is not None:
            parameters.append(object_type)
        rows = _rows(
            self.connection.execute(_annotation_query(object_type, keyed=requested is not None), parameters),
            _OccurrenceRow._make,
        )
        for row in rows:
            if row.name is None:
                raise TimeFFormatError(
                    f"annotation occurrence {row.occurrence_id!r} refers to missing content {row.content_id!r}"
                )
            if row.object_type not in ANNOTATED_OBJECT_TYPES:
                raise TimeFFormatError(
                    f"annotation occurrence {row.occurrence_id!r} has unknown object type {row.object_type!r}"
                )
            if (row.object_type == "Dataset") != (row.object_id is None):
                raise TimeFFormatError(
                    f"annotation occurrence {row.occurrence_id!r} on {row.object_type} has object id {row.object_id!r}"
                )
        grouped: dict[tuple[str, int | None], list[Annotation]] = defaultdict(list)
        for row in rows:
            annotation = Annotation(
                occurrence_id=row.occurrence_id,
                id=_required(row.content_id, "annotation content"),
                key=_required(row.name, "annotation name"),
                value=decode_annotation_value(
                    row.value_kind,
                    row.text_value,
                    row.integer_value,
                    row.float_value,
                    row.boolean_value,
                    row.text_list_value,
                ),
                unit=row.unit,
                description=row.description,
                metadata=_decode_json(row.content_metadata, default={}),
                span=_occurrence_span(row),
                source=_decode_json(row.provenance),
                confidence=row.confidence,
                occurrence_metadata=_decode_json(row.metadata, default={}),
            )
            grouped[row.object_type, row.object_id].append(annotation)
        return {key: tuple(value) for key, value in grouped.items()}


class _TaskHydration:
    """State for one :meth:`DuckDBControlReader.iter_tasks` call.

    It holds the caller's Records and the lookups built from them, loads Records and Signals a
    task refers to outside that set on demand, and keeps every task that another task derives
    from until its dependants are built.
    """

    def __init__(  # noqa: PLR0913 - one argument per task selection axis
        self,
        reader: DuckDBControlReader,
        records: tuple[Record, ...] | None,
        *,
        cached_records: Iterable[Record] = (),
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
    ) -> None:
        """Index the caller's Records and decide whether the task table needs filtering.

        Raises:
            TimeFValidationError: If one of ``records`` is not stored in the control database.
        """
        self.reader = reader
        self.required_modalities = required_modalities
        self.supported_modalities = supported_modalities
        self.split = split
        self.connection = reader.connection
        self.records_by_id: dict[int, Record] = {}
        self.signals_by_id: dict[int, Signal] = {}
        self._records_without_signal_index: list[Record] = []
        self.annotations_by_id: dict[int, Annotation] = {}
        self.retained: dict[int, Task] = {}
        self._hydrating: set[int] = set()
        self._dataset_annotations_indexed = False
        facts = reader.task_facts()
        self.parent_ids = facts.parent_task_ids
        self.tasks_annotated = facts.tasks_annotated
        given = {record.id: record for record in cached_records}
        given.update((record.id, record) for record in records or ())
        if given:
            stored = {
                row[0]
                for row in self.connection.execute(
                    "SELECT record_id FROM records WHERE record_id IN (SELECT unnest(?))", [list(given)]
                ).fetchall()
            }
            missing = sorted(given.keys() - stored)
            if missing:
                raise TimeFValidationError(f"no such record(s) in control.duckdb: {missing}")
        for record in given.values():
            self._index_record(record)
        self.record_ids: list[int] | None = None if records is None else [record.id for record in records]

    def iter_tasks(self) -> Iterator[Task]:
        """Yield the selected tasks in task ID order, one bounded batch at a time.

        Yields:
            Each hydrated task.
        """
        with self.reader._task_row_cursor(
            self.record_ids,
            self.required_modalities,
            self.supported_modalities,
            self.split,
        ) as selection:
            while rows := selection.fetchmany(_TASK_BATCH_SIZE):
                task_rows = [_TaskRow._make(row) for row in rows]
                yield from self._hydrate([row.task_id for row in task_rows], task_rows=task_rows)

    def read_all(self) -> list[Task]:
        """Build all selected tasks, using batches for filtered reads.

        Returns:
            The tasks in task ID order.
        """
        unfiltered = self.required_modalities is None and self.supported_modalities is None and self.split is None
        if self.record_ids is None and unfiltered:
            return self._hydrate(None)
        return list(self.iter_tasks())

    def _hydrate(self, task_ids: list[int] | None, *, task_rows: list[_TaskRow] | None = None) -> list[Task]:
        """Build one batch, or the whole table for ``None``, hydrating unbuilt parents first.

        Rows from the selection cursor avoid a second task-table scan.

        Returns:
            The tasks in the order of ``task_ids``, or in task ID order for the whole table.

        Raises:
            TimeFFormatError: If the stored derivations contain a cycle.
        """
        pending = None if task_ids is None else [task_id for task_id in task_ids if task_id not in self.retained]
        if task_rows is not None:
            task_rows = [row for row in task_rows if row.task_id not in self.retained]
        dependencies = self.reader._task_dependencies(pending) if self.parent_ids else {}
        in_batch = None if pending is None else set(pending)
        missing_parents = (
            []
            if in_batch is None
            else sorted(
                {
                    parent_id
                    for parent_ids in dependencies.values()
                    for parent_id in parent_ids
                    if parent_id not in self.retained and parent_id not in in_batch
                }
            )
        )
        if missing_parents:
            cyclic = self._hydrating.intersection(missing_parents)
            if cyclic:
                raise TimeFFormatError(f"task dependencies contain a cycle through tasks {sorted(cyclic)}")
            self._hydrating.update(pending or ())
            try:
                self._hydrate(missing_parents)
            finally:
                self._hydrating.difference_update(pending or ())
        built = self._build(pending, task_rows=task_rows)
        for task_id, parent_ids in dependencies.items():
            built[task_id].from_tasks = tuple(self._task(parent_id, built) for parent_id in parent_ids)
        if task_ids is None:
            return list(built.values())
        for task_id, task in built.items():
            if task_id in self.parent_ids:
                self.retained[task_id] = task
        return [self._task(task_id, built) for task_id in task_ids]

    def _task(self, task_id: int, built: dict[int, Task]) -> Task:
        """Return a task from this batch or the retained parents.

        Raises:
            TimeFFormatError: If the task was never hydrated.
        """
        task = built.get(task_id) or self.retained.get(task_id)
        if task is None:
            raise TimeFFormatError(f"task dependency refers to missing task {task_id!r}")
        return task

    def _build(self, task_ids: list[int] | None, *, task_rows: list[_TaskRow] | None = None) -> dict[int, Task]:
        """Read every table for one batch of task IDs and construct the tasks.

        Returns:
            The constructed tasks by ID, without ``from_tasks`` wired.

        Raises:
            TimeFFormatError: If a task has an unknown type.
        """
        if task_ids is not None and not task_ids:
            return {}
        reader = self.reader
        if task_rows is None:
            task_rows = reader._task_rows(task_ids)
        record_refs = reader._task_relationships("task_record_refs", "record_id", task_ids)
        annotation_refs = reader._task_relationships("task_annotation_refs", "occurrence_id", task_ids)
        target_rows = reader._target_rows(task_ids)
        task_annotations: dict[tuple[str, int | None], tuple[Annotation, ...]] = {}
        if self.tasks_annotated:
            task_annotations = reader._read_annotations(task_ids, object_type="Task")
            if task_ids is None:
                _check_owners(task_annotations.keys(), {"Task": {row.task_id for row in task_rows}})

        self._ensure_records(
            {record_id for refs in record_refs.values() for ids in refs.values() for record_id in ids}
            | {row.record_id for row in target_rows if row.record_id is not None}
        )
        self._ensure_signals(
            {row.signal_id for row in target_rows if row.target_kind == "signal" and row.signal_id is not None}
        )
        reader._check_signal_ids_stored(
            {
                row.signal_id
                for row in target_rows
                if row.target_kind in {"step_point", "step_interval"} and row.signal_id is not None
            }
            | {signal_id for row in target_rows for signal_id in row.span_signal_ids or ()}
            | {signal_id for row in task_rows for signal_id in row.scope_signal_ids or ()}
        )
        annotations_by_occurrence = self._annotations_by_occurrence(
            {occurrence_id for refs in annotation_refs.values() for ids in refs.values() for occurrence_id in ids}
        )

        targets_by_task: dict[int, list[object]] = defaultdict(list)
        for target in target_rows:
            targets_by_task[target.task_id].append(
                decode_target(target._asdict(), records=self.records_by_id, signals=self.signals_by_id)
            )

        tasks: dict[int, Task] = {}
        for row in task_rows:
            try:
                task_type = TaskType(row.task_type)
                cls = TASKS[task_type]
            except (ValueError, KeyError) as exc:
                raise TimeFFormatError(f"task {row.task_id!r} has unknown type {row.task_type!r}") from exc
            refs = record_refs.get(row.task_id, {})
            kwargs: dict[str, Any] = {
                "id": row.task_id,
                "inputs": reader._resolve(row.task_id, "inputs", refs, self.records_by_id, "record"),
                "targets": tuple(targets_by_task.get(row.task_id, ())) if row.has_inline_targets else None,
                "prompt": row.prompt,
                "input_modalities": decode_input_modalities(row.input_modalities),
                "split": decode_split(row.split),
                "scope": decode_span(row.scope_type, row.scope_start, row.scope_end, row.scope_signal_ids),
                "rationale": row.rationale,
                "annotations": task_annotations.get(("Task", row.task_id), ()),
                "metadata": _decode_json(row.metadata, default={}),
                **decode_task_payload(task_type, row.payload_columns),
            }
            if "candidate_records" in cls.__dataclass_fields__:
                kwargs["candidate_records"] = reader._resolve(
                    row.task_id, "candidate_records", refs, self.records_by_id, "record"
                )
            task_annotation_refs = annotation_refs.get(row.task_id, {})
            kwargs["input_annotations"] = reader._resolve(
                row.task_id,
                "input_annotations",
                task_annotation_refs,
                annotations_by_occurrence,
                "annotation occurrence",
            )
            kwargs["target_annotations"] = reader._resolve(
                row.task_id,
                "target_annotations",
                task_annotation_refs,
                annotations_by_occurrence,
                "annotation occurrence",
            )
            task = cls(**kwargs)
            try:
                TimeFDataset._check_task_record_scope(task)
            except TimeFValidationError as exc:
                raise TimeFFormatError(f"stored task {task.id!r} has an invalid Record scope: {exc}") from exc
            tasks[row.task_id] = task
        return tasks

    def _index_record(self, record: Record) -> None:
        """Add one Record and everything hanging off it to the lookups."""
        self.records_by_id[record.id] = record
        # Text and scalar targets never need Signals, so the Signal walk waits for a target that does.
        self._records_without_signal_index.append(record)
        annotations = [*record.annotations]
        for source in record.walk_sources():
            annotations.extend(source.annotations)
            for signal in source.signals:
                annotations.extend(signal.annotations)
        for annotation in annotations:
            if annotation.occurrence_id is not None:
                self.annotations_by_id[annotation.occurrence_id] = annotation

    def _ensure_records(self, record_ids: set[int]) -> None:
        """Hydrate Records a task refers to that the caller did not pass.

        Raises:
            TimeFFormatError: If an ID names no stored Record.
        """
        missing = sorted(record_ids - self.records_by_id.keys())
        if not missing:
            return
        try:
            hydrated = self.reader.read_records(missing)
        except TimeFValidationError as exc:
            raise TimeFFormatError(f"task refers to a missing record: {exc}") from exc
        for record in hydrated:
            self._index_record(record)

    def _ensure_signals(self, signal_ids: set[int]) -> None:
        """Index the Signals of the hydrated Records when one batch's targets refer to Signals.

        Raises:
            TimeFFormatError: If an ID names no Signal on the hydrated Records.
        """
        if not signal_ids:
            return
        for record in self._records_without_signal_index:
            for signal in record.signals:
                self.signals_by_id[signal.id] = signal
        self._records_without_signal_index.clear()
        outside = sorted(signal_ids - self.signals_by_id.keys())
        if outside:
            raise TimeFFormatError(f"task target refers to Signal {outside[0]!r} on a Record outside the hydrated set")

    def _annotations_by_occurrence(self, occurrence_ids: set[int]) -> dict[int, Annotation]:
        """Resolve referenced annotation occurrences to the objects on the hydrated Records.

        Returns:
            Annotations by occurrence ID.

        Raises:
            TimeFFormatError: If an ID names no occurrence on the hydrated Records or the Dataset.
        """
        resolved: dict[int, Annotation] = {}
        for occurrence_id in occurrence_ids:
            annotation = self.annotations_by_id.get(occurrence_id)
            if annotation is None and not self._dataset_annotations_indexed:
                self._index_dataset_annotations()
                annotation = self.annotations_by_id.get(occurrence_id)
            if annotation is None:
                raise TimeFFormatError(
                    f"task refers to annotation occurrence {occurrence_id!r} that no hydrated object carries"
                )
            resolved[occurrence_id] = annotation
        return resolved

    def _index_dataset_annotations(self) -> None:
        """Add the Dataset-level annotation occurrences to the lookups, once."""
        self._dataset_annotations_indexed = True
        for annotation in self.reader._read_annotations(object_type="Dataset").get(("Dataset", None), ()):
            if annotation.occurrence_id is not None:
                self.annotations_by_id[annotation.occurrence_id] = annotation
