"""Hydrate the TimeF object hierarchy from ``control.duckdb``."""

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from fractions import Fraction
from functools import partial
import json
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import duckdb
import numpy as np
import pyarrow as pa

from timenet.dataset import IrregularAxis, OrdinalAxis, Record, RegularAxis, Signal, Source
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.annotation_codec import decode_annotation_value
from timenet.format.duckdb import check_control_schema, connect_control
from timenet.format.task_codec import decode_input_modalities, decode_span, decode_target, decode_task_payload
from timenet.types import (
    TASKS,
    Annotation,
    InputModality,
    Task,
    TaskType,
    TimeInterval,
    TimePoint,
    TimeSeriesSpec,
    ureg,
)


ValueLoader = Callable[[str, TimeSeriesSpec], pa.Array]
ValueLoaderFactory = Callable[[int, str, TimeSeriesSpec], Callable[[], pa.Array]]
OffsetsLoader = Callable[[int, str], pa.Array]


class _RecordRow(NamedTuple):
    """One ``records`` row in schema column order."""

    record_key: int
    record_id: str
    start_time_us: int | None
    time_span_start_us: int | None
    time_span_end_us: int | None
    metadata: str


class _SourceRow(NamedTuple):
    """One ``sources`` row in schema column order."""

    source_key: int
    source_id: str
    record_key: int
    parent_source_key: int | None
    name: str
    metadata: str


class _SignalRow(NamedTuple):
    """One ``signals`` row in schema column order."""

    signal_key: int
    signal_id: str
    source_key: int
    name: str
    axis_key: int
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
        )


class _OccurrenceRow(NamedTuple):
    """One annotation occurrence joined with its content and the object it annotates."""

    occurrence_key: int
    occurrence_id: str
    object_type: str
    object_id: str | None
    span_type: str
    start_us: int | None
    end_us: int | None
    signal_keys: list[int] | None
    provenance: str | None
    confidence: float | None
    metadata: str
    content_id: str | None
    name: str | None
    value_kind: str | None
    text_value: str | None
    integer_value: int | None
    float_value: float | None
    boolean_value: bool | None
    text_list_value: list[str] | None
    unit: str | None
    content_metadata: str | None
    content_key: int
    object_key: int | None


class _TaskRow(NamedTuple):
    """One ``tasks`` row in schema column order."""

    task_key: int
    task_id: str
    task_type: str
    prompt: str | None
    input_modalities: list[str] | None
    scope_type: str | None
    scope_start: int | None
    scope_end: int | None
    scope_signal_keys: list[int] | None
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

    task_key: int
    target_kind: str
    text_value: str | None
    integer_value: int | None
    float_value: float | None
    boolean_value: bool | None
    record_key: int | None
    signal_key: int | None
    span_start: int | None
    span_end: int | None
    signal_keys: list[int] | None


_TASK_BATCH_SIZE = 5_000
"""Tasks hydrated per round of keyed queries."""

_OBJECT_ID_SELECTS = {
    "Dataset": "SELECT dataset_key AS object_key, 'Dataset' AS object_type, dataset_id AS object_id FROM datasets",
    "Record": "SELECT record_key AS object_key, 'Record' AS object_type, record_id AS object_id FROM records",
    "Source": "SELECT source_key AS object_key, 'Source' AS object_type, source_id AS object_id FROM sources",
    "Signal": "SELECT signal_key AS object_key, 'Signal' AS object_type, signal_id AS object_id FROM signals",
    "Task": "SELECT task_key AS object_key, 'Task' AS object_type, task_id AS object_id FROM tasks",
}
"""Per object type, the query that maps its internal keys to public IDs for annotation lookups."""


def _signal_ids_cte(name: str, table: str, keys_column: str, group_by: tuple[str, ...], ids_column: str) -> str:
    """Build a CTE that turns a ``BIGINT[]`` column of Signal keys into a list of Signal IDs.

    Args:
        name: The CTE's name.
        table: The table holding the key list.
        keys_column: The ``BIGINT[]`` column of Signal keys.
        group_by: The columns that identify one row of ``table``.
        ids_column: The name of the produced ``VARCHAR[]`` column.

    Returns:
        ``name AS (...)``, preserving each list's order.
    """
    grouped = ", ".join(group_by)
    return f"""{name} AS (
        SELECT {grouped}, list(signal_id ORDER BY ordinal) AS {ids_column}
        FROM (
            SELECT {grouped}, unnest({keys_column}) AS signal_key, generate_subscripts({keys_column}, 1) AS ordinal
            FROM {table}
        ) JOIN signals USING (signal_key)
        GROUP BY {grouped}
    )"""  # noqa: S608 - fixed identifiers


def _annotation_query(object_type: str | None, *, keyed: bool) -> str:
    """Build the occurrence query, restricted to one object type and to requested keys when asked.

    Args:
        object_type: Keep one object type, or ``None`` for all.
        keyed: Whether the caller passes object keys. The object lookup then scans only the rows
            with those keys instead of unioning every object in the database.

    Returns:
        SQL with one positional parameter for the requested keys (an empty list when not keyed),
        plus a second for the object type when it is given.

    Raises:
        TimeFFormatError: If ``object_type`` is not an annotated object type.
    """
    if object_type is not None and object_type not in _OBJECT_ID_SELECTS:
        raise TimeFFormatError(f"unknown annotated object type {object_type!r}")
    key_filter = " WHERE object_key IN (SELECT object_key FROM requested)" if keyed else ""
    object_ids = " UNION ALL ".join(
        f"SELECT * FROM ({select}){key_filter}"  # noqa: S608 - fixed fragments
        for name, select in _OBJECT_ID_SELECTS.items()
        if object_type is None or name == object_type
    )
    # Two fixed shapes rather than one ``(? OR key IN ...)`` predicate: a parameter inside an OR keeps
    # DuckDB from pushing the key filter below the joins, which made a one-Record read scan every
    # occurrence in the database.
    # The key filter sits inside the occurrences subquery so it applies before the joins.
    occurrences = (
        "(SELECT * FROM annotation_occurrences WHERE object_key IN (SELECT object_key FROM requested))"
        if keyed
        else "annotation_occurrences"
    )
    where = "" if object_type is None else "WHERE o.object_type = ?"
    return f"""WITH requested AS (SELECT unnest(?) AS object_key),
        object_ids AS ({object_ids})
        SELECT o.occurrence_key, o.occurrence_id, o.object_type, objects.object_id, o.span_type,
               o.start_us, o.end_us, o.signal_keys, o.provenance, o.confidence,
               o.metadata, c.content_id, c.name, c.value_kind, c.text_value, c.integer_value,
               c.float_value, c.boolean_value, c.text_list_value, c.unit, c.metadata,
               o.content_key, objects.object_key
        FROM {occurrences} o
        LEFT JOIN annotation_contents c USING (content_key)
        LEFT JOIN object_ids objects
          ON objects.object_key = o.object_key AND objects.object_type = o.object_type
        {where}
        ORDER BY o.object_type, objects.object_id, c.content_id, o.span_type,
                 o.start_us NULLS FIRST, o.end_us NULLS FIRST, o.occurrence_id"""  # noqa: S608 - fixed fragments


class _TaskFacts(NamedTuple):
    """Facts about one control database that every task hydration reuses."""

    record_count: int
    parent_task_keys: frozenset[int]
    tasks_annotated: bool


_HierarchyRow = TypeVar("_HierarchyRow", _RecordRow, _SourceRow, _SignalRow)
_Row = TypeVar("_Row", bound=tuple[Any, ...])
_Ref = TypeVar("_Ref", Record, Annotation)
_T = TypeVar("_T")

_HIERARCHY_TABLES: dict[type[tuple[Any, ...]], tuple[str, str, str]] = {
    _RecordRow: ("records", "record_id", "record_id"),
    _SourceRow: ("sources", "record_key", "source_id"),
    _SignalRow: ("signals", "source_key", "signal_id"),
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


def _occurrence_span(row: _OccurrenceRow, scope: tuple[str, ...] | None) -> TimePoint | TimeInterval | None:
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
        return TimePoint(start_us=start_us, time_series_ids=scope)
    if row.span_type == "interval":
        end_us = _required(row.end_us, f"the end of annotation occurrence {row.occurrence_id!r}")
        return TimeInterval(start_us=start_us, end_us=end_us, time_series_ids=scope)
    raise TimeFFormatError(f"annotation occurrence {row.occurrence_id!r} has unknown span type {row.span_type!r}")


def _missing_values(signal_id: str, _spec: TimeSeriesSpec) -> pa.Array:
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
        self._record_keys: dict[str, int] = {}
        self._task_facts: _TaskFacts | None = None

    def close(self) -> None:
        """Close the reader's DuckDB connection."""
        self.connection.close()

    def __enter__(self) -> "DuckDBControlReader":
        """Return this open reader.

        Returns:
            This reader.
        """
        return self

    def __exit__(self, *_: object) -> None:
        """Close the connection when leaving a context manager."""
        self.close()

    def record_ids(self) -> tuple[str, ...]:
        """Return every stored Record ID in the order :meth:`read_records` yields them, without hydrating.

        Returns:
            The Record IDs sorted by ID.
        """
        return tuple(
            row[0] for row in self.connection.execute("SELECT record_id FROM records ORDER BY record_id").fetchall()
        )

    def read_records(  # noqa: PLR0914, PLR0915 - related row sets stay together
        self,
        record_ids: Iterable[str] | None = None,
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
        requested = None if record_ids is None else tuple(record_ids)
        read_all = requested is None
        rows = self._read_rows(_RecordRow, requested)
        by_id = self._index_rows(rows, "record", key=lambda row: row.record_id, id_of=lambda row: row.record_id)
        self._record_keys.update((row.record_id, row.record_key) for row in rows)
        order = tuple(by_id) if requested is None else requested
        missing = [record_id for record_id in order if record_id not in by_id]
        if missing:
            raise TimeFValidationError(f"no such record(s) in control.duckdb: {missing}")

        record_keys = [row.record_key for row in rows]
        source_rows = self._read_rows(_SourceRow, None if read_all else record_keys)
        source_data = self._index_rows(
            source_rows, "source", key=lambda row: row.source_key, id_of=lambda row: row.source_id
        )
        source_keys = [row.source_key for row in source_rows]
        signal_rows = self._read_rows(_SignalRow, None if read_all else source_keys)
        signal_data = self._index_rows(
            signal_rows, "signal", key=lambda row: row.signal_key, id_of=lambda row: row.signal_id
        )
        signal_keys = [row.signal_key for row in signal_rows]
        for signal_row in signal_data.values():
            if signal_row.source_key not in source_data:
                raise TimeFFormatError(
                    f"signal {signal_row.signal_id!r} refers to missing source key {signal_row.source_key!r}"
                )
        object_keys = [*record_keys, *source_keys, *signal_keys]
        annotation_keys = None if read_all else object_keys
        annotations = self._read_annotations(annotation_keys) if with_annotations else {}
        axes = self._read_axes({row.axis_key for row in signal_rows})

        signals_by_source: dict[int, list[Signal]] = defaultdict(list)
        specs: dict[tuple[Any, ...], TimeSeriesSpec] = {}
        for signal_row in signal_rows:
            axis = axes.get(signal_row.axis_key)
            if axis is None:
                raise TimeFFormatError(
                    f"signal {signal_row.signal_id!r} refers to missing axis key {signal_row.axis_key!r}"
                )
            spec = specs.get(signal_row.spec_fields)
            if spec is None:
                spec = self._spec_from_row(signal_row)
                specs[signal_row.spec_fields] = spec
            offsets_loader = None
            if isinstance(axis, IrregularAxis):
                offsets_loader = (
                    partial(self._offsets_loader, signal_row.axis_key, axis.axis_id)
                    if self._offsets_loader is not None
                    else partial(self.load_axis_offsets_by_key, signal_row.axis_key, axis.axis_id)
                )
            signals_by_source[signal_row.source_key].append(
                Signal.from_loader(
                    id=signal_row.signal_id,
                    name=signal_row.name,
                    spec=spec,
                    time_axis=axis,
                    n_values=signal_row.n_values,
                    loader=(
                        self._value_loader_factory(signal_row.signal_key, signal_row.signal_id, spec)
                        if self._value_loader_factory is not None
                        else partial(self._value_loader, signal_row.signal_id, spec)
                    ),
                    time_offsets_loader=offsets_loader,
                    annotations=annotations.get(("Signal", signal_row.signal_id), ()),
                    metadata=_decode_json(signal_row.metadata, default={}),
                )
            )

        children, roots_by_record = self._group_sources(source_rows)
        hydrated_sources: set[int] = set()

        def hydrate_source(source_key: int, record_key: int) -> Source:
            source_row = source_data[source_key]
            if source_row.record_key != record_key:
                raise TimeFFormatError(f"source {source_row.source_id!r} crosses record boundaries")
            hydrated_sources.add(source_key)
            return Source(
                id=source_row.source_id,
                name=source_row.name,
                sources=tuple(hydrate_source(child, record_key) for child in children[source_key]),
                signals=tuple(signals_by_source[source_key]),
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
            root_sources = tuple(
                hydrate_source(source_key, record_row.record_key)
                for source_key in roots_by_record[record_row.record_key]
            )
            records.append(
                Record(
                    record_id=record_id,
                    sources=root_sources,
                    annotations=annotations.get(("Record", record_id), ()),
                    start_time=record_row.start_time_us,
                    time_span=span,
                    metadata=_decode_json(record_row.metadata, default={}),
                )
            )
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
        *,
        key: Callable[[_HierarchyRow], Any],
        id_of: Callable[[_HierarchyRow], str],
    ) -> dict[Any, _HierarchyRow]:
        """Return rows by key and reject duplicate keys or public IDs.

        Args:
            rows: The rows to index.
            kind: The object kind named in the error message.
            key: The value each row is indexed by.
            id_of: The public ID of each row.

        Raises:
            TimeFFormatError: If a key or public ID occurs more than once.
        """
        indexed = {key(row): row for row in rows}
        if len(indexed) != len(rows) or len({id_of(row) for row in rows}) != len(rows):
            raise TimeFFormatError(f"control.duckdb contains duplicate {kind} IDs or keys")
        return indexed

    @staticmethod
    def _group_sources(
        rows: list[_SourceRow],
    ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        """Group child and root Source keys.

        Returns:
            Child keys by parent key and root keys by Record key.
        """
        children: dict[int, list[int]] = defaultdict(list)
        roots: dict[int, list[int]] = defaultdict(list)
        for row in rows:
            if row.parent_source_key is None:
                roots[row.record_key].append(row.source_key)
            else:
                children[row.parent_source_key].append(row.source_key)
        return children, roots

    def read_dataset_annotations(self, dataset_id: str) -> tuple[Annotation, ...]:
        """Return annotations attached directly to a dataset.

        Returns:
            Dataset-level annotation occurrences in stable occurrence-ID order.
        """
        row = self.connection.execute(
            "SELECT dataset_key FROM datasets WHERE dataset_id = ?",
            [dataset_id],
        ).fetchone()
        if row is None:
            return ()
        return self._read_annotations((row[0],)).get(("Dataset", dataset_id), ())

    def chunk_rows_by_key(self, signal_key: int, signal_id: str) -> list[dict[str, Any]]:
        """Return one Signal's chunk locations in chunk order.

        Raises:
            TimeFFormatError: If the Signal has no stored chunks.
        """
        rows = self.chunk_rows_by_keys((signal_key,))[signal_key]
        if not rows:
            raise TimeFFormatError(f"signal {signal_id!r} has no stored value chunks")
        return rows

    def chunk_rows_by_keys(self, signal_keys: Iterable[int]) -> dict[int, list[dict[str, Any]]]:
        """Return chunk locators for a batch of internal Signal keys.

        Args:
            signal_keys: Internal integer Signal keys.

        Returns:
            Chunk rows keyed by internal Signal key. Each Signal's rows are in chunk-index order.
        """
        keys = tuple(signal_keys)
        if not keys:
            return {}
        rows = self.connection.execute(
            """SELECT signal_key, chunk_index, value_path, chunk_major_index, chunk_minor_index, n_values
               FROM signal_chunks
               WHERE signal_key IN (SELECT unnest(?))
               ORDER BY signal_key, chunk_index""",
            [list(keys)],
        ).fetchall()
        grouped: dict[int, list[tuple[Any, ...]]] = defaultdict(list)
        for signal_key, *chunk_row in rows:
            grouped[signal_key].append(tuple(chunk_row))
        return {signal_key: self._chunk_dicts(grouped[signal_key]) for signal_key in keys}

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

    def read_tasks(self, records: Iterable[Record]) -> tuple[Task, ...]:
        """Hydrate every task attached to ``records`` and restore all in-memory object references.

        Args:
            records: Records hydrated by this reader. Tasks reference these objects, so the same
                ``Record`` instance a caller holds is the one its tasks point at.

        Returns:
            Concrete tasks in stable ID order.

        Raises:
            TimeFFormatError: If a relationship refers to a missing object or has an invalid field.
        """  # noqa: DOC502 - raised by _TaskHydration
        hydration = _TaskHydration(self, tuple(records))
        return tuple(hydration.read_all())

    def iter_tasks(
        self,
        records: Iterable[Record] | None = None,
        *,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
    ) -> Iterator[Task]:
        """Yield the tasks whose inputs include one of ``records``, a bounded batch at a time.

        Each batch runs a handful of keyed queries and builds only that batch's tasks, so a caller
        that walks one Record at a time never holds the whole task table in memory. Passing every
        Record of the dataset skips the input filter and streams the full table.

        Tasks a batch derives from through ``from_tasks`` are hydrated first when they have not
        been yielded yet, and kept until their dependants are built. Records a task refers to
        outside ``records`` are hydrated on demand.

        Args:
            records: Records hydrated by this reader.

        Yields:
            Concrete tasks in stable ID order.

        Raises:
            TimeFFormatError: If a relationship refers to a missing object or has an invalid field.
        """  # noqa: DOC502 - raised by _TaskHydration
        hydration = _TaskHydration(
            self,
            () if records is None else tuple(records),
            all_records=records is None,
            required_modalities=required_modalities,
            supported_modalities=supported_modalities,
        )
        yield from hydration.iter_tasks()

    def task_facts(self) -> _TaskFacts:
        """Return the per-database facts task hydration needs, read once per reader.

        Returns:
            The record count, the keys of tasks other tasks derive from, and whether any task
            carries annotations.
        """
        if self._task_facts is None:
            (record_count,) = self.connection.execute("SELECT count(*) FROM records").fetchone() or (0,)
            parent_keys = {
                row[0]
                for row in self.connection.execute("SELECT DISTINCT parent_task_key FROM task_dependencies").fetchall()
            }
            (annotated,) = self.connection.execute(
                "SELECT count(*) FROM annotation_occurrences WHERE object_type = 'Task'"
            ).fetchone() or (0,)
            self._task_facts = _TaskFacts(record_count, frozenset(parent_keys), annotated > 0)
        return self._task_facts

    def task_table(
        self,
        record_ids: Iterable[str] | None = None,
        *,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
    ) -> pa.Table:
        """Return the task table as Arrow, with object references as public IDs, without building Tasks.

        One row per task with its type, prompt, rationale, the four scope columns (Signal keys
        resolved to IDs), the configuration columns, ordered ``input_record_ids``, ordered
        ``from_task_ids``, and the metadata JSON string. Targets are in :meth:`target_table`.

        Args:
            record_ids: Keep only tasks whose inputs include one of these Records, or ``None`` for all.

        Returns:
            The tasks in task ID order.
        """
        where, parameters = self._selection_filter(record_ids, required_modalities, supported_modalities)
        return self.connection.execute(
            f"""WITH {_signal_ids_cte("scope_ids", "tasks", "scope_signal_keys", ("task_key",), "scope_signal_ids")},
                inputs AS (
                    SELECT task_key, list(record_id ORDER BY position) AS input_record_ids
                    FROM task_record_refs JOIN records USING (record_key)
                    WHERE field = 'inputs'
                    GROUP BY task_key
                ),
                parents AS (
                    SELECT d.task_key, list(parent.task_id ORDER BY d.position) AS from_task_ids
                    FROM task_dependencies d JOIN tasks parent ON parent.task_key = d.parent_task_key
                    GROUP BY d.task_key
                )
                SELECT t.task_id, t.task_type, t.prompt, t.rationale,
                       t.scope_type, t.scope_start, t.scope_end, scope_ids.scope_signal_ids,
                       t.target_schema, t.unit, t.target_name, t.mode, t.has_inline_targets,
                       t.input_modalities,
                       coalesce(inputs.input_record_ids, []) AS input_record_ids,
                       coalesce(parents.from_task_ids, []) AS from_task_ids,
                       t.metadata
                FROM tasks t
                LEFT JOIN scope_ids USING (task_key)
                LEFT JOIN inputs USING (task_key)
                LEFT JOIN parents USING (task_key)
                {where}
                ORDER BY t.task_id""",  # noqa: S608 - fixed fragments
            parameters,
        ).to_arrow_table()

    def target_table(
        self,
        record_ids: Iterable[str] | None = None,
        *,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
    ) -> pa.Table:
        """Return every inline target as Arrow, one typed row each, with references as public IDs.

        Args:
            record_ids: Keep only targets of tasks whose inputs include one of these Records.

        Returns:
            Targets in task ID and position order.
        """
        where, parameters = self._selection_filter(record_ids, required_modalities, supported_modalities)
        return self.connection.execute(
            f"""WITH {_signal_ids_cte("span_ids", "task_targets", "signal_keys", ("task_key", "position"), "span_signal_ids")}
                SELECT t.task_id, x.position, x.target_kind, x.text_value, x.integer_value, x.float_value,
                       x.boolean_value, records.record_id, signals.signal_id,
                       x.span_start, x.span_end, span_ids.span_signal_ids
                FROM task_targets x
                JOIN tasks t USING (task_key)
                LEFT JOIN records ON records.record_key = x.record_key
                LEFT JOIN signals ON signals.signal_key = x.signal_key
                LEFT JOIN span_ids ON span_ids.task_key = x.task_key AND span_ids.position = x.position
                {where}
                ORDER BY t.task_id, x.position""",  # noqa: S608 - fixed fragments
            parameters,
        ).to_arrow_table()

    def annotation_table(self, object_type: str | None = None) -> pa.Table:
        """Return every annotation occurrence joined with its content as Arrow.

        Args:
            object_type: Keep only occurrences on this object type, or ``None`` for all.

        Returns:
            One row per occurrence with the annotated object's type and ID, the content's key,
            typed value columns, unit, span columns with Signal IDs, confidence, and JSON metadata.

        Raises:
            TimeFFormatError: If ``object_type`` is not an annotated object type.
        """
        if object_type is not None and object_type not in _OBJECT_ID_SELECTS:
            raise TimeFFormatError(f"unknown annotated object type {object_type!r}")
        object_ids = " UNION ALL ".join(_OBJECT_ID_SELECTS.values())
        where = "" if object_type is None else "WHERE o.object_type = ?"
        return self.connection.execute(
            f"""WITH object_ids AS ({object_ids}),
                {_signal_ids_cte("span_ids", "annotation_occurrences", "signal_keys", ("occurrence_key",), "signal_ids")}
                SELECT o.occurrence_id, o.object_type, objects.object_id, c.content_id, c.name AS key,
                       c.value_kind, c.text_value, c.integer_value, c.float_value, c.boolean_value,
                       c.text_list_value, c.unit, o.span_type, o.start_us, o.end_us, span_ids.signal_ids,
                       o.confidence, o.provenance, c.metadata AS content_metadata, o.metadata
                FROM annotation_occurrences o
                JOIN annotation_contents c USING (content_key)
                LEFT JOIN object_ids objects
                  ON objects.object_key = o.object_key AND objects.object_type = o.object_type
                LEFT JOIN span_ids USING (occurrence_key)
                {where}
                ORDER BY o.object_type, objects.object_id, o.occurrence_id""",  # noqa: S608 - fixed fragments
            [] if object_type is None else [object_type],
        ).to_arrow_table()

    @staticmethod
    def _record_filter(record_ids: Iterable[str] | None) -> tuple[str, list[object]]:
        """Return the WHERE clause that keeps tasks whose inputs include one of ``record_ids``.

        Returns:
            The SQL fragment, empty for every task, and its parameters.
        """
        if record_ids is None:
            return "", []
        return (
            """WHERE t.task_key IN (
                   SELECT task_key FROM task_record_refs
                   WHERE field = 'inputs' AND record_key IN (
                       SELECT record_key FROM records WHERE record_id IN (SELECT unnest(?))
                   )
               )""",
            [list(record_ids)],
        )

    def _selection_filter(
        self,
        record_ids: Iterable[str] | None,
        required_modalities: Iterable[InputModality] | None,
        supported_modalities: Iterable[InputModality] | None,
    ) -> tuple[str, list[object]]:
        """Build the task selection predicate for Arrow table reads.

        Returns:
            The SQL WHERE clause and its bound values.
        """
        record_where, record_params = self._record_filter(record_ids)
        modality_where, modality_params = self._modality_filter(required_modalities, supported_modalities)
        predicates = [record_where.removeprefix("WHERE ")] if record_where else []
        if modality_where:
            predicates.append(modality_where)
        return ("WHERE " + " AND ".join(predicates) if predicates else "", [*record_params, *modality_params])

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
        predicates = ["t.input_modalities IS NOT NULL"]
        parameters: list[object] = []
        if required:
            predicates.append("list_has_all(t.input_modalities, ?)")
            parameters.append(required)
        if supported is not None:
            predicates.append("list_has_all(?, t.input_modalities)")
            parameters.append(supported)
        return " AND ".join(predicates), parameters

    def _task_keys(
        self,
        record_keys: list[int] | None,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
    ) -> list[int]:
        """Return the keys of the tasks to hydrate, in task ID order.

        Args:
            record_keys: Keys of the Records whose input tasks to select, or ``None`` for every task.

        Returns:
            The ordered task keys.
        """
        modality_where, modality_params = self._modality_filter(required_modalities, supported_modalities)
        predicates = []
        params: list[object] = []
        if record_keys is not None:
            predicates.append(
                """t.task_key IN (
                    SELECT task_key FROM task_record_refs
                    WHERE field = 'inputs' AND record_key IN (SELECT unnest(?))
                )"""
            )
            params.append(record_keys)
        if modality_where:
            predicates.append(modality_where)
            params.extend(modality_params)
        where = "WHERE " + " AND ".join(predicates) if predicates else ""
        rows = self.connection.execute(
            f"SELECT t.task_key FROM tasks t {where} ORDER BY t.task_id",  # noqa: S608 - fixed predicates
            params,
        ).fetchall()
        return [row[0] for row in rows]

    @staticmethod
    def _task_filter(task_keys: list[int] | None) -> tuple[str, list[object]]:
        """Return the WHERE clause and parameters that select one batch, or nothing for every task.

        Returns:
            The SQL fragment, empty for the whole table, and its parameters.
        """
        if task_keys is None:
            return "", []
        return "WHERE task_key IN (SELECT unnest(?))", [task_keys]

    def _task_rows(self, task_keys: list[int] | None) -> list[_TaskRow]:
        """Read the task rows for one batch of keys, or every task, in task ID order.

        Returns:
            The task rows.
        """
        columns = ", ".join(_TaskRow._fields)
        where, parameters = self._task_filter(task_keys)
        return _rows(
            self.connection.execute(
                f"SELECT {columns} FROM tasks {where} ORDER BY task_id",  # noqa: S608 - fixed identifiers
                parameters,
            ),
            _TaskRow._make,
        )

    def _target_rows(self, task_keys: list[int] | None) -> list[_TargetRow]:
        """Read the target rows for one batch of tasks, or every task, in task and position order.

        Returns:
            The target rows.
        """
        columns = ", ".join(_TargetRow._fields)
        where, parameters = self._task_filter(task_keys)
        return _rows(
            self.connection.execute(
                f"SELECT {columns} FROM task_targets {where} ORDER BY task_key, position",  # noqa: S608 - fixed identifiers
                parameters,
            ),
            _TargetRow._make,
        )

    def _signal_ids_by_key(self, signal_keys: Iterable[int]) -> dict[int, str]:
        """Map internal Signal keys to public IDs without touching the hydrated hierarchy.

        Returns:
            Signal IDs by key.

        Raises:
            TimeFFormatError: If a key names no stored Signal.
        """
        requested = sorted(set(signal_keys))
        if not requested:
            return {}
        rows = self.connection.execute(
            "SELECT signal_key, signal_id FROM signals WHERE signal_key IN (SELECT unnest(?))",
            [requested],
        ).fetchall()
        if len(rows) != len(requested):
            missing = sorted(set(requested) - {key for key, _ in rows})
            raise TimeFFormatError(f"task refers to missing signal key {missing[0]!r}")
        return dict(rows)

    def _task_relationships(
        self, table: str, column: str, task_keys: list[int] | None
    ) -> dict[int, dict[str, tuple[int, ...]]]:
        """Read one ordered task relationship table for a batch, or for every task, through integer keys.

        Args:
            table: ``task_record_refs`` or ``task_annotation_refs``.
            column: The referenced key column of that table.
            task_keys: The batch, or ``None`` for the whole table.

        Returns:
            ``task_key -> field -> ordered referenced keys``.
        """
        where, parameters = self._task_filter(task_keys)
        rows = self.connection.execute(
            f"SELECT task_key, field, {column} FROM {table} {where} ORDER BY task_key, field, position",  # noqa: S608
            parameters,
        ).fetchall()
        return self._group_relationships(rows)

    def _task_dependencies(self, task_keys: list[int] | None) -> dict[int, tuple[int, ...]]:
        """Read the ordered parent keys of each task in one batch, or of every task.

        Returns:
            ``task_key -> ordered parent task keys`` for tasks that have parents.
        """
        where, parameters = self._task_filter(task_keys)
        rows = self.connection.execute(
            f"SELECT task_key, parent_task_key FROM task_dependencies {where} ORDER BY task_key, position",  # noqa: S608
            parameters,
        ).fetchall()
        parents: dict[int, list[int]] = defaultdict(list)
        for task_key, parent_key in rows:
            parents[task_key].append(parent_key)
        return {task_key: tuple(keys) for task_key, keys in parents.items()}

    @staticmethod
    def _group_relationships(rows: list[tuple[int, str, int]]) -> dict[int, dict[str, tuple[int, ...]]]:
        """Group ordered relationship query rows.

        Returns:
            Relationship values grouped by task key and field.
        """
        grouped: dict[int, dict[str, tuple[int, ...]]] = {}
        current_task: int | None = None
        current_field: str | None = None
        current_values: list[int] = []
        for task_key, field, value in rows:
            if task_key != current_task or field != current_field:
                if current_task is not None and current_field is not None:
                    grouped.setdefault(current_task, {})[current_field] = tuple(current_values)
                current_task = task_key
                current_field = field
                current_values = [value]
            else:
                current_values.append(value)
        if current_task is not None and current_field is not None:
            grouped.setdefault(current_task, {})[current_field] = tuple(current_values)
        return grouped

    @staticmethod
    def _resolve(
        task_id: str,
        field: str,
        refs: dict[str, tuple[int, ...]],
        objects: dict[int, _Ref],
        label: str,
    ) -> tuple[_Ref, ...]:
        """Resolve one ordered reference field to hydrated objects.

        Args:
            task_id: The task the field belongs to, for the error message.
            field: The relationship field name.
            refs: The task's ordered keys by field.
            objects: Hydrated objects by internal key.
            label: What the keys name in the error message, such as ``"record"``.

        Returns:
            The referenced objects in stored order.

        Raises:
            TimeFFormatError: If a key names no hydrated object.
        """
        keys = refs.get(field)
        if not keys:
            return ()
        try:
            return tuple(map(objects.__getitem__, keys))
        except KeyError as exc:
            raise TimeFFormatError(
                f"task {task_id!r} field {field!r} refers to missing {label} key {exc.args[0]!r}"
            ) from exc

    def _read_axes(
        self,
        axis_keys: Iterable[int] | None = None,
    ) -> dict[int, RegularAxis | IrregularAxis | OrdinalAxis]:
        """Hydrate each shared axis exactly once.

        Returns:
            Axes keyed by their stored IDs.

        Raises:
            TimeFFormatError: If an axis has an unknown type.
        """
        requested = None if axis_keys is None else tuple(axis_keys)
        if requested == ():
            return {}
        axes: dict[int, RegularAxis | IrregularAxis | OrdinalAxis] = {}
        query = """SELECT axis_key, axis_id, axis_type, period_numerator_us, period_denominator,
                          origin_us, first_us, last_us FROM axes"""
        rows = (
            self.connection.execute(query).fetchall()
            if requested is None
            else self.connection.execute(
                f"{query} WHERE axis_key IN (SELECT unnest(?))",
                [list(requested)],
            ).fetchall()
        )
        for axis_key, axis_id, axis_type, numerator, denominator, origin, first, last in rows:
            if axis_type == "regular":
                axes[axis_key] = RegularAxis(
                    axis_id=axis_id,
                    period_us=Fraction(numerator, denominator),
                    start_index=origin,
                )
            elif axis_type == "irregular":
                axes[axis_key] = IrregularAxis(axis_id=axis_id, first_us=first, last_us=last)
            elif axis_type == "ordinal":
                axes[axis_key] = OrdinalAxis(axis_id=axis_id)
            else:
                raise TimeFFormatError(f"axis {axis_id!r} has unknown type {axis_type!r}")
        return axes

    def load_axis_offsets_by_key(self, axis_key: int, axis_id: str) -> pa.Array:
        """Load one shared irregular axis through its internal integer key.

        Returns:
            The axis offsets as Arrow int64 values.

        Raises:
            TimeFFormatError: If the offsets disagree with the axis or its Signals.
        """
        rows = self.connection.execute(
            """SELECT offsets.offset_us
               FROM axis_offsets offsets
               WHERE axis_key = ? ORDER BY offsets.position""",
            [axis_key],
        ).fetchall()
        offsets = np.asarray([row[0] for row in rows], dtype=np.int64)
        axis = self.connection.execute(
            "SELECT first_us, last_us FROM axes WHERE axis_key = ? AND axis_type = 'irregular'",
            [axis_key],
        ).fetchone()
        if axis is None:
            raise TimeFFormatError(f"axis {axis_id!r} has offsets but is missing or not irregular")
        lengths = {
            row[0]
            for row in self.connection.execute(
                "SELECT DISTINCT n_values FROM signals WHERE axis_key = ?", [axis_key]
            ).fetchall()
        }
        if len(lengths) != 1 or len(offsets) not in lengths:
            raise TimeFFormatError(
                f"axis {axis_id!r} stores {len(offsets)} offsets but its Signals declare lengths {sorted(lengths)}"
            )
        endpoints = axis[:2]
        if len(offsets) == 0 or (int(offsets[0]), int(offsets[-1])) != endpoints:
            raise TimeFFormatError(f"axis {axis_id!r} has offsets disagreeing with its axis endpoints {endpoints}")
        if np.any(np.diff(offsets) < 0):
            raise TimeFFormatError(f"axis {axis_id!r} has decreasing offsets")
        return pa.array(offsets, type=pa.int64())

    @staticmethod
    def _spec_from_row(row: _SignalRow) -> TimeSeriesSpec:
        """Reconstruct an inline signal specification.

        Returns:
            The typed specification.
        """
        return TimeSeriesSpec(
            spec_type=row.spec_type,
            name=row.spec_name,
            unit_value=ureg.Unit(_required(row.unit, f"the unit of signal {row.signal_id!r}")),
            dtype=row.dtype,
            categories=tuple(row.categories),
            value_shape=tuple(row.value_shape),
            dimension_names=tuple(row.dimension_names),
            nullable=row.nullable,
        )

    def _read_annotations(
        self,
        object_keys: Iterable[int] | None = None,
        *,
        object_type: str | None = None,
        annotations_by_occurrence: dict[int, Annotation] | None = None,
    ) -> dict[tuple[str, str], tuple[Annotation, ...]]:
        """Hydrate annotation content and occurrences for hierarchy objects.

        Args:
            object_keys: Internal keys of the annotated objects, or ``None`` for every occurrence.
            object_type: Restrict the read to occurrences on one object type, which also limits
                the object lookup to that type's table.
            annotations_by_occurrence: Filled with every hydrated occurrence by its internal key.

        Returns:
            Attached occurrences keyed by object type and object ID.

        Raises:
            TimeFFormatError: If an occurrence has an unknown span type.
        """
        requested = None if object_keys is None else tuple(object_keys)
        if requested == ():
            return {}
        grouped: dict[tuple[str, str], list[Annotation]] = defaultdict(list)
        parameters: list[object] = [list(requested or ())]
        if object_type is not None:
            parameters.append(object_type)
        rows = _rows(
            self.connection.execute(_annotation_query(object_type, keyed=requested is not None), parameters),
            _OccurrenceRow._make,
        )
        for row in rows:
            if row.content_id is None or row.name is None:
                raise TimeFFormatError(
                    f"annotation occurrence {row.occurrence_id!r} refers to missing content key {row.content_key!r}"
                )
            if row.object_id is None:
                if row.object_type not in {"Dataset", "Record", "Source", "Signal", "Task"}:
                    raise TimeFFormatError(
                        f"annotation occurrence {row.occurrence_id!r} has unknown object type {row.object_type!r}"
                    )
                raise TimeFFormatError(
                    f"annotation occurrence {row.occurrence_id!r} refers to missing {row.object_type} key"
                )
        referenced_signal_keys = sorted({signal_key for row in rows for signal_key in (row.signal_keys or ())})
        signal_ids_by_key = (
            dict(
                self.connection.execute(
                    """SELECT signal_key, signal_id FROM signals
                       WHERE signal_key IN (SELECT unnest(?))""",
                    [referenced_signal_keys],
                ).fetchall()
            )
            if referenced_signal_keys
            else {}
        )
        for row in rows:
            try:
                scope = (
                    None
                    if row.signal_keys is None
                    else tuple(signal_ids_by_key[signal_key] for signal_key in row.signal_keys)
                )
            except KeyError as exc:
                raise TimeFFormatError(
                    f"annotation occurrence {row.occurrence_id!r} refers to missing signal key {exc.args[0]!r}"
                ) from exc
            span = _occurrence_span(row, scope)
            content_metadata = _decode_json(row.content_metadata, default={})
            description = content_metadata.pop("description", None)
            # The first pass over ``rows`` already rejected NULL content and object columns.
            content_id = _required(row.content_id, "annotation content")
            name = _required(row.name, "annotation name")
            object_id = _required(row.object_id, "annotated object")
            annotation = Annotation(
                occurrence_id=row.occurrence_id,
                id=content_id,
                key=name,
                value=decode_annotation_value(
                    row.value_kind,
                    row.text_value,
                    row.integer_value,
                    row.float_value,
                    row.boolean_value,
                    row.text_list_value,
                ),
                unit=row.unit,
                description=description,
                metadata=content_metadata,
                span=span,
                source=_decode_json(row.provenance),
                confidence=row.confidence,
                occurrence_metadata=_decode_json(row.metadata, default={}),
            )
            grouped[row.object_type, object_id].append(annotation)
            if annotations_by_occurrence is not None:
                annotations_by_occurrence[row.occurrence_key] = annotation
        return {key: tuple(value) for key, value in grouped.items()}


class _TaskHydration:
    """State for one :meth:`DuckDBControlReader.iter_tasks` call.

    It holds the caller's Records and the lookups built from them, loads Records and Signals a
    task refers to outside that set on demand, and keeps every task that another task derives
    from until its dependants are built.
    """

    def __init__(
        self,
        reader: DuckDBControlReader,
        records: tuple[Record, ...],
        *,
        all_records: bool = False,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
    ) -> None:
        """Index the caller's Records and decide whether the task table needs filtering.

        Raises:
            TimeFValidationError: If one of ``records`` is not stored in the control database.
        """
        self.reader = reader
        self.required_modalities = required_modalities
        self.supported_modalities = supported_modalities
        self.connection = reader.connection
        self.records_by_id: dict[str, Record] = {record.id: record for record in records}
        self.records_by_key: dict[int, Record] = {}
        self.signals_by_id: dict[str, Signal] = {}
        self._records_without_signal_index: list[Record] = []
        self.annotations_by_id: dict[str, Annotation] = {}
        self.retained: dict[int, Task] = {}
        self._hydrating: set[int] = set()
        self._dataset_annotations_indexed = False
        facts = reader.task_facts()
        self.parent_keys = facts.parent_task_keys
        self.tasks_annotated = facts.tasks_annotated
        unknown = [record_id for record_id in self.records_by_id if record_id not in reader._record_keys]
        if unknown:
            # Records hydrated by another reader instance: look their keys up once.
            rows = self.connection.execute(
                "SELECT record_key, record_id FROM records WHERE record_id IN (SELECT unnest(?))",
                [unknown],
            ).fetchall()
            if len(rows) != len(unknown):
                missing = sorted(set(unknown) - {record_id for _, record_id in rows})
                raise TimeFValidationError(f"no such record(s) in control.duckdb: {missing}")
            reader._record_keys.update((record_id, record_key) for record_key, record_id in rows)
        for record_id, record in list(self.records_by_id.items()):
            self._index_record(reader._record_keys[record_id], record)
        self.record_keys: list[int] | None = (
            None if all_records or len(self.records_by_id) == facts.record_count else list(self.records_by_key)
        )

    def iter_tasks(self) -> Iterator[Task]:
        """Yield the selected tasks in task ID order, one bounded batch at a time.

        Yields:
            Each hydrated task.
        """
        task_keys = self.reader._task_keys(
            self.record_keys,
            self.required_modalities,
            self.supported_modalities,
        )
        for start in range(0, len(task_keys), _TASK_BATCH_SIZE):
            yield from self._hydrate(task_keys[start : start + _TASK_BATCH_SIZE])

    def read_all(self) -> list[Task]:
        """Build every selected task in one pass of unfiltered or single-batch queries.

        Returns:
            The tasks in task ID order.
        """
        if self.record_keys is None and self.required_modalities is None and self.supported_modalities is None:
            return self._hydrate(None)
        return self._hydrate(
            self.reader._task_keys(self.record_keys, self.required_modalities, self.supported_modalities)
        )

    def _hydrate(self, task_keys: list[int] | None) -> list[Task]:
        """Build one batch, or the whole table for ``None``, hydrating unbuilt parents first.

        Returns:
            The tasks in the order of ``task_keys``, or in task ID order for the whole table.

        Raises:
            TimeFFormatError: If the stored derivations contain a cycle.
        """
        pending = None if task_keys is None else [key for key in task_keys if key not in self.retained]
        dependencies = self.reader._task_dependencies(pending) if self.parent_keys else {}
        in_batch = None if pending is None else set(pending)
        missing_parents = (
            []
            if in_batch is None
            else sorted(
                {
                    parent_key
                    for parent_keys in dependencies.values()
                    for parent_key in parent_keys
                    if parent_key not in self.retained and parent_key not in in_batch
                }
            )
        )
        if missing_parents:
            cyclic = self._hydrating.intersection(missing_parents)
            if cyclic:
                raise TimeFFormatError(f"task dependencies contain a cycle through task keys {sorted(cyclic)}")
            self._hydrating.update(pending or ())
            try:
                self._hydrate(missing_parents)
            finally:
                self._hydrating.difference_update(pending or ())
        built = self._build(pending)
        for task_key, parent_keys in dependencies.items():
            built[task_key].from_tasks = tuple(self._task(parent_key, built) for parent_key in parent_keys)
        if task_keys is None:
            return list(built.values())
        for task_key, task in built.items():
            if task_key in self.parent_keys:
                self.retained[task_key] = task
        return [self._task(task_key, built) for task_key in task_keys]

    def _task(self, task_key: int, built: dict[int, Task]) -> Task:
        """Return a task from this batch or the retained parents.

        Raises:
            TimeFFormatError: If the key was never hydrated.
        """
        task = built.get(task_key) or self.retained.get(task_key)
        if task is None:
            raise TimeFFormatError(f"task dependency refers to missing task key {task_key!r}")
        return task

    def _build(self, task_keys: list[int] | None) -> dict[int, Task]:  # noqa: PLR0914 - one batch joins several tables
        """Read every table for one batch of task keys and construct the tasks.

        Returns:
            The constructed tasks by key, without ``from_tasks`` wired.

        Raises:
            TimeFFormatError: If a task has an unknown type.
        """
        if task_keys is not None and not task_keys:
            return {}
        reader = self.reader
        task_rows = reader._task_rows(task_keys)
        record_refs = reader._task_relationships("task_record_refs", "record_key", task_keys)
        annotation_refs = reader._task_relationships("task_annotation_refs", "occurrence_key", task_keys)
        target_rows = reader._target_rows(task_keys)
        task_annotations = reader._read_annotations(task_keys, object_type="Task") if self.tasks_annotated else {}

        self._ensure_records(
            {key for refs in record_refs.values() for keys in refs.values() for key in keys}
            | {row.record_key for row in target_rows if row.record_key is not None}
        )
        signals_by_key = self._signals_by_key({row.signal_key for row in target_rows if row.signal_key is not None})
        signal_ids_by_key = reader._signal_ids_by_key(
            [key for row in target_rows for key in (row.signal_keys or ())]
            + [key for row in task_rows for key in (row.scope_signal_keys or ())]
        )
        annotations_by_occurrence = self._annotations_by_occurrence(
            {key for refs in annotation_refs.values() for keys in refs.values() for key in keys}
        )

        targets_by_task: dict[int, list[object]] = defaultdict(list)
        for target in target_rows:
            span_signal_ids = (
                None if target.signal_keys is None else tuple(signal_ids_by_key[key] for key in target.signal_keys)
            )
            signal_id = None if target.signal_key is None else signals_by_key[target.signal_key].id
            if target.target_kind in {"step_point", "step_interval"}:
                signal_id = None if not span_signal_ids else span_signal_ids[0]
            target_values: dict[str, object] = {
                "target_kind": target.target_kind,
                "text_value": target.text_value,
                "integer_value": target.integer_value,
                "float_value": target.float_value,
                "boolean_value": target.boolean_value,
                "record_id": None if target.record_key is None else self.records_by_key[target.record_key].id,
                "signal_id": signal_id,
                "span_start": target.span_start,
                "span_end": target.span_end,
                "span_signal_ids": span_signal_ids,
            }
            targets_by_task[target.task_key].append(
                decode_target(target_values, records=self.records_by_id, signals=self.signals_by_id)
            )

        tasks: dict[int, Task] = {}
        for row in task_rows:
            try:
                task_type = TaskType(row.task_type)
                cls = TASKS[task_type]
            except (ValueError, KeyError) as exc:
                raise TimeFFormatError(f"task {row.task_id!r} has unknown type {row.task_type!r}") from exc
            refs = record_refs.get(row.task_key, {})
            kwargs: dict[str, Any] = {
                "id": row.task_id,
                "inputs": reader._resolve(row.task_id, "inputs", refs, self.records_by_key, "record"),
                "targets": tuple(targets_by_task.get(row.task_key, ())) if row.has_inline_targets else None,
                "prompt": row.prompt,
                "input_modalities": decode_input_modalities(row.input_modalities),
                "scope": decode_span(
                    row.scope_type,
                    row.scope_start,
                    row.scope_end,
                    None if row.scope_signal_keys is None else [signal_ids_by_key[k] for k in row.scope_signal_keys],
                ),
                "rationale": row.rationale,
                "annotations": task_annotations.get(("Task", row.task_id), ()),
                "metadata": _decode_json(row.metadata, default={}),
                **decode_task_payload(task_type, row.payload_columns),
            }
            if "candidate_records" in cls.__dataclass_fields__:
                kwargs["candidate_records"] = reader._resolve(
                    row.task_id, "candidate_records", refs, self.records_by_key, "record"
                )
            task_annotation_refs = annotation_refs.get(row.task_key, {})
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
            tasks[row.task_key] = cls(**kwargs)
        return tasks

    def _index_record(self, record_key: int, record: Record) -> None:
        """Add one Record and everything hanging off it to the lookups."""
        self.records_by_key[record_key] = record
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

    def _ensure_records(self, record_keys: set[int]) -> None:
        """Hydrate Records a task refers to that the caller did not pass.

        Raises:
            TimeFFormatError: If a key names no stored Record.
        """
        missing = sorted(record_keys - self.records_by_key.keys())
        if not missing:
            return
        rows = self.connection.execute(
            "SELECT record_key, record_id FROM records WHERE record_key IN (SELECT unnest(?))",
            [missing],
        ).fetchall()
        if len(rows) != len(missing):
            unknown = sorted(set(missing) - {key for key, _ in rows})
            raise TimeFFormatError(f"task refers to missing record key {unknown[0]!r}")
        ids_by_key = dict(rows)
        hydrated = self.reader.read_records([ids_by_key[key] for key in missing])
        for record_key, record in zip(missing, hydrated, strict=True):
            self._index_record(record_key, record)

    def _signals_by_key(self, signal_keys: set[int]) -> dict[int, Signal]:
        """Resolve the Signals one batch's targets refer to.

        Returns:
            Signals by internal key.

        Raises:
            TimeFFormatError: If a key names no Signal on the hydrated Records.
        """
        if not signal_keys:
            return {}
        for record in self._records_without_signal_index:
            for signal in record.signals:
                self.signals_by_id[signal.id] = signal
        self._records_without_signal_index.clear()
        resolved: dict[int, Signal] = {}
        for signal_key, signal_id in self.reader._signal_ids_by_key(signal_keys).items():
            signal = self.signals_by_id.get(signal_id)
            if signal is None:
                raise TimeFFormatError(
                    f"task target refers to Signal {signal_id!r} on a Record outside the hydrated set"
                )
            resolved[signal_key] = signal
        return resolved

    def _annotations_by_occurrence(self, occurrence_keys: set[int]) -> dict[int, Annotation]:
        """Resolve referenced annotation occurrences to the objects on the hydrated Records.

        Returns:
            Annotations by internal occurrence key.

        Raises:
            TimeFFormatError: If a key names no occurrence on the hydrated Records or the Dataset.
        """
        if not occurrence_keys:
            return {}
        rows = self.connection.execute(
            "SELECT occurrence_key, occurrence_id FROM annotation_occurrences WHERE occurrence_key IN (SELECT unnest(?))",
            [sorted(occurrence_keys)],
        ).fetchall()
        resolved: dict[int, Annotation] = {}
        for occurrence_key, occurrence_id in rows:
            annotation = self.annotations_by_id.get(occurrence_id)
            if annotation is None and not self._dataset_annotations_indexed:
                self._index_dataset_annotations()
                annotation = self.annotations_by_id.get(occurrence_id)
            if annotation is None:
                raise TimeFFormatError(
                    f"task refers to annotation occurrence {occurrence_id!r} that no hydrated object carries"
                )
            resolved[occurrence_key] = annotation
        return resolved

    def _index_dataset_annotations(self) -> None:
        """Add the Dataset-level annotation occurrences to the lookups, once."""
        self._dataset_annotations_indexed = True
        for (dataset_key,) in self.connection.execute("SELECT dataset_key FROM datasets").fetchall():
            for annotations in self.reader._read_annotations((dataset_key,)).values():
                for annotation in annotations:
                    if annotation.occurrence_id is not None:
                        self.annotations_by_id[annotation.occurrence_id] = annotation
