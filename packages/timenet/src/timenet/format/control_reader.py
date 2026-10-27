"""Hydrate the TimeF object hierarchy from ``control.duckdb``.

The objects hold their stored values behind value handles, unloaded.
"""

from collections import ChainMap, defaultdict
from collections.abc import Callable, Collection, Container, Iterable, Iterator, Mapping, MutableMapping
from fractions import Fraction
from functools import partial
from graphlib import CycleError, TopologicalSorter
import json
import os
from pathlib import Path
import threading
from typing import Any, NamedTuple, TypeVar, cast
import weakref

import duckdb
import pyarrow as pa

from timenet.dataset import IrregularAxis, OrdinalAxis, Record, RegularAxis, Signal, Source, TimeAxis, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFNotFoundError, TimeFReaderError, TimeFValidationError
from timenet.format.annotation_codec import decode_annotation_value
from timenet.format.control_schema import ANNOTATED_OBJECT_TYPES, LAST_ID_PREFIX
from timenet.format.duckdb import CATALOG_ERRORS, check_control_schema, connect_control
from timenet.format.stored_array import ChunkLocation
from timenet.format.stored_values import (
    ExternalObjectFieldKey,
    PackedField,
    PackedLocation,
    PackedValueReader,
    StoredOffsets,
    StoredText,
    StoredTextList,
)
from timenet.format.task_codec import (
    ANNOTATION_FIELDS,
    RECORD_FIELDS,
    TaskRelationshipField,
    decode_input_modalities,
    decode_span,
    decode_split,
    decode_target,
    decode_task_payload,
)
from timenet.types import (
    TASKS,
    Annotation,
    ArrayHandle,
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
from timenet.types.handles import CallableArray
from timenet.types.ids import EntityId
from timenet.types.object_kind import ObjectKind
from timenet.types.reference_keys import SourceKey
from timenet.types.splits import Split, parse_split
from timenet.types.wire import ValueDtype
from timenet.values_backends.types import ChunkDataIndex, StoredChunk


ValueLoaderFactory = Callable[[int, TimeSeriesSpec], ArrayHandle]


class _RecordRow(NamedTuple):
    """One ``records`` row in schema column order."""

    record_id: EntityId
    clock_id: int
    time_span_start_us: int | None
    time_span_end_us: int | None
    metadata: str

    @property
    def id(self) -> EntityId:
        """Return the Record id."""
        return self.record_id


class _SourceRow(NamedTuple):
    """One ``sources`` row in schema column order."""

    source_id: EntityId
    record_id: EntityId
    parent_source_id: EntityId | None
    name: str
    metadata: str

    @property
    def id(self) -> EntityId:
        """Return the Source id."""
        return self.source_id


class _SignalRow(NamedTuple):
    """One ``signals`` row in schema column order."""

    signal_id: EntityId
    source_id: EntityId
    name: str
    axis_id: EntityId
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
    def id(self) -> EntityId:
        """Return the Signal id."""
        return self.signal_id

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


class _PreparedSignal(NamedTuple):
    """A signal row with its axis, specification, and value handles."""

    row: _SignalRow
    axis: TimeAxis
    spec: TimeSeriesSpec
    values: ArrayHandle
    time_offsets: StoredOffsets | None


class _RecordHierarchy(NamedTuple):
    """Selected rows with checked relationships and the signals' value handles."""

    records: list[_RecordRow]
    """The record rows in result order, a row again for each repeated request."""
    sources: dict[EntityId, _SourceRow]
    children: dict[EntityId, list[EntityId]]
    roots_by_record: dict[EntityId, list[EntityId]]
    signals: list[_PreparedSignal]
    owners: dict[ObjectKind, Collection[EntityId]]
    """The IDs of the selected Records, Sources, and Signals, the owners of their annotations."""
    read_all: bool


class _ChunkRow(NamedTuple):
    """One chunk locator in the control catalog's column order."""

    signal_id: EntityId
    chunk_index: int
    value_path: str
    source_key: SourceKey | None
    chunk_major_index: int
    chunk_minor_index: int | None
    n_values: int

    def stored_chunk(self, strings: dict[str, str]) -> StoredChunk:
        """Return the value-reader locator represented by this catalog row.

        Args:
            strings: File paths already in use, so chunks in one file share one string.

        Returns:
            The chunk's file, coordinates, and length.
        """
        return StoredChunk(
            chunk_idx=self.chunk_index,
            chunk_file=strings.setdefault(self.value_path, self.value_path),
            data_index=ChunkDataIndex(self.chunk_major_index, self.chunk_minor_index),
            n_values=self.n_values,
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
    content_id: int
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
    """One ``task_targets`` row in schema column order."""

    task_id: int
    position: int
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
"""Tasks hydrated per round of keyed queries, unless a stream asks for another batch size."""


def _annotation_query(owners: Mapping[ObjectKind, Collection[int] | None]) -> tuple[str, list[object]]:
    """Build the occurrence query, restricted to the requested owners.

    Args:
        owners: The owner IDs to read, by object type. If the IDs are ``None``, the query reads
            every occurrence of that type. The Dataset's occurrences have no object ID, so its IDs
            are occurrence IDs.

    Returns:
        The SQL and its positional parameters.
    """
    parameters: list[object] = []
    # With a parameter inside an OR, DuckDB does not push the filter below the join. A one-Record
    # read then scans every occurrence. For this reason, each owner type is one subquery of plain
    # predicates.
    selects: list[str] = []
    every: list[str] = []
    for object_type, object_ids in owners.items():
        if object_ids is None:
            every.append(object_type)
        else:
            # The Dataset's occurrences have no object ID, so its IDs select occurrences by occurrence ID.
            selects.append(
                "SELECT * FROM annotation_occurrences WHERE object_type = ? AND occurrence_id IN (SELECT unnest(?))"
                if object_type == ObjectKind.DATASET
                else "SELECT * FROM annotation_occurrences WHERE object_type = ? AND object_id IN (SELECT unnest(?))"
            )
            parameters.extend((object_type, list(object_ids)))
    if every:
        selects.append("SELECT * FROM annotation_occurrences WHERE object_type IN (SELECT unnest(?))")
        parameters.append(every)
    occurrences = f"({' UNION ALL '.join(selects)})"
    query = f"""SELECT o.occurrence_id, o.object_type, o.object_id, o.span_type,
               o.start_us, o.end_us, o.provenance, o.confidence,
               o.metadata, o.content_id, c.name, c.value_kind, c.text_value, c.integer_value,
               c.float_value, c.boolean_value, c.text_list_value, c.unit, c.description, c.metadata
        FROM {occurrences} o
        LEFT JOIN annotation_contents c USING (content_id)
        ORDER BY o.occurrence_id"""  # noqa: S608 - fixed fragments
    return query, parameters


class _TaskFacts(NamedTuple):
    """Facts about one control database that every task hydration reuses."""

    parent_task_ids: frozenset[int]
    tasks_annotated: bool


_HierarchyRow = TypeVar("_HierarchyRow", _RecordRow, _SourceRow, _SignalRow)
_Row = TypeVar("_Row", bound=tuple[Any, ...])
_Ref = TypeVar("_Ref", Record, Annotation)
_T = TypeVar("_T")

_HIERARCHY_TABLES: dict[type[tuple[Any, ...]], tuple[str, str, str, ObjectKind]] = {
    _RecordRow: ("records", "record_id", "record_id", ObjectKind.RECORD),
    _SourceRow: ("sources", "source_id", "record_id", ObjectKind.SOURCE),
    _SignalRow: ("signals", "signal_id", "source_id", ObjectKind.SIGNAL),
}
"""Table name, ID column, relation column, and object kind of each hierarchy row type."""


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


def _require_found(kind: ObjectKind, requested: Iterable[EntityId], found: Container[EntityId]) -> None:
    """Check that every requested ID names a stored object of one kind.

    Raises:
        TimeFNotFoundError: If an ID names none, listing every such ID.
    """
    missing = [identity for identity in requested if identity not in found]
    if missing:
        raise TimeFNotFoundError(f"no {kind.value} {', '.join(map(repr, missing))}")


def _missing_values(signal_id: int, _spec: TimeSeriesSpec) -> pa.Array:
    """Fail when metadata-only hydration is asked to load values.

    Raises:
        TimeFFormatError: Always, because no values loader was configured.
    """
    raise TimeFFormatError(f"no values loader is configured for signal {signal_id!r}")


def _missing_values_factory(signal_id: int, spec: TimeSeriesSpec) -> ArrayHandle:
    """Create a value handle for a metadata-only reader.

    Returns:
        A handle that raises TimeFFormatError when asked to read values.
    """
    return CallableArray(partial(_missing_values, signal_id, spec))


def _shared_source(source: SourceKey | None, strings: dict[str, str]) -> SourceKey | None:
    """Return a source key as the string already in use for it, so chunk locations share one string.

    Returns:
        The source key, or ``None`` for the release whose catalog records the chunks.
    """
    return None if source is None else SourceKey(strings.setdefault(source, source))


def _stores_locators(connection: duckdb.DuckDBPyConnection) -> bool:
    """Return whether the catalog stores any packed-value locator.

    Raises:
        TimeFFormatError: If the catalog has no locator table.
    """
    try:
        return connection.execute("SELECT 1 FROM external_values LIMIT 1").fetchone() is not None
    except CATALOG_ERRORS as exc:
        raise TimeFFormatError(f"cannot read external values: {exc}") from exc


class _ConnectionRegistry:
    """The control readers that hold an open DuckDB connection, so each one closes before a fork.

    A forked child hangs in DuckDB when its parent holds the same database open, even on a new
    connection. So the before-fork hook closes every registered connection, and each reader opens
    its connection again on its next query, in the parent and in the child. Forking while other
    threads use readers is not supported: use the spawn or forkserver start method there.
    """

    def __init__(self) -> None:
        self.readers: weakref.WeakSet[DuckDBControlReader] = weakref.WeakSet()
        # Guards ``readers``, which readers in different threads change.
        self.lock = threading.Lock()

    def before_fork(self) -> None:
        """Close every registered connection."""
        with self.lock:
            readers = list(self.readers)
        for reader in readers:
            reader._disconnect()

    def after_fork_in_child(self) -> None:
        """Give the child a fresh lock, since a thread that held the inherited one is not in it."""
        self.lock = threading.Lock()


_CONNECTIONS = _ConnectionRegistry()
"""The open control-reader connections of this process."""

if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_CONNECTIONS.before_fork, after_in_child=_CONNECTIONS.after_fork_in_child)


class DuckDBControlReader:  # noqa: PLR0904 - one public method per kind of catalog read
    """Keep one read-only DuckDB connection and hydrate requested records from it.

    The connection opens again on the next query after :meth:`close`, and after the process forks,
    since every connection closes before a fork.
    """

    def __init__(
        self,
        path: Path,
        *,
        value_loader_factory: ValueLoaderFactory = _missing_values_factory,
        external: PackedValueReader | None = None,
        fetch: Callable[[], Path] | None = None,
    ) -> None:
        """Open and validate an immutable control database.

        Args:
            path: Local path to ``control.duckdb``.
            value_loader_factory: Factory for range-aware per-Signal value handles. The default
                rejects value reads while allowing metadata-only reads.
            external: Payload reader used to resolve locators for each requested object batch.
                Required when the selected objects have packed values.
            fetch: Returns a verified local path to the database again, such as a fresh copy of
                a remote catalog. A reopen calls it when ``path`` no longer exists.

        Raises:
            TimeFFormatError: If the database cannot be opened or has an unsupported schema.
        """  # noqa: DOC502 - raised by _open
        self.path = Path(path)
        self._fetch = fetch
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._open()
        self.external = external
        self._packed_values_stored: bool | None = None
        self._value_loader_factory = value_loader_factory
        self._clock_cache: dict[int, TimeOrigin] = {}
        self._record_occurrences: dict[int, dict[int, Annotation]] = {}
        """The occurrences on each live hydrated Record by occurrence id, keyed by ``id(record)``."""
        self._task_facts: _TaskFacts | None = None
        self._known_record_ids: set[int] = set()
        # Chunk files and sources named by the chunk locations read so far, so equal names share one string.
        self._chunk_strings: dict[str, str] = {}

    def run(self, query: Callable[[duckdb.DuckDBPyConnection], _T]) -> _T:
        """Run one read through the connection, which opens again first if it was closed.

        Args:
            query: Reads through the connection it is given.

        Returns:
            What ``query`` returns.

        Raises:
            TimeFReaderError: If the connection closed while the read ran, through :meth:`close` or
                through the before-fork hook in another thread.
            TimeFFormatError: If the catalog lacks a table, column, or type that the read expects.
            duckdb.Error: Any other DuckDB failure, such as I/O, memory, or an interrupt, unchanged.
        """
        connection = self._open()
        try:
            return query(connection)
        except duckdb.Error as exc:
            if self._connection is not connection:
                raise TimeFReaderError(
                    f"the connection to control database {self.path} was closed while a read was running "
                    "on it. Forking while other threads use readers is not supported; use the spawn or "
                    "forkserver start method."
                ) from exc
            if isinstance(exc, CATALOG_ERRORS):
                raise TimeFFormatError(f"could not read control database {self.path}: {exc}") from exc
            raise

    def _query(self, sql: str, parameters: object = None) -> list[Any]:
        """Run one query through :meth:`run`.

        Returns:
            Every row of the result.
        """
        return self.run(lambda connection: connection.execute(sql, parameters).fetchall())

    def _query_table(self, sql: str, parameters: object = None) -> pa.Table:
        """Run one query through :meth:`run`.

        Returns:
            The result as an Arrow table.
        """
        return self.run(lambda connection: connection.execute(sql, parameters).to_arrow_table())

    def _query_named(self, sql: str, parameters: object, make: Callable[[Iterable[Any]], _Row]) -> list[_Row]:
        """Run one query through :meth:`run`.

        Returns:
            Every row of the result as a named tuple.
        """
        return self.run(lambda connection: _rows(connection.execute(sql, parameters), make))

    def _open(self) -> duckdb.DuckDBPyConnection:
        """Open the database read-only unless it is open, check its schema, and register the connection.

        Returns:
            The open connection.

        Raises:
            TimeFFormatError: If the database cannot be opened or has an unsupported schema.
        """
        connection = self._connection
        if connection is not None:
            return connection
        if self._fetch is not None and not self.path.exists():
            self.path = Path(self._fetch())
        try:
            connection = connect_control(self.path, read_only=True)
        except duckdb.Error as exc:
            raise TimeFFormatError(f"could not open control database {self.path}: {exc}") from exc
        try:
            check_control_schema(connection)
        except BaseException:
            connection.close()
            raise
        with _CONNECTIONS.lock:
            self._connection = connection
            _CONNECTIONS.readers.add(self)
        return connection

    def _disconnect(self) -> None:
        """Close the DuckDB connection, keeping what the reader knows about the catalog."""
        with _CONNECTIONS.lock:
            _CONNECTIONS.readers.discard(self)
            connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()

    def _external_values(
        self, kind: ObjectKind, object_ids: Iterable[int]
    ) -> dict[ExternalObjectFieldKey, StoredText | StoredTextList]:
        """Return text handles for the packed values of only the objects in this read.

        Args:
            kind: The kind of the objects that own the values.
            object_ids: The objects' catalog ids: task ids, or annotation content ids.

        Returns:
            Handles indexed by object, field, and position, or no handles for an inline catalog.
        """
        locations = self._external_locations(kind, object_ids)
        external = self.external
        return {} if external is None else {key: external.handle(location) for key, location in locations.items()}

    def _external_locations(
        self, kind: ObjectKind, object_ids: Iterable[int]
    ) -> dict[ExternalObjectFieldKey, PackedLocation]:
        """Return the packed-value locations of only the objects in this read.

        A release that lists no payload files has no payload reader, so nothing is looked up per
        batch. The catalog is checked once for any locator instead.

        Returns:
            Locations indexed by object, field, and position, or none for an inline catalog.

        Raises:
            TimeFFormatError: If packed values need a payload reader that was not configured.
        """
        if self.external is not None:
            ids = tuple(object_ids)
            return self.run(lambda connection: PackedValueReader.locations(connection, kind, ids))
        if self._stores_packed_values():
            raise TimeFFormatError("control.duckdb stores packed values, but the release lists no payload files")
        return {}

    def _stores_packed_values(self) -> bool:
        """Return whether the catalog stores any packed-value locator, checked once per reader.

        Raises:
            TimeFFormatError: If the catalog has no locator table.
        """  # noqa: DOC502 - raised by _stores_locators
        if self._packed_values_stored is None:
            self._packed_values_stored = self.run(_stores_locators)
        return self._packed_values_stored

    def _stored_offsets(
        self, axes: Mapping[int, TimeAxis], shared: dict[int, StoredOffsets] | None = None
    ) -> dict[int, StoredOffsets]:
        """Return one handle to the packed time offsets of each irregular axis, without reading them.

        Args:
            axes: The axes by axis ID.
            shared: Handles that earlier record reads of the same call built, by axis ID. Their
                axes reuse them, and new handles are added, so one call has one handle per axis.

        Returns:
            Handles keyed by axis ID. Each checks its time offsets against the axis on its first read
            and keeps them.

        Raises:
            TimeFFormatError: If an irregular axis has no packed time offsets, or another axis has some.
        """
        handles: dict[int, StoredOffsets] = {} if shared is None else shared
        new_axes = {axis_id: axis for axis_id, axis in axes.items() if axis_id not in handles}
        locations = self._external_locations(ObjectKind.AXIS, new_axes)
        for axis_id, axis in new_axes.items():
            location = locations.get(
                ExternalObjectFieldKey(object_kind=ObjectKind.AXIS, object_id=axis_id, field=PackedField.TIME_OFFSETS)
            )
            if not isinstance(axis, IrregularAxis):
                if location is not None:
                    raise TimeFFormatError(f"axis {axis_id!r} is not irregular but stores packed offsets")
                continue
            if location is None or self.external is None:
                raise TimeFFormatError(f"irregular axis {axis_id!r} has no packed offsets")
            handles[axis_id] = self.external.offsets(location, axis_id=axis_id, first=axis.first_us, last=axis.last_us)
        return handles

    def close(self) -> None:
        """Close the reader's DuckDB connection and drop its cached catalog facts."""
        self._disconnect()
        self._clock_cache.clear()
        self._task_facts = None
        self._known_record_ids.clear()
        self._chunk_strings.clear()

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
        return tuple(row[0] for row in self._query("SELECT record_id FROM records ORDER BY record_id"))

    def read_records(
        self,
        record_ids: Iterable[int] | None = None,
        *,
        with_annotations: bool = True,
        shared_offsets: dict[int, StoredOffsets] | None = None,
    ) -> tuple[Record, ...]:
        """Hydrate complete recursive records, with their stored values behind unloaded handles.

        Source links, clocks, axes, and time-offset locations are checked before any object is
        built. No payload file is opened.

        Args:
            record_ids: Requested IDs in result order, or ``None`` for every record.
            with_annotations: Whether to hydrate annotations on records, sources, and signals.
            shared_offsets: Time-offset handles by axis ID, shared with earlier record reads of
                the same call, and extended with this read's new ones. ``None`` shares nothing.

        Returns:
            The hydrated records.

        Raises:
            TimeFValidationError: If a requested ID is not an int.
            TimeFNotFoundError: If a requested record is absent.
            TimeFFormatError: If the stored hierarchy is inconsistent.
        """  # noqa: DOC502 - row selection and hierarchy helpers validate the stored data
        requested = None if record_ids is None else tuple(check_id(item, "record id") for item in record_ids)
        hierarchy = self._read_record_hierarchy(requested, shared_offsets)
        occurrences: dict[int, Annotation] = {}
        annotations = self._read_record_annotations(hierarchy, occurrences) if with_annotations else {}
        signals_by_source = self._read_signals(hierarchy.signals, annotations)
        records = self._assemble_records(hierarchy, signals_by_source, annotations, occurrences)
        self._known_record_ids.update(hierarchy.owners[ObjectKind.RECORD])
        return records

    def _read_record_hierarchy(
        self, requested: tuple[EntityId, ...] | None, shared_offsets: dict[int, StoredOffsets] | None
    ) -> _RecordHierarchy:
        """Read and validate the catalog rows without opening payload files.

        Returns:
            Selected rows, source links, annotation owners, and the signals' value handles.

        Raises:
            TimeFFormatError: If a signal names a missing source.
        """
        read_all = requested is None
        record_rows = self._read_record_rows(requested)
        record_ids = dict.fromkeys(row.record_id for row in record_rows).keys()
        sources_by_id = self._query_rows(_SourceRow, None if read_all else record_ids)
        signals_by_id = self._query_rows(_SignalRow, None if read_all else sources_by_id)
        for row in signals_by_id.values():
            if row.source_id not in sources_by_id:
                raise TimeFFormatError(f"signal {row.signal_id!r} refers to missing source {row.source_id!r}")
        children, roots = self._group_sources(sources_by_id, record_ids)
        self._cache_record_clocks(record_rows)
        return _RecordHierarchy(
            records=record_rows,
            sources=sources_by_id,
            children=children,
            roots_by_record=roots,
            signals=self._prepare_signals(list(signals_by_id.values()), shared_offsets),
            owners={
                ObjectKind.RECORD: record_ids,
                ObjectKind.SOURCE: sources_by_id.keys(),
                ObjectKind.SIGNAL: signals_by_id.keys(),
            },
            read_all=read_all,
        )

    def _read_record_rows(self, requested: tuple[EntityId, ...] | None) -> list[_RecordRow]:
        """Select and validate record rows, preserving the requested order.

        Args:
            requested: Record IDs in result order, or ``None`` for every record.

        Returns:
            Record rows in result order, including repeated requests for the same record.

        Raises:
            TimeFNotFoundError: If a requested record is absent.
        """
        records_by_id = self._query_rows(_RecordRow, requested)
        if requested is None:
            return list(records_by_id.values())
        missing = [record_id for record_id in requested if record_id not in records_by_id]
        if missing:
            raise TimeFNotFoundError(f"no such record(s) in control.duckdb: {missing}")
        return [records_by_id[record_id] for record_id in requested]

    def _prepare_signals(
        self, signal_rows: list[_SignalRow], shared_offsets: dict[int, StoredOffsets] | None
    ) -> list[_PreparedSignal]:
        """Validate axes and create the signals' value handles without reading payloads.

        Returns:
            Signal metadata and value handles, ready to build the signals.

        Raises:
            TimeFFormatError: If a signal refers to a missing axis, or its axis stores a different
                number of time offsets.
        """
        axes = self._read_axes({row.axis_id for row in signal_rows})
        # Signals that share an axis share its handle, so its time offsets are read and checked once.
        time_offsets_by_axis = self._stored_offsets(axes, shared_offsets)
        specs: dict[tuple[Any, ...], TimeSeriesSpec] = {}
        prepared: list[_PreparedSignal] = []
        for row in signal_rows:
            axis = axes.get(row.axis_id)
            if axis is None:
                raise TimeFFormatError(f"signal {row.signal_id!r} refers to missing axis {row.axis_id!r}")
            spec = specs.get(row.spec_fields)
            if spec is None:
                spec = self._spec_from_row(row)
                specs[row.spec_fields] = spec
            time_offsets = time_offsets_by_axis.get(row.axis_id)
            if time_offsets is not None and time_offsets.length != row.n_values:
                raise TimeFFormatError(
                    f"axis {row.axis_id!r} stores {time_offsets.length} offsets but signal "
                    f"{row.signal_id!r} declares {row.n_values} values"
                )
            prepared.append(
                _PreparedSignal(row, axis, spec, self._value_loader_factory(row.signal_id, spec), time_offsets)
            )
        return prepared

    def _read_record_annotations(
        self, hierarchy: _RecordHierarchy, by_occurrence: dict[int, Annotation]
    ) -> dict[tuple[str, int | None], tuple[Annotation, ...]]:
        """Hydrate the annotations on the selected Records, Sources, and Signals.

        A full read fetches every occurrence on those object types, so it also rejects any it cannot
        attach.

        Args:
            hierarchy: The selected rows.
            by_occurrence: Filled with every hydrated occurrence by its occurrence id.

        Returns:
            Attached occurrences keyed by object type and object ID.
        """
        if hierarchy.read_all:
            annotations = self._read_annotations(dict.fromkeys(hierarchy.owners), by_occurrence)
            self._check_owners(annotations.keys(), hierarchy.owners)
            return annotations
        return self._read_annotations(hierarchy.owners, by_occurrence)

    @staticmethod
    def _read_signals(
        prepared: list[_PreparedSignal],
        annotations: Mapping[tuple[str, int | None], tuple[Annotation, ...]],
    ) -> dict[EntityId, list[Signal]]:
        """Build signals from validated metadata and their value handles.

        Returns:
            Signals grouped by their source ID.
        """
        signals_by_source: dict[EntityId, list[Signal]] = defaultdict(list)
        for signal in prepared:
            row = signal.row
            built = Signal.from_loader(
                name=row.name,
                spec=signal.spec,
                time_axis=signal.axis,
                n_values=row.n_values,
                loader=signal.values,
                time_offsets_loader=signal.time_offsets,
                annotations=annotations.get((ObjectKind.SIGNAL, row.signal_id), ()),
                metadata=_decode_json(row.metadata, default={}),
            )
            object.__setattr__(built, "_id", row.signal_id)  # noqa: PLC2801 (Signal is frozen)
            signals_by_source[row.source_id].append(built)
        return signals_by_source

    def _cache_record_clocks(self, rows: list[_RecordRow]) -> None:
        """Cache the selected records' clock origins in one query.

        Raises:
            TimeFFormatError: If a record refers to a missing clock.
        """
        missing_clocks = {row.clock_id for row in rows} - self._clock_cache.keys()
        if missing_clocks:
            for clock_id, timestamp in self._query(
                "SELECT clock_id, start_time_us FROM clocks WHERE clock_id IN (SELECT unnest(?))",
                [list(missing_clocks)],
            ):
                self._clock_cache[clock_id] = TimeOrigin(timestamp)
        if missing_clocks - self._clock_cache.keys():
            raise TimeFFormatError(
                f"records reference missing clocks {sorted(missing_clocks - self._clock_cache.keys())}"
            )

    def _assemble_records(
        self,
        hierarchy: _RecordHierarchy,
        signals_by_source: dict[EntityId, list[Signal]],
        annotations: Mapping[tuple[str, int | None], tuple[Annotation, ...]],
        occurrences: Mapping[int, Annotation],
    ) -> tuple[Record, ...]:
        """Attach signals and annotations to the recursive source trees and records.

        Returns:
            Records in the supplied row order.

        Raises:
            TimeFFormatError: If a stored record span lacks its required end.
        """  # noqa: DOC502 - the span helper rejects missing bounds
        sources_by_id = hierarchy.sources
        children = hierarchy.children
        occurrence_ids = {id(annotation): key for key, annotation in occurrences.items()}

        def hydrate_source(source_id: EntityId) -> Source:
            source_row = sources_by_id[source_id]
            return Source(
                name=source_row.name,
                sources=tuple(hydrate_source(child) for child in children[source_id]),
                signals=tuple(signals_by_source[source_id]),
                annotations=annotations.get((ObjectKind.SOURCE, source_id), ()),
                metadata=_decode_json(source_row.metadata, default={}),
            )

        records: list[Record] = []
        for record_row in hierarchy.records:
            record_id = record_row.record_id
            span = None
            if record_row.time_span_start_us is not None:
                span = TimeInterval(
                    start_us=record_row.time_span_start_us,
                    end_us=_required(record_row.time_span_end_us, f"the time span end of record {record_id!r}"),
                )
            record = Record(
                sources=tuple(hydrate_source(source_id) for source_id in hierarchy.roots_by_record[record_id]),
                start_time=self._clock_cache[record_row.clock_id],
                annotations=annotations.get((ObjectKind.RECORD, record_id), ()),
                time_span=span,
                metadata=_decode_json(record_row.metadata, default={}),
            )
            record._id = record_id
            records.append(record)
            self._record_occurrences[id(record)] = {
                occurrence_ids[id(annotation)]: annotation for annotation in record.walk_annotations()
            }
            weakref.finalize(record, self._record_occurrences.pop, id(record), None)
        return tuple(records)

    def _query_rows(
        self,
        row_type: type[_HierarchyRow],
        related_ids: Iterable[object] | None,
    ) -> dict[EntityId, _HierarchyRow]:
        """Read all rows or only rows related to the supplied IDs, indexed by ID.

        Args:
            row_type: The hierarchy row type, which names the table and its columns.
            related_ids: Values of the table's relation column to keep, or ``None`` for every row.

        Returns:
            Rows from the requested hierarchy table in ID order.

        Raises:
            TimeFFormatError: If an ID occurs more than once.
        """
        table, id_column, relation_column, kind = _HIERARCHY_TABLES[row_type]
        columns = ", ".join(row_type._fields)
        query = (
            f"SELECT {columns} FROM {table} WHERE ? OR {relation_column} "  # noqa: S608 - fixed identifiers
            f"IN (SELECT unnest(?)) ORDER BY {id_column}"
        )
        rows = self._query_named(query, [related_ids is None, list(related_ids or ())], row_type._make)
        indexed = {row.id: row for row in rows}
        if len(indexed) != len(rows):
            raise TimeFFormatError(f"control.duckdb contains duplicate {kind.value.lower()} IDs")
        return indexed

    @staticmethod
    def _group_sources(
        rows: Mapping[EntityId, _SourceRow], record_ids: Collection[EntityId]
    ) -> tuple[dict[EntityId, list[EntityId]], dict[EntityId, list[EntityId]]]:
        """Validate source relationships and group child and root Source IDs.

        Returns:
            Child IDs by parent Source ID and root IDs by Record ID.

        Raises:
            TimeFFormatError: If a source names a missing record or parent, crosses record
                boundaries, or forms a cycle.
        """
        children: dict[EntityId, list[EntityId]] = defaultdict(list)
        roots: dict[EntityId, list[EntityId]] = defaultdict(list)
        parents: dict[EntityId, tuple[EntityId, ...]] = {}
        for row in rows.values():
            if row.record_id not in record_ids:
                raise TimeFFormatError(f"source {row.source_id!r} refers to missing record {row.record_id!r}")
            if row.parent_source_id is None:
                roots[row.record_id].append(row.source_id)
                parents[row.source_id] = ()
            else:
                parent = rows.get(row.parent_source_id)
                if parent is None:
                    raise TimeFFormatError(
                        f"source {row.source_id!r} refers to missing parent source {row.parent_source_id!r}"
                    )
                if parent.record_id != row.record_id:
                    raise TimeFFormatError(f"source {row.source_id!r} crosses record boundaries")
                children[row.parent_source_id].append(row.source_id)
                parents[row.source_id] = (row.parent_source_id,)
        try:
            TopologicalSorter(parents).prepare()
        except CycleError as error:
            raise TimeFFormatError("source hierarchy contains a cycle") from error
        return children, roots

    def read_last_ids(self) -> dict[ObjectKind, int]:
        """Return the last used id per entity kind, as the writer stored it.

        Returns:
            The stored counters. The result does not include a kind without a stored value.

        Raises:
            TimeFFormatError: If a stored counter names an unknown kind or is not an integer.
        """
        rows = self._query("SELECT key, value FROM control_metadata WHERE starts_with(key, ?)", [LAST_ID_PREFIX])
        try:
            return {ObjectKind(key.removeprefix(LAST_ID_PREFIX)): int(value) for key, value in rows}
        except ValueError as exc:
            raise TimeFFormatError("control.duckdb stores an invalid last id counter") from exc

    def read_dataset_annotations(
        self, dataset_id: str, *, by_occurrence: dict[int, Annotation] | None = None
    ) -> tuple[Annotation, ...]:
        """Return annotations attached directly to a dataset, as new objects on every call.

        Args:
            dataset_id: The dataset whose annotations to read.
            by_occurrence: Filled with every annotation by its occurrence id, so that a task read
                of the same call can reference these objects.

        Returns:
            Dataset-level annotation occurrences in storage order.
        """
        if not self._query("SELECT 1 FROM datasets WHERE dataset_id = ?", [dataset_id]):
            return ()
        occurrences: dict[int, Annotation] = {} if by_occurrence is None else by_occurrence
        self._read_annotations({ObjectKind.DATASET: None}, occurrences)
        return tuple(occurrences.values())

    def chunk_location(self, signal_id: int) -> ChunkLocation:
        """Return where one Signal's chunks are stored, from its catalog rows.

        Raises:
            TimeFFormatError: If the Signal has no stored chunks.
        """
        location = self.chunk_locations_by_ids((signal_id,))[signal_id]
        if location is None:
            raise TimeFFormatError(f"signal {signal_id!r} has no stored value chunks")
        return location

    def chunk_locations_by_ids(self, signal_ids: Iterable[int]) -> dict[int, ChunkLocation | None]:
        """Return where a batch of Signals' chunks are stored, from their catalog rows.

        Args:
            signal_ids: Signal IDs.

        Returns:
            Locations keyed by Signal ID, each with the Signal's chunks in chunk-index order, or
            ``None`` for a Signal without stored chunks.

        Raises:
            TimeFFormatError: If one Signal's chunks are stored by more than one release.
        """
        ids = tuple(signal_ids)
        if not ids:
            return {}
        columns = ", ".join(_ChunkRow._fields)
        rows = self._query_named(
            f"""SELECT {columns} FROM signal_chunks
               WHERE signal_id IN (SELECT unnest(?))
               ORDER BY signal_id, chunk_index""",  # noqa: S608 - fixed column names
            [list(ids)],
            _ChunkRow._make,
        )
        chunks: dict[int, list[StoredChunk]] = {signal_id: [] for signal_id in ids}
        sources: dict[int, SourceKey | None] = {}
        strings = self._chunk_strings
        for row in rows:
            # The writer stores or reuses a Signal's chunks as a unit, so one release stores them all.
            if sources.setdefault(row.signal_id, row.source_key) != row.source_key:
                raise TimeFFormatError(f"signal {row.signal_id!r} has chunks stored by more than one release")
            chunks[row.signal_id].append(row.stored_chunk(strings))
        return {
            signal_id: ChunkLocation(source=_shared_source(sources[signal_id], strings), chunks=tuple(stored))
            if stored
            else None
            for signal_id, stored in chunks.items()
        }

    def imported_ids(
        self, alias: str, kind: ObjectKind, parent_ids: Iterable[EntityId]
    ) -> tuple[str | None, dict[EntityId, EntityId]]:
        """Return the parent release of an input alias and the local IDs of objects imported under it.

        Args:
            alias: The input alias the objects were imported under.
            kind: The kind of the objects.
            parent_ids: The objects' IDs in the parent release.

        Returns:
            The parent release the catalog names for the alias, or ``None`` if it names none, and
            the local object ID of each parent ID the catalog knows.
        """
        found = self._query("SELECT parent_ref FROM imports WHERE alias = ?", [alias])
        if not found:
            return None, {}
        rows = self._query(
            "SELECT parent_id, object_id FROM imported_objects "
            "WHERE alias = ? AND object_kind = ? AND parent_id IN (SELECT unnest(?))",
            [alias, kind.value, list(parent_ids)],
        )
        return found[0][0], dict(rows)

    def signal_record_ids(self, signal_ids: Collection[EntityId]) -> dict[EntityId, EntityId]:
        """Return the ID of the Record that holds each of the given Signals, building no object.

        Args:
            signal_ids: The Signals' IDs.

        Returns:
            The Record ID of each Signal, by Signal ID.

        Raises:
            TimeFNotFoundError: If an ID names no stored Signal.
        """  # noqa: DOC502 - raised by _require_found
        owners: dict[EntityId, EntityId] = dict(
            self._query(
                "SELECT signals.signal_id, sources.record_id FROM signals JOIN sources USING (source_id) "
                "WHERE signals.signal_id IN (SELECT unnest(?))",
                [list(signal_ids)],
            )
        )
        _require_found(ObjectKind.SIGNAL, signal_ids, owners)
        return owners

    def read_signals_by_id(self, signal_ids: Collection[EntityId]) -> dict[EntityId, Signal]:
        """Build the Records that hold the given Signals, and return those Signals.

        Each Record is built once, however many of its Signals are asked for, and the Records of
        one call share one time-offsets handle per axis.

        Args:
            signal_ids: The Signals' IDs.

        Returns:
            The Signals by ID, each inside its Record, with stored values behind unloaded handles.

        Raises:
            TimeFNotFoundError: If an ID names no stored Signal.
        """  # noqa: DOC502 - raised by signal_record_ids
        owners = self.signal_record_ids(signal_ids)
        signals: dict[EntityId, Signal] = {}
        for record in self.read_records(list(dict.fromkeys(owners.values()))):
            signals.update((signal.id, signal) for signal in record.signals if signal.id in owners)
        return signals

    def read_tasks_by_id(self, task_ids: Collection[EntityId]) -> dict[EntityId, Task]:
        """Build the given tasks, sharing the Records they reference, a bounded batch at a time.

        The tasks they derive from through ``from_tasks`` are built too.

        Args:
            task_ids: The task IDs, each once.

        Returns:
            The tasks by ID.

        Raises:
            TimeFNotFoundError: If an ID names no stored task.
        """  # noqa: DOC502 - raised by _require_found
        stored = {
            task_id
            for (task_id,) in self._query(
                "SELECT task_id FROM tasks WHERE task_id IN (SELECT unnest(?))", [list(task_ids)]
            )
        }
        _require_found(ObjectKind.TASK, task_ids, stored)
        tasks = _TaskHydration(self, None).read_ids(list(task_ids))
        return {task.id: task for task in tasks}

    def read_tasks(  # noqa: PLR0913 - one argument per task selection axis
        self,
        records: Iterable[Record] | None = None,
        *,
        cached_records: Iterable[Record] = (),
        dataset_annotations: Mapping[int, Annotation] | None = None,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
    ) -> tuple[Task, ...]:
        """Hydrate every selected task and restore all in-memory object references.

        Args:
            records: Input Records to select, or ``None`` for every task. Tasks reuse these objects.
            cached_records: Hydrated Records to reuse without restricting the selection.
            dataset_annotations: The dataset's hydrated annotations by occurrence id, which tasks
                then reference instead of their own copies, or ``None`` to read them when a task
                needs them.
            split: Keep only tasks of this split, or ``None`` for every task.

        Returns:
            Concrete tasks in stable ID order.

        Raises:
            TimeFFormatError: If a relationship refers to a missing object or has an invalid field.
        """  # noqa: DOC502 - raised by _TaskHydration
        hydration = _TaskHydration(
            self,
            None if records is None else tuple(records),
            cached_records=cached_records,
            dataset_annotations=dataset_annotations,
            required_modalities=required_modalities,
            supported_modalities=supported_modalities,
            split=split,
        )
        return tuple(hydration.read_all())

    def iter_task_batches(
        self,
        records: Iterable[Record] | None = None,
        *,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
        batch_size: int | None = None,
    ) -> Iterator[list[Task]]:
        """Yield the tasks whose inputs include one of ``records``, a bounded batch at a time.

        Each batch runs a handful of keyed queries and builds only that batch's tasks, so a caller
        that walks one Record at a time never holds the whole task table in memory. Only ``None``
        skips the input Record filter; an empty collection selects no tasks.

        Each batch is built on its own. The parents its tasks derive from through ``from_tasks``,
        and the Records, Signals, and annotations they refer to outside ``records``, are built for
        the batch and shared only among its tasks, so nothing of a yielded batch stays with the
        stream.

        Args:
            records: Input Records to select, or ``None`` for every task. Tasks reuse these objects.
            split: Keep only tasks of this split, or ``None`` for every task.
            batch_size: The most tasks a batch holds, or ``None`` for 5,000.

        Yields:
            Each batch's concrete tasks, in stable ID order across batches.

        Raises:
            TimeFFormatError: If a relationship refers to a missing object or has an invalid field.
        """  # noqa: DOC502 - raised by _TaskHydration
        hydration = _TaskHydration(
            self,
            None if records is None else tuple(records),
            required_modalities=required_modalities,
            supported_modalities=supported_modalities,
            split=split,
            batch_size=batch_size,
        )
        yield from hydration.iter_batches()

    def task_facts(self) -> _TaskFacts:
        """Return the per-database facts task hydration needs, read once per reader.

        Returns:
            The IDs of tasks other tasks derive from, and whether any task carries annotations.

        Raises:
            TimeFFormatError: If two task rows share an ID.
        """
        if self._task_facts is None:
            if self._query("SELECT 1 FROM tasks GROUP BY task_id HAVING count(*) > 1 LIMIT 1"):
                raise TimeFFormatError("control.duckdb contains duplicate task IDs")
            parent_ids = {row[0] for row in self._query("SELECT DISTINCT parent_task_id FROM task_dependencies")}
            ((annotated,),) = self._query(
                "SELECT count(*) FROM annotation_occurrences WHERE object_type = ?", [ObjectKind.TASK]
            )
            self._task_facts = _TaskFacts(frozenset(parent_ids), annotated > 0)
        return self._task_facts

    def read_task_table(
        self,
        record_ids: Iterable[int] | None = None,
        *,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
    ) -> pa.Table:
        """Return the task table as Arrow, with object references as public IDs, without building Tasks.

        The table has one row per task. Each row holds the type, prompt, rationale, the four scope
        columns, the configuration columns, ordered ``input_record_ids``, ordered ``from_task_ids``,
        and the metadata JSON string. Targets are in :meth:`read_target_table`.

        Args:
            record_ids: Keep only tasks whose inputs include one of these Records, or ``None`` for all.
            split: Keep only tasks of this split, or ``None`` for all.

        Returns:
            The tasks in task ID order.
        """
        where, parameters = self._selection_filter(record_ids, required_modalities, supported_modalities, split)
        return self._query_table(
            f"""WITH inputs AS (
                    SELECT task_id, list(record_id ORDER BY position) AS input_record_ids
                    FROM task_record_refs
                    WHERE field = '{TaskRelationshipField.INPUTS}'
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
        )

    def read_target_table(
        self,
        record_ids: Iterable[int] | None = None,
        *,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
    ) -> pa.Table:
        """Return every inline target as Arrow, one typed row each, with references as public IDs.

        The columns follow :class:`timenet.format.task_target_row.TargetValueRow`: a step span names
        its one series in ``span_signal_ids``.

        Args:
            record_ids: Keep only targets of tasks whose inputs include one of these Records.
            split: Keep only targets of tasks of this split, or ``None`` for all.

        Returns:
            Targets in task ID and position order.
        """
        where, parameters = self._selection_filter(record_ids, required_modalities, supported_modalities, split)
        return self._query_table(
            f"""SELECT t.task_id, x.position, x.target_kind, x.text_value, x.integer_value, x.float_value,
                       x.boolean_value, x.record_id, x.signal_id, x.span_start, x.span_end, x.span_signal_ids
                FROM task_targets x
                JOIN tasks t USING (task_id)
                {where}
                ORDER BY t.task_id, x.position""",  # noqa: S608 - fixed fragments
            parameters,
        )

    def read_annotation_table(self, object_type: ObjectKind | None = None) -> pa.Table:
        """Return every annotation occurrence joined with its content as Arrow.

        Args:
            object_type: Keep only occurrences on this object type, or ``None`` for all.

        Returns:
            One row per occurrence. Each row has the annotated object's type and ID, the content's
            ID and key, typed value columns, unit, span columns, confidence, and JSON metadata. The
            object ID is NULL for the Dataset, which has a string ``dataset_id`` instead.

        Raises:
            TimeFFormatError: If ``object_type`` is not an annotated object type.
        """
        if object_type is not None and object_type not in ANNOTATED_OBJECT_TYPES:
            raise TimeFFormatError(f"unknown annotated object type {object_type!r}")
        where = "" if object_type is None else "WHERE o.object_type = ?"
        return self._query_table(
            f"""SELECT o.occurrence_id, o.object_type, o.object_id, c.content_id, c.name AS key,
                       c.value_kind, c.text_value, c.integer_value, c.float_value, c.boolean_value,
                       c.text_list_value, c.unit, c.description, o.span_type, o.start_us, o.end_us,
                       o.confidence, o.provenance, c.metadata AS content_metadata, o.metadata
                FROM annotation_occurrences o
                JOIN annotation_contents c USING (content_id)
                {where}
                ORDER BY o.object_type, o.object_id, o.occurrence_id""",  # noqa: S608 - fixed fragments
            [] if object_type is None else [object_type],
        )

    @staticmethod
    def _record_filter(record_ids: Iterable[int] | None) -> tuple[str, list[object]]:
        """Return the WHERE clause that keeps tasks whose inputs include one of ``record_ids``.

        Returns:
            The SQL fragment, empty for every task, and its parameters.
        """
        if record_ids is None:
            return "", []
        return (
            f"""WHERE t.task_id IN (
                   SELECT task_id FROM task_record_refs
                   WHERE field = '{TaskRelationshipField.INPUTS}' AND record_id IN (SELECT unnest(?))
               )""",  # noqa: S608 - fixed field name
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

    def _selected_task_ids(  # noqa: PLR0913 - one argument per task selection axis, and the page
        self,
        record_ids: list[int] | None,
        required_modalities: Iterable[InputModality] | None,
        supported_modalities: Iterable[InputModality] | None,
        split: Split | str | None,
        *,
        after: int | None,
        limit: int,
    ) -> list[int]:
        """Return one page of the IDs of the selected tasks, in task ID order, in one query.

        The query's result is read in full, so no result stays open while the caller builds the
        page's tasks.

        Args:
            record_ids: IDs of the Records whose input tasks to select, or ``None`` for every task.
            split: Keep only tasks of this split, or ``None`` for every task.
            after: The last task ID of the page before, or ``None`` for the first page.
            limit: The most IDs the page holds.

        Returns:
            The page's task IDs.
        """
        where, params = self._selection_filter(record_ids, required_modalities, supported_modalities, split)
        if after is not None:
            where = f"{where} AND t.task_id > ?" if where else "WHERE t.task_id > ?"
            params = [*params, after]
        sql = f"SELECT t.task_id FROM tasks t {where} ORDER BY t.task_id LIMIT ?"  # noqa: S608 - fixed predicates
        return [task_id for (task_id,) in self._query(sql, [*params, limit])]

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
        return self._query_named(
            f"SELECT {columns} FROM tasks {where} ORDER BY task_id",  # noqa: S608 - fixed identifiers
            parameters,
            _TaskRow._make,
        )

    def _target_rows(self, task_ids: list[int] | None) -> list[_TargetRow]:
        """Read the target rows for one batch of tasks, or every task, in task and position order.

        Returns:
            The target rows.
        """
        columns = ", ".join(_TargetRow._fields)
        where, parameters = self._task_filter(task_ids)
        return self._query_named(
            f"SELECT {columns} FROM task_targets {where} ORDER BY task_id, position",  # noqa: S608 - fixed identifiers
            parameters,
            _TargetRow._make,
        )

    def _missing_ids(self, table: str, column: str, ids: Collection[int]) -> list[int]:
        """Return the ids that name no row of ``table``.

        Returns:
            The missing ids in ascending order.
        """
        if not ids:
            return []
        stored = {
            row[0]
            for row in self._query(
                f"SELECT {column} FROM {table} WHERE {column} IN (SELECT unnest(?))",  # noqa: S608 - fixed identifiers
                [list(ids)],
            )
        }
        return sorted(set(ids) - stored)

    def _task_relationships(
        self, table: str, column: str, fields: frozenset[TaskRelationshipField], task_ids: list[int] | None
    ) -> dict[int, dict[str, tuple[int, ...]]]:
        """Read one ordered task relationship table for a batch, or for every task.

        Args:
            table: ``task_record_refs`` or ``task_annotation_refs``.
            column: The referenced ID column of that table.
            fields: The fields that the table can hold.
            task_ids: The batch, or ``None`` for the whole table.

        Returns:
            ``task_id -> field -> ordered referenced IDs``.

        Raises:
            TimeFFormatError: If a row names a field that the table cannot hold.
        """
        where, parameters = self._task_filter(task_ids)
        rows = self._query(
            f"SELECT task_id, field, {column} FROM {table} {where} ORDER BY task_id, field, position",  # noqa: S608
            parameters,
        )
        grouped = self._group_relationships(rows)
        unknown = {field for refs in grouped.values() for field in refs} - fields
        if unknown:
            raise TimeFFormatError(f"{table} has unknown fields {sorted(unknown)}")
        return grouped

    def _task_dependencies(self, task_ids: list[int] | None) -> dict[int, tuple[int, ...]]:
        """Read the ordered parent IDs of each task in one batch, or of every task.

        Returns:
            ``task_id -> ordered parent task IDs`` for tasks that have parents.
        """
        where, parameters = self._task_filter(task_ids)
        rows = self._query(
            f"SELECT task_id, parent_task_id FROM task_dependencies {where} ORDER BY task_id, position",  # noqa: S608
            parameters,
        )
        parents: dict[int, list[int]] = defaultdict(list)
        for task_id, parent_id in rows:
            parents[task_id].append(parent_id)
        return {task_id: tuple(ids) for task_id, ids in parents.items()}

    @staticmethod
    def _group_relationships(
        rows: list[tuple[int, str, int]],
    ) -> dict[int, dict[str, tuple[int, ...]]]:
        """Group ordered relationship query rows.

        Returns:
            Relationship values grouped by task ID and stored field name.
        """
        grouped: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for task_id, field, value in rows:
            grouped[task_id][field].append(value)
        return {task_id: {field: tuple(ids) for field, ids in fields.items()} for task_id, fields in grouped.items()}

    @staticmethod
    def _resolve(
        task_id: int,
        field: TaskRelationshipField,
        refs: dict[str, tuple[int, ...]],
        objects: Mapping[int, _Ref],
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
                f"task {task_id!r} field {field.value!r} refers to missing {label} {exc.args[0]!r}"
            ) from exc

    def _read_axes(
        self,
        axis_ids: Iterable[int] | None = None,
    ) -> dict[int, TimeAxis]:
        """Hydrate each shared axis exactly once.

        Returns:
            Axes keyed by axis ID.

        Raises:
            TimeFFormatError: If an axis has an unknown type, or an irregular axis has no Signals or
                Signals of different lengths.
        """
        requested = None if axis_ids is None else tuple(axis_ids)
        if requested == ():
            return {}
        axes: dict[int, TimeAxis] = {}
        query = """SELECT axis_id, axis_type, period_numerator_us, period_denominator,
                          start_index, offset_us, first_us, last_us,
                          lengths.min_n_values, lengths.max_n_values
                   FROM axes
                   LEFT JOIN (
                       SELECT axis_id, min(n_values) AS min_n_values, max(n_values) AS max_n_values
                       FROM signals GROUP BY axis_id
                   ) lengths USING (axis_id)"""
        rows = (
            self._query(query)
            if requested is None
            else self._query(f"{query} WHERE axis_id IN (SELECT unnest(?))", [list(requested)])
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
                regular = RegularAxis(
                    period_us=Fraction(numerator, denominator), start_index=start_index, offset_us=offset
                )
                axes[axis_id] = regular
            elif axis_type == "irregular":
                if min_n_values is None or max_n_values is None:
                    raise TimeFFormatError(f"irregular axis {axis_id!r} has no Signals")
                if min_n_values != max_n_values:
                    raise TimeFFormatError(
                        f"axis {axis_id!r} is shared by Signals with different lengths "
                        f"{sorted((min_n_values, max_n_values))}"
                    )
                axes[axis_id] = IrregularAxis(first_us=first, last_us=last)
            elif axis_type == "ordinal":
                axes[axis_id] = OrdinalAxis()
            else:
                raise TimeFFormatError(f"axis {axis_id!r} has unknown type {axis_type!r}")
        return axes

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

    def _check_owners(
        self,
        annotated: Iterable[tuple[str, int | None]],
        owners: Mapping[ObjectKind, Collection[int]],
    ) -> None:
        """Reject occurrences that a full read fetched but cannot attach.

        Args:
            annotated: The ``(object_type, object_id)`` owners that the fetched occurrences name.
            owners: The hydrated IDs of each object type the read fetched in full.

        Raises:
            TimeFFormatError: If an occurrence names an unknown object type, or an ID outside the
                hydrated set of its type.
        """
        unknown = self._query(
            "SELECT occurrence_id, object_type FROM annotation_occurrences "
            "WHERE object_type NOT IN (SELECT unnest(?)) LIMIT 1",
            [sorted(ANNOTATED_OBJECT_TYPES)],
        )
        if unknown:
            ((occurrence_id, object_type),) = unknown
            raise TimeFFormatError(f"annotation occurrence {occurrence_id!r} has unknown object type {object_type!r}")
        for object_type, object_id in annotated:
            if object_id not in owners[ObjectKind(object_type)]:
                raise TimeFFormatError(f"annotation occurrence refers to missing {object_type} {object_id!r}")

    def _read_annotations(
        self,
        owners: Mapping[ObjectKind, Collection[int] | None],
        by_occurrence: MutableMapping[int, Annotation] | None = None,
    ) -> dict[tuple[str, int | None], tuple[Annotation, ...]]:
        """Hydrate annotation content and occurrences.

        Args:
            owners: IDs of the annotated objects by object type. If the IDs of a type are ``None``,
                the read gets every occurrence of that type. IDs match only within their type. The
                Dataset's occurrences have no object ID, so its IDs are occurrence IDs.
            by_occurrence: Filled with every hydrated occurrence by its occurrence id.

        Returns:
            Attached occurrences keyed by object type and object ID. The object ID is ``None`` for
            the Dataset.

        Raises:
            TimeFFormatError: If an occurrence has unknown content, an owner ID that does not fit its
                type, or an unknown span type.
        """
        owners = {object_type: ids for object_type, ids in owners.items() if ids is None or ids}
        if not owners:
            return {}
        query, parameters = _annotation_query(owners)
        rows = self._query_named(query, parameters, _OccurrenceRow._make)
        external = self._external_values(ObjectKind.ANNOTATION_CONTENT, {row.content_id for row in rows})
        grouped: dict[tuple[str, int | None], list[Annotation]] = defaultdict(list)
        for row in rows:
            if row.name is None:
                raise TimeFFormatError(
                    f"annotation occurrence {row.occurrence_id!r} refers to missing content {row.content_id!r}"
                )
            if (row.object_type == ObjectKind.DATASET) != (row.object_id is None):
                raise TimeFFormatError(
                    f"annotation occurrence {row.occurrence_id!r} on {row.object_type} has object id {row.object_id!r}"
                )
            handle = external.get(
                ExternalObjectFieldKey(
                    object_kind=ObjectKind.ANNOTATION_CONTENT, object_id=row.content_id, field=PackedField.VALUE
                )
            )
            annotation = Annotation(
                key=row.name,
                value=handle
                if handle is not None
                else decode_annotation_value(
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
            if by_occurrence is not None:
                by_occurrence[row.occurrence_id] = annotation
        return {key: tuple(value) for key, value in grouped.items()}


class _Lookups(NamedTuple):
    """The caller's Records, their Signals, and the annotations a stream gives every batch to look up.

    Batches write what they build to lookups of their own in front of these. Only the first batch
    whose targets need Signals adds the caller's Records' Signals here, and only the first batch
    that needs an annotation of a Record another reader built matches those Records here, once for
    the stream.
    """

    records_by_id: MutableMapping[EntityId, Record]
    signals_by_id: MutableMapping[EntityId, Signal]
    unindexed: list[Record]
    """The caller's Records whose Signals are not in ``signals_by_id`` yet."""
    annotations_by_id: MutableMapping[int, Annotation]
    foreign: list[Record]
    """The caller's Records that another reader built, whose annotations are not matched yet."""
    dataset_annotations_indexed: bool


class _TaskHydration:
    """State for one task read of a :class:`DuckDBControlReader`.

    It holds the caller's Records and the lookups built from them. Within one scope, it builds the
    Records, Signals, and annotations that tasks refer to outside that set once, and keeps every
    task that another task derives from until its dependants are built. A one-pass read is one
    scope. A stream starts a new scope for each batch, so its batches share only the caller's
    objects.
    """

    def __init__(  # noqa: PLR0913 - one argument per task selection axis
        self,
        reader: DuckDBControlReader,
        records: tuple[Record, ...] | None,
        *,
        cached_records: Iterable[Record] = (),
        dataset_annotations: Mapping[int, Annotation] | None = None,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Index the caller's Records and decide whether the task table needs filtering.

        Raises:
            TimeFValidationError: If ``batch_size`` is not a positive int.
            TimeFNotFoundError: If one of ``records`` is not stored in the control database.
        """
        if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
            raise TimeFValidationError(f"batch_size must be a positive int, got {batch_size!r}")
        self.reader = reader
        self.required_modalities = required_modalities
        self.supported_modalities = supported_modalities
        self.split = split
        self.batch_size = _TASK_BATCH_SIZE if batch_size is None else batch_size
        self.records_by_id: MutableMapping[EntityId, Record] = {}
        # Records one scope reads share one time-offsets handle per axis.
        self.offsets_by_axis: dict[int, StoredOffsets] = {}
        self.signals_by_id: MutableMapping[EntityId, Signal] = {}
        self._records_without_signal_index: list[Record] = []
        self.annotations_by_id: MutableMapping[int, Annotation] = {}
        self._foreign_records: list[Record] = []
        self.retained: dict[int, Task] = {}
        self._hydrating: set[int] = set()
        # The lookups of the caller's objects that a stream's batches share, or None for one pass.
        self._given: _Lookups | None = None
        self._dataset_annotations_indexed = dataset_annotations is not None
        self.annotations_by_id.update(dataset_annotations or {})
        facts = reader.task_facts()
        self.parent_ids = facts.parent_task_ids
        self.tasks_annotated = facts.tasks_annotated
        given = {record.id: record for record in cached_records}
        given.update((record.id, record) for record in records or ())
        missing = reader._missing_ids("records", "record_id", given.keys() - reader._known_record_ids)
        if missing:
            raise TimeFNotFoundError(f"no such record(s) in control.duckdb: {missing}")
        for record in given.values():
            self._index_record(record)
        self.record_ids: list[int] | None = None if records is None else [record.id for record in records]

    def iter_batches(self) -> Iterator[list[Task]]:
        """Yield the selected tasks in task ID order, one bounded batch at a time, each in its own scope.

        The tasks of a batch share the objects the batch builds, and no batch shares them with
        another. The hydration forgets a batch's objects before it yields them, so it holds nothing
        of a batch it has handed out.

        Yields:
            Each batch's hydrated tasks.
        """
        given = self._share_given()
        self._start_scope(given)
        for task_ids in self._pages():
            yield self._hydrate_scope(task_ids, given)

    def read_ids(self, task_ids: list[int]) -> list[Task]:
        """Build the tasks with these IDs, each given once, in one scope, a bounded batch at a time.

        Returns:
            The tasks in the order of ``task_ids``.
        """
        return [
            task
            for start in range(0, len(task_ids), _TASK_BATCH_SIZE)
            for task in self._hydrate(task_ids[start : start + _TASK_BATCH_SIZE])
        ]

    def read_all(self) -> list[Task]:
        """Build all selected tasks in one scope, using batches for filtered reads.

        Returns:
            The tasks in task ID order.
        """
        unfiltered = self.required_modalities is None and self.supported_modalities is None and self.split is None
        if self.record_ids is None and unfiltered:
            return self._hydrate(None)
        return [task for task_ids in self._pages() for task in self._hydrate(task_ids)]

    def _pages(self) -> Iterator[list[int]]:
        """Select the tasks a batch at a time and yield their IDs in task ID order.

        Each page is a query of its own that starts after the last ID of the page before, so no
        result stays open while a batch is built, and a connection closed between batches, as a
        fork closes it, opens again for the next page.

        Yields:
            Each batch's task IDs.
        """
        after = None
        while True:
            ids = self.reader._selected_task_ids(
                self.record_ids,
                self.required_modalities,
                self.supported_modalities,
                self.split,
                after=after,
                limit=self.batch_size,
            )
            if ids:
                yield ids
            if len(ids) < self.batch_size:
                return
            after = ids[-1]

    def _share_given(self) -> _Lookups:
        """Keep the lookups of the caller's objects, as they stand before any batch, for every batch.

        Returns:
            The lookups, which the batches then look through.
        """
        self._given = _Lookups(
            self.records_by_id,
            self.signals_by_id,
            self._records_without_signal_index,
            self.annotations_by_id,
            self._foreign_records,
            self._dataset_annotations_indexed,
        )
        return self._given

    def _start_scope(self, given: _Lookups) -> None:
        """Forget what earlier batches built, and look up the caller's objects in ``given``."""
        self.records_by_id = ChainMap({}, given.records_by_id)
        self.offsets_by_axis = {}
        self.signals_by_id = ChainMap({}, given.signals_by_id)
        self._records_without_signal_index = []
        self.annotations_by_id = ChainMap({}, given.annotations_by_id)
        # Records a batch builds come from this reader, so only the caller's can need matching.
        self._foreign_records = given.foreign
        self._dataset_annotations_indexed = given.dataset_annotations_indexed
        self.retained = {}

    def _hydrate_scope(self, task_ids: list[int], given: _Lookups) -> list[Task]:
        """Build one batch in a scope of its own, which ends when the batch is built.

        Returns:
            The batch's tasks in the order of ``task_ids``.
        """
        try:
            return self._hydrate(task_ids)
        finally:
            self._start_scope(given)

    def _hydrate(self, task_ids: list[int] | None) -> list[Task]:
        """Build one batch, or the whole table for ``None``, hydrating unbuilt parents first.

        Returns:
            The tasks in the order of ``task_ids``, or in task ID order for the whole table.

        Raises:
            TimeFFormatError: If the stored derivations contain a cycle.
        """
        pending = None if task_ids is None else [task_id for task_id in task_ids if task_id not in self.retained]
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
        built = self._build(pending)
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

    def _build(self, task_ids: list[int] | None) -> dict[int, Task]:  # noqa: PLR0914 - one batch joins several tables
        """Read every table for one batch of task IDs and construct the tasks.

        Returns:
            The constructed tasks by ID, without ``from_tasks`` wired.

        Raises:
            TimeFFormatError: If a task has an unknown type.
        """
        if task_ids is not None and not task_ids:
            return {}
        reader = self.reader
        task_rows = reader._task_rows(task_ids)
        external = reader._external_values(ObjectKind.TASK, (row.task_id for row in task_rows))
        record_refs = reader._task_relationships("task_record_refs", "record_id", RECORD_FIELDS, task_ids)
        annotation_refs = reader._task_relationships(
            "task_annotation_refs", "occurrence_id", ANNOTATION_FIELDS, task_ids
        )
        target_rows = reader._target_rows(task_ids)
        task_annotations: dict[tuple[str, int | None], tuple[Annotation, ...]] = {}
        if self.tasks_annotated:
            task_annotations = reader._read_annotations({ObjectKind.TASK: task_ids})
        if task_ids is None:
            reader._check_owners(task_annotations.keys(), {ObjectKind.TASK: {row.task_id for row in task_rows}})

        self._ensure_records(
            {record_id for refs in record_refs.values() for ids in refs.values() for record_id in ids}
            | {row.record_id for row in target_rows if row.record_id is not None}
        )
        self._ensure_signals(
            {row.signal_id for row in target_rows if row.signal_id is not None}
            | {signal_id for row in target_rows for signal_id in row.span_signal_ids or ()}
            | {signal_id for row in task_rows for signal_id in row.scope_signal_ids or ()}
        )
        annotations_by_occurrence = self._annotations_by_occurrence(
            {occurrence_id for refs in annotation_refs.values() for ids in refs.values() for occurrence_id in ids}
        )

        targets_by_task: dict[int, list[object]] = defaultdict(list)
        for target in target_rows:
            values = target._asdict()
            task_id = values.pop("task_id")
            # A locator names the target's stored position, which need not be its index in the task.
            position = values.pop("position")
            handle = external.get(
                ExternalObjectFieldKey(
                    object_kind=ObjectKind.TASK, object_id=task_id, field=PackedField.TARGETS, position=position
                )
            )
            targets_by_task[task_id].append(
                handle
                if handle is not None
                else decode_target(values, records=self.records_by_id, signals=self.signals_by_id)
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
                "inputs": reader._resolve(
                    row.task_id, TaskRelationshipField.INPUTS, refs, self.records_by_id, "record"
                ),
                "targets": tuple(targets_by_task.get(row.task_id, ())) if row.has_inline_targets else None,
                "prompt": external.get(
                    ExternalObjectFieldKey(
                        object_kind=ObjectKind.TASK, object_id=row.task_id, field=PackedField.PROMPT
                    ),
                    row.prompt,
                ),
                "input_modalities": decode_input_modalities(row.input_modalities),
                "split": decode_split(row.split),
                "scope": decode_span(
                    row.scope_type, row.scope_start, row.scope_end, row.scope_signal_ids, signals=self.signals_by_id
                ),
                "rationale": external.get(
                    ExternalObjectFieldKey(
                        object_kind=ObjectKind.TASK, object_id=row.task_id, field=PackedField.RATIONALE
                    ),
                    row.rationale,
                ),
                "annotations": task_annotations.get((ObjectKind.TASK, row.task_id), ()),
                "metadata": _decode_json(row.metadata, default={}),
                **decode_task_payload(task_type, row.payload_columns),
            }
            if TaskRelationshipField.CANDIDATE_RECORDS in cls.__dataclass_fields__:
                kwargs["candidate_records"] = reader._resolve(
                    row.task_id, TaskRelationshipField.CANDIDATE_RECORDS, refs, self.records_by_id, "record"
                )
            task_annotation_refs = annotation_refs.get(row.task_id, {})
            kwargs["input_annotations"] = reader._resolve(
                row.task_id,
                TaskRelationshipField.INPUT_ANNOTATIONS,
                task_annotation_refs,
                annotations_by_occurrence,
                "annotation occurrence",
            )
            kwargs["target_annotations"] = reader._resolve(
                row.task_id,
                TaskRelationshipField.TARGET_ANNOTATIONS,
                task_annotation_refs,
                annotations_by_occurrence,
                "annotation occurrence",
            )
            task = cls(**kwargs)
            task._id = row.task_id
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
        occurrences = self.reader._record_occurrences.get(id(record))
        if occurrences is None:
            self._foreign_records.append(record)
        else:
            self.annotations_by_id.update(occurrences)

    def _match_foreign_occurrences(self) -> None:
        """Find the occurrence ids of the Records that another reader instance hydrated, once.

        The method reads these Records again from storage only once. A fresh read builds their
        annotations in the same order, so the two walks line up one to one. Only the caller's
        Records can be foreign, so a stream matches them once, into the lookups every batch shares.
        """
        foreign = list(self._foreign_records)
        self._foreign_records.clear()
        if not foreign:
            return
        given = self._given
        index = self.annotations_by_id if given is None else given.annotations_by_id
        fresh_records = self.reader.read_records([record.id for record in foreign])
        for record, fresh in zip(foreign, fresh_records, strict=True):
            index.update(self._match_occurrences(record, fresh))

    def _match_occurrences(self, record: Record, fresh: Record) -> dict[int, Annotation]:
        """Pair the annotations of ``record`` with the annotations of ``fresh``.

        ``fresh`` is the same Record, read again from storage.

        Returns:
            The annotations of ``record`` by occurrence id. The result is empty if any pair differs
            in key, span or content.
        """
        ids = {
            id(annotation): occurrence_id
            for occurrence_id, annotation in self.reader._record_occurrences[id(fresh)].items()
        }
        stored = list(fresh.walk_annotations())
        passed = list(record.walk_annotations())
        if len(stored) != len(passed) or any(
            original._content_key != annotation._content_key or original.span != annotation.span
            for original, annotation in zip(stored, passed, strict=True)
        ):
            return {}
        return {ids[id(original)]: annotation for original, annotation in zip(stored, passed, strict=True)}

    def _ensure_records(self, record_ids: set[int]) -> None:
        """Hydrate Records a task refers to that the caller did not pass.

        Raises:
            TimeFFormatError: If an ID names no stored Record.
        """
        missing = sorted(record_ids - self.records_by_id.keys())
        if not missing:
            return
        try:
            hydrated = self.reader.read_records(missing, shared_offsets=self.offsets_by_axis)
        except TimeFValidationError as exc:
            raise TimeFFormatError(f"task refers to a missing record: {exc}") from exc
        for record in hydrated:
            self._index_record(record)

    def _ensure_signals(self, signal_ids: set[int]) -> None:
        """Index the Signals of the hydrated Records when one batch's targets or scopes refer to Signals.

        Args:
            signal_ids: Every Signal ID the batch refers to.

        Raises:
            TimeFFormatError: If an ID names no stored Signal, or no Signal on the hydrated Records.
        """
        if not signal_ids:
            return
        self._index_signals()
        outside = sorted(signal_ids - self.signals_by_id.keys())
        if outside:
            missing = self.reader._missing_ids("signals", "signal_id", outside)
            if missing:
                raise TimeFFormatError(f"task refers to missing signal {missing[0]!r}")
            raise TimeFFormatError(f"task target refers to Signal {outside[0]!r} on a Record outside the hydrated set")

    def _index_signals(self) -> None:
        """Add the Signals of the Records indexed since the last walk to the lookup by ID."""
        given = self._given
        if given is not None and given.unindexed:
            # A stream walks the caller's Records once, into the lookups every batch shares.
            for record in given.unindexed:
                for signal in record.signals:
                    given.signals_by_id[signal.id] = signal
            given.unindexed.clear()
        for record in self._records_without_signal_index:
            for signal in record.signals:
                self.signals_by_id[signal.id] = signal
        self._records_without_signal_index.clear()

    def _annotations_by_occurrence(self, occurrence_ids: set[int]) -> dict[int, Annotation]:
        """Resolve referenced annotation occurrences to the objects on the hydrated Records or the Dataset.

        An occurrence on no hydrated Record is looked up among the Dataset's occurrences: the
        scope reads only those its tasks reference, unless the caller gave every one of them.

        Returns:
            Annotations by occurrence ID.

        Raises:
            TimeFFormatError: If an ID names no occurrence on the hydrated Records or the Dataset.
        """
        resolved: dict[int, Annotation] = {}
        unresolved: list[int] = []
        for occurrence_id in sorted(occurrence_ids):
            annotation = self.annotations_by_id.get(occurrence_id)
            if annotation is None and self._foreign_records:
                self._match_foreign_occurrences()
                annotation = self.annotations_by_id.get(occurrence_id)
            if annotation is None:
                unresolved.append(occurrence_id)
            else:
                resolved[occurrence_id] = annotation
        if unresolved and not self._dataset_annotations_indexed:
            # The scope's own lookup keeps them, so a later batch of a stream reads its own copies.
            self.reader._read_annotations({ObjectKind.DATASET: unresolved}, self.annotations_by_id)
        for occurrence_id in unresolved:
            annotation = self.annotations_by_id.get(occurrence_id)
            if annotation is None:
                raise TimeFFormatError(
                    f"task refers to annotation occurrence {occurrence_id!r} that no hydrated object carries"
                )
            resolved[occurrence_id] = annotation
        return resolved
