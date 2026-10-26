"""Hydrate the TimeF object hierarchy from ``control.duckdb``.

The objects hold their stored values behind value handles, unloaded.
"""

from collections import ChainMap, defaultdict
from collections.abc import Callable, Collection, Container, Iterable, Iterator, Mapping, MutableMapping
from fractions import Fraction
from functools import partial
from graphlib import CycleError, TopologicalSorter
from itertools import chain
import json
import os
from pathlib import Path
import threading
from typing import Any, NamedTuple, TypeVar, cast
import weakref

import duckdb
import pyarrow as pa

from timenet.dataset import IrregularAxis, OrdinalAxis, Record, RegularAxis, Signal, Source, TimeAxis, TimeFDataset
from timenet.dataset.composition import ImportMap
from timenet.errors import TimeFFormatError, TimeFNotFoundError, TimeFReaderError, TimeFValidationError
from timenet.format.annotation_codec import decode_annotation_value
from timenet.format.duckdb import check_control_schema, connect_control
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
    decode_input_modalities,
    decode_span,
    decode_split,
    decode_target,
    decode_task_payload,
)
from timenet.format.task_target_kind import TaskTargetKind
from timenet.format.task_target_row import TargetValueRow
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
    ureg,
)
from timenet.types.handles import CallableArray
from timenet.types.object_kind import ObjectKind
from timenet.types.reference_keys import InputAlias, OwnerKey
from timenet.types.splits import Split, parse_split
from timenet.types.wire import ValueDtype
from timenet.values_backends.types import ChunkDataIndex, StoredChunk


ValueLoaderFactory = Callable[[int, str, TimeSeriesSpec], ArrayHandle]


class _RecordRow(NamedTuple):
    """The ``records`` columns that hydrate one Record, in schema column order."""

    record_key: int
    record_id: str
    clock_id: int
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
    sources: dict[int, _SourceRow]
    children: dict[int, list[int]]
    roots_by_record: dict[int, list[int]]
    signals: list[_PreparedSignal]
    annotation_keys: tuple[int, ...] | None


class _ChunkRow(NamedTuple):
    """One chunk locator in the control catalog's column order."""

    signal_key: int
    chunk_index: int
    value_path: str
    chunk_major_index: int
    chunk_minor_index: int | None
    n_values: int
    owner_key: OwnerKey | None

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
    """One annotation occurrence joined with its content and the object it annotates."""

    occurrence_key: int
    occurrence_id: str
    object_type: str
    object_id: str | None
    span_type: str
    start_us: int | None
    end_us: int | None
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
    description: str | None
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
    split: str | None
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
    ObjectKind.DATASET: "SELECT dataset_key AS object_key, 'Dataset' AS object_type, dataset_id AS object_id FROM datasets",
    ObjectKind.RECORD: "SELECT record_key AS object_key, 'Record' AS object_type, record_id AS object_id FROM records",
    ObjectKind.SOURCE: "SELECT source_key AS object_key, 'Source' AS object_type, source_id AS object_id FROM sources",
    ObjectKind.SIGNAL: "SELECT signal_key AS object_key, 'Signal' AS object_type, signal_id AS object_id FROM signals",
    ObjectKind.TASK: "SELECT task_key AS object_key, 'Task' AS object_type, task_id AS object_id FROM tasks",
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


def _annotation_query(object_type: ObjectKind | None, *, keyed: bool) -> str:
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
               o.start_us, o.end_us, o.provenance, o.confidence,
               o.metadata, c.content_id, c.name, c.value_kind, c.text_value, c.integer_value,
               c.float_value, c.boolean_value, c.text_list_value, c.unit, c.description, c.metadata,
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

    parent_task_keys: frozenset[int]
    tasks_annotated: bool


_HierarchyRow = TypeVar("_HierarchyRow", _RecordRow, _SourceRow, _SignalRow)
_HierarchyKey = TypeVar("_HierarchyKey", int, str)
_Row = TypeVar("_Row", bound=tuple[Any, ...])
_Ref = TypeVar("_Ref", Record, Annotation)
_T = TypeVar("_T")

_HIERARCHY_TABLES: dict[type[tuple[Any, ...]], tuple[str, str, str]] = {
    _RecordRow: ("records", "record_id", "read_order"),
    _SourceRow: ("sources", "record_key", "source_id"),
    _SignalRow: ("signals", "source_key", "signal_id"),
}
"""Table name, relation column, and ordering column of each hierarchy row type."""


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


def _require_found(kind: ObjectKind, requested: Iterable[str], found: Container[str]) -> None:
    """Check that every requested ID names a stored object of one kind.

    Raises:
        TimeFNotFoundError: If an ID names none, listing every such ID.
    """
    missing = [identity for identity in requested if identity not in found]
    if missing:
        raise TimeFNotFoundError(f"no {kind.value} {', '.join(map(repr, missing))}")


def _missing_values(signal_id: str, _spec: TimeSeriesSpec) -> pa.Array:
    """Fail when metadata-only hydration is asked to load values.

    Raises:
        TimeFFormatError: Always, because no values loader was configured.
    """
    raise TimeFFormatError(f"no values loader is configured for signal {signal_id!r}")


def _shared_owner(owner: OwnerKey | None, strings: dict[str, str]) -> OwnerKey | None:
    """Return an owner key as the string already in use for it, so chunk locations share one string.

    Returns:
        The owner key, or ``None`` for the release whose catalog records the chunks.
    """
    return None if owner is None else OwnerKey(strings.setdefault(owner, owner))


def _missing_values_factory(_key: int, signal_id: str, spec: TimeSeriesSpec) -> ArrayHandle:
    """Create a value handle for a metadata-only reader.

    Returns:
        A handle that raises TimeFFormatError when asked to read values.
    """
    return CallableArray(partial(_missing_values, signal_id, spec))


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

_CATALOG_ERRORS = (duckdb.CatalogException, duckdb.BinderException, duckdb.ConversionException)
"""DuckDB errors that mean the catalog lacks a table, column, or type that a read expects."""


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
        self._value_loader_factory = value_loader_factory
        self._record_keys: dict[str, int] = {}
        self._clock_cache: dict[int, TimeOrigin] = {}
        self._task_facts: _TaskFacts | None = None
        # Chunk files and owners named by the chunk locations read so far, so equal names share one string.
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
            if isinstance(exc, _CATALOG_ERRORS):
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

    def _query_rows(self, sql: str, parameters: object, make: Callable[[Iterable[Any]], _Row]) -> list[_Row]:
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
        self, kind: ObjectKind, object_ids: Iterable[str]
    ) -> dict[ExternalObjectFieldKey, StoredText | StoredTextList]:
        """Return text handles for the packed values of only the objects in this read.

        Returns:
            Handles indexed by object, field, and position, or no handles for an inline catalog.
        """
        external = self.external
        locations = self._external_locations(kind, object_ids)
        return {} if external is None else {key: external.handle(location) for key, location in locations.items()}

    def _external_locations(
        self, kind: ObjectKind, object_ids: Iterable[str]
    ) -> dict[ExternalObjectFieldKey, PackedLocation]:
        """Return the packed-value locations of only the objects in this read.

        Returns:
            Locations indexed by object, field, and position, or none for an inline catalog.

        Raises:
            TimeFFormatError: If packed values need a payload reader that was not configured.
        """
        ids = tuple(object_ids)
        locations = self.run(lambda connection: PackedValueReader.locations(connection, kind, ids))
        if locations and self.external is None:
            key = next(iter(locations))
            raise TimeFFormatError(f"no packed-value reader is configured for {kind.value} {key.object_id!r}")
        return locations

    def _stored_offsets(
        self, axes: dict[int, TimeAxis], shared: dict[int, StoredOffsets] | None = None
    ) -> dict[int, StoredOffsets]:
        """Return one handle to the packed time offsets of each irregular axis, without reading them.

        Args:
            axes: The axes by internal axis key.
            shared: Handles that earlier record reads of the same call built, by axis key. Their
                axes reuse them, and new handles are added, so one call has one handle per axis.

        Returns:
            Handles keyed by internal axis key. Each checks its time offsets against the axis on its
            first read and keeps them.

        Raises:
            TimeFFormatError: If an irregular axis has no packed time offsets, or another axis has some.
        """
        handles: dict[int, StoredOffsets] = {} if shared is None else shared
        new_axes = {axis_key: axis for axis_key, axis in axes.items() if axis_key not in handles}
        locations = self._external_locations(ObjectKind.AXIS, (axis.axis_id for axis in new_axes.values()))
        for axis_key, axis in new_axes.items():
            location = locations.get(
                ExternalObjectFieldKey(
                    object_kind=ObjectKind.AXIS, object_id=axis.axis_id, field=PackedField.TIME_OFFSETS
                )
            )
            if not isinstance(axis, IrregularAxis):
                if location is not None:
                    raise TimeFFormatError(f"axis {axis.axis_id!r} is not irregular but stores packed offsets")
                continue
            if location is None or self.external is None:
                raise TimeFFormatError(f"irregular axis {axis.axis_id!r} has no packed offsets")
            handles[axis_key] = StoredOffsets(
                self.external, location, axis_id=axis.axis_id, first=axis.first_us, last=axis.last_us
            )
        return handles

    def close(self) -> None:
        """Close the reader's DuckDB connection and drop its cached catalog facts."""
        self._disconnect()
        self._record_keys.clear()
        self._clock_cache.clear()
        self._task_facts = None
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

    def record_ids(self) -> tuple[str, ...]:
        """Return every stored Record ID in the order :meth:`read_records` yields them, without hydrating.

        Returns:
            The Record IDs in their stored read order.
        """
        return tuple(row[0] for row in self._query("SELECT record_id FROM records ORDER BY read_order"))

    def read_records(
        self,
        record_ids: Iterable[str] | None = None,
        *,
        with_annotations: bool = True,
        shared_offsets: dict[int, StoredOffsets] | None = None,
    ) -> tuple[Record, ...]:
        """Hydrate complete recursive records, with their stored values behind unloaded handles.

        Source links, clocks, axes, and time-offset locations are checked before any object is
        built. No payload file is opened.

        Args:
            record_ids: Requested IDs in result order, or ``None`` for every record in read order.
            with_annotations: Whether to hydrate annotations on records, sources, and signals.
            shared_offsets: Time-offset handles by axis key, shared with earlier record reads of
                the same call, and extended with this read's new ones. ``None`` shares nothing.

        Returns:
            The hydrated records.

        Raises:
            TimeFNotFoundError: If a requested record is absent.
            TimeFFormatError: If the stored hierarchy is inconsistent.
        """  # noqa: DOC502 - row selection and hierarchy helpers validate the stored data
        hierarchy = self._read_record_hierarchy(None if record_ids is None else tuple(record_ids), shared_offsets)
        annotations = self._read_annotations(hierarchy.annotation_keys) if with_annotations else {}
        signals_by_source = self._read_signals(hierarchy.signals, annotations)
        return self._assemble_records(hierarchy, signals_by_source, annotations)

    def _read_record_hierarchy(
        self, requested: tuple[str, ...] | None, shared_offsets: dict[int, StoredOffsets] | None
    ) -> _RecordHierarchy:
        """Read and validate the catalog rows without opening payload files.

        Returns:
            Selected rows, source links, annotation keys, and the signals' value handles.

        Raises:
            TimeFFormatError: If a signal names a missing source.
        """
        read_all = requested is None
        record_rows = self._read_record_rows(requested)
        sources_by_key = self._query_sources_by_key(
            record_keys=None if read_all else [row.record_key for row in record_rows]
        )
        signals_by_key = self._query_signals_by_key(source_keys=None if read_all else sources_by_key)
        signal_rows = list(signals_by_key.values())
        for row in signal_rows:
            if row.source_key not in sources_by_key:
                raise TimeFFormatError(f"signal {row.signal_id!r} refers to missing source key {row.source_key!r}")
        children, roots = self._group_sources(sources_by_key, {row.record_key for row in record_rows})
        self._cache_record_clocks(record_rows)
        return _RecordHierarchy(
            records=record_rows,
            sources=sources_by_key,
            children=children,
            roots_by_record=roots,
            signals=self._prepare_signals(signal_rows, shared_offsets),
            annotation_keys=None
            if read_all
            else tuple(chain((row.record_key for row in record_rows), sources_by_key, signals_by_key)),
        )

    def _read_record_rows(self, requested: tuple[str, ...] | None) -> list[_RecordRow]:
        """Select and validate record rows, preserving the requested order.

        Args:
            requested: Record IDs in result order, or ``None`` for every record in read order.

        Returns:
            Record rows in result order, including repeated requests for the same record.

        Raises:
            TimeFNotFoundError: If a requested record is absent.
        """
        records_by_id = self._query_records_by_id(record_ids=requested)
        self._record_keys.update((row.record_id, row.record_key) for row in records_by_id.values())
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
        axes = self._read_axes({row.axis_key for row in signal_rows})
        # Signals that share an axis share its handle, so its time offsets are read and checked once.
        time_offsets_by_axis = self._stored_offsets(axes, shared_offsets)
        specs: dict[tuple[Any, ...], TimeSeriesSpec] = {}
        prepared: list[_PreparedSignal] = []
        for row in signal_rows:
            axis = axes.get(row.axis_key)
            if axis is None:
                raise TimeFFormatError(f"signal {row.signal_id!r} refers to missing axis key {row.axis_key!r}")
            spec = specs.get(row.spec_fields)
            if spec is None:
                spec = self._spec_from_row(row)
                specs[row.spec_fields] = spec
            time_offsets = time_offsets_by_axis.get(row.axis_key)
            if time_offsets is not None and time_offsets.length != row.n_values:
                raise TimeFFormatError(
                    f"axis {axis.axis_id!r} stores {time_offsets.length} offsets but signal "
                    f"{row.signal_id!r} declares {row.n_values} values"
                )
            prepared.append(
                _PreparedSignal(
                    row,
                    axis,
                    spec,
                    self._value_loader_factory(row.signal_key, row.signal_id, spec),
                    time_offsets,
                )
            )
        return prepared

    @staticmethod
    def _read_signals(
        prepared: list[_PreparedSignal],
        annotations: dict[tuple[ObjectKind, str], tuple[Annotation, ...]],
    ) -> dict[int, list[Signal]]:
        """Build signals from validated metadata and their value handles.

        Returns:
            Signals grouped by their source key.
        """
        signals_by_source: dict[int, list[Signal]] = defaultdict(list)
        for signal in prepared:
            row = signal.row
            signals_by_source[row.source_key].append(
                Signal.from_loader(
                    id=row.signal_id,
                    name=row.name,
                    spec=signal.spec,
                    time_axis=signal.axis,
                    n_values=row.n_values,
                    loader=signal.values,
                    time_offsets_loader=signal.time_offsets,
                    annotations=annotations.get((ObjectKind.SIGNAL, row.signal_id), ()),
                    metadata=_decode_json(row.metadata, default={}),
                )
            )
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
        signals_by_source: dict[int, list[Signal]],
        annotations: dict[tuple[ObjectKind, str], tuple[Annotation, ...]],
    ) -> tuple[Record, ...]:
        """Attach signals and annotations to the recursive source trees and records.

        Returns:
            Records in the supplied row order.

        Raises:
            TimeFFormatError: If a stored record span lacks its required end.
        """  # noqa: DOC502 - the span helper rejects missing bounds
        rows = hierarchy.records
        sources_by_key = hierarchy.sources
        children = hierarchy.children
        roots_by_record = hierarchy.roots_by_record

        def hydrate_source(source_key: int) -> Source:
            source_row = sources_by_key[source_key]
            return Source(
                id=source_row.source_id,
                name=source_row.name,
                sources=tuple(hydrate_source(child) for child in children[source_key]),
                signals=tuple(signals_by_source[source_key]),
                annotations=annotations.get((ObjectKind.SOURCE, source_row.source_id), ()),
                metadata=_decode_json(source_row.metadata, default={}),
            )

        records: list[Record] = []
        for record_row in rows:
            record_id = record_row.record_id
            span = None
            if record_row.time_span_start_us is not None:
                span = TimeInterval(
                    start_us=record_row.time_span_start_us,
                    end_us=_required(record_row.time_span_end_us, f"the time span end of record {record_id!r}"),
                )
            root_sources = tuple(hydrate_source(source_key) for source_key in roots_by_record[record_row.record_key])
            record = Record(
                record_id=record_id,
                sources=root_sources,
                start_time=self._clock_cache[record_row.clock_id],
                annotations=annotations.get((ObjectKind.RECORD, record_id), ()),
                time_span=span,
                metadata=_decode_json(record_row.metadata, default={}),
            )
            records.append(record)
        return tuple(records)

    def _query_records_by_id(self, record_ids: Iterable[str] | None) -> dict[str, _RecordRow]:
        """Read matching Record rows indexed by public ID.

        Returns:
            Record rows in their stored read order, with duplicate IDs rejected.
        """
        return self._index_rows(
            self._read_rows(_RecordRow, record_ids),
            ObjectKind.RECORD,
            key=lambda row: row.record_id,
            id_of=lambda row: row.record_id,
        )

    def _query_sources_by_key(self, record_keys: Iterable[int] | None) -> dict[int, _SourceRow]:
        """Read Source rows for the supplied Record keys, indexed by internal Source key.

        Returns:
            Source rows in public-ID order, with duplicate keys and IDs rejected.
        """
        return self._index_rows(
            self._read_rows(_SourceRow, record_keys),
            ObjectKind.SOURCE,
            key=lambda row: row.source_key,
            id_of=lambda row: row.source_id,
        )

    def _query_signals_by_key(self, source_keys: Iterable[int] | None) -> dict[int, _SignalRow]:
        """Read Signal rows for the supplied Source keys, indexed by internal Signal key.

        Returns:
            Signal rows in public-ID order, with duplicate keys and IDs rejected.
        """
        return self._index_rows(
            self._read_rows(_SignalRow, source_keys),
            ObjectKind.SIGNAL,
            key=lambda row: row.signal_key,
            id_of=lambda row: row.signal_id,
        )

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
            Rows from the requested hierarchy table, sorted by its ordering column.
        """
        table, relation_column, order_column = _HIERARCHY_TABLES[row_type]
        columns = ", ".join(row_type._fields)
        query = (
            f"SELECT {columns} FROM {table} WHERE ? OR {relation_column} "  # noqa: S608 - fixed identifiers
            f"IN (SELECT unnest(?)) ORDER BY {order_column}"
        )
        return self._query_rows(query, [related_ids is None, list(related_ids or ())], row_type._make)

    @staticmethod
    def _index_rows(
        rows: list[_HierarchyRow],
        kind: ObjectKind,
        *,
        key: Callable[[_HierarchyRow], _HierarchyKey],
        id_of: Callable[[_HierarchyRow], str],
    ) -> dict[_HierarchyKey, _HierarchyRow]:
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
            raise TimeFFormatError(f"control.duckdb contains duplicate {kind.value.lower()} IDs or keys")
        return indexed

    @staticmethod
    def _group_sources(
        rows: dict[int, _SourceRow], record_keys: set[int]
    ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        """Validate source relationships and group child and root keys.

        Returns:
            Child keys by parent key and root keys by Record key.

        Raises:
            TimeFFormatError: If a source names a missing record or parent, crosses record
                boundaries, or forms a cycle.
        """
        children: dict[int, list[int]] = defaultdict(list)
        roots: dict[int, list[int]] = defaultdict(list)
        parents: dict[int, tuple[int, ...]] = {}
        for row in rows.values():
            if row.record_key not in record_keys:
                raise TimeFFormatError(f"source {row.source_id!r} refers to missing record key {row.record_key!r}")
            if row.parent_source_key is None:
                roots[row.record_key].append(row.source_key)
                parents[row.source_key] = ()
            else:
                parent = rows.get(row.parent_source_key)
                if parent is None:
                    raise TimeFFormatError(
                        f"source {row.source_id!r} refers to missing parent key {row.parent_source_key!r}"
                    )
                if parent.record_key != row.record_key:
                    raise TimeFFormatError(f"source {row.source_id!r} crosses record boundaries")
                children[row.parent_source_key].append(row.source_key)
                parents[row.source_key] = (row.parent_source_key,)
        try:
            TopologicalSorter(parents).prepare()
        except CycleError as error:
            raise TimeFFormatError("source hierarchy contains a cycle") from error
        return children, roots

    def read_dataset_annotations(self, dataset_id: str) -> tuple[Annotation, ...]:
        """Return annotations attached directly to a dataset.

        Returns:
            Dataset-level annotation occurrences in stable occurrence-ID order.
        """
        rows = self._query("SELECT dataset_key FROM datasets WHERE dataset_id = ?", [dataset_id])
        if not rows:
            return ()
        return self._read_annotations((rows[0][0],)).get((ObjectKind.DATASET, dataset_id), ())

    def chunk_location_by_key(self, signal_key: int, signal_id: str) -> ChunkLocation:
        """Return where one Signal's chunks are stored, from its catalog rows.

        Raises:
            TimeFFormatError: If the Signal has no stored chunks.
        """
        location = self.chunk_locations_by_keys((signal_key,))[signal_key]
        if location is None:
            raise TimeFFormatError(f"signal {signal_id!r} has no stored value chunks")
        return location

    def chunk_locations_by_keys(self, signal_keys: Iterable[int]) -> dict[int, ChunkLocation | None]:
        """Return where a batch of Signals' chunks are stored, from their catalog rows.

        Args:
            signal_keys: Internal integer Signal keys.

        Returns:
            Locations keyed by internal Signal key, each with the Signal's chunks in chunk-index
            order, or ``None`` for a Signal without stored chunks.

        Raises:
            TimeFFormatError: If one Signal's chunks are stored by more than one release.
        """
        keys = tuple(signal_keys)
        if not keys:
            return {}
        columns = ", ".join(_ChunkRow._fields)
        rows = self._query_rows(
            f"""SELECT {columns} FROM signal_chunks
               WHERE signal_key IN (SELECT unnest(?))
               ORDER BY signal_key, chunk_index""",  # noqa: S608 - fixed column names
            [list(keys)],
            _ChunkRow._make,
        )
        chunks: dict[int, list[StoredChunk]] = {key: [] for key in keys}
        owners: dict[int, OwnerKey | None] = {}
        strings = self._chunk_strings
        for row in rows:
            # The writer stores or reuses a Signal's chunks as a unit, so one release stores them all.
            if owners.setdefault(row.signal_key, row.owner_key) != row.owner_key:
                found = self._query("SELECT signal_id FROM signals WHERE signal_key = ?", [row.signal_key])
                signal = found[0][0] if found else row.signal_key
                raise TimeFFormatError(f"signal {signal!r} has chunks stored by more than one release")
            chunks[row.signal_key].append(row.stored_chunk(strings))
        return {
            key: ChunkLocation(owner=_shared_owner(owners[key], strings), chunks=tuple(stored)) if stored else None
            for key, stored in chunks.items()
        }

    def imported_ids(
        self, alias: str, kind: ObjectKind, parent_ids: Iterable[str]
    ) -> tuple[str | None, dict[str, str]]:
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

    def _occurrence_record_ids(self, occurrence_ids: Collection[str]) -> dict[str, str]:
        """Return the ID of the Record whose hierarchy holds each annotation occurrence, building no object.

        An occurrence on the Dataset or a task, or one the catalog lacks, has no entry.

        Returns:
            The Record ID of each occurrence, by occurrence ID.
        """
        rows = self._query(
            """SELECT occurrences.occurrence_id,
                      coalesce(on_records.record_id, of_sources.record_id, of_signals.record_id)
               FROM annotation_occurrences AS occurrences
               LEFT JOIN records AS on_records
                 ON occurrences.object_type = ? AND on_records.record_key = occurrences.object_key
               LEFT JOIN sources ON occurrences.object_type = ? AND sources.source_key = occurrences.object_key
               LEFT JOIN records AS of_sources ON of_sources.record_key = sources.record_key
               LEFT JOIN signals ON occurrences.object_type = ? AND signals.signal_key = occurrences.object_key
               LEFT JOIN sources AS signal_sources ON signal_sources.source_key = signals.source_key
               LEFT JOIN records AS of_signals ON of_signals.record_key = signal_sources.record_key
               WHERE occurrences.occurrence_id IN (SELECT unnest(?))""",
            [ObjectKind.RECORD.value, ObjectKind.SOURCE.value, ObjectKind.SIGNAL.value, list(occurrence_ids)],
        )
        return {occurrence_id: record_id for occurrence_id, record_id in rows if record_id is not None}

    def import_map(self) -> tuple[dict[str, str], ImportMap]:
        """Return the parent release of each input alias, and the import map of the catalog.

        Returns:
            The parent release that ``imports`` names for each input alias, and the local ID of each
            object in ``imported_objects``, keyed by input alias, kind, and parent ID.
        """
        releases: dict[str, str] = dict(self._query("SELECT alias, parent_ref FROM imports"))
        rows = self._query("SELECT alias, object_kind, parent_id, object_id FROM imported_objects")
        return releases, {
            (InputAlias(alias), ObjectKind(kind), parent_id): object_id for alias, kind, parent_id, object_id in rows
        }

    def signal_record_ids(self, signal_ids: Collection[str]) -> dict[str, str]:
        """Return the ID of the Record that holds each of the given Signals, building no object.

        Args:
            signal_ids: The Signals' IDs.

        Returns:
            The Record ID of each Signal, by Signal ID.

        Raises:
            TimeFNotFoundError: If an ID names no stored Signal.
        """  # noqa: DOC502 - raised by _require_found
        owners: dict[str, str] = dict(
            self._query(
                "SELECT signal_id, record_id FROM signals JOIN sources USING (source_key) "
                "JOIN records USING (record_key) WHERE signal_id IN (SELECT unnest(?))",
                [list(signal_ids)],
            )
        )
        _require_found(ObjectKind.SIGNAL, signal_ids, owners)
        return owners

    def read_signals_by_id(self, signal_ids: Collection[str]) -> dict[str, Signal]:
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
        signals: dict[str, Signal] = {}
        for record in self.read_records(list(dict.fromkeys(owners.values()))):
            signals.update((signal.id, signal) for signal in record.walk_signals() if signal.id in owners)
        return signals

    def read_tasks_by_id(self, task_ids: Collection[str]) -> dict[str, Task]:
        """Build the given tasks, sharing the Records they reference, a bounded batch at a time.

        The tasks they derive from through ``from_tasks`` are built too.

        Args:
            task_ids: The task IDs, each once.

        Returns:
            The tasks by ID.

        Raises:
            TimeFNotFoundError: If an ID names no stored task.
        """  # noqa: DOC502 - raised by _require_found
        keys: dict[str, int] = dict(
            self._query("SELECT task_id, task_key FROM tasks WHERE task_id IN (SELECT unnest(?))", [list(task_ids)])
        )
        _require_found(ObjectKind.TASK, task_ids, keys)
        tasks = _TaskHydration(self, None).read_keys([keys[identity] for identity in task_ids])
        return {task.id: task for task in tasks}

    def read_tasks(  # noqa: PLR0913 - one argument per task selection axis
        self,
        records: Iterable[Record] | None = None,
        *,
        cached_records: Iterable[Record] = (),
        dataset_annotations: Iterable[Annotation] | None = None,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
    ) -> tuple[Task, ...]:
        """Hydrate every selected task and restore all in-memory object references.

        Args:
            records: Input Records to select, or ``None`` for every task. Tasks reuse these objects.
            cached_records: Hydrated Records to reuse without restricting the selection.
            dataset_annotations: The dataset's hydrated annotations, which tasks then reference
                instead of their own copies, or ``None`` to read them when a task needs them.
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
        )
        yield from hydration.iter_batches()

    def task_facts(self) -> _TaskFacts:
        """Return the per-database facts task hydration needs, read once per reader.

        Returns:
            The keys of tasks other tasks derive from, and whether any task carries annotations.
        """
        if self._task_facts is None:
            parent_keys = {row[0] for row in self._query("SELECT DISTINCT parent_task_key FROM task_dependencies")}
            ((annotated,),) = self._query("SELECT count(*) FROM annotation_occurrences WHERE object_type = 'Task'")
            self._task_facts = _TaskFacts(frozenset(parent_keys), annotated > 0)
        return self._task_facts

    def read_task_table(
        self,
        record_ids: Iterable[str] | None = None,
        *,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
    ) -> pa.Table:
        """Return the task table as Arrow, with object references as public IDs, without building Tasks.

        One row per task with its type, prompt, rationale, the four scope columns (Signal keys
        resolved to IDs), the configuration columns, ordered ``input_record_ids``, ordered
        ``from_task_ids``, and the metadata JSON string. Targets are in :meth:`read_target_table`.

        Args:
            record_ids: Keep only tasks whose inputs include one of these Records, or ``None`` for all.
            split: Keep only tasks of this split, or ``None`` for all.

        Returns:
            The tasks in task ID order.
        """
        where, parameters = self._selection_filter(record_ids, required_modalities, supported_modalities, split)
        return self._query_table(
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
                       t.input_modalities, t.split,
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
        )

    def read_target_table(
        self,
        record_ids: Iterable[str] | None = None,
        *,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
    ) -> pa.Table:
        """Return every inline target as Arrow, one typed row each, with references as public IDs.

        Args:
            record_ids: Keep only targets of tasks whose inputs include one of these Records.
            split: Keep only targets of tasks of this split, or ``None`` for all.

        Returns:
            Targets in task ID and position order.
        """
        where, parameters = self._selection_filter(record_ids, required_modalities, supported_modalities, split)
        return self._query_table(
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
        )

    def read_annotation_table(self, object_type: ObjectKind | None = None) -> pa.Table:
        """Return every annotation occurrence joined with its content as Arrow.

        Args:
            object_type: Keep only occurrences on this object type, or ``None`` for all.

        Returns:
            One row per occurrence with the annotated object's type and ID, the content's key,
            typed value columns, unit, span columns, confidence, and JSON metadata.

        Raises:
            TimeFFormatError: If ``object_type`` is not an annotated object type.
        """
        if object_type is not None and object_type not in _OBJECT_ID_SELECTS:
            raise TimeFFormatError(f"unknown annotated object type {object_type!r}")
        object_ids = " UNION ALL ".join(_OBJECT_ID_SELECTS.values())
        where = "" if object_type is None else "WHERE o.object_type = ?"
        return self._query_table(
            f"""WITH object_ids AS ({object_ids})
                SELECT o.occurrence_id, o.object_type, objects.object_id, c.content_id, c.name AS key,
                       c.value_kind, c.text_value, c.integer_value, c.float_value, c.boolean_value,
                       c.text_list_value, c.unit, c.description, o.span_type, o.start_us, o.end_us,
                       o.confidence, o.provenance, c.metadata AS content_metadata, o.metadata
                FROM annotation_occurrences o
                JOIN annotation_contents c USING (content_key)
                LEFT JOIN object_ids objects
                  ON objects.object_key = o.object_key AND objects.object_type = o.object_type
                {where}
                ORDER BY o.object_type, objects.object_id, o.occurrence_id""",  # noqa: S608 - fixed fragments
            [] if object_type is None else [object_type.value],
        )

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

    def _selected_task_keys(
        self,
        record_keys: list[int] | None,
        required_modalities: Iterable[InputModality] | None,
        supported_modalities: Iterable[InputModality] | None,
        split: Split | str | None,
    ) -> list[int]:
        """Return the keys of the selected tasks in task ID order, in one query.

        The query's result is read in full, so no result stays open while the caller builds tasks
        a batch at a time.

        Args:
            record_keys: Keys of the Records whose input tasks to select, or ``None`` for every task.
            split: Keep only tasks of this split, or ``None`` for every task.

        Returns:
            The selected task keys.
        """
        modality_where, modality_params = self._modality_filter(required_modalities, supported_modalities)
        split_where, split_params = self._split_filter(split)
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
        if split_where:
            predicates.append(split_where)
            params.extend(split_params)
        where = "WHERE " + " AND ".join(predicates) if predicates else ""
        sql = f"SELECT t.task_key FROM tasks t {where} ORDER BY t.task_id"  # noqa: S608 - fixed predicates
        return [key for (key,) in self._query(sql, params)]

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
        return self._query_rows(
            f"SELECT {columns} FROM tasks {where} ORDER BY task_id",  # noqa: S608 - fixed identifiers
            parameters,
            _TaskRow._make,
        )

    def _target_rows(self, task_keys: list[int] | None) -> list[_TargetRow]:
        """Read the target rows for one batch of tasks, or every task, in task and position order.

        Returns:
            The target rows.
        """
        columns = ", ".join(_TargetRow._fields)
        where, parameters = self._task_filter(task_keys)
        return self._query_rows(
            f"SELECT {columns} FROM task_targets {where} ORDER BY task_key, position",  # noqa: S608 - fixed identifiers
            parameters,
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
        rows = self._query(
            "SELECT signal_key, signal_id FROM signals WHERE signal_key IN (SELECT unnest(?))", [requested]
        )
        if len(rows) != len(requested):
            missing = sorted(set(requested) - {key for key, _ in rows})
            raise TimeFFormatError(f"task refers to missing signal key {missing[0]!r}")
        return dict(rows)

    def _record_keys_of_occurrences(self, occurrence_keys: Iterable[int]) -> set[int]:
        """Return the keys of the Records whose hierarchy holds the given annotation occurrences.

        An occurrence on the Dataset or a task has no Record.

        Returns:
            The Record keys, each once.
        """
        rows = self._query(
            """SELECT DISTINCT coalesce(records.record_key, sources.record_key, signal_sources.record_key)
               FROM annotation_occurrences AS occurrences
               LEFT JOIN records ON occurrences.object_type = ? AND records.record_key = occurrences.object_key
               LEFT JOIN sources ON occurrences.object_type = ? AND sources.source_key = occurrences.object_key
               LEFT JOIN signals ON occurrences.object_type = ? AND signals.signal_key = occurrences.object_key
               LEFT JOIN sources AS signal_sources ON signal_sources.source_key = signals.source_key
               WHERE occurrences.occurrence_key IN (SELECT unnest(?))""",
            [ObjectKind.RECORD.value, ObjectKind.SOURCE.value, ObjectKind.SIGNAL.value, sorted(set(occurrence_keys))],
        )
        return {record_key for (record_key,) in rows if record_key is not None}

    def _record_keys_of_signals(self, signal_keys: Iterable[int]) -> set[int]:
        """Return the keys of the Records that hold the given Signals.

        Returns:
            The Record keys, each once.
        """
        rows = self._query(
            "SELECT DISTINCT record_key FROM signals JOIN sources USING (source_key) "
            "WHERE signal_key IN (SELECT unnest(?))",
            [sorted(set(signal_keys))],
        )
        return {record_key for (record_key,) in rows}

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
        rows = self._query(
            f"SELECT task_key, field, {column} FROM {table} {where} ORDER BY task_key, field, position",  # noqa: S608
            parameters,
        )
        return self._group_relationships(rows)

    def _task_dependencies(self, task_keys: list[int] | None) -> dict[int, tuple[int, ...]]:
        """Read the ordered parent keys of each task in one batch, or of every task.

        Returns:
            ``task_key -> ordered parent task keys`` for tasks that have parents.
        """
        where, parameters = self._task_filter(task_keys)
        rows = self._query(
            f"SELECT task_key, parent_task_key FROM task_dependencies {where} ORDER BY task_key, position",  # noqa: S608
            parameters,
        )
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
        objects: Mapping[int, _Ref],
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
    ) -> dict[int, TimeAxis]:
        """Hydrate each shared axis exactly once.

        Returns:
            Axes keyed by their internal keys.

        Raises:
            TimeFFormatError: If an axis has an unknown type, or an irregular axis has no Signals or
                Signals of different lengths.
        """
        requested = None if axis_keys is None else tuple(axis_keys)
        if requested == ():
            return {}
        axes: dict[int, TimeAxis] = {}
        query = """SELECT axes.axis_key, axis_id, axis_type, period_numerator_us, period_denominator,
                          start_index, offset_us, first_us, last_us,
                          lengths.min_n_values, lengths.max_n_values
                   FROM axes
                   LEFT JOIN (
                       SELECT axis_key, min(n_values) AS min_n_values, max(n_values) AS max_n_values
                       FROM signals GROUP BY axis_key
                   ) lengths USING (axis_key)"""
        rows = (
            self._query(query)
            if requested is None
            else self._query(f"{query} WHERE axes.axis_key IN (SELECT unnest(?))", [list(requested)])
        )
        for (
            axis_key,
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
                axes[axis_key] = RegularAxis(
                    axis_id=axis_id,
                    period_us=Fraction(numerator, denominator),
                    start_index=start_index,
                    offset_us=offset,
                )
            elif axis_type == "irregular":
                if min_n_values is None or max_n_values is None:
                    raise TimeFFormatError(f"irregular axis {axis_id!r} has no Signals")
                if min_n_values != max_n_values:
                    raise TimeFFormatError(
                        f"axis {axis_id!r} is shared by Signals with different lengths "
                        f"{sorted((min_n_values, max_n_values))}"
                    )
                axes[axis_key] = IrregularAxis(axis_id=axis_id, first_us=first, last_us=last)
            elif axis_type == "ordinal":
                axes[axis_key] = OrdinalAxis(axis_id=axis_id)
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

    def _read_annotations(
        self,
        object_keys: Iterable[int] | None = None,
        *,
        object_type: ObjectKind | None = None,
        annotations_by_occurrence: dict[int, Annotation] | None = None,
    ) -> dict[tuple[ObjectKind, str], tuple[Annotation, ...]]:
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
        grouped: dict[tuple[ObjectKind, str], list[Annotation]] = defaultdict(list)
        parameters: list[object] = [list(requested or ())]
        if object_type is not None:
            parameters.append(object_type.value)
        rows = self._query_rows(
            _annotation_query(object_type, keyed=requested is not None), parameters, _OccurrenceRow._make
        )
        for row in rows:
            if row.content_id is None or row.name is None:
                raise TimeFFormatError(
                    f"annotation occurrence {row.occurrence_id!r} refers to missing content key {row.content_key!r}"
                )
            if row.object_id is None:
                if row.object_type not in _OBJECT_ID_SELECTS:
                    raise TimeFFormatError(
                        f"annotation occurrence {row.occurrence_id!r} has unknown object type {row.object_type!r}"
                    )
                raise TimeFFormatError(
                    f"annotation occurrence {row.occurrence_id!r} refers to missing {row.object_type} key"
                )
        external = self._external_values(
            ObjectKind.ANNOTATION, (_required(row.content_id, "annotation content") for row in rows)
        )
        for row in rows:
            span = _occurrence_span(row)
            # The first pass over ``rows`` already rejected NULL content and object columns.
            content_id = _required(row.content_id, "annotation content")
            name = _required(row.name, "annotation name")
            object_id = _required(row.object_id, "annotated object")
            handle = external.get(
                ExternalObjectFieldKey(object_kind=ObjectKind.ANNOTATION, object_id=content_id, field=PackedField.VALUE)
            )
            annotation = Annotation(
                occurrence_id=row.occurrence_id,
                id=content_id,
                key=name,
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
                span=span,
                source=_decode_json(row.provenance),
                confidence=row.confidence,
                occurrence_metadata=_decode_json(row.metadata, default={}),
            )
            grouped[ObjectKind(row.object_type), object_id].append(annotation)
            if annotations_by_occurrence is not None:
                annotations_by_occurrence[row.occurrence_key] = annotation
        return {key: tuple(value) for key, value in grouped.items()}


class _Lookups(NamedTuple):
    """The caller's Records, their Signals, and the annotations a stream gives every batch to look up.

    Batches write what they build to lookups of their own in front of these. Only the first batch
    whose targets need Signals adds the caller's Records' Signals here, once for the stream.
    """

    records_by_key: MutableMapping[int, Record]
    records_by_id: MutableMapping[str, Record]
    signals_by_id: MutableMapping[str, Signal]
    unindexed: list[Record]
    """The caller's Records whose Signals are not in ``signals_by_id`` yet."""
    annotations_by_id: MutableMapping[str, Annotation]
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
        dataset_annotations: Iterable[Annotation] | None = None,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
    ) -> None:
        """Index the caller's Records and decide whether the task table needs filtering.

        Raises:
            TimeFNotFoundError: If one of ``records`` is not stored in the control database.
        """
        self.reader = reader
        self.required_modalities = required_modalities
        self.supported_modalities = supported_modalities
        self.split = split
        self.records_by_id: MutableMapping[str, Record] = {record.id: record for record in cached_records}
        self.records_by_id.update((record.id, record) for record in records or ())
        self.records_by_key: MutableMapping[int, Record] = {}
        # Records one scope reads share one time-offsets handle per axis.
        self.offsets_by_axis: dict[int, StoredOffsets] = {}
        self.signals_by_id: MutableMapping[str, Signal] = {}
        self._records_without_signal_index: list[Record] = []
        self.annotations_by_id: MutableMapping[str, Annotation] = {}
        self.retained: dict[int, Task] = {}
        self._hydrating: set[int] = set()
        # The lookups of the caller's objects that a stream's batches share, or None for one pass.
        self._given: _Lookups | None = None
        self._dataset_annotations_indexed = dataset_annotations is not None
        self._index_annotations(dataset_annotations or ())
        facts = reader.task_facts()
        self.parent_keys = facts.parent_task_keys
        self.tasks_annotated = facts.tasks_annotated
        unknown = [record_id for record_id in self.records_by_id if record_id not in reader._record_keys]
        if unknown:
            # Records hydrated by another reader instance: look their keys up once.
            rows = reader._query(
                "SELECT record_key, record_id FROM records WHERE record_id IN (SELECT unnest(?))", [unknown]
            )
            if len(rows) != len(unknown):
                missing = sorted(set(unknown) - {record_id for _, record_id in rows})
                raise TimeFNotFoundError(f"no such record(s) in control.duckdb: {missing}")
            reader._record_keys.update((record_id, record_key) for record_key, record_id in rows)
        for record_id, record in list(self.records_by_id.items()):
            self._index_record(reader._record_keys[record_id], record)
        self.record_keys: list[int] | None = (
            None if records is None else [reader._record_keys[record.id] for record in records]
        )

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
        for task_keys in self._pages():
            yield self._hydrate_scope(task_keys, given)

    def read_keys(self, task_keys: list[int]) -> list[Task]:
        """Build the tasks with these keys, each given once, a bounded batch at a time.

        Returns:
            The tasks in the order of ``task_keys``.
        """
        return [
            task
            for start in range(0, len(task_keys), _TASK_BATCH_SIZE)
            for task in self._hydrate(task_keys[start : start + _TASK_BATCH_SIZE])
        ]

    def read_all(self) -> list[Task]:
        """Build all selected tasks in one scope, using batches for filtered reads.

        Returns:
            The tasks in task ID order.
        """
        unfiltered = self.required_modalities is None and self.supported_modalities is None and self.split is None
        if self.record_keys is None and unfiltered:
            return self._hydrate(None)
        return [task for task_keys in self._pages() for task in self._hydrate(task_keys)]

    def _pages(self) -> Iterator[list[int]]:
        """Select the tasks once and yield their keys in task ID order, a bounded batch at a time.

        Yields:
            Each batch's task keys.
        """
        keys = self.reader._selected_task_keys(
            self.record_keys, self.required_modalities, self.supported_modalities, self.split
        )
        for start in range(0, len(keys), _TASK_BATCH_SIZE):
            # Each batch reads its own rows, so a connection closed between batches opens again.
            yield keys[start : start + _TASK_BATCH_SIZE]

    def _share_given(self) -> _Lookups:
        """Keep the lookups of the caller's objects, as they stand before any batch, for every batch.

        Returns:
            The lookups, which the batches then look through.
        """
        self._given = _Lookups(
            self.records_by_key,
            self.records_by_id,
            self.signals_by_id,
            self._records_without_signal_index,
            self.annotations_by_id,
            self._dataset_annotations_indexed,
        )
        return self._given

    def _start_scope(self, given: _Lookups) -> None:
        """Forget what earlier batches built, and look up the caller's objects in ``given``."""
        self.records_by_key = ChainMap({}, given.records_by_key)
        self.records_by_id = ChainMap({}, given.records_by_id)
        self.offsets_by_axis = {}
        self.signals_by_id = ChainMap({}, given.signals_by_id)
        self._records_without_signal_index = []
        self.annotations_by_id = ChainMap({}, given.annotations_by_id)
        self._dataset_annotations_indexed = given.dataset_annotations_indexed
        self.retained = {}

    def _hydrate_scope(self, task_keys: list[int], given: _Lookups) -> list[Task]:
        """Build one batch in a scope of its own, which ends when the batch is built.

        Returns:
            The batch's tasks in the order of ``task_keys``.
        """
        try:
            return self._hydrate(task_keys)
        finally:
            self._start_scope(given)

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
        external = reader._external_values(ObjectKind.TASK, (row.task_id for row in task_rows))
        record_refs = reader._task_relationships("task_record_refs", "record_key", task_keys)
        annotation_refs = reader._task_relationships("task_annotation_refs", "occurrence_key", task_keys)
        target_rows = reader._target_rows(task_keys)
        task_annotations = (
            reader._read_annotations(task_keys, object_type=ObjectKind.TASK) if self.tasks_annotated else {}
        )

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

        ids_by_key = {row.task_key: row.task_id for row in task_rows}
        targets_by_task: dict[int, list[object]] = defaultdict(list)
        for target in target_rows:
            span_signal_ids = (
                None if target.signal_keys is None else tuple(signal_ids_by_key[key] for key in target.signal_keys)
            )
            signal_id = None if target.signal_key is None else signals_by_key[target.signal_key].id
            target_values = TargetValueRow(
                target_kind=TaskTargetKind(target.target_kind),
                text_value=target.text_value,
                integer_value=target.integer_value,
                float_value=target.float_value,
                boolean_value=target.boolean_value,
                record_id=None if target.record_key is None else self.records_by_key[target.record_key].id,
                signal_id=signal_id,
                span_start=target.span_start,
                span_end=target.span_end,
                span_signal_ids=span_signal_ids,
            )
            handle = external.get(
                ExternalObjectFieldKey(
                    object_kind=ObjectKind.TASK,
                    object_id=ids_by_key[target.task_key],
                    field=PackedField.TARGETS,
                    position=len(targets_by_task[target.task_key]),
                )
            )
            targets_by_task[target.task_key].append(
                handle
                if handle is not None
                else decode_target(target_values, records=self.records_by_id, signals=self.signals_by_id)
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
                "prompt": external.get(
                    ExternalObjectFieldKey(
                        object_kind=ObjectKind.TASK, object_id=row.task_id, field=PackedField.PROMPT
                    ),
                    row.prompt,
                ),
                "input_modalities": decode_input_modalities(row.input_modalities),
                "split": decode_split(row.split),
                "scope": decode_span(
                    row.scope_type,
                    row.scope_start,
                    row.scope_end,
                    None if row.scope_signal_keys is None else [signal_ids_by_key[k] for k in row.scope_signal_keys],
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
            task = cls(**kwargs)
            try:
                TimeFDataset._check_task_record_scope(task)
            except TimeFValidationError as exc:
                raise TimeFFormatError(f"stored task {task.id!r} has an invalid Record scope: {exc}") from exc
            tasks[row.task_key] = task
        return tasks

    def _index_record(self, record_key: int, record: Record) -> None:
        """Add one Record and everything hanging off it to the lookups."""
        self.records_by_key[record_key] = record
        self.records_by_id[record.id] = record
        # Text and scalar targets never need Signals, so the Signal walk waits for a target that does.
        self._records_without_signal_index.append(record)
        self._index_annotations(record.walk_annotations())

    def _index_annotations(self, annotations: Iterable[Annotation]) -> None:
        """Add annotation occurrences to the lookup by occurrence ID."""
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
        rows = self.reader._query(
            "SELECT record_key, record_id FROM records WHERE record_key IN (SELECT unnest(?))", [missing]
        )
        if len(rows) != len(missing):
            unknown = sorted(set(missing) - {key for key, _ in rows})
            raise TimeFFormatError(f"task refers to missing record key {unknown[0]!r}")
        ids_by_key = dict(rows)
        hydrated = self.reader.read_records([ids_by_key[key] for key in missing], shared_offsets=self.offsets_by_axis)
        for record_key, record in zip(missing, hydrated, strict=True):
            self._index_record(record_key, record)

    def _signals_by_key(self, signal_keys: set[int]) -> dict[int, Signal]:
        """Resolve the Signals one batch's targets refer to, hydrating the Records that hold them.

        A Signal target can lie on a Record that the task references in no other way.

        Returns:
            Signals by internal key.

        Raises:
            TimeFFormatError: If a key names no Signal on the hydrated Records.
        """
        if not signal_keys:
            return {}
        self._index_signals()
        signal_ids = self.reader._signal_ids_by_key(signal_keys)
        unbuilt = [signal_key for signal_key, signal_id in signal_ids.items() if signal_id not in self.signals_by_id]
        if unbuilt:
            self._ensure_records(self.reader._record_keys_of_signals(unbuilt))
            self._index_signals()
        resolved: dict[int, Signal] = {}
        for signal_key, signal_id in signal_ids.items():
            signal = self.signals_by_id.get(signal_id)
            if signal is None:
                raise TimeFFormatError(
                    f"task target refers to Signal {signal_id!r} on a Record outside the hydrated set"
                )
            resolved[signal_key] = signal
        return resolved

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

    def _annotations_by_occurrence(self, occurrence_keys: set[int]) -> dict[int, Annotation]:
        """Resolve referenced annotation occurrences, hydrating the Records that hold them.

        A task can reference an annotation on a Record that it references in no other way.

        Returns:
            Annotations by internal occurrence key.

        Raises:
            TimeFFormatError: If a key names no occurrence on the hydrated Records or the Dataset.
        """
        if not occurrence_keys:
            return {}
        rows = self.reader._query(
            "SELECT occurrence_key, occurrence_id FROM annotation_occurrences WHERE occurrence_key IN (SELECT unnest(?))",
            [sorted(occurrence_keys)],
        )
        unbuilt = [
            occurrence_key for occurrence_key, occurrence_id in rows if occurrence_id not in self.annotations_by_id
        ]
        if unbuilt:
            self._ensure_records(self.reader._record_keys_of_occurrences(unbuilt))
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
        for (dataset_key,) in self.reader._query("SELECT dataset_key FROM datasets"):
            for annotations in self.reader._read_annotations((dataset_key,)).values():
                self._index_annotations(annotations)
