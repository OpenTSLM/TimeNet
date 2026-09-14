"""Read a dataset version's control plane with SQL, and its values through whichever backend wrote them.

The queries here are the reason for the whole exercise. Reconstructing one task means walking
``Task -> Records -> recursive Sources -> Signals -> axis and spec -> annotations at every level``.
Against normalized tables that is a recursive traversal plus several joins. In SQL it is one
statement the engine plans; in Python it is an index-building pass and a hand-written walk.

The reader opens the database read-only. A read-only connection does not create a WAL beside the
file, so a downloaded version stays exactly as it was checksummed in the manifest.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

import duckdb
import numpy as np

from timenet.control_plane import schema as ddl
from timenet.control_plane.values import ChunkLocator, read_values
from timenet.errors import TimeFFormatError
from timenet.format.constants import CONTROL_DB_FILE


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion


@dataclass(frozen=True)
class ResolvedAnnotation:
    """One annotation as a caller sees it: payload and placement together, ids resolved away."""

    name: str
    value: str
    unit: str | None
    span_type: str
    start_us: int | None
    end_us: int | None
    provenance: str | None
    confidence: float | None

    def to_text(self) -> str:
        """Render the annotation as one line of text for a prompt.

        Returns:
            The annotation as ``name=value`` with its unit and time placement, when it has them.
        """
        text = f"{self.name}={self.value}"
        if self.unit:
            text += f" {self.unit}"
        if self.span_type == "point" and self.start_us is not None:
            text += f" @{self.start_us / 1e6:g}s"
        elif self.span_type == "interval" and self.start_us is not None and self.end_us is not None:
            text += f" @[{self.start_us / 1e6:g}s..{self.end_us / 1e6:g}s]"
        return text


@dataclass
class SignalView:
    """One signal, with its axis and spec resolved and its annotations attached."""

    signal_id: str
    name: str
    spec_type: str
    unit: str
    dtype: str
    axis_type: str
    n_values: int
    metadata: dict[str, Any] = field(default_factory=dict)
    annotations: list[ResolvedAnnotation] = field(default_factory=list)


@dataclass
class SourceView:
    """One source, its child sources, and the signals it produces."""

    source_id: str
    name: str
    depth: int
    metadata: dict[str, Any] = field(default_factory=dict)
    sources: list["SourceView"] = field(default_factory=list)
    signals: list[SignalView] = field(default_factory=list)
    annotations: list[ResolvedAnnotation] = field(default_factory=list)


@dataclass
class RecordView:
    """One recording session, rebuilt from the tables into the shape a caller thinks in."""

    record_id: str
    start_time_us: int | None
    metadata: dict[str, Any] = field(default_factory=dict)
    sources: list[SourceView] = field(default_factory=list)
    annotations: list[ResolvedAnnotation] = field(default_factory=list)

    def walk_sources(self) -> list[SourceView]:
        """Return every source in this record, parents before children.

        Returns:
            The sources in breadth-first order.
        """
        found: list[SourceView] = []
        queue = list(self.sources)
        while queue:
            current = queue.pop(0)
            found.append(current)
            queue.extend(current.sources)
        return found

    def signals(self) -> list[SignalView]:
        """Return every signal in this record.

        Returns:
            The signals, in the order their sources are walked.
        """
        return [signal for source in self.walk_sources() for signal in source.signals]


@dataclass
class TaskView:
    """One task with its inputs and targets resolved in order."""

    task_id: str
    prompt: str
    inputs: list[RecordView | str] = field(default_factory=list)
    target: list[RecordView | str] = field(default_factory=list)
    annotations: list[ResolvedAnnotation] = field(default_factory=list)


# The materialized path makes this a plain sorted scan: ORDER BY path is already depth-first
# display order, and a parent always sorts before its children, so the tree can be rebuilt in one
# pass. The draft's parent pointer alone needs a recursive CTE to get the same thing.
_SOURCE_TREE = """
SELECT source_id, parent_source_id, name, position, metadata, depth
FROM sources WHERE record_id = ? ORDER BY path
"""

# A whole subtree, without recursion: every descendant's path starts with the ancestor's.
_SUBTREE = """
SELECT source_id, parent_source_id, name, position, metadata, depth
FROM sources
WHERE record_id = ? AND (path = ? OR starts_with(path, ? || '.'))
ORDER BY path
"""

# Signals hang off sources through a link table, because one series can be used by several records.
_RECORD_SIGNALS = """
SELECT sig.signal_id, ss.source_id, sig.name, ss.position, sig.n_values, sig.metadata,
       sp.spec_id, sp.unit, sp.dtype, ax.axis_type
FROM sources src
JOIN source_signals ss ON ss.source_id = src.source_id
JOIN signals sig ON sig.signal_id = ss.signal_id
JOIN specs sp ON sp.spec_id = sig.spec_id
JOIN axes  ax ON ax.axis_id = sig.axis_id
WHERE src.record_id = ?
ORDER BY ss.source_id, ss.position
"""

# Every annotation anywhere in one record: on the record, on any source in its tree, on any signal
# under those sources. The denormalized scope makes this one indexed equality. Against the draft's
# polymorphic object_type/object_id it takes a recursive walk of the source tree unioned with two
# more id lookups, which is what the first version of this query had to do.
# Annotations on the record and on its sources are record-scoped, so they come back on one indexed
# equality. Annotations on a signal are not: a shared series carries the same statement in every
# record that uses it, so those are reached through the link table.
_RECORD_ANNOTATIONS = """
SELECT o.object_type, coalesce(o.on_record_id, o.on_source_id) AS object_id,
       c.name, c.value, c.unit, o.span_type, o.start_us, o.end_us, o.provenance, o.confidence,
       o.occurrence_id
FROM entities_to_annotations o
JOIN annotations c ON c.content_id = o.content_id
WHERE o.scope_record_id = ?
UNION ALL
SELECT o.object_type, o.on_signal_id AS object_id,
       c.name, c.value, c.unit, o.span_type, o.start_us, o.end_us, o.provenance, o.confidence,
       o.occurrence_id
FROM entities_to_annotations o
JOIN annotations c ON c.content_id = o.content_id
JOIN source_signals ss ON ss.signal_id = o.on_signal_id
JOIN sources src ON src.source_id = ss.source_id
WHERE src.record_id = ?
ORDER BY occurrence_id
"""

# The batched forms of the three queries above. Hydrating a record at a time costs 34 ms on a
# 618,508-record corpus, because each of these scans the whole table; asking for a thousand records
# at once costs 0.111 ms per record, because the same scan answers all of them.
_BATCH_RECORDS = """
SELECT record_id, start_time_us, metadata FROM records WHERE record_id IN (SELECT unnest(?))
"""

_BATCH_SOURCES = """
SELECT record_id, source_id, parent_source_id, name, position, metadata, depth
FROM sources WHERE record_id IN (SELECT unnest(?)) ORDER BY record_id, path
"""

_BATCH_SIGNALS = """
SELECT src.record_id, sig.signal_id, ss.source_id, sig.name, ss.position, sig.n_values, sig.metadata,
       sp.spec_id, sp.unit, sp.dtype, ax.axis_type
FROM sources src
JOIN source_signals ss ON ss.source_id = src.source_id
JOIN signals sig ON sig.signal_id = ss.signal_id
JOIN specs sp ON sp.spec_id = sig.spec_id
JOIN axes  ax ON ax.axis_id = sig.axis_id
WHERE src.record_id IN (SELECT unnest(?))
ORDER BY src.record_id, ss.source_id, ss.position
"""

_BATCH_ANNOTATIONS = """
SELECT o.scope_record_id, o.object_type, coalesce(o.on_record_id, o.on_source_id) AS object_id,
       c.name, c.value, c.unit, o.span_type, o.start_us, o.end_us, o.provenance, o.confidence,
       o.occurrence_id
FROM entities_to_annotations o
JOIN annotations c ON c.content_id = o.content_id
WHERE o.scope_record_id IN (SELECT unnest(?))
UNION ALL
SELECT src.record_id, o.object_type, o.on_signal_id AS object_id,
       c.name, c.value, c.unit, o.span_type, o.start_us, o.end_us, o.provenance, o.confidence,
       o.occurrence_id
FROM entities_to_annotations o
JOIN annotations c ON c.content_id = o.content_id
JOIN source_signals ss ON ss.signal_id = o.on_signal_id
JOIN sources src ON src.source_id = ss.source_id
WHERE src.record_id IN (SELECT unnest(?))
ORDER BY 1, occurrence_id
"""

# Every n-th row, so N workers split a corpus into disjoint slices without coordinating.
_PARTITIONED_IDS = """
SELECT {column} FROM (
    SELECT {column}, row_number() OVER (ORDER BY {column}) - 1 AS ordinal FROM {table}
) WHERE ordinal % ? = ?
"""

_ANNOTATIONS_FOR = """
SELECT c.name, c.value, c.unit, o.span_type, o.start_us, o.end_us, o.provenance, o.confidence
FROM entities_to_annotations o
JOIN annotations c ON c.content_id = o.content_id
WHERE o.{column} = ?
ORDER BY o.occurrence_id
"""

# The reverse direction: start from a statement, find every object that carries it. This is what the
# draft's derived annotation_objects_index table exists for; the index on content_id serves it.
_OBJECTS_WITH = """
SELECT o.object_type,
       coalesce(o.on_dataset_id, o.on_task_id, o.on_record_id, o.on_source_id, o.on_signal_id) AS object_id
FROM entities_to_annotations o
JOIN annotations c ON c.content_id = o.content_id
WHERE c.name = ? AND (? IS NULL OR c.value = ?)
ORDER BY o.object_type, object_id
"""

# Reverse lookup all the way back to records. The scope column collapses what the draft needs three
# joins for: an annotation on a signal already knows which record it belongs to.
_RECORDS_WITH = """
SELECT DISTINCT record_id FROM (
    SELECT o.scope_record_id AS record_id
    FROM entities_to_annotations o
    JOIN annotations c ON c.content_id = o.content_id
    WHERE c.name = ? AND (? IS NULL OR c.value = ?) AND o.scope_record_id IS NOT NULL
    UNION ALL
    SELECT src.record_id
    FROM entities_to_annotations o
    JOIN annotations c ON c.content_id = o.content_id
    JOIN source_signals ss ON ss.signal_id = o.on_signal_id
    JOIN sources src ON src.source_id = ss.source_id
    WHERE c.name = ? AND (? IS NULL OR c.value = ?)
)
ORDER BY record_id
"""

# The tasks a record answers. The proposal keeps a record_tasks table for this; the index on
# task_items.record_id gives the same lookup without a second copy of the links.
_TASKS_FOR_RECORD = """
SELECT DISTINCT task_id FROM task_items WHERE record_id = ? ORDER BY task_id
"""


class TimeFReader:  # noqa: PLR0904 - the read surface is wide because the hierarchy is
    """Opens a version's control database and answers questions about its hierarchy."""

    def __init__(self, root: Path, database: str | Path | None = None) -> None:
        """Open the control plane of a version directory.

        Args:
            root: The version directory. The values plane is read relative to it.
            database: The control database to open. It defaults to the file inside ``root``. Pass a
                URL here to open a remote database in place, without downloading it.

        Raises:
            TimeFFormatError: If the database is missing, or was written by a newer schema.
        """
        self._root = Path(root)
        self._target = str(database) if database is not None else str(self._root / CONTROL_DB_FILE)
        if database is None and not (self._root / CONTROL_DB_FILE).exists():
            raise TimeFFormatError(f"no control database at {self._root / CONTROL_DB_FILE}")
        self._backends: dict[str, str] | None = None
        self._opened: duckdb.DuckDBPyConnection | None = _connect(self._target)
        version = self.connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if version is None:
            raise TimeFFormatError(f"{self._target} has no schema_version in its meta table")

    def __getstate__(self) -> dict[str, Any]:
        """Drop the open connection so the reader can cross a process boundary.

        A ``DuckDBPyConnection`` does not pickle, and a torch ``DataLoader`` ships its dataset to
        every worker. The reader therefore carries only what it needs to reopen, and each worker
        gets its own connection on first use.

        Returns:
            The reader's state, without the connection.
        """
        return {**self.__dict__, "_opened": None}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a reader without opening anything yet.

        Args:
            state: The state produced by :meth:`__getstate__`.
        """
        self.__dict__.update(state)

    @classmethod
    def open_version(cls, version: "DatasetVersion") -> Self:
        """Open the control plane of a version handed over by a registry.

        Args:
            version: The opened version. Its manifest must name a control database.

        Returns:
            A reader for that version.

        Raises:
            TimeFFormatError: If the version's manifest has no control database.
        """
        if version.manifest.files.control_db is None:
            raise TimeFFormatError(f"{version.manifest.dataset_id} does not use a DuckDB control plane")
        return cls(Path(version.root))

    def __enter__(self) -> Self:
        """Return this reader, so it can be used as a context manager.

        Returns:
            This reader.
        """
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """Close the database connection."""
        self.close()

    def close(self) -> None:
        """Close the database connection, if one is open."""
        if self._opened is not None:
            self._opened.close()
            self._opened = None

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """The open connection, for a caller that wants to run its own SQL.

        The connection is opened on first use, so a reader restored in a DataLoader worker connects
        there rather than carrying a handle across the process boundary.
        """
        if self._opened is None:
            self._opened = _connect(self._target)
        return self._opened

    def record_ids(self) -> list[str]:
        """Return every record id in the dataset.

        Returns:
            The record ids, sorted.
        """
        return [
            row[0] for row in self.connection.execute("SELECT record_id FROM records ORDER BY record_id").fetchall()
        ]

    def task_ids(self) -> list[str]:
        """Return every task id in the dataset.

        Returns:
            The task ids, sorted.
        """
        return [row[0] for row in self.connection.execute("SELECT task_id FROM tasks ORDER BY task_id").fetchall()]

    def counts(self) -> dict[str, int]:
        """Return the row count of every control-plane table.

        Returns:
            A mapping from table name to row count.
        """
        counted: dict[str, int] = {}
        for table in ddl.TABLES:
            # The table names come from this module's own schema, never from input.
            row = self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()  # noqa: S608
            counted[table] = 0 if row is None else row[0]
        return counted

    def record(self, record_id: str) -> RecordView:
        """Rebuild one record: its source tree, its signals, and every annotation in it.

        Reading many records one call at a time is the slow way round; see :meth:`records` and
        :meth:`iter_records`.

        Args:
            record_id: The record to read.

        Returns:
            The reconstructed record.

        Raises:
            TimeFFormatError: If no record has this id.
        """
        found = self.records([record_id])
        if not found:
            raise TimeFFormatError(f"no record {record_id!r}")
        return found[0]

    def records(self, record_ids: Sequence[str]) -> list[RecordView]:  # noqa: PLR0914
        """Rebuild many records with four queries, whatever the batch size.

        Each query scans its table once and answers for the whole batch, so the per-record cost
        falls with the batch. Measured on a 618,508-record corpus: 34.49 ms per record one at a
        time, 0.111 ms per record in batches of a thousand.

        Args:
            record_ids: The records to read. Ids that name no record are skipped.

        Returns:
            The reconstructed records, in the order the ids were given.
        """
        wanted = list(record_ids)
        if not wanted:
            return []
        built = {
            row[0]: RecordView(record_id=row[0], start_time_us=row[1], metadata=_metadata(row[2]))
            for row in self.connection.execute(_BATCH_RECORDS, [wanted]).fetchall()
        }

        sources: dict[str, SourceView] = {}
        for record_id, source_id, parent_id, name, _position, metadata, depth in self.connection.execute(
            _BATCH_SOURCES, [wanted]
        ).fetchall():
            view = SourceView(source_id=source_id, name=name, depth=depth, metadata=_metadata(metadata))
            sources[source_id] = view
            if parent_id is None:
                built[record_id].sources.append(view)
            else:
                sources[parent_id].sources.append(view)

        signals: dict[str, SignalView] = {}
        for row in self.connection.execute(_BATCH_SIGNALS, [wanted]).fetchall():
            (_record_id, signal_id, source_id, name, _position, n_values, metadata, spec_id, unit, dtype, axis) = row
            view = SignalView(
                signal_id=signal_id,
                name=name,
                spec_type=spec_id,
                unit=unit,
                dtype=dtype,
                axis_type=axis,
                n_values=n_values,
                metadata=_metadata(metadata),
            )
            signals[signal_id] = view
            sources[source_id].signals.append(view)

        for record_id, object_type, object_id, *payload in self.connection.execute(
            _BATCH_ANNOTATIONS, [wanted, wanted]
        ).fetchall():
            annotation = _annotation(payload[:-1])
            if object_type == "record":
                built[record_id].annotations.append(annotation)
            elif object_type == "source":
                sources[object_id].annotations.append(annotation)
            else:
                signals[object_id].annotations.append(annotation)
        return [built[record_id] for record_id in wanted if record_id in built]

    def iter_records(
        self, *, batch_size: int = 512, worker_index: int = 0, num_workers: int = 1
    ) -> Iterator[RecordView]:
        """Walk every record, hydrating them in batches.

        Args:
            batch_size: How many records to hydrate per round of queries.
            worker_index: Which slice of the corpus this caller wants, from 0.
            num_workers: How many slices the corpus is split into. Each worker sees a disjoint
                slice, so a DataLoader's workers together see every record exactly once.

        Yields:
            Each record in this worker's slice.

        Raises:
            ValueError: If the worker index and count do not describe a slice.
        """  # noqa: DOC502 - raised by _id_batches
        for batch in self._id_batches("records", "record_id", batch_size, worker_index, num_workers):
            yield from self.records(batch)

    def iter_tasks(self, *, batch_size: int = 512, worker_index: int = 0, num_workers: int = 1) -> Iterator[TaskView]:
        """Walk every task, hydrating each batch's records together.

        Args:
            batch_size: How many tasks to hydrate per round of queries.
            worker_index: Which slice of the corpus this caller wants, from 0.
            num_workers: How many slices the corpus is split into.

        Yields:
            Each task in this worker's slice, with its input and target records rebuilt.

        Raises:
            ValueError: If the worker index and count do not describe a slice.
        """  # noqa: DOC502 - raised by _id_batches
        for batch in self._id_batches("tasks", "task_id", batch_size, worker_index, num_workers):
            yield from self.tasks(batch)

    def _id_batches(
        self, table: str, column: str, batch_size: int, worker_index: int, num_workers: int
    ) -> Iterator[list[str]]:
        """Yield this worker's ids in batches.

        Args:
            table: The table to draw ids from.
            column: The id column.
            batch_size: How many ids per batch.
            worker_index: Which slice this caller wants.
            num_workers: How many slices there are.

        Yields:
            One batch of ids at a time.

        Raises:
            ValueError: If the worker index and count do not describe a slice, or the batch is empty.
        """
        if num_workers < 1 or not 0 <= worker_index < num_workers:
            raise ValueError(f"worker {worker_index} of {num_workers} is not a slice of the corpus")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        query = _PARTITIONED_IDS.format(table=table, column=column)
        # The ids stream from their own cursor: hydrating a batch runs more queries, and a second
        # query on the same connection would discard the result this one is still reading.
        cursor = self.connection.cursor()
        try:
            found = cursor.execute(query, [num_workers, worker_index])
            while rows := found.fetchmany(batch_size):
                yield [row[0] for row in rows]
        finally:
            cursor.close()

    def task(self, task_id: str) -> TaskView:
        """Rebuild one task, resolving each input and target in order.

        Args:
            task_id: The task to read.

        Returns:
            The reconstructed task, with every referenced record rebuilt in full.

        Raises:
            TimeFFormatError: If no task has this id.
        """
        row = self.connection.execute("SELECT task_id, prompt FROM tasks WHERE task_id = ?", [task_id]).fetchone()
        if row is None:
            raise TimeFFormatError(f"no task {task_id!r}")
        task = TaskView(task_id=row[0], prompt=row[1], annotations=self.annotations_for("task", task_id))
        for role, _position, item_type, text_value, record_id in self.connection.execute(
            "SELECT role, position, item_type, text_value, record_id FROM task_items "
            "WHERE task_id = ? ORDER BY role, position",
            [task_id],
        ).fetchall():
            item = text_value if item_type == "text" else self.record(record_id)
            (task.inputs if role == "input" else task.target).append(item)
        return task

    def tasks(self, task_ids: Sequence[str]) -> list[TaskView]:
        """Rebuild many tasks, hydrating every record they refer to in one batch.

        Args:
            task_ids: The tasks to read. Ids that name no task are skipped.

        Returns:
            The reconstructed tasks, in the order the ids were given.
        """
        wanted = list(task_ids)
        if not wanted:
            return []
        built = {
            row[0]: TaskView(task_id=row[0], prompt=row[1])
            for row in self.connection.execute(
                "SELECT task_id, prompt FROM tasks WHERE task_id IN (SELECT unnest(?))", [wanted]
            ).fetchall()
        }
        items = self.connection.execute(
            "SELECT task_id, role, position, item_type, text_value, record_id FROM task_items "
            "WHERE task_id IN (SELECT unnest(?)) ORDER BY task_id, role, position",
            [wanted],
        ).fetchall()
        # Every record any of these tasks names, rebuilt together rather than one task at a time.
        referenced = {row[5] for row in items if row[3] == "record" and row[5] is not None}
        records = {record.record_id: record for record in self.records(sorted(referenced))}
        for task_id, role, _position, item_type, text_value, record_id in items:
            item = text_value if item_type == "text" else records.get(record_id)
            if item is None:
                continue
            (built[task_id].inputs if role == "input" else built[task_id].target).append(item)
        for annotation_row in self.connection.execute(
            "SELECT o.on_task_id, c.name, c.value, c.unit, o.span_type, o.start_us, o.end_us, "
            "o.provenance, o.confidence FROM entities_to_annotations o "
            "JOIN annotations c ON c.content_id = o.content_id "
            "WHERE o.on_task_id IN (SELECT unnest(?)) ORDER BY o.occurrence_id",
            [wanted],
        ).fetchall():
            built[annotation_row[0]].annotations.append(_annotation(annotation_row[1:]))
        return [built[task_id] for task_id in wanted if task_id in built]

    def annotations_for(self, object_type: str, object_id: str) -> list[ResolvedAnnotation]:
        """Return every annotation attached to one object.

        Args:
            object_type: One of ``dataset``, ``task``, ``record``, ``source``, ``signal``.
            object_id: The object's id.

        Returns:
            The annotations, payload and placement resolved together.

        Raises:
            TimeFFormatError: If ``object_type`` is not one of the five annotatable kinds.
        """
        column = ddl.TARGET_COLUMNS.get(object_type)
        if column is None:
            raise TimeFFormatError(f"{object_type!r} is not an annotatable object type")
        query = _ANNOTATIONS_FOR.format(column=column)
        return [_annotation(row) for row in self.connection.execute(query, [object_id]).fetchall()]

    def subtree(self, record_id: str, source_id: str) -> list[tuple[str, str, int]]:
        """Return one source and everything beneath it, without walking the tree edge by edge.

        Args:
            record_id: The record the source belongs to.
            source_id: The source at the top of the subtree.

        Returns:
            One ``(source_id, name, depth)`` tuple per source, in depth-first order.

        Raises:
            TimeFFormatError: If the record has no such source.
        """
        row = self.connection.execute(
            "SELECT path FROM sources WHERE record_id = ? AND source_id = ?", [record_id, source_id]
        ).fetchone()
        if row is None:
            raise TimeFFormatError(f"record {record_id!r} has no source {source_id!r}")
        found = self.connection.execute(_SUBTREE, [record_id, row[0], row[0]]).fetchall()
        return [(item[0], item[2], item[5]) for item in found]

    def objects_with(self, name: str, value: str | None = None) -> list[tuple[str, str]]:
        """Find every object carrying an annotation, starting from the annotation.

        Args:
            name: The annotation's name, for example ``patient_sex``.
            value: The annotation's value, or ``None`` to match any value.

        Returns:
            One ``(object_type, object_id)`` pair per occurrence.
        """
        return [tuple(row) for row in self.connection.execute(_OBJECTS_WITH, [name, value, value]).fetchall()]

    def records_with(self, name: str, value: str | None = None) -> list[str]:
        """Find every record carrying an annotation, directly or on one of its sources or signals.

        Args:
            name: The annotation's name.
            value: The annotation's value, or ``None`` to match any value.

        Returns:
            The matching record ids, sorted.
        """
        return [
            row[0]
            for row in self.connection.execute(_RECORDS_WITH, [name, value, value, name, value, value]).fetchall()
        ]

    def tasks_for_record(self, record_id: str) -> list[str]:
        """Return every task that refers to a record.

        Args:
            record_id: The record.

        Returns:
            The task ids, sorted.
        """
        return [row[0] for row in self.connection.execute(_TASKS_FOR_RECORD, [record_id]).fetchall()]

    def chunk_locators(self, signal_id: str) -> list[ChunkLocator]:
        """Return where a signal's values live in the values plane.

        Args:
            signal_id: The signal.

        Returns:
            The signal's chunk locators, in chunk order.
        """
        return [
            ChunkLocator(*row)
            for row in self.connection.execute(
                "SELECT signal_id, chunk_idx, chunk_file, chunk_major_idx, chunk_minor_idx, n_values "
                "FROM signal_chunks WHERE signal_id = ? ORDER BY chunk_idx",
                [signal_id],
            ).fetchall()
        ]

    def values_backends(self) -> dict[str, str]:
        """Return which backend wrote each values artifact.

        A locator's two indexes mean different things per backend, so a reader has to know this
        before it can follow one. The table is one row per shard or array, not per chunk, and it is
        read once and kept: dispatching per chunk would cost a lookup on every values read.

        Returns:
            A mapping from artifact path to backend name.
        """
        if self._backends is None:
            self._backends = dict(
                self.connection.execute("SELECT chunk_file, backend FROM values_artifacts").fetchall()
            )
        return self._backends

    def values(self, signal_id: str) -> np.ndarray:
        """Read one signal's values out of the values plane, whichever backend wrote it.

        Args:
            signal_id: The signal.

        Returns:
            The signal's values.
        """
        return read_values(self._root, self.chunk_locators(signal_id), self.values_backends())


_REMOTE_SCHEMES = ("http://", "https://", "s3://", "gs://", "az://")


def _connect(target: str) -> duckdb.DuckDBPyConnection:
    """Open a control database, whether it is a local file or a URL.

    A local file opens directly. A URL is attached through ``httpfs`` instead, which reads it with
    range requests: the client pulls the blocks its query touches rather than the whole file. That
    makes it possible to ask which records carry an annotation before deciding to download the
    dataset at all.

    Args:
        target: A local path, or a URL on a supported remote scheme.

    Returns:
        A read-only connection whose default database is the control plane.
    """
    if not target.startswith(_REMOTE_SCHEMES):
        return duckdb.connect(target, read_only=True)
    connection = duckdb.connect()
    connection.execute("INSTALL httpfs")
    connection.execute("LOAD httpfs")
    if target.startswith(("s3://", "gs://", "az://")):
        # Let DuckDB resolve credentials the way every other AWS tool does: environment, shared
        # config, SSO. Without a secret an object store answers 403 and DuckDB reports it as a
        # database that does not exist, which is a confusing way to learn you are not signed in.
        connection.execute("CREATE SECRET IF NOT EXISTS (TYPE s3, PROVIDER credential_chain)")
    connection.execute(f"ATTACH '{target}' AS control (READ_ONLY)")
    connection.execute("USE control")
    return connection


def _metadata(raw: str | None) -> dict[str, Any]:
    """Parse a stored metadata blob.

    Returns:
        The parsed mapping, empty when the blob is absent.
    """
    return json.loads(raw) if raw else {}


def _annotation(row: Sequence[Any]) -> ResolvedAnnotation:
    """Build a resolved annotation from a query row.

    Returns:
        The annotation.
    """
    name, value, unit, span_type, start_us, end_us, provenance, confidence = row
    return ResolvedAnnotation(
        name=name,
        value=value,
        unit=unit,
        span_type=span_type,
        start_us=start_us,
        end_us=end_us,
        provenance=provenance,
        confidence=confidence,
    )


def render_task(reader: TimeFReader, task_id: str, *, include_values: bool = False) -> str:
    """Render one task as the text a training example starts from.

    This is the proposal's renderer: it walks the task's inputs, each record's source tree, each
    signal's spec and axis, and the annotations at every level, in order.

    Args:
        reader: The reader to pull from.
        task_id: The task to render.
        include_values: Also summarize each signal's values.

    Returns:
        The rendered example.
    """
    task = reader.task(task_id)
    lines = [f"PROMPT: {task.prompt}"]
    for annotation in task.annotations:
        lines.append(f"  [task] {annotation.to_text()}")
    for item in task.inputs:
        if isinstance(item, str):
            lines.append(f"INPUT text: {item}")
            continue
        lines.append(f"INPUT record {item.record_id}")
        for annotation in item.annotations:
            lines.append(f"  [record] {annotation.to_text()}")
        for source in item.walk_sources():
            lines.append(f"  {'  ' * source.depth}source: {source.name}")
            for annotation in source.annotations:
                lines.append(f"  {'  ' * source.depth}  [source] {annotation.to_text()}")
            for signal in source.signals:
                summary = f"{signal.name} ({signal.spec_type}, {signal.unit}, {signal.n_values} values)"
                if include_values:
                    values = reader.values(signal.signal_id)
                    summary += f" first={values[:3].tolist()}"
                lines.append(f"  {'  ' * source.depth}  signal: {summary}")
                for annotation in signal.annotations:
                    lines.append(f"  {'  ' * source.depth}    [signal] {annotation.to_text()}")
    for item in task.target:
        lines.append(f"TARGET: {item if isinstance(item, str) else f'record {item.record_id}'}")
    return "\n".join(lines)
