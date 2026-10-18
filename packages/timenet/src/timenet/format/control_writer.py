"""Write the in-memory TimeF hierarchy into ``control.duckdb``."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from hashlib import sha256
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, assert_never, cast

import duckdb
import numpy as np
import pyarrow as pa

from timenet.dataset import IrregularAxis, OrdinalAxis, RegularAxis, Signal, TimeAxis, TimeFDataset
from timenet.dataset.axis import to_time_offsets_us
from timenet.dataset.storage_order import ID_KINDS, iter_hierarchy_occurrences, iter_record_sources
from timenet.errors import TimeFValidationError
from timenet.format.annotation_codec import encode_annotation_value
from timenet.format.checksums import CHECKSUM_PREFIX
from timenet.format.control_schema import ID_WATERMARK_PREFIX, TABLES, Table
from timenet.format.duckdb import connect_control, create_control_schema, transaction
from timenet.format.final_ids import IdAssigner, IdKind, plan_final_ids
from timenet.format.task_codec import encode_span, encode_target, encode_task_payload
from timenet.json import canonical_json
from timenet.types import (
    Annotation,
    Task,
    TSCorrespondenceTask,
    annotation_type_of,
)
from timenet.types.ids import FINAL_ID_LIMIT
from timenet.types.splits import parse_split


if TYPE_CHECKING:
    from timenet.values_backends.writer import ChunkPlacement


_ROW_BATCH_SIZE = 10_000
_TASK_BATCH_SIZE = _ROW_BATCH_SIZE
_AXIS_OFFSETS_STAGING = "_timenet_axis_offsets_staging"

_ID_COLUMNS: tuple[tuple[str, str], ...] = (
    ("records", "record_id"),
    ("sources", "source_id"),
    ("sources", "record_id"),
    ("sources", "parent_source_id"),
    ("axes", "axis_id"),
    ("axis_offsets", "axis_id"),
    ("signals", "signal_id"),
    ("signals", "source_id"),
    ("signals", "axis_id"),
    ("signal_chunks", "signal_id"),
    ("annotation_contents", "content_id"),
    ("annotation_occurrences", "occurrence_id"),
    ("annotation_occurrences", "content_id"),
    ("annotation_occurrences", "object_id"),
    ("tasks", "task_id"),
    ("tasks", "list_max(scope_signal_ids)"),
    ("task_targets", "task_id"),
    ("task_targets", "record_id"),
    ("task_targets", "signal_id"),
    ("task_targets", "list_max(span_signal_ids)"),
    ("task_record_refs", "task_id"),
    ("task_record_refs", "record_id"),
    ("task_annotation_refs", "task_id"),
    ("task_annotation_refs", "occurrence_id"),
    ("task_dependencies", "task_id"),
    ("task_dependencies", "parent_task_id"),
)
"""Every stored id, as ``(table, expression)``. A stored version holds no provisional id."""

_WATERMARK_COLUMNS: dict[str, tuple[str, str]] = {
    "record": ("records", "record_id"),
    "source": ("sources", "source_id"),
    "signal": ("signals", "signal_id"),
    "axis": ("axes", "axis_id"),
    "content": ("annotation_contents", "content_id"),
    "occurrence": ("annotation_occurrences", "occurrence_id"),
    "task": ("tasks", "task_id"),
}
"""The primary id column of each id kind."""

_OWNER_KINDS: dict[str, IdKind] = {"Record": "record", "Source": "source", "Signal": "signal", "Task": "task"}
"""The id kind of each annotated object type except the Dataset."""


def _validated_irregular_offsets(signal: Signal) -> np.ndarray:
    """Load and validate the shared offsets declared by one irregular Signal.

    Returns:
        Canonical contiguous int64 offsets ready for shared storage.

    Raises:
        TimeFValidationError: If the offsets are absent, malformed, a different length from the
            Signal, or disagree with the axis endpoints.
    """
    loader = signal.time_offsets_loader
    axis = signal.time_axis
    if loader is None or not isinstance(axis, IrregularAxis):
        raise TimeFValidationError(f"irregular signal {signal.id!r} has no time-offset loader")
    offsets = to_time_offsets_us(loader().to_numpy(zero_copy_only=False))
    if len(offsets) != signal.n_values:
        raise TimeFValidationError(
            f"series {signal.id!r}: its time_offsets_loader returned {len(offsets)} time offsets "
            f"but it declares n_values={signal.n_values}"
        )
    if int(offsets[0]) != axis.first_us or int(offsets[-1]) != axis.last_us:
        raise TimeFValidationError(
            f"series {signal.id!r}: its axis claims the stream runs "
            f"{axis.first_us}..{axis.last_us} us but the stream runs "
            f"{int(offsets[0])}..{int(offsets[-1])} us"
        )
    return offsets


def _task_batches(tasks: Iterable[Task]) -> Iterator[tuple[Task, ...]]:
    """Yield bounded task batches without materializing a streamed source."""
    iterator = iter(tasks)
    while batch := tuple(islice(iterator, _TASK_BATCH_SIZE)):
        yield batch


def _insert_rows(connection: duckdb.DuckDBPyConnection, table: Table, rows: list[dict[str, object]]) -> None:
    """Insert one Arrow batch, typed by the table's schema, into a control table."""
    if not rows:
        return
    relation_name = f"_timenet_{table.name}_batch"
    names = ", ".join(table.column_names)
    connection.register(relation_name, pa.Table.from_pylist(rows, schema=table.arrow_schema))
    try:
        connection.execute(
            f"INSERT INTO {table.name} ({names}) SELECT {names} FROM {relation_name}"  # noqa: S608
        )
    finally:
        connection.unregister(relation_name)


class _TableBatch:
    """Insert bounded Arrow batches into one control table."""

    def __init__(self, connection: duckdb.DuckDBPyConnection, table: Table) -> None:
        self.connection = connection
        self.table = table
        self.rows: list[dict[str, object]] = []

    def add(self, row: dict[str, object]) -> None:
        """Add one row and write the batch when it reaches the size limit."""
        self.rows.append(row)
        if len(self.rows) >= _ROW_BATCH_SIZE:
            self.flush()

    def flush(self) -> None:
        """Write all buffered rows and clear the buffer."""
        _insert_rows(self.connection, self.table, self.rows)
        self.rows.clear()


class _OffsetSignature(NamedTuple):
    """Bounded identity for one validated irregular offset stream."""

    n_values: int
    checksum: str


def _offset_signature(offsets: np.ndarray) -> _OffsetSignature:
    """Return the length and algorithm-qualified digest of canonical offsets.

    Returns:
        A bounded identity for the offset stream.
    """
    checksum = CHECKSUM_PREFIX + sha256(memoryview(offsets)).hexdigest()
    return _OffsetSignature(len(offsets), checksum)


class _StoredAxis(NamedTuple):
    """One axis buffered for the control plane and shared by its Signals."""

    axis: TimeAxis
    offsets: _OffsetSignature | None
    offsets_loader: Callable[[], pa.Array] | None


class _HierarchyBatches:
    """Own the bounded row buffers for the stored hierarchy.

    Axis offsets go to a temporary table first. :meth:`finish` copies them into ``axis_offsets`` in
    ``axis_id`` order, so each axis's rows share row groups and a one-axis lookup skips the rest.
    """

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.connection = connection
        self.records = _TableBatch(connection, TABLES["records"])
        self.clocks = _TableBatch(connection, TABLES["clocks"])
        self.sources = _TableBatch(connection, TABLES["sources"])
        self.axes = _TableBatch(connection, TABLES["axes"])
        connection.execute(f"CREATE TEMP TABLE {_AXIS_OFFSETS_STAGING} AS FROM axis_offsets LIMIT 0")
        self.axis_offsets = _TableBatch(connection, replace(TABLES["axis_offsets"], name=_AXIS_OFFSETS_STAGING))
        self.signals = _TableBatch(connection, TABLES["signals"])

    def finish(self) -> None:
        """Write all remaining hierarchy rows and move the staged axis offsets into place."""
        self.records.flush()
        self.clocks.flush()
        self.sources.flush()
        self.axes.flush()
        self.axis_offsets.flush()
        self.signals.flush()
        self.connection.execute(f"INSERT INTO axis_offsets FROM {_AXIS_OFFSETS_STAGING} ORDER BY axis_id, position")
        self.connection.execute(f"DROP TABLE {_AXIS_OFFSETS_STAGING}")


class DuckDBControlWriter:
    """Serialize the relational part of one immutable TimeF version."""

    def __init__(self, path: Path) -> None:
        """Configure a new local control database.

        Args:
            path: Destination path. It must not already exist.

        Raises:
            FileExistsError: If ``path`` already exists.
        """
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"control database already exists at {self.path}")

    def write_hierarchy(
        self,
        dataset: TimeFDataset,
        placements: dict[tuple[int, int], ChunkPlacement] | None = None,
        *,
        tasks: Iterable[Task] | None = None,
        ids: IdAssigner | None = None,
    ) -> dict[str, int]:
        """Write records, sources, signals, axes, and their annotations atomically.

        Every stored id is final. The objects keep their in-memory ids, and each id is translated
        through ``ids`` as it is stored. The largest id of each kind is recorded so a later edit
        does not reuse it.

        Args:
            dataset: The complete in-memory hierarchy.
            placements: Values-plane chunks keyed by in-memory ``(signal_id, chunk_index)``.
            tasks: A previously validated task stream, or ``None`` to use the dataset's tasks. The
                tasks of a stream get their final ids as they pass.
            ids: The final-id plan from :func:`~timenet.format.final_ids.plan_final_ids`, or
                ``None`` to plan here. It must leave out the tasks of a stream.

        Returns:
            Counts keyed by concrete task type.

        Raises:
            TimeFValidationError: If the hierarchy repeats a public ID, refers to an object outside
                the dataset, shares one axis ID between different definitions, carries an unbound
                annotation, holds a value that cannot be stored as JSON, or declares Signal lengths
                that its axis offsets or value chunks do not match.
        """
        streamed = tasks is not None or dataset.has_task_stream
        if ids is None:
            ids = plan_final_ids(dataset, with_tasks=not streamed)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with connect_control(self.path) as connection:
                create_control_schema(connection)
                with transaction(connection):
                    task_counts = self._write_objects(
                        connection,
                        dataset,
                        dataset.iter_tasks() if tasks is None else tasks,
                        ids,
                        streamed=streamed,
                    )
                    self._write_chunks(connection, placements, ids)
                    self._check_source_tree(connection)
                    ids.check_unique()
                    self._check_final_ids(connection)
                    self._write_watermarks(connection, dataset._id_watermarks)
                connection.execute("CHECKPOINT")
        except duckdb.Error as exc:
            self.path.unlink(missing_ok=True)
            raise TimeFValidationError(f"could not write TimeF control database: {exc}") from exc
        except BaseException:
            self.path.unlink(missing_ok=True)
            raise
        return task_counts

    def _write_objects(
        self,
        connection: duckdb.DuckDBPyConnection,
        dataset: TimeFDataset,
        tasks: Iterable[Task],
        ids: IdAssigner,
        *,
        streamed: bool,
    ) -> dict[str, int]:
        """Insert hierarchy rows into an open transaction.

        Returns:
            Counts keyed by concrete task type.

        Raises:
            TimeFValidationError: If two records share an ID.
        """
        batches = _HierarchyBatches(connection)
        axes: dict[int, _StoredAxis] = {}
        source_ids: set[int] = set()
        clock_keys: dict[int, int] = {}
        signal_ids: set[int] = set()
        connection.execute("INSERT INTO datasets (dataset_id) VALUES (?)", [dataset.metadata.dataset_id])
        record_ids: set[int] = set()
        for record in dataset.records:
            if record.id in record_ids:
                raise TimeFValidationError(f"record id {record.id!r} is not unique")
            identity = id(record.start_time)
            if identity not in clock_keys:
                clock_keys[identity] = len(clock_keys) + 1
                batches.clocks.add({"clock_id": clock_keys[identity], "start_time_us": record.start_time.timestamp})
            span = record.time_span
            batches.records.add(
                {
                    "record_id": ids.final("record", record.id),
                    "clock_id": clock_keys[identity],
                    "time_span_start_us": None if span is None else span.start_us,
                    "time_span_end_us": None if span is None else span.end_us,
                    "metadata": canonical_json(record.metadata),
                }
            )
            record_ids.add(record.id)
            self._write_record_sources(record, axes, source_ids, signal_ids, batches, ids)
        batches.finish()
        # Occurrences are stored grouped by owner type, then in owner order, so keyed reads skip row groups.
        annotations: list[tuple[str, int | None, Annotation]] = [
            (object_type, None if owner is None else ids.final(_OWNER_KINDS[object_type], owner.id), annotation)
            for object_type, owner, annotation in iter_hierarchy_occurrences(dataset)
        ]
        task_counts = self._write_tasks(connection, ids, annotations, tasks, streamed=streamed)
        self._write_annotations(connection, annotations, ids)
        self._check_task_references(connection)
        return task_counts

    @staticmethod
    def _write_chunks(
        connection: duckdb.DuckDBPyConnection,
        placements: dict[tuple[int, int], ChunkPlacement] | None,
        ids: IdAssigner,
    ) -> None:
        """Insert values-plane chunk locations and check they cover every Signal exactly.

        Args:
            connection: The open write transaction.
            placements: Chunks keyed by in-memory ``(signal_id, chunk_index)``, or ``None`` for a
                metadata-only database that stores no values.
            ids: The final-id plan.

        Raises:
            TimeFValidationError: If a chunk names an unknown Signal, is empty, or the chunks of a
                Signal do not add up to its declared number of values.
        """
        if placements is None:
            return
        translated: dict[tuple[int, int], ChunkPlacement] = {}
        for (signal_id, chunk_index), placement in placements.items():
            final = ids.get("signal", signal_id)
            if final is None:
                raise TimeFValidationError(f"chunk {chunk_index} refers to unknown signal {signal_id!r}")
            translated[final, chunk_index] = placement
        placements = translated
        signals: dict[int, int] = dict(connection.execute("SELECT signal_id, n_values FROM signals").fetchall())
        stored_values: dict[int, int] = dict.fromkeys(signals, 0)
        for (signal_id, chunk_index), placement in placements.items():
            if signal_id not in signals:
                raise TimeFValidationError(f"chunk {chunk_index} refers to unknown signal {signal_id!r}")
            if placement.n_values <= 0:
                raise TimeFValidationError(f"chunk {chunk_index} of signal {signal_id!r} is empty")
            stored_values[signal_id] += placement.n_values
        for signal_id, n_values in signals.items():
            if stored_values[signal_id] != n_values:
                raise TimeFValidationError(
                    f"signal {signal_id!r} declares {n_values} values but its chunks store {stored_values[signal_id]}"
                )
        chunks = _TableBatch(connection, TABLES["signal_chunks"])
        for (signal_id, chunk_index), placement in sorted(placements.items()):
            chunks.add(
                {
                    "signal_id": signal_id,
                    "chunk_index": chunk_index,
                    "value_path": placement.chunk_file,
                    "chunk_major_index": placement.data_index.major_idx,
                    "chunk_minor_index": placement.data_index.minor_idx,
                    "n_values": placement.n_values,
                }
            )
        chunks.flush()

    def _write_tasks(  # noqa: PLR0912 - one block per task relationship table
        self,
        connection: duckdb.DuckDBPyConnection,
        ids: IdAssigner,
        annotations: list[tuple[str, int | None, Annotation]],
        tasks_source: Iterable[Task],
        *,
        streamed: bool,
    ) -> dict[str, int]:
        """Insert tasks and their relationships, one bounded batch at a time.

        A streamed task gets its final ids as it passes. A streamed task may derive from a task that
        streams later, so a dependency on a task without a final id yet waits until the stream ends.

        Returns:
            Counts keyed by concrete task type.

        Raises:
            TimeFValidationError: If a task ID repeats, a task refers to a Record, Signal, task, or
                annotation occurrence outside the dataset or to an unbound annotation, or a task
                derives from itself.
        """
        task_counts: dict[str, int] = {}
        waiting: list[tuple[Task, int, int, int]] = []
        for batch in _task_batches(tasks_source):
            task_rows: list[dict[str, object]] = []
            target_rows: list[dict[str, object]] = []
            record_ref_rows: list[dict[str, object]] = []
            annotation_ref_rows: list[dict[str, object]] = []
            dependency_rows: list[dict[str, object]] = []
            for task in batch:
                if streamed:
                    ids.stream_task(task)
                task_id = ids.get("task", task.id)
                if task_id is None:
                    raise TimeFValidationError(f"task {task.id!r} is not in the dataset")
                task_type = str(task.task_type)
                task_counts[task_type] = task_counts.get(task_type, 0) + 1
                task_rows.append(
                    {
                        "task_id": task_id,
                        "task_type": task_type,
                        "prompt": task.prompt,
                        "input_modalities": sorted(item.value for item in task.resolved_input_modalities),
                        "split": None if task.split is None else parse_split(task.split).value,
                        **self._scope_columns(task, ids),
                        "has_inline_targets": task.targets is not None,
                        **encode_task_payload(task),
                        "rationale": task.rationale,
                        "metadata": canonical_json(task.metadata),
                    }
                )
                target_rows.extend(
                    self._target_row(task, task_id, position, target, ids)
                    for position, target in enumerate(task.targets or ())
                )
                annotations.extend(("Task", task_id, annotation) for annotation in task.annotations)
                for field, records in self._task_record_refs(task):
                    for position, record_id in enumerate(records):
                        final_record = ids.get("record", record_id)
                        if final_record is None:
                            raise TimeFValidationError(
                                f"task {task.id!r} field {field!r} refers to unknown record {record_id!r}"
                            )
                        record_ref_rows.append(
                            {"task_id": task_id, "field": field, "position": position, "record_id": final_record}
                        )
                for field, values in (
                    ("input_annotations", task.input_annotations),
                    ("target_annotations", task.target_annotations),
                ):
                    for position, annotation in enumerate(values):
                        if annotation.occurrence_id is None:
                            raise TimeFValidationError(
                                f"task {task.id!r} field {field!r} refers to an unbound annotation"
                            )
                        occurrence_id = ids.get("occurrence", annotation.occurrence_id)
                        if occurrence_id is None:
                            raise TimeFValidationError(
                                f"task {task.id!r} field {field!r} refers to unknown annotation occurrence "
                                f"{annotation.occurrence_id!r}"
                            )
                        annotation_ref_rows.append(
                            {"task_id": task_id, "field": field, "position": position, "occurrence_id": occurrence_id}
                        )
                for position, parent in enumerate(task.from_tasks):
                    if parent.id == task.id:
                        raise TimeFValidationError(f"task {task.id!r} depends on itself")
                    parent_id = ids.get("task", parent.id)
                    if parent_id is None:
                        waiting.append((task, task_id, position, parent.id))
                    else:
                        dependency_rows.append({"task_id": task_id, "position": position, "parent_task_id": parent_id})
            _insert_rows(connection, TABLES["tasks"], task_rows)
            _insert_rows(connection, TABLES["task_targets"], target_rows)
            _insert_rows(connection, TABLES["task_record_refs"], record_ref_rows)
            _insert_rows(connection, TABLES["task_annotation_refs"], annotation_ref_rows)
            _insert_rows(connection, TABLES["task_dependencies"], dependency_rows)
        dependency_rows = []
        for task, task_id, position, parent in waiting:
            parent_id = ids.get("task", parent)
            if parent_id is None:
                raise TimeFValidationError(f"task {task.id!r} depends on task {parent!r}, which is not in the dataset")
            dependency_rows.append({"task_id": task_id, "position": position, "parent_task_id": parent_id})
        _insert_rows(connection, TABLES["task_dependencies"], dependency_rows)
        return task_counts

    @staticmethod
    def _check_final_ids(connection: duckdb.DuckDBPyConnection) -> None:
        """Confirm no stored id, key or reference, is provisional.

        Raises:
            TimeFValidationError: If a column holds an id at or above ``2**52``.
        """
        query = " UNION ALL ".join(
            f"SELECT {position} AS position, '{table}.{column}' AS name, max({column}) AS largest FROM {table}"  # noqa: S608
            for position, (table, column) in enumerate(_ID_COLUMNS)
        )
        row = connection.execute(
            f"SELECT name, largest FROM ({query}) WHERE largest >= ? ORDER BY position LIMIT 1",  # noqa: S608
            [FINAL_ID_LIMIT],
        ).fetchone()
        if row is not None:
            raise TimeFValidationError(f"{row[0]} stores the provisional id {row[1]!r}; the writer missed a reference")

    @staticmethod
    def _write_watermarks(connection: duckdb.DuckDBPyConnection, floors: dict[str, int]) -> None:
        """Record, per kind, the largest id this version or any version it derives from stored."""
        for kind in ID_KINDS:
            table, column = _WATERMARK_COLUMNS[kind]
            row = connection.execute(f"SELECT coalesce(max({column}), 0) FROM {table}").fetchone()  # noqa: S608
            largest = max(floors.get(kind, 0), 0 if row is None else int(row[0]))
            connection.execute("INSERT INTO control_metadata VALUES (?, ?)", [ID_WATERMARK_PREFIX + kind, str(largest)])

    @staticmethod
    def _final_signal_ids(signal_ids: tuple[int, ...] | None, ids: IdAssigner) -> tuple[list[int] | None, list[int]]:
        """Translate the Signal ids of one span.

        Returns:
            The final ids, ``None`` for a span over every Signal, and the ids outside the dataset.
        """
        if signal_ids is None:
            return None, []
        finals = [ids.get("signal", item) for item in signal_ids]
        missing = [item for item, final in zip(signal_ids, finals, strict=True) if final is None]
        return [final for final in finals if final is not None], missing

    @classmethod
    def _scope_columns(cls, task: Task, ids: IdAssigner) -> dict[str, object]:
        """Project a task's scope onto the four scope columns.

        Returns:
            The scope columns, all ``None`` for a task without a scope.

        Raises:
            TimeFValidationError: If the scope names a Signal outside the dataset.
        """
        encoded = encode_span(task.scope)
        if encoded is None:
            return {"scope_type": None, "scope_start": None, "scope_end": None, "scope_signal_ids": None}
        scope_signal_ids, missing = cls._final_signal_ids(encoded["signal_ids"], ids)
        if missing:
            raise TimeFValidationError(f"task {task.id!r} scope refers to unknown Signals {missing}")
        return {
            "scope_type": encoded["span_type"],
            "scope_start": encoded["span_start"],
            "scope_end": encoded["span_end"],
            "scope_signal_ids": scope_signal_ids,
        }

    @classmethod
    def _target_row(
        cls,
        task: Task,
        task_id: int,
        position: int,
        target: object,
        ids: IdAssigner,
    ) -> dict[str, object]:
        """Validate one target and project it onto a ``task_targets`` row.

        Returns:
            The row, with object references as final IDs.

        Raises:
            TimeFValidationError: If the target refers to an object outside the dataset.
        """
        row = encode_target(target)
        label = f"task {task.id!r} target {position}"
        record_id = cast("int | None", row["record_id"])
        signal_id = cast("int | None", row["signal_id"])
        if record_id is not None:
            row["record_id"] = ids.get("record", record_id)
            if row["record_id"] is None:
                raise TimeFValidationError(f"{label} refers to unknown Record {record_id!r}")
        if signal_id is not None:
            row["signal_id"] = ids.get("signal", signal_id)
            if row["signal_id"] is None:
                raise TimeFValidationError(f"{label} refers to unknown Signal {signal_id!r}")
        span_signal_ids, missing_signals = cls._final_signal_ids(
            cast("tuple[int, ...] | None", row["span_signal_ids"]), ids
        )
        if missing_signals:
            raise TimeFValidationError(f"{label} refers to unknown Signals {missing_signals}")
        row["span_signal_ids"] = span_signal_ids
        return {"task_id": task_id, "position": position, **row}

    @staticmethod
    def _task_record_refs(task: Task) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """Project task record objects into normalized field rows.

        Returns:
            Pairs of public field name and ordered record IDs.
        """
        refs: list[tuple[str, tuple[int, ...]]] = [("inputs", tuple(record.id for record in task.inputs))]
        if isinstance(task, TSCorrespondenceTask):
            refs.append(("candidate_records", tuple(record.id for record in task.candidate_records)))
        return tuple(refs)

    @staticmethod
    def _check_task_references(connection: duckdb.DuckDBPyConnection) -> None:
        """Confirm every stored task dependency and annotation reference names a stored row.

        Raises:
            TimeFValidationError: If a task depends on a task outside the dataset, or refers to an
                annotation occurrence that no stored object carries.
        """
        unknown_parent = connection.execute(
            """SELECT d.task_id, d.parent_task_id
               FROM task_dependencies d
               LEFT JOIN tasks parent ON parent.task_id = d.parent_task_id
               WHERE parent.task_id IS NULL
               ORDER BY d.task_id, d.position
               LIMIT 1"""
        ).fetchone()
        if unknown_parent is not None:
            task_id, parent_id = unknown_parent
            raise TimeFValidationError(f"task {task_id!r} depends on task {parent_id!r}, which is not in the dataset")
        unknown_occurrence = connection.execute(
            """SELECT r.task_id, r.field, r.occurrence_id
               FROM task_annotation_refs r
               LEFT JOIN annotation_occurrences o ON o.occurrence_id = r.occurrence_id
               WHERE o.occurrence_id IS NULL
               ORDER BY r.task_id, r.field, r.position
               LIMIT 1"""
        ).fetchone()
        if unknown_occurrence is not None:
            task_id, field, occurrence_id = unknown_occurrence
            raise TimeFValidationError(
                f"task {task_id!r} field {field!r} refers to unknown annotation occurrence {occurrence_id!r}"
            )

    def _write_record_sources(  # noqa: PLR0913, PLR0917 - one argument per lookup and buffer
        self,
        record: Any,
        axes: dict[int, _StoredAxis],
        source_ids: set[int],
        signal_ids: set[int],
        batches: _HierarchyBatches,
        ids: IdAssigner,
    ) -> None:
        """Insert one record's source tree and signals, in the storage order final ids follow.

        Raises:
            TimeFValidationError: If the record has no Sources, or a Source or Signal ID is already
                used elsewhere in the dataset.
        """
        if not record.sources:
            raise TimeFValidationError(
                f"record {record.id!r} has no Sources; a Record stores its Signals under Sources"
            )
        parent_ids: dict[int, int] = {}
        for source in iter_record_sources(record):
            if source.id in source_ids:
                raise TimeFValidationError(f"source id {source.id!r} is not unique")
            source_ids.add(source.id)
            final_source = ids.final("source", source.id)
            batches.sources.add(
                {
                    "source_id": final_source,
                    "record_id": ids.final("record", record.id),
                    "parent_source_id": parent_ids.get(id(source)),
                    "name": source.name,
                    "metadata": canonical_json(source.metadata),
                }
            )
            for signal in source.signals:
                if signal.id in signal_ids:
                    raise TimeFValidationError(f"signal id {signal.id!r} is not unique")
                signal_ids.add(signal.id)
                self._write_signal(signal, final_source, axes, batches, ids)
            for child in source.sources:
                parent_ids[id(child)] = final_source

    def _write_signal(
        self,
        signal: Signal,
        source_id: int,
        axes: dict[int, _StoredAxis],
        batches: _HierarchyBatches,
        ids: IdAssigner,
    ) -> None:
        """Insert one signal and its shared axis.

        Args:
            signal: The signal.
            source_id: The final id of its source.
            axes: The axes stored so far, by final id.
            batches: The row buffers.
            ids: The final-id plan.

        Raises:
            TimeFValidationError: If one axis ID identifies different axis definitions, or an
                irregular axis has a different number of offsets than the Signal has values.
        """
        axis = signal.time_axis
        axis_id = ids.final("axis", axis.id)
        existing = axes.get(axis_id)
        if existing is None:
            offsets = self._write_axis(signal, axis_id, batches)
            axes[axis_id] = _StoredAxis(axis, offsets, signal.time_offsets_loader)
        elif existing.axis != axis:
            raise TimeFValidationError(f"axis id {axis.id!r} is reused with different definitions")
        if existing is not None and isinstance(axis, IrregularAxis):
            if signal.time_offsets_loader is None:
                raise TimeFValidationError(f"irregular signal {signal.id!r} has no time-offset loader")
            stored_offsets = existing.offsets
            if stored_offsets is None:
                raise TimeFValidationError(f"irregular axis {axis.id!r} stores no time offsets")
            if signal.time_offsets_loader is not existing.offsets_loader:
                candidate = _validated_irregular_offsets(signal)
                candidate_signature = _offset_signature(candidate)
                if candidate_signature != stored_offsets:
                    raise TimeFValidationError(f"axis id {axis.id!r} is shared by signals with different time offsets")
            if stored_offsets.n_values != signal.n_values:
                raise TimeFValidationError(
                    f"signal {signal.id!r} declares {signal.n_values} values but its axis "
                    f"{axis.id!r} has {stored_offsets.n_values} time offsets"
                )

        spec = signal.spec
        batches.signals.add(
            {
                "signal_id": ids.final("signal", signal.id),
                "source_id": source_id,
                "name": signal.name,
                "axis_id": axis_id,
                "spec_type": spec.spec_type,
                "spec_name": spec.name,
                "unit": None if spec.unit_value is None else str(spec.unit_value),
                "dtype": spec.dtype,
                "categories": list(spec.categories),
                "value_shape": list(spec.value_shape),
                "dimension_names": list(spec.dimension_names),
                "nullable": spec.nullable,
                "n_values": signal.n_values,
                "metadata": canonical_json(signal.metadata),
                "modality": spec.modality.value,
            }
        )

    @staticmethod
    def _write_axis(signal: Signal, axis_id: int, batches: _HierarchyBatches) -> _OffsetSignature | None:
        """Buffer an axis and any irregular offsets.

        Returns:
            A bounded signature of the irregular offsets, or ``None`` for other axis types.

        Raises:
            TimeFValidationError: If an irregular signal has no offsets loader.
        """
        axis = signal.time_axis
        if isinstance(axis, RegularAxis):
            batches.axes.add(
                {
                    "axis_id": axis_id,
                    "axis_type": str(axis.axis_type),
                    "period_numerator_us": axis.period_us.numerator,
                    "period_denominator": axis.period_us.denominator,
                    "start_index": axis.start_index,
                    "offset_us": axis.offset_us,
                    "first_us": None,
                    "last_us": None,
                }
            )
            return None
        if isinstance(axis, IrregularAxis):
            batches.axes.add(
                {
                    "axis_id": axis_id,
                    "axis_type": str(axis.axis_type),
                    "period_numerator_us": None,
                    "period_denominator": None,
                    "start_index": None,
                    "offset_us": None,
                    "first_us": axis.first_us,
                    "last_us": axis.last_us,
                }
            )
            if signal.time_offsets_loader is None:
                raise TimeFValidationError(f"irregular signal {signal.id!r} has no time-offset loader")
            offsets = _validated_irregular_offsets(signal)
            for position, offset in enumerate(offsets):
                batches.axis_offsets.add({"axis_id": axis_id, "position": position, "offset_us": int(offset)})
            return _offset_signature(offsets)
        if isinstance(axis, OrdinalAxis):
            batches.axes.add(
                {
                    "axis_id": axis_id,
                    "axis_type": str(axis.axis_type),
                    "period_numerator_us": None,
                    "period_denominator": None,
                    "start_index": None,
                    "offset_us": None,
                    "first_us": None,
                    "last_us": None,
                }
            )
            return None
        assert_never(axis)

    @staticmethod
    def _write_annotations(
        connection: duckdb.DuckDBPyConnection,
        annotations: Iterable[tuple[str, int | None, Annotation]],
        ids: IdAssigner,
    ) -> None:
        """Insert reusable content once and every occurrence separately, in Arrow batches.

        Args:
            connection: The open write transaction.
            annotations: ``(object_type, object_id, annotation)`` triples. ``object_id`` is the
                owner's final id, or ``None`` for the Dataset.
            ids: The final-id plan, which already holds every occurrence and content.

        Raises:
            TimeFValidationError: If an annotation is unbound.
        """
        stored_contents: set[int] = set()
        content_batch = _TableBatch(connection, TABLES["annotation_contents"])
        occurrence_batch = _TableBatch(connection, TABLES["annotation_occurrences"])
        for object_type, object_id, annotation in annotations:
            if annotation.occurrence_id is None:
                owner = object_type if object_id is None else f"{object_type} {object_id!r}"
                raise TimeFValidationError(
                    f"annotation {annotation.id!r} on {owner} has no occurrence id; attach it with annotate()"
                )
            content_id = ids.final("content", annotation.id)
            if content_id not in stored_contents:
                stored_contents.add(content_id)
                content_batch.add(
                    {
                        "content_id": content_id,
                        "name": annotation.name,
                        **encode_annotation_value(annotation.value, label=f"annotation {annotation.id!r}"),
                        "unit": annotation.unit,
                        "description": annotation.description,
                        "metadata": canonical_json(annotation.metadata),
                    }
                )
            span = annotation.span
            occurrence_batch.add(
                {
                    "occurrence_id": ids.final("occurrence", annotation.occurrence_id),
                    "content_id": content_id,
                    "object_type": object_type,
                    "object_id": object_id,
                    "span_type": str(annotation_type_of(annotation)),
                    "start_us": None if span is None else span.start_us,
                    "end_us": None if span is None else span.exclusive_end,
                    "provenance": None if annotation.source is None else canonical_json(annotation.source),
                    "confidence": annotation.confidence,
                    "metadata": canonical_json(annotation.occurrence_metadata),
                }
            )
        content_batch.flush()
        occurrence_batch.flush()

    @staticmethod
    def _check_source_tree(connection: duckdb.DuckDBPyConnection) -> None:
        """Confirm the stored Sources form one tree per Record.

        The writer walks each Record's Source tree recursively, so both conditions hold by
        construction. They are checked in SQL anyway because ``sources`` is small and the recursive
        query is the clearest statement of the invariant.

        Raises:
            TimeFValidationError: If a Source and its parent belong to different Records, or the
                parent relationship contains a cycle.
        """
        wrong_parent = connection.execute(
            """SELECT child.source_id
               FROM sources child
               JOIN sources parent ON parent.source_id = child.parent_source_id
               WHERE child.record_id <> parent.record_id
               LIMIT 1"""
        ).fetchone()
        if wrong_parent is not None:
            raise TimeFValidationError(f"source {wrong_parent[0]!r} and its parent belong to different records")

        cycle = connection.execute(
            """WITH RECURSIVE ancestors(source_id, parent_source_id, path, cyclic) AS (
                   SELECT source_id, parent_source_id, [source_id], false FROM sources
                   UNION ALL
                   SELECT parent.source_id, parent.parent_source_id,
                          list_append(ancestors.path, parent.source_id),
                          list_contains(ancestors.path, parent.source_id)
                   FROM ancestors
                   JOIN sources parent ON parent.source_id = ancestors.parent_source_id
                   WHERE NOT ancestors.cyclic
               )
               SELECT source_id FROM ancestors WHERE cyclic LIMIT 1"""
        ).fetchone()
        if cycle is not None:
            raise TimeFValidationError(f"source hierarchy contains a cycle at {cycle[0]!r}")
