"""Write the in-memory TimeF hierarchy into ``control.duckdb``."""

from __future__ import annotations

from collections.abc import Callable, Container, Iterable, Iterator
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, assert_never

import duckdb
import numpy as np
import pyarrow as pa

from timenet.dataset import IrregularAxis, OrdinalAxis, Record, RegularAxis, Signal, TimeAxis, TimeFDataset
from timenet.dataset.axis import to_time_offsets_us
from timenet.errors import TimeFValidationError
from timenet.format.annotation_codec import encode_annotation_value
from timenet.format.checksums import CHECKSUM_PREFIX
from timenet.format.control_schema import LAST_ID_PREFIX, TABLES, Table
from timenet.format.duckdb import connect_control, create_control_schema, transaction
from timenet.format.task_codec import TaskRelationshipField, encode_span, encode_target, encode_task_payload
from timenet.json import validate_json_value
from timenet.types import (
    Annotation,
    IdKind,
    ObjectKind,
    Task,
    TSCorrespondenceTask,
    annotation_type_of,
)
from timenet.types.ids import EntityId, IdCounters
from timenet.types.splits import parse_split


if TYPE_CHECKING:
    from timenet.values_backends.writer import ChunkPlacement


_ROW_BATCH_SIZE = 10_000
_TASK_BATCH_SIZE = _ROW_BATCH_SIZE


def _json(value: object) -> str:
    """Return deterministic JSON after validating its value."""
    validated = validate_json_value(value)
    return json.dumps(validated, sort_keys=True, separators=(",", ":"), allow_nan=False)


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


def _reserve_keys(connection: duckdb.DuckDBPyConnection, sequence: str, count: int) -> list[int]:
    """Reserve generated integer keys from a known control-schema sequence.

    Returns:
        The reserved keys in sequence order.
    """
    if count == 0:
        return []
    return [
        int(row[0])
        for row in connection.execute(
            f"SELECT nextval('{sequence}') FROM range(?)",  # noqa: S608 - sequence is an internal constant
            [count],
        ).fetchall()
    ]


def _reserve_hierarchy_keys(
    connection: duckdb.DuckDBPyConnection,
    records: Iterable[Any],
) -> tuple[Iterator[int], Iterator[int]]:
    """Reserve exact key ranges for hierarchy objects and axes.

    Returns:
        Iterators over object keys and axis keys.
    """
    record_count = 0
    source_count = 0
    signal_count = 0
    axes: set[tuple[str, int]] = set()
    for record in records:
        record_count += 1
        for source in record.walk_sources():
            source_count += 1
            signal_count += len(source.signals)
            axes.update(_axis_identity(signal.time_axis) for signal in source.signals)
    object_keys = _reserve_keys(
        connection,
        "object_key_sequence",
        record_count + source_count + signal_count,
    )
    axis_keys = _reserve_keys(connection, "axis_key_sequence", len(axes))
    return iter(object_keys), iter(axis_keys)


def _axis_identity(axis: TimeAxis) -> tuple[str, int]:
    """Return what decides which Signals share one stored axis row.

    An axis read from storage keeps its stored id. Any other axis is shared by object.

    Returns:
        A hashable identity for the axis row.
    """
    if axis._stored_id is not None:
        return ("stored", axis._stored_id)
    return ("object", id(axis))


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


class _HierarchyKeys(NamedTuple):
    """Generated keys of the Sources and Signals written so far, by public ID."""

    sources: dict[int, int]
    signals: dict[int, int]


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
    key: int
    axis_id: EntityId
    offsets: _OffsetSignature | None
    offsets_loader: Callable[[], pa.Array] | None


class _HierarchyBatches:
    """Own the bounded row buffers for the stored hierarchy."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.records = _TableBatch(connection, TABLES["records"])
        self.clocks = _TableBatch(connection, TABLES["clocks"])
        self.sources = _TableBatch(connection, TABLES["sources"])
        self.axes = _TableBatch(connection, TABLES["axes"])
        self.axis_offsets = _TableBatch(connection, TABLES["axis_offsets"])
        self.signals = _TableBatch(connection, TABLES["signals"])

    def flush(self) -> None:
        """Write all remaining hierarchy rows."""
        self.records.flush()
        self.clocks.flush()
        self.sources.flush()
        self.axes.flush()
        self.axis_offsets.flush()
        self.signals.flush()


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
        self._ids = IdCounters()
        self._occurrence_ids: dict[int, EntityId] = {}

    def write_hierarchy(
        self,
        dataset: TimeFDataset,
        placements: dict[tuple[int, int], ChunkPlacement] | None = None,
        *,
        tasks: Iterable[Task] | None = None,
    ) -> dict[str, int]:
        """Write records, sources, signals, axes, and their annotations atomically.

        Args:
            dataset: The complete in-memory hierarchy.
            placements: Values-plane chunks keyed by ``(signal_id, chunk_index)``.
            tasks: A previously validated task stream, or ``None`` to use the dataset's tasks.

        Returns:
            Counts keyed by concrete task type.

        Raises:
            TimeFValidationError: If the hierarchy refers to an object outside the dataset, holds a
                value that cannot be stored as JSON, or declares Signal lengths that its axis offsets
                or value chunks do not match.
        """
        self._ids = IdCounters(dataset.last_ids)
        self._occurrence_ids: dict[int, EntityId] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with connect_control(self.path) as connection:
                create_control_schema(connection)
                with transaction(connection):
                    task_counts = self._write_objects(
                        connection,
                        dataset,
                        dataset.iter_tasks() if tasks is None else tasks,
                    )
                    self._write_chunks(connection, placements)
                    self._check_source_tree(connection)
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
    ) -> dict[str, int]:
        """Insert hierarchy rows into an open transaction.

        Sources, axes, and annotations have no public id. Each gets the storage id it was read
        with, or the next id of its kind.

        Returns:
            Counts keyed by concrete task type.
        """
        records = dataset.records
        object_keys, axis_keys = _reserve_hierarchy_keys(connection, records)
        batches = _HierarchyBatches(connection)
        axes: dict[tuple[str, int], _StoredAxis] = {}
        source_keys: dict[int, int] = {}
        clock_keys: dict[int, int] = {}
        signal_keys: dict[int, int] = {}
        connection.execute("INSERT INTO datasets (dataset_id) VALUES (?)", [dataset.metadata.dataset_id])
        annotations: list[tuple[ObjectKind, int | None, Annotation]] = [
            (ObjectKind.DATASET, None, annotation) for annotation in dataset.annotations
        ]
        record_keys: dict[int, int] = {}
        for record in records:
            identity = id(record.start_time)
            if identity not in clock_keys:
                clock_keys[identity] = len(clock_keys) + 1
                batches.clocks.add({"clock_id": clock_keys[identity], "start_time_us": record.start_time.timestamp})
            span = record.time_span
            record_key = next(object_keys)
            batches.records.add(
                {
                    "record_key": record_key,
                    "record_id": record.id,
                    "clock_id": clock_keys[identity],
                    "time_span_start_us": None if span is None else span.start_us,
                    "time_span_end_us": None if span is None else span.end_us,
                    "metadata": _json(record.metadata),
                }
            )
            record_keys[record.id] = record_key
            annotations.extend((ObjectKind.RECORD, record.id, annotation) for annotation in record.annotations)
            self._write_record_sources(
                record,
                record_key,
                axes,
                _HierarchyKeys(source_keys, signal_keys),
                annotations,
                object_keys,
                axis_keys,
                batches,
            )
        batches.flush()
        task_counts = self._write_tasks(connection, record_keys.keys(), signal_keys.keys(), annotations, tasks)
        # Clustering occurrences on object_id lets keyed reads skip row groups.
        annotations.sort(key=lambda item: (item[1] is not None, item[1] or 0))
        self._write_annotations(connection, annotations)
        self._check_task_references(connection)
        stored = self._ids.last()
        connection.executemany(
            "INSERT INTO control_metadata VALUES (?, ?)",
            [[f"{LAST_ID_PREFIX}{kind}", str(max(value, stored[kind]))] for kind, value in dataset.last_ids.items()],
        )
        return task_counts

    @staticmethod
    def _write_chunks(
        connection: duckdb.DuckDBPyConnection,
        placements: dict[tuple[int, int], ChunkPlacement] | None,
    ) -> None:
        """Insert values-plane chunk locations and check they cover every Signal exactly.

        Args:
            connection: The open write transaction.
            placements: Chunks keyed by ``(signal_id, chunk_index)``, or ``None`` for a
                metadata-only database that stores no values.

        Raises:
            TimeFValidationError: If a chunk names an unknown Signal, is empty, or the chunks of a
                Signal do not add up to its declared number of values.
        """
        if placements is None:
            return
        signals = {
            signal_id: (signal_key, n_values)
            for signal_id, signal_key, n_values in connection.execute(
                "SELECT signal_id, signal_key, n_values FROM signals"
            ).fetchall()
        }
        stored_values: dict[int, int] = dict.fromkeys(signals, 0)
        for (signal_id, chunk_index), placement in placements.items():
            if signal_id not in signals:
                raise TimeFValidationError(f"chunk {chunk_index} refers to unknown signal {signal_id!r}")
            if placement.n_values <= 0:
                raise TimeFValidationError(f"chunk {chunk_index} of signal {signal_id!r} is empty")
            stored_values[signal_id] += placement.n_values
        for signal_id, (_, n_values) in signals.items():
            if stored_values[signal_id] != n_values:
                raise TimeFValidationError(
                    f"signal {signal_id!r} declares {n_values} values but its chunks store {stored_values[signal_id]}"
                )
        chunks = _TableBatch(connection, TABLES["signal_chunks"])
        for (signal_id, chunk_index), placement in sorted(placements.items()):
            chunks.add(
                {
                    "signal_key": signals[signal_id][0],
                    "chunk_index": chunk_index,
                    "value_path": placement.chunk_file,
                    "chunk_major_index": placement.data_index.major_idx,
                    "chunk_minor_index": placement.data_index.minor_idx,
                    "n_values": placement.n_values,
                }
            )
        chunks.flush()

    def _write_tasks(
        self,
        connection: duckdb.DuckDBPyConnection,
        record_ids: Container[int],
        signal_ids: Container[int],
        annotations: list[tuple[ObjectKind, int | None, Annotation]],
        tasks_source: Iterable[Task],
    ) -> dict[str, int]:
        """Insert tasks and their relationships, one bounded batch at a time.

        Dependencies and annotation references may name tasks and occurrences that are written
        later, so :meth:`_check_task_references` checks them once everything is stored.

        Returns:
            Counts keyed by concrete task type.

        Raises:
            TimeFValidationError: If a task refers to a Record or Signal outside the dataset,
                derives from a task that is not in the dataset, or derives from itself.
        """
        # A streamed task can name a parent that streams later and gets its id then, so
        # dependencies are stored once every task is written.
        dependencies: list[tuple[int, int, Task]] = []
        task_counts: dict[str, int] = {}
        for batch in _task_batches(tasks_source):
            task_rows: list[dict[str, object]] = []
            target_rows: list[dict[str, object]] = []
            record_ref_rows: list[dict[str, object]] = []
            annotation_ref_rows: list[dict[str, object]] = []
            for task in batch:
                task_id = task.id
                task_type = str(task.task_type)
                task_counts[task_type] = task_counts.get(task_type, 0) + 1
                task_rows.append(
                    {
                        "task_id": task_id,
                        "task_type": task_type,
                        "prompt": task.prompt,
                        "input_modalities": sorted(item.value for item in task.resolved_input_modalities),
                        "split": None if task.split is None else parse_split(task.split).value,
                        **self._scope_columns(task, signal_ids),
                        "has_inline_targets": task.targets is not None,
                        **encode_task_payload(task),
                        "rationale": task.rationale,
                        "metadata": _json(task.metadata),
                    }
                )
                target_rows.extend(
                    self._target_row(task_id, position, target, record_ids, signal_ids)
                    for position, target in enumerate(task.targets or ())
                )
                annotations.extend((ObjectKind.TASK, task_id, annotation) for annotation in task.annotations)
                for field, records in self._task_record_refs(task):
                    for position, record_id in enumerate(records):
                        if record_id not in record_ids:
                            raise TimeFValidationError(
                                f"task {task_id!r} field {field.value!r} refers to unknown record {record_id!r}"
                            )
                        record_ref_rows.append(
                            {"task_id": task_id, "field": field, "position": position, "record_id": record_id}
                        )
                for field, values in (
                    (TaskRelationshipField.INPUT_ANNOTATIONS, task.input_annotations),
                    (TaskRelationshipField.TARGET_ANNOTATIONS, task.target_annotations),
                ):
                    for position, annotation in enumerate(values):
                        annotation_ref_rows.append(
                            {
                                "task_id": task_id,
                                "field": field,
                                "position": position,
                                "occurrence_id": self._occurrence_id(annotation),
                            }
                        )
                dependencies.extend((task_id, position, parent) for position, parent in enumerate(task.from_tasks))
            _insert_rows(connection, TABLES["tasks"], task_rows)
            _insert_rows(connection, TABLES["task_targets"], target_rows)
            _insert_rows(connection, TABLES["task_record_refs"], record_ref_rows)
            _insert_rows(connection, TABLES["task_annotation_refs"], annotation_ref_rows)
        dependency_rows: list[dict[str, object]] = []
        for task_id, position, parent in dependencies:
            if parent._id is None:
                raise TimeFValidationError(f"task {task_id!r} depends on a task that is not in the dataset")
            if parent.id == task_id:
                raise TimeFValidationError(f"task {task_id!r} depends on itself")
            dependency_rows.append({"task_id": task_id, "position": position, "parent_task_id": parent.id})
        _insert_rows(connection, TABLES["task_dependencies"], dependency_rows)
        return task_counts

    @staticmethod
    def _scope_columns(task: Task, signal_ids: Container[int]) -> dict[str, object]:
        """Project a task's scope onto the four scope columns.

        Returns:
            The scope columns, all ``None`` for a task without a scope.

        Raises:
            TimeFValidationError: If the scope names a Signal outside the dataset.
        """
        encoded = encode_span(task.scope)
        if encoded is None:
            return {"scope_type": None, "scope_start": None, "scope_end": None, "scope_signal_ids": None}
        scope_signal_ids = encoded["signal_ids"]
        missing = [item for item in scope_signal_ids or () if item not in signal_ids]
        if missing:
            raise TimeFValidationError(f"task {task.id!r} scope refers to unknown Signals {missing}")
        return {
            "scope_type": encoded["span_type"],
            "scope_start": encoded["span_start"],
            "scope_end": encoded["span_end"],
            "scope_signal_ids": scope_signal_ids,
        }

    @staticmethod
    def _target_row(
        task_id: int,
        position: int,
        target: object,
        record_ids: Container[int],
        signal_ids: Container[int],
    ) -> dict[str, object]:
        """Validate one target and project it onto a ``task_targets`` row.

        Returns:
            The row, with object references as public IDs.

        Raises:
            TimeFValidationError: If the target refers to an object outside the dataset.
        """
        row = encode_target(target)
        label = f"task {task_id!r} target {position}"
        record_id = row["record_id"]
        signal_id = row["signal_id"]
        if record_id is not None and record_id not in record_ids:
            raise TimeFValidationError(f"{label} refers to unknown Record {record_id!r}")
        if signal_id is not None and signal_id not in signal_ids:
            raise TimeFValidationError(f"{label} refers to unknown Signal {signal_id!r}")
        missing_signals = [item for item in row["span_signal_ids"] or () if item not in signal_ids]
        if missing_signals:
            raise TimeFValidationError(f"{label} refers to unknown Signals {missing_signals}")
        return {"task_id": task_id, "position": position, **row}

    @staticmethod
    def _task_record_refs(task: Task) -> tuple[tuple[TaskRelationshipField, tuple[int, ...]], ...]:
        """Project task record objects into normalized field rows.

        Returns:
            Pairs of field and ordered record IDs.
        """
        refs = [(TaskRelationshipField.INPUTS, tuple(record.id for record in task.inputs))]
        if isinstance(task, TSCorrespondenceTask):
            refs.append(
                (
                    TaskRelationshipField.CANDIDATE_RECORDS,
                    tuple(record.id for record in task.candidate_records),
                )
            )
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

    def _write_record_sources(  # noqa: PLR0913, PLR0917 - hierarchy maps stay explicit
        self,
        record: Record,
        record_key: int,
        axes: dict[tuple[str, int], _StoredAxis],
        keys: _HierarchyKeys,
        annotations: list[tuple[ObjectKind, int | None, Annotation]],
        object_keys: Iterator[int],
        axis_keys: Iterator[int],
        batches: _HierarchyBatches,
    ) -> None:
        """Insert one record's source tree and signals.

        Raises:
            TimeFValidationError: If the record has no Sources, or a Source or Signal ID is already
                used elsewhere in the dataset.
        """
        if not record.sources:
            raise TimeFValidationError(
                f"record {record.id!r} has no Sources; a Record stores its Signals under Sources"
            )

        def write_source(source: Any, parent_key: int | None) -> None:
            source_id = source._stored_id or self._ids.next(IdKind.SOURCE)
            if source_id in keys.sources:
                raise TimeFValidationError(f"source id {source_id!r} is not unique")
            source_key = next(object_keys)
            keys.sources[source_id] = source_key
            batches.sources.add(
                {
                    "source_key": source_key,
                    "source_id": source_id,
                    "record_key": record_key,
                    "parent_source_key": parent_key,
                    "name": source.name,
                    "metadata": _json(source.metadata),
                }
            )
            annotations.extend((ObjectKind.SOURCE, source_id, annotation) for annotation in source.annotations)
            for signal in source.signals:
                if signal.id in keys.signals:
                    raise TimeFValidationError(f"signal id {signal.id!r} is not unique")
                signal_key = self._write_signal(signal, source_key, axes, object_keys, axis_keys, batches)
                keys.signals[signal.id] = signal_key
                annotations.extend((ObjectKind.SIGNAL, signal.id, annotation) for annotation in signal.annotations)
            for child in source.sources:
                write_source(child, source_key)

        for source in record.sources:
            write_source(source, None)

    def _write_signal(  # noqa: PLR0913, PLR0917 - the hierarchy write state stays explicit
        self,
        signal: Signal,
        source_key: int,
        axes: dict[tuple[str, int], _StoredAxis],
        object_keys: Iterator[int],
        axis_keys: Iterator[int],
        batches: _HierarchyBatches,
    ) -> int:
        """Insert one signal and its shared axis.

        Returns:
            The generated Signal key.

        Raises:
            TimeFValidationError: If one stored axis id identifies different axis definitions, or an
                irregular axis has a different number of offsets than the Signal has values.
        """
        axis = signal.time_axis
        identity = _axis_identity(axis)
        existing = axes.get(identity)
        if existing is None:
            axis_key = next(axis_keys)
            axis_id = axis._stored_id or self._ids.next(IdKind.AXIS)
            offsets = self._write_axis(signal, axis_key, axis_id, batches)
            axes[identity] = _StoredAxis(axis, axis_key, axis_id, offsets, signal.time_offsets_loader)
        elif existing.axis != axis:
            raise TimeFValidationError(f"axis id {existing.axis_id!r} is reused with different definitions")
        else:
            axis_key = existing.key
            axis_id = existing.axis_id
        if existing is not None and isinstance(axis, IrregularAxis):
            if signal.time_offsets_loader is None:
                raise TimeFValidationError(f"irregular signal {signal.id!r} has no time-offset loader")
            stored_offsets = existing.offsets
            if stored_offsets is None:
                raise TimeFValidationError(f"irregular axis {axis_id!r} stores no time offsets")
            if signal.time_offsets_loader is not existing.offsets_loader:
                candidate = _validated_irregular_offsets(signal)
                candidate_signature = _offset_signature(candidate)
                if candidate_signature != stored_offsets:
                    raise TimeFValidationError(f"axis id {axis_id!r} is shared by signals with different time offsets")
            if stored_offsets.n_values != signal.n_values:
                raise TimeFValidationError(
                    f"signal {signal.id!r} declares {signal.n_values} values but its axis "
                    f"{axis_id!r} has {stored_offsets.n_values} time offsets"
                )

        spec = signal.spec
        signal_key = next(object_keys)
        batches.signals.add(
            {
                "signal_key": signal_key,
                "signal_id": signal.id,
                "source_key": source_key,
                "name": signal.name,
                "axis_key": axis_key,
                "spec_type": spec.spec_type,
                "spec_name": spec.name,
                "unit": None if spec.unit_value is None else str(spec.unit_value),
                "dtype": spec.dtype,
                "categories": list(spec.categories),
                "value_shape": list(spec.value_shape),
                "dimension_names": list(spec.dimension_names),
                "nullable": spec.nullable,
                "n_values": signal.n_values,
                "metadata": _json(signal.metadata),
                "modality": spec.modality.value,
            }
        )
        return signal_key

    @staticmethod
    def _write_axis(
        signal: Signal, axis_key: int, axis_id: EntityId, batches: _HierarchyBatches
    ) -> _OffsetSignature | None:
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
                    "axis_key": axis_key,
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
                    "axis_key": axis_key,
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
                batches.axis_offsets.add({"axis_key": axis_key, "position": position, "offset_us": int(offset)})
            return _offset_signature(offsets)
        if isinstance(axis, OrdinalAxis):
            batches.axes.add(
                {
                    "axis_key": axis_key,
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

    def _write_annotations(
        self,
        connection: duckdb.DuckDBPyConnection,
        annotations: Iterable[tuple[ObjectKind, int | None, Annotation]],
    ) -> None:
        """Insert reusable content once and every occurrence separately, in Arrow batches.

        Args:
            connection: The open write transaction.
            annotations: ``(object_type, object_id, annotation)`` triples. ``object_id`` is ``None``
                for the Dataset.

        Raises:
            TimeFValidationError: If one occurrence is attached twice, or one stored content id has
                two payloads.
        """
        contents: dict[tuple[str, int], tuple[EntityId, tuple[object, ...]]] = {}
        written: set[int] = set()
        written_ids: set[int] = set()
        content_batch = _TableBatch(connection, TABLES["annotation_contents"])
        occurrence_batch = _TableBatch(connection, TABLES["annotation_occurrences"])
        for object_type, object_id, annotation in annotations:
            if id(annotation) in written:
                raise TimeFValidationError(
                    f"an occurrence of annotation {annotation.key!r} is attached in more than one place"
                )
            written.add(id(annotation))
            occurrence_id = self._occurrence_id(annotation)
            if occurrence_id in written_ids:
                raise TimeFValidationError(f"annotation occurrence id {occurrence_id!r} is not unique")
            written_ids.add(occurrence_id)
            content = annotation._content_fields
            known = contents.get(annotation._content_identity)
            if known is not None and known[1] != content:
                raise TimeFValidationError(f"annotation content id {known[0]!r} is reused with different content")
            if known is None:
                known = (annotation._content_id or self._ids.next(IdKind.CONTENT), content)
                contents[annotation._content_identity] = known
                content_batch.add(
                    {
                        "content_id": known[0],
                        "name": annotation.name,
                        **encode_annotation_value(annotation.value, label=f"annotation {annotation.key!r}"),
                        "unit": annotation.unit,
                        "description": annotation.description,
                        "metadata": _json(annotation.metadata),
                    }
                )
            span = annotation.span
            occurrence_batch.add(
                {
                    "occurrence_id": occurrence_id,
                    "content_id": known[0],
                    "object_type": object_type,
                    "object_id": object_id,
                    "span_type": str(annotation_type_of(annotation)),
                    "start_us": None if span is None else span.start_us,
                    "end_us": None if span is None else span.exclusive_end,
                    "provenance": None if annotation.source is None else _json(annotation.source),
                    "confidence": annotation.confidence,
                    "metadata": _json(annotation.occurrence_metadata),
                }
            )
        content_batch.flush()
        occurrence_batch.flush()

    def _occurrence_id(self, annotation: Annotation) -> EntityId:
        """Return the storage id of one occurrence, numbering it the first time it is asked for.

        A task stores its annotation references before the occurrence rows are written, so both
        ask here and get the same id.

        Returns:
            The stored id the occurrence was read with, or the next occurrence id.
        """
        known = self._occurrence_ids.get(id(annotation))
        if known is None:
            known = annotation._occurrence_id or self._ids.next(IdKind.OCCURRENCE)
            self._occurrence_ids[id(annotation)] = known
        return known

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
               JOIN sources parent ON parent.source_key = child.parent_source_key
               WHERE child.record_key <> parent.record_key
               LIMIT 1"""
        ).fetchone()
        if wrong_parent is not None:
            raise TimeFValidationError(f"source {wrong_parent[0]!r} and its parent belong to different records")

        cycle = connection.execute(
            """WITH RECURSIVE ancestors(source_key, parent_source_key, path, cyclic) AS (
                   SELECT source_key, parent_source_key, [source_key], false FROM sources
                   UNION ALL
                   SELECT parent.source_key, parent.parent_source_key,
                          list_append(ancestors.path, parent.source_key),
                          list_contains(ancestors.path, parent.source_key)
                   FROM ancestors
                   JOIN sources parent ON parent.source_key = ancestors.parent_source_key
                   WHERE NOT ancestors.cyclic
               )
               SELECT source_key FROM ancestors WHERE cyclic LIMIT 1"""
        ).fetchone()
        if cycle is not None:
            raise TimeFValidationError(f"source hierarchy contains a cycle at {cycle[0]!r}")
