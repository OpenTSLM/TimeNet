"""Write the in-memory TimeF hierarchy into ``control.duckdb``."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, assert_never, cast

import duckdb
import pyarrow as pa

from timenet.dataset import IrregularAxis, OrdinalAxis, RegularAxis, Signal, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.format.annotation_codec import encode_annotation_value
from timenet.format.control_schema import TABLES, Table
from timenet.format.duckdb import connect_control, create_control_schema, transaction
from timenet.format.task_codec import encode_span, encode_target, encode_task_payload
from timenet.types import (
    Annotation,
    Task,
    TSCorrespondenceTask,
    annotation_type_of,
)


if TYPE_CHECKING:
    from timenet.values_backends.writer import ChunkPlacement


_ROW_BATCH_SIZE = 10_000
_TASK_BATCH_SIZE = _ROW_BATCH_SIZE


def _json(value: object) -> str:
    """Return deterministic JSON and name unsupported values as validation failures.

    Raises:
        TimeFValidationError: If ``value`` is not JSON-compatible.
    """
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TimeFValidationError(f"value is not JSON-compatible: {value!r}") from exc


def _returned_key(cursor: duckdb.DuckDBPyConnection) -> int:
    """Read one generated integer key from an ``INSERT ... RETURNING`` statement.

    Returns:
        The generated key.

    Raises:
        TimeFValidationError: If DuckDB does not return a key.
    """
    row = cursor.fetchone()
    if row is None:
        raise TimeFValidationError("DuckDB did not return a generated internal key")
    return int(row[0])


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
    axis_ids: set[str] = set()
    for record in records:
        record_count += 1
        for source in record.walk_sources():
            source_count += 1
            signal_count += len(source.signals)
            axis_ids.update(signal.time_axis.axis_id for signal in source.signals)
    object_keys = _reserve_keys(
        connection,
        "object_key_sequence",
        record_count + source_count + signal_count,
    )
    axis_keys = _reserve_keys(connection, "axis_key_sequence", len(axis_ids))
    return iter(object_keys), iter(axis_keys)


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


class _SequenceKeys:
    """Hand out keys from a control-schema sequence, reserving them in blocks."""

    def __init__(self, connection: duckdb.DuckDBPyConnection, sequence: str) -> None:
        self.connection = connection
        self.sequence = sequence
        self._reserved: Iterator[int] = iter(())

    def next(self) -> int:
        """Return the next key, reserving another block when the current one is used up.

        Returns:
            One key that no other row will get.
        """
        for key in self._reserved:
            return key
        self._reserved = iter(_reserve_keys(self.connection, self.sequence, _ROW_BATCH_SIZE))
        return next(self._reserved)


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

    sources: dict[str, int]
    signals: dict[str, int]


class _HierarchyBatches:
    """Own the bounded row buffers for the stored hierarchy."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.records = _TableBatch(connection, TABLES["records"])
        self.sources = _TableBatch(connection, TABLES["sources"])
        self.axes = _TableBatch(connection, TABLES["axes"])
        self.axis_offsets = _TableBatch(connection, TABLES["axis_offsets"])
        self.signals = _TableBatch(connection, TABLES["signals"])

    def flush(self) -> None:
        """Write all remaining hierarchy rows."""
        self.records.flush()
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

    def write_hierarchy(
        self,
        dataset: TimeFDataset,
        placements: dict[tuple[str, int], ChunkPlacement] | None = None,
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
            TimeFValidationError: If the hierarchy repeats a public ID, refers to an object outside
                the dataset, shares one axis ID between different definitions, carries an unbound
                annotation, holds a value that cannot be stored as JSON, or declares Signal lengths
                that its axis offsets or value chunks do not match.
        """
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

        Returns:
            Counts keyed by concrete task type.

        Raises:
            TimeFValidationError: If two records share an ID.
        """
        records = dataset.records
        object_keys, axis_keys = _reserve_hierarchy_keys(connection, records)
        batches = _HierarchyBatches(connection)
        axes: dict[str, tuple[object, int, pa.Array | None]] = {}
        source_keys: dict[str, int] = {}
        signal_keys: dict[str, int] = {}
        dataset_key = _returned_key(
            connection.execute(
                "INSERT INTO datasets (dataset_id) VALUES (?) RETURNING dataset_key",
                [dataset.metadata.dataset_id],
            )
        )
        annotations: list[tuple[str, int, Annotation]] = [
            ("Dataset", dataset_key, annotation) for annotation in dataset.annotations
        ]
        record_keys: dict[str, int] = {}
        for record in records:
            if record.id in record_keys:
                raise TimeFValidationError(f"record id {record.id!r} is not unique")
            span = record.time_span
            record_key = next(object_keys)
            batches.records.add(
                {
                    "record_key": record_key,
                    "record_id": record.record_id,
                    "start_time_us": record.start_time,
                    "time_span_start_us": None if span is None else span.start_us,
                    "time_span_end_us": None if span is None else span.end_us,
                    "metadata": _json(record.metadata),
                }
            )
            record_keys[record.id] = record_key
            annotations.extend(("Record", record_key, annotation) for annotation in record.annotations)
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
        annotation_refs, task_counts = self._write_tasks(
            connection,
            record_keys,
            signal_keys,
            annotations,
            tasks,
        )
        occurrence_keys = self._write_annotations(connection, annotations, signal_keys)
        self._write_task_annotation_refs(connection, annotation_refs, occurrence_keys)
        return task_counts

    @staticmethod
    def _write_chunks(
        connection: duckdb.DuckDBPyConnection,
        placements: dict[tuple[str, int], ChunkPlacement] | None,
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
        stored_values: dict[str, int] = dict.fromkeys(signals, 0)
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
        record_keys: dict[str, int],
        signal_keys: dict[str, int],
        annotations: list[tuple[str, int, Annotation]],
        tasks_source: Iterable[Task],
    ) -> tuple[list[tuple[int, str, int, str]], dict[str, int]]:
        """Insert tasks and normalized object relationships.

        Returns:
            Deferred task-to-annotation rows and counts keyed by concrete task type.

        Raises:
            TimeFValidationError: If a task refers to an object outside the dataset or derives from itself.
        """
        task_ids: set[str] = set()
        dependency_rows: list[tuple[str, int, str]] = []
        annotation_refs: list[tuple[int, str, int, str]] = []
        task_keys: dict[str, int] = {}
        task_counts: dict[str, int] = {}
        for batch in _task_batches(tasks_source):
            batch_task_keys = _reserve_keys(connection, "object_key_sequence", len(batch))
            task_rows: list[dict[str, object]] = []
            target_rows: list[dict[str, object]] = []
            record_ref_rows: list[dict[str, object]] = []
            for task, task_key in zip(batch, batch_task_keys, strict=True):
                if task.id in task_ids:
                    raise TimeFValidationError(f"task id {task.id!r} is not unique")
                task_ids.add(task.id)
                task_keys[task.id] = task_key
                task_type = str(task.task_type)
                task_counts[task_type] = task_counts.get(task_type, 0) + 1
                task_rows.append(
                    {
                        "task_key": task_key,
                        "task_id": task.id,
                        "task_type": task_type,
                        "prompt": task.prompt,
                        "input_modalities": sorted(item.value for item in task.input_modalities),
                        **self._scope_columns(task, signal_keys),
                        "has_inline_targets": task.targets is not None,
                        **encode_task_payload(task),
                        "rationale": task.rationale,
                        "metadata": _json(task.metadata),
                    }
                )
                target_rows.extend(
                    self._target_row(
                        task_key=task_key,
                        position=position,
                        target_label=f"task {task.id!r} target {position}",
                        target=target,
                        record_keys=record_keys,
                        signal_keys=signal_keys,
                    )
                    for position, target in enumerate(task.targets or ())
                )
                annotations.extend(("Task", task_key, annotation) for annotation in task.annotations)
                for field, records in self._task_record_refs(task):
                    for position, record_id in enumerate(records):
                        if record_id not in record_keys:
                            raise TimeFValidationError(
                                f"task {task.id!r} field {field!r} refers to unknown record {record_id!r}"
                            )
                        record_ref_rows.append(
                            {
                                "task_key": task_key,
                                "field": field,
                                "position": position,
                                "record_key": record_keys[record_id],
                            }
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
                        annotation_refs.append((task_key, field, position, annotation.occurrence_id))
                dependency_rows.extend(
                    (task.id, position, parent.id) for position, parent in enumerate(task.from_tasks)
                )
            _insert_rows(connection, TABLES["tasks"], task_rows)
            _insert_rows(connection, TABLES["task_targets"], target_rows)
            _insert_rows(connection, TABLES["task_record_refs"], record_ref_rows)
        stored_dependencies: list[dict[str, object]] = []
        for task_id, position, parent_id in dependency_rows:
            if parent_id == task_id:
                raise TimeFValidationError(f"task {task_id!r} depends on itself")
            if parent_id not in task_ids:
                raise TimeFValidationError(
                    f"task {task_id!r} depends on task {parent_id!r}, which is not in the dataset"
                )
            stored_dependencies.append(
                {
                    "task_key": task_keys[task_id],
                    "position": position,
                    "parent_task_key": task_keys[parent_id],
                }
            )
        _insert_rows(connection, TABLES["task_dependencies"], stored_dependencies)
        return annotation_refs, task_counts

    @staticmethod
    def _scope_columns(task: Task, signal_keys: dict[str, int]) -> dict[str, object]:
        """Project a task's scope onto the four scope columns.

        Returns:
            The scope columns, all ``None`` for a task without a scope.

        Raises:
            TimeFValidationError: If the scope names a Signal outside the dataset.
        """
        encoded = encode_span(task.scope)
        if encoded is None:
            return {"scope_type": None, "scope_start": None, "scope_end": None, "scope_signal_keys": None}
        signal_ids = encoded["signal_ids"]
        missing = [] if signal_ids is None else [item for item in signal_ids if item not in signal_keys]
        if missing:
            raise TimeFValidationError(f"task {task.id!r} scope refers to unknown Signals {missing}")
        return {
            "scope_type": encoded["span_type"],
            "scope_start": encoded["span_start"],
            "scope_end": encoded["span_end"],
            "scope_signal_keys": None if signal_ids is None else [signal_keys[item] for item in signal_ids],
        }

    @staticmethod
    def _target_row(  # noqa: PLR0913 - one target projects onto keys from two lookups
        *,
        task_key: int,
        position: int,
        target_label: str,
        target: object,
        record_keys: dict[str, int],
        signal_keys: dict[str, int],
    ) -> dict[str, object]:
        """Validate one target and project it onto a ``task_targets`` row.

        Returns:
            The row, with object references as internal keys.

        Raises:
            TimeFValidationError: If the target refers to an object outside the dataset.
        """
        row = encode_target(target)
        record_id = cast("str | None", row["record_id"])
        signal_id = cast("str | None", row["signal_id"])
        if record_id is not None and record_id not in record_keys:
            raise TimeFValidationError(f"{target_label} refers to unknown Record {record_id!r}")
        if signal_id is not None and signal_id not in signal_keys:
            raise TimeFValidationError(f"{target_label} refers to unknown Signal {signal_id!r}")
        span_signal_ids = cast("tuple[str, ...] | list[str] | None", row["span_signal_ids"])
        missing_signals = [item for item in span_signal_ids or () if item not in signal_keys]
        if missing_signals:
            raise TimeFValidationError(f"{target_label} refers to unknown Signals {missing_signals}")
        kind = row["target_kind"]
        # A step span names its one series in signal_id; it is stored in signal_keys like a time span's scope.
        stored_signal_keys = (
            [signal_keys[signal_id]]
            if signal_id is not None and kind in {"step_point", "step_interval"}
            else None
            if span_signal_ids is None
            else [signal_keys[item] for item in span_signal_ids]
        )
        return {
            "task_key": task_key,
            "position": position,
            "target_kind": kind,
            "text_value": row["text_value"],
            "integer_value": row["integer_value"],
            "float_value": row["float_value"],
            "boolean_value": row["boolean_value"],
            "record_key": None if record_id is None else record_keys[record_id],
            "signal_key": signal_keys[signal_id] if signal_id is not None and kind == "signal" else None,
            "span_start": row["span_start"],
            "span_end": row["span_end"],
            "signal_keys": stored_signal_keys,
        }

    @staticmethod
    def _task_record_refs(task: Task) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Project task record objects into normalized field rows.

        Returns:
            Pairs of public field name and ordered record IDs.
        """
        refs: list[tuple[str, tuple[str, ...]]] = [("inputs", tuple(record.id for record in task.inputs))]
        if isinstance(task, TSCorrespondenceTask):
            refs.append(("candidate_records", tuple(record.id for record in task.candidate_records)))
        return tuple(refs)

    @staticmethod
    def _write_task_annotation_refs(
        connection: duckdb.DuckDBPyConnection,
        rows: list[tuple[int, str, int, str]],
        occurrence_keys: dict[str, int],
    ) -> None:
        """Insert task annotation references after occurrence rows exist.

        Raises:
            TimeFValidationError: If a referenced occurrence is not attached to a stored object.
        """
        for start in range(0, len(rows), _TASK_BATCH_SIZE):
            stored: list[dict[str, object]] = []
            for task_key, field, position, occurrence_id in rows[start : start + _TASK_BATCH_SIZE]:
                occurrence_key = occurrence_keys.get(occurrence_id)
                if occurrence_key is None:
                    raise TimeFValidationError(
                        f"task key {task_key} field {field!r} refers to unknown annotation occurrence {occurrence_id!r}"
                    )
                stored.append(
                    {
                        "task_key": task_key,
                        "field": field,
                        "position": position,
                        "occurrence_key": occurrence_key,
                    }
                )
            _insert_rows(connection, TABLES["task_annotation_refs"], stored)

    def _write_record_sources(  # noqa: PLR0913, PLR0917 - hierarchy maps stay explicit
        self,
        record: Any,
        record_key: int,
        axes: dict[str, tuple[object, int, pa.Array | None]],
        keys: _HierarchyKeys,
        annotations: list[tuple[str, int, Annotation]],
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
                f"record {record.record_id!r} has no Sources; a Record stores its Signals under Sources"
            )

        def write_source(source: Any, parent_key: int | None) -> None:
            if source.id in keys.sources:
                raise TimeFValidationError(f"source id {source.id!r} is not unique")
            source_key = next(object_keys)
            keys.sources[source.id] = source_key
            batches.sources.add(
                {
                    "source_key": source_key,
                    "source_id": source.id,
                    "record_key": record_key,
                    "parent_source_key": parent_key,
                    "name": source.name,
                    "metadata": _json(source.metadata),
                }
            )
            annotations.extend(("Source", source_key, annotation) for annotation in source.annotations)
            for signal in source.signals:
                if signal.id in keys.signals:
                    raise TimeFValidationError(f"signal id {signal.id!r} is not unique")
                signal_key = self._write_signal(signal, source_key, axes, object_keys, axis_keys, batches)
                keys.signals[signal.id] = signal_key
                annotations.extend(("Signal", signal_key, annotation) for annotation in signal.annotations)
            for child in source.sources:
                write_source(child, source_key)

        for source in record.sources:
            write_source(source, None)

    def _write_signal(  # noqa: PLR0913, PLR0917 - the hierarchy write state stays explicit
        self,
        signal: Signal,
        source_key: int,
        axes: dict[str, tuple[object, int, pa.Array | None]],
        object_keys: Iterator[int],
        axis_keys: Iterator[int],
        batches: _HierarchyBatches,
    ) -> int:
        """Insert one signal and its shared axis.

        Returns:
            The generated Signal key.

        Raises:
            TimeFValidationError: If one axis ID identifies different axis definitions, or an
                irregular axis has a different number of offsets than the Signal has values.
        """
        axis = signal.time_axis
        existing = axes.get(axis.axis_id)
        if existing is None:
            axis_key = next(axis_keys)
            offsets = self._write_axis(signal, axis_key, batches)
            if offsets is not None and len(offsets) != signal.n_values:
                raise TimeFValidationError(
                    f"signal {signal.id!r} declares {signal.n_values} values but its axis "
                    f"{axis.axis_id!r} has {len(offsets)} time offsets"
                )
            axes[axis.axis_id] = (axis, axis_key, offsets)
        elif existing[0] != axis:
            raise TimeFValidationError(f"axis id {axis.axis_id!r} is reused with different definitions")
        else:
            axis_key = existing[1]
        if existing is not None and isinstance(axis, IrregularAxis):
            if signal.time_offsets_loader is None:
                raise TimeFValidationError(f"irregular signal {signal.id!r} has no time-offset loader")
            stored_offsets = existing[2]
            if stored_offsets is None or not signal.time_offsets_loader().equals(stored_offsets):
                raise TimeFValidationError(f"axis id {axis.axis_id!r} is shared by signals with different time offsets")
            if len(stored_offsets) != signal.n_values:
                raise TimeFValidationError(
                    f"signal {signal.id!r} declares {signal.n_values} values but its axis "
                    f"{axis.axis_id!r} has {len(stored_offsets)} time offsets"
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
                "unit": str(spec.unit_value),
                "dtype": spec.dtype,
                "categories": list(spec.categories),
                "value_shape": list(spec.value_shape),
                "dimension_names": list(spec.dimension_names),
                "nullable": spec.nullable,
                "n_values": signal.n_values,
                "metadata": _json(signal.metadata),
            }
        )
        return signal_key

    @staticmethod
    def _write_axis(signal: Signal, axis_key: int, batches: _HierarchyBatches) -> pa.Array | None:
        """Buffer an axis and any irregular offsets.

        Returns:
            The irregular offsets, or ``None`` for other axis types.

        Raises:
            TimeFValidationError: If an irregular signal has no offsets loader.
        """
        axis = signal.time_axis
        if isinstance(axis, RegularAxis):
            batches.axes.add(
                {
                    "axis_key": axis_key,
                    "axis_id": axis.axis_id,
                    "axis_type": str(axis.axis_type),
                    "period_numerator_us": axis.period_us.numerator,
                    "period_denominator": axis.period_us.denominator,
                    "origin_us": axis.start_index,
                    "first_us": None,
                    "last_us": None,
                }
            )
            return None
        if isinstance(axis, IrregularAxis):
            batches.axes.add(
                {
                    "axis_key": axis_key,
                    "axis_id": axis.axis_id,
                    "axis_type": str(axis.axis_type),
                    "period_numerator_us": None,
                    "period_denominator": None,
                    "origin_us": None,
                    "first_us": axis.first_us,
                    "last_us": axis.last_us,
                }
            )
            if signal.time_offsets_loader is None:
                raise TimeFValidationError(f"irregular signal {signal.id!r} has no time-offset loader")
            offsets = signal.time_offsets_loader()
            for position, offset in enumerate(offsets.to_pylist()):
                batches.axis_offsets.add({"axis_key": axis_key, "position": position, "offset_us": offset})
            return offsets
        if isinstance(axis, OrdinalAxis):
            batches.axes.add(
                {
                    "axis_key": axis_key,
                    "axis_id": axis.axis_id,
                    "axis_type": str(axis.axis_type),
                    "period_numerator_us": None,
                    "period_denominator": None,
                    "origin_us": None,
                    "first_us": None,
                    "last_us": None,
                }
            )
            return None
        assert_never(axis)

    @staticmethod
    def _write_annotations(
        connection: duckdb.DuckDBPyConnection,
        annotations: Iterable[tuple[str, int, Annotation]],
        signal_keys: dict[str, int],
    ) -> dict[str, int]:
        """Insert reusable content once and every occurrence separately, in Arrow batches.

        Keys come from the content and occurrence sequences up front, so both tables are written
        through bounded batches rather than one ``INSERT ... RETURNING`` per row.

        Returns:
            Internal occurrence keys indexed by public occurrence ID.

        Raises:
            TimeFValidationError: If an annotation is unbound, one occurrence ID is attached twice,
                one content ID has two payloads, or a span names an unknown Signal.
        """
        contents: dict[str, tuple[object, ...]] = {}
        content_keys: dict[str, int] = {}
        occurrence_keys: dict[str, int] = {}
        content_batch = _TableBatch(connection, TABLES["annotation_contents"])
        occurrence_batch = _TableBatch(connection, TABLES["annotation_occurrences"])
        content_key_source = _SequenceKeys(connection, "content_key_sequence")
        occurrence_key_source = _SequenceKeys(connection, "occurrence_key_sequence")
        for object_type, object_key, annotation in annotations:
            if annotation.occurrence_id is None:
                raise TimeFValidationError(
                    f"annotation {annotation.content_id!r} on {object_type} key {object_key} has no occurrence id; "
                    "attach it with annotate()"
                )
            if annotation.occurrence_id in occurrence_keys:
                raise TimeFValidationError(f"annotation occurrence id {annotation.occurrence_id!r} is not unique")
            content = (
                annotation.name,
                annotation.value,
                annotation.unit,
                annotation.description,
                annotation.metadata,
            )
            previous = contents.get(annotation.content_id)
            if previous is not None and previous != content:
                raise TimeFValidationError(
                    f"annotation content id {annotation.content_id!r} is reused with different content"
                )
            if previous is None:
                contents[annotation.content_id] = content
                metadata = dict(annotation.metadata)
                if annotation.description is not None:
                    metadata["description"] = annotation.description
                content_key = content_key_source.next()
                content_keys[annotation.content_id] = content_key
                content_batch.add(
                    {
                        "content_key": content_key,
                        "content_id": annotation.content_id,
                        "name": annotation.name,
                        **encode_annotation_value(annotation.value, label=f"annotation {annotation.content_id!r}"),
                        "unit": annotation.unit,
                        "metadata": _json(metadata),
                    }
                )
            span = annotation.span
            span_signal_ids = () if span is None else span.time_series_ids or ()
            missing_signals = [signal_id for signal_id in span_signal_ids if signal_id not in signal_keys]
            if missing_signals:
                raise TimeFValidationError(
                    f"annotation occurrence {annotation.occurrence_id!r} refers to unknown Signals {missing_signals}"
                )
            occurrence_key = occurrence_key_source.next()
            occurrence_keys[annotation.occurrence_id] = occurrence_key
            occurrence_batch.add(
                {
                    "occurrence_key": occurrence_key,
                    "occurrence_id": annotation.occurrence_id,
                    "content_key": content_keys[annotation.content_id],
                    "object_type": object_type,
                    "object_key": object_key,
                    "span_type": str(annotation_type_of(annotation)),
                    "start_us": None if span is None else span.start_us,
                    "end_us": None if span is None else span.exclusive_end,
                    "signal_keys": (
                        None
                        if span is None or span.time_series_ids is None
                        else [signal_keys[s] for s in span_signal_ids]
                    ),
                    "provenance": None if annotation.source is None else _json(annotation.source),
                    "confidence": annotation.confidence,
                    "metadata": _json(annotation.occurrence_metadata),
                }
            )
        content_batch.flush()
        occurrence_batch.flush()
        return occurrence_keys

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
