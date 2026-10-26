"""Write the in-memory TimeF hierarchy into ``control.duckdb``."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Container, Hashable, Iterable, Iterator
from itertools import count, islice
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, assert_never, cast
from weakref import WeakValueDictionary

import duckdb
import numpy as np
import pyarrow as pa
from pydantic import ValidationError

from timenet.dataset import IrregularAxis, OrdinalAxis, Record, RegularAxis, Signal, TimeFDataset
from timenet.dataset.axis import to_time_offsets_us
from timenet.dataset.dataset import _TaskValidationIndex
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.annotation_codec import encode_annotation_value
from timenet.format.control_audit import CROSS_RECORD_PARENT, SOURCE_CYCLE
from timenet.format.control_schema import LAST_ID_PREFIX, TABLES, Table
from timenet.format.duckdb import connect_control, create_control_schema, transaction
from timenet.format.stored_values import (
    CatalogValue,
    PackedField,
    PackedValueKind,
    PackedValueWriter,
    StoredOffsets,
    offsets_digest,
)
from timenet.format.table_writer import insert_rows
from timenet.format.task_codec import TaskRelationshipField, encode_span, encode_task_payload
from timenet.format.task_target_row import TargetValueRow
from timenet.json import validate_json_value
from timenet.types import (
    Annotation,
    ArrayHandle,
    ObjectKind,
    Task,
    TSCorrespondenceTask,
    annotation_type_of,
)
from timenet.types.ids import EntityId
from timenet.types.splits import parse_split


if TYPE_CHECKING:
    from timenet.values_backends.writer import ChunkPlacement


_ROW_BATCH_SIZE = 10_000
_TASK_BATCH_SIZE = _ROW_BATCH_SIZE
_AnnotationRef = tuple[int, TaskRelationshipField, int, Annotation]


def _packed_placeholder(kind: PackedValueKind) -> str | list[str]:
    """Return the catalog value written in place of a packed value.

    The placeholder is an empty string or an empty list, so the catalog column keeps its type. Only
    the ``external_values`` locator tells packed text apart from genuinely empty text.

    Returns:
        The empty value of the packed value's catalog type.

    Raises:
        TimeFValidationError: If the value kind has no catalog column, as for time offsets.
    """
    match kind:
        case PackedValueKind.TEXT:
            return ""
        case PackedValueKind.TEXT_LIST:
            return []
        case PackedValueKind.TIME_OFFSETS:
            # Time offsets are packed directly and have no catalog column to hold a placeholder.
            raise TimeFValidationError("packed time offsets have no catalog placeholder")
        case _:
            assert_never(kind)


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
    time_offsets = signal._time_offsets
    axis = signal.time_axis
    if time_offsets is None or not isinstance(axis, IrregularAxis):
        raise TimeFValidationError(f"irregular signal {signal.id!r} has no time offsets")
    offsets = to_time_offsets_us(time_offsets.read_numpy())
    if len(offsets) != signal.n_values:
        raise TimeFValidationError(
            f"series {signal.id!r}: reading its time offsets returned {len(offsets)} time offsets "
            f"but it declares n_values={signal.n_values}"
        )
    if int(offsets[0]) != axis.first_us or int(offsets[-1]) != axis.last_us:
        raise TimeFValidationError(
            f"series {signal.id!r}: its axis claims the stream runs "
            f"{axis.first_us}..{axis.last_us} us but the stream runs "
            f"{int(offsets[0])}..{int(offsets[-1])} us"
        )
    return offsets


def _check_stored_offsets(signal: Signal, axis: IrregularAxis, time_offsets: StoredOffsets) -> None:
    """Check stored time offsets against a Signal and its axis without reading them.

    Raises:
        TimeFValidationError: If the time offsets are a different length from the Signal, or were
            read with an axis whose endpoints differ from the Signal's axis.
    """
    if time_offsets.length != signal.n_values:
        raise TimeFValidationError(
            f"series {signal.id!r}: its stored time offsets number {time_offsets.length} "
            f"but it declares n_values={signal.n_values}"
        )
    if (time_offsets.first, time_offsets.last) != (axis.first_us, axis.last_us):
        raise TimeFValidationError(
            f"series {signal.id!r}: its axis claims the stream runs {axis.first_us}..{axis.last_us} us "
            f"but its stored time offsets run {time_offsets.first}..{time_offsets.last} us"
        )


def _task_batches(tasks: Iterable[Task]) -> Iterator[tuple[Task, ...]]:
    """Yield bounded task batches without materializing a streamed source."""
    iterator = iter(tasks)
    while batch := tuple(islice(iterator, _TASK_BATCH_SIZE)):
        yield batch


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
        insert_rows(self.connection, self.table, self.rows)
        self.rows.clear()


class _OffsetSignature(NamedTuple):
    """Bounded identity for one validated irregular offset stream."""

    n_values: int
    digest: bytes


def _offset_signature(offsets: np.ndarray) -> _OffsetSignature:
    """Return the length and digest of canonical offsets.

    Returns:
        A bounded identity for the offset stream.
    """
    return _OffsetSignature(len(offsets), offsets_digest(offsets))


class _StoredAxis(NamedTuple):
    """One axis buffered for the control plane and shared by its Signals."""

    axis_id: EntityId
    signal_name: str
    offsets: _OffsetSignature | None
    offsets_handle: ArrayHandle | None


class _HierarchyBatches:
    """Own the bounded row buffers for the stored hierarchy."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.records = _TableBatch(connection, TABLES["records"])
        self.clocks = _TableBatch(connection, TABLES["clocks"])
        self.sources = _TableBatch(connection, TABLES["sources"])
        self.axes = _TableBatch(connection, TABLES["axes"])
        self.signals = _TableBatch(connection, TABLES["signals"])

    def flush(self) -> None:
        """Write all remaining hierarchy rows."""
        self.records.flush()
        self.clocks.flush()
        self.sources.flush()
        self.axes.flush()
        self.signals.flush()


class DuckDBControlWriter:
    """Serialize the relational part of one immutable TimeF version."""

    def __init__(self, path: Path, *, external: PackedValueWriter) -> None:
        """Configure a new local control database.

        Args:
            path: Destination path. It must not already exist.
            external: Writer for packed text and irregular time offsets beside the catalog.

        Raises:
            FileExistsError: If ``path`` already exists.
        """
        self.path = Path(path)
        self.external = external
        if self.path.exists():
            raise FileExistsError(f"control database already exists at {self.path}")
        self._storage_ids: defaultdict[str, Iterator[int]] = defaultdict(lambda: count(1))

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
            TimeFValidationError: If the hierarchy refers to an object outside the dataset, repeats a
                task id, holds a value that JSON cannot store, or declares Signal lengths that do not
                match its axis offsets or value chunks.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with connect_control(self.path) as connection:
                create_control_schema(connection)
                with transaction(connection):
                    task_counts, signal_values = self._write_objects(
                        connection,
                        dataset,
                        dataset.iter_tasks() if tasks is None else tasks,
                    )
                    self._write_chunks(connection, placements, signal_values)
                    self._check_source_tree(connection)
                    self.external.finish(connection)
                connection.execute("CHECKPOINT")
        except duckdb.Error as exc:
            self.path.unlink(missing_ok=True)
            raise TimeFValidationError(f"could not write TimeF control database: {exc}") from exc
        except BaseException:
            self.path.unlink(missing_ok=True)
            raise
        return task_counts

    def _pack(self, kind: ObjectKind, object_id: int, field: PackedField, value: Any, position: int = 0) -> Any:
        """Construct a catalog value from the explicit packing result.

        Args:
            kind: The kind of the object that owns the value.
            object_id: The object's id in this write: a task id, or an annotation content id.
            field: The object's field that holds the value.
            value: The value to store.
            position: The value's position in an ordered field, such as a task's targets.

        Returns:
            The original inline value, or the placeholder of a packed value's kind.
        """
        result = self.external.pack(kind, object_id, field, value, position)
        if isinstance(result, CatalogValue):
            return result.value
        return _packed_placeholder(result.locator.value_kind)

    def _write_objects(
        self,
        connection: duckdb.DuckDBPyConnection,
        dataset: TimeFDataset,
        tasks: Iterable[Task],
    ) -> tuple[dict[str, int], dict[int, int]]:
        """Insert hierarchy rows into an open transaction.

        Sources, axes, and annotations have no public id, so each write gives their rows new ids.

        Returns:
            Counts keyed by concrete task type, and the declared number of values of each stored
            Signal by id.
        """
        batches = _HierarchyBatches(connection)
        axes: dict[Hashable, _StoredAxis] = {}
        clock_keys: dict[int, int] = {}
        signal_values: dict[int, int] = {}
        connection.execute("INSERT INTO datasets (dataset_id) VALUES (?)", [dataset.metadata.dataset_id])
        annotations: list[tuple[ObjectKind, int | None, Annotation]] = [
            (ObjectKind.DATASET, None, annotation) for annotation in dataset.annotations
        ]
        record_ids: set[int] = set()
        for record in dataset.records:
            identity = id(record.start_time)
            if identity not in clock_keys:
                clock_keys[identity] = len(clock_keys) + 1
                batches.clocks.add({"clock_id": clock_keys[identity], "start_time_us": record.start_time.timestamp})
            span = record.time_span
            batches.records.add(
                {
                    "record_id": record.id,
                    "clock_id": clock_keys[identity],
                    "time_span_start_us": None if span is None else span.start_us,
                    "time_span_end_us": None if span is None else span.end_us,
                    "metadata": _json(record.metadata),
                }
            )
            record_ids.add(record.id)
            annotations.extend((ObjectKind.RECORD, record.id, annotation) for annotation in record.annotations)
            self._write_record_sources(record, axes, signal_values, annotations, batches)
            # Irregular axes pack their time offsets here, so store those locators before they pile up.
            self.external.write_locators(connection, at_least=_ROW_BATCH_SIZE)
        batches.flush()
        task_counts, annotation_refs = self._write_tasks(
            connection, dataset, record_ids, signal_values, annotations, tasks
        )
        # Clustering occurrences on object_id lets keyed reads skip row groups.
        annotations.sort(key=lambda item: (item[1] is not None, item[1] or 0))
        self._write_annotations(connection, annotations, annotation_refs)
        connection.executemany(
            "INSERT INTO control_metadata VALUES (?, ?)",
            [[f"{LAST_ID_PREFIX}{kind}", str(value)] for kind, value in dataset.last_ids.items()],
        )
        return task_counts, signal_values

    @staticmethod
    def _write_chunks(
        connection: duckdb.DuckDBPyConnection,
        placements: dict[tuple[int, int], ChunkPlacement] | None,
        signals: dict[int, int],
    ) -> None:
        """Insert values-plane chunk locations and check they cover every Signal exactly.

        Args:
            connection: The open write transaction.
            placements: Chunks keyed by ``(signal_id, chunk_index)``, or ``None`` for a
                metadata-only database that stores no values.
            signals: The declared number of values of each stored Signal, by id.

        Raises:
            TimeFValidationError: If a chunk names an unknown Signal, is empty, or the chunks of a
                Signal do not add up to its declared number of values.
        """
        if placements is None:
            return
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
                    "source_key": placement.source_key,
                    "chunk_major_index": placement.data_index.major_idx,
                    "chunk_minor_index": placement.data_index.minor_idx,
                    "n_values": placement.n_values,
                }
            )
        chunks.flush()

    def _write_tasks(  # noqa: PLR0913, PLR0917 - reference sets stay explicit
        self,
        connection: duckdb.DuckDBPyConnection,
        dataset: TimeFDataset,
        record_ids: Container[int],
        signal_ids: Container[int],
        annotations: list[tuple[ObjectKind, int | None, Annotation]],
        tasks_source: Iterable[Task],
    ) -> tuple[dict[str, int], list[_AnnotationRef]]:
        """Insert tasks and their relationships, one bounded batch at a time.

        A dependency can name a task that the writer writes later. So the writer checks and stores
        dependencies after it writes every task. The method returns the annotation references,
        because occurrences get their ids only when :meth:`_write_annotations` writes them.

        Returns:
            Counts keyed by concrete task type, and the annotation references.

        Raises:
            TimeFValidationError: If a task refers to a Record or Signal outside the dataset, or to
                a Record or Signal object that the dataset did not register under that id. Also if
                two tasks share an id, or a task derives from a task that is not in the dataset or
                from itself.
        """
        # Weak: if no other task names a streamed task, Python can free it after the writer writes it.
        written: WeakValueDictionary[int, Task] = WeakValueDictionary()
        dependencies: list[tuple[int, int, Task]] = []
        annotation_refs: list[_AnnotationRef] = []
        task_counts: dict[str, int] = {}
        index = _TaskValidationIndex(dataset)
        for batch in _task_batches(tasks_source):
            task_rows: list[dict[str, object]] = []
            target_rows: list[dict[str, object]] = []
            record_ref_rows: list[dict[str, object]] = []
            for task in batch:
                TimeFDataset._check_record_refs(task, index)
                TimeFDataset._check_signal_refs(task, index)
                task_id = task.id
                written[task_id] = task
                task_type = str(task.task_type)
                task_counts[task_type] = task_counts.get(task_type, 0) + 1
                task_rows.append(
                    {
                        "task_id": task_id,
                        "task_type": task_type,
                        "prompt": self._pack(ObjectKind.TASK, task_id, PackedField.PROMPT, task.prompt),
                        "input_modalities": sorted(item.value for item in task.resolved_input_modalities),
                        "split": None if task.split is None else parse_split(task.split).value,
                        **self._scope_columns(task, signal_ids),
                        "has_inline_targets": task.targets is not None,
                        **encode_task_payload(task),
                        "rationale": self._pack(ObjectKind.TASK, task_id, PackedField.RATIONALE, task.rationale),
                        "metadata": _json(task.metadata),
                    }
                )
                target_rows.extend(
                    self._target_row(
                        task_id,
                        position,
                        self._pack(ObjectKind.TASK, task_id, PackedField.TARGETS, target, position),
                        record_ids,
                        signal_ids,
                    )
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
                    annotation_refs.extend((task_id, field, position, item) for position, item in enumerate(values))
                dependencies.extend((task_id, position, parent) for position, parent in enumerate(task.from_tasks))
            insert_rows(connection, TABLES["tasks"], task_rows)
            insert_rows(connection, TABLES["task_targets"], target_rows)
            insert_rows(connection, TABLES["task_record_refs"], record_ref_rows)
            # Store this batch's locators now, so a streamed task source never holds them all.
            self.external.write_locators(connection)
        duplicate = connection.execute(
            "SELECT task_id FROM tasks GROUP BY task_id HAVING count(*) > 1 LIMIT 1"
        ).fetchone()
        if duplicate is not None:
            raise TimeFValidationError(f"task id {duplicate[0]!r} is not unique")
        dependency_rows: list[dict[str, object]] = []
        for task_id, position, parent in dependencies:
            if parent._id is None or written.get(parent._id) is not parent:
                with_id = "" if parent._id is None else f" with id {parent._id!r}"
                raise TimeFValidationError(f"task {task_id!r} depends on a task{with_id} that is not in the dataset")
            if parent.id == task_id:
                raise TimeFValidationError(f"task {task_id!r} depends on itself")
            dependency_rows.append({"task_id": task_id, "position": position, "parent_task_id": parent.id})
        insert_rows(connection, TABLES["task_dependencies"], dependency_rows)
        return task_counts, annotation_refs

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
        target_label = f"task {task_id!r} target {position}"
        try:
            row = TargetValueRow.from_target(target)
            if row.record_id is not None and row.record_id not in record_ids:
                raise TimeFValidationError(f"{target_label} refers to unknown Record {row.record_id!r}")
            if row.signal_id is not None and row.signal_id not in signal_ids:
                raise TimeFValidationError(f"{target_label} refers to unknown Signal {row.signal_id!r}")
            missing_signals = [item for item in row.span_signal_ids or () if item not in signal_ids]
            if missing_signals:
                raise TimeFValidationError(f"{target_label} refers to unknown Signals {missing_signals}")
            return {"task_id": task_id, "position": position, **row.model_dump()}
        except (TimeFFormatError, ValidationError) as exc:
            raise TimeFValidationError(f"{target_label}: {exc}") from exc

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

    def _write_record_sources(
        self,
        record: Record,
        axes: dict[Hashable, _StoredAxis],
        signal_values: dict[int, int],
        annotations: list[tuple[ObjectKind, int | None, Annotation]],
        batches: _HierarchyBatches,
    ) -> None:
        """Insert one record's source tree and signals.

        Raises:
            TimeFValidationError: If the record has no Sources, or a Signal ID is already used
                elsewhere in the dataset.
        """
        if not record.sources:
            raise TimeFValidationError(
                f"record {record.id!r} has no Sources; a Record stores its Signals under Sources"
            )

        def write_source(source: Any, parent_id: int | None) -> None:
            source_id = next(self._storage_ids["source"])
            batches.sources.add(
                {
                    "source_id": source_id,
                    "record_id": record.id,
                    "parent_source_id": parent_id,
                    "name": source.name,
                    "metadata": _json(source.metadata),
                }
            )
            annotations.extend((ObjectKind.SOURCE, source_id, annotation) for annotation in source.annotations)
            for signal in source.signals:
                if signal.id in signal_values:
                    raise TimeFValidationError(f"signal id {signal.id!r} is not unique")
                signal_values[signal.id] = signal.n_values
                self._write_signal(signal, source_id, axes, batches)
                annotations.extend((ObjectKind.SIGNAL, signal.id, annotation) for annotation in signal.annotations)
            for child in source.sources:
                write_source(child, source_id)

        for source in record.sources:
            write_source(source, None)

    def _write_signal(
        self,
        signal: Signal,
        source_id: int,
        axes: dict[Hashable, _StoredAxis],
        batches: _HierarchyBatches,
    ) -> None:
        """Insert one signal and its shared axis.

        Raises:
            TimeFValidationError: If an irregular axis has a different number of offsets than the
                Signal has values.
        """
        axis = signal.time_axis
        identity = id(axis) if isinstance(axis, IrregularAxis) else axis
        existing = axes.get(identity)
        if existing is None:
            axis_id = next(self._storage_ids["axis"])
            offsets = self._write_axis(signal, axis_id, batches)
            axes[identity] = _StoredAxis(axis_id, signal.name, offsets, signal._time_offsets)
        else:
            axis_id = existing.axis_id
            if isinstance(axis, IrregularAxis):
                stored_offsets = cast("_OffsetSignature", existing.offsets)
                # Handles over one callable compare equal, so offsets shared through it are read once.
                if signal._time_offsets != existing.offsets_handle:
                    time_offsets = signal._time_offsets
                    # Stored offsets record their length and digest, so comparing them reads and hashes nothing.
                    if isinstance(time_offsets, StoredOffsets):
                        candidate_signature = _OffsetSignature(time_offsets.length, time_offsets.digest)
                    else:
                        candidate_signature = _offset_signature(_validated_irregular_offsets(signal))
                    if candidate_signature != stored_offsets:
                        raise TimeFValidationError(
                            f"one IrregularAxis object is shared by signals {existing.signal_name!r} "
                            f"and {signal.name!r} with different time offsets"
                        )
                if stored_offsets.n_values != signal.n_values:
                    raise TimeFValidationError(
                        f"signal {signal.name!r} declares {signal.n_values} values but the IrregularAxis "
                        f"it shares with signal {existing.signal_name!r} has {stored_offsets.n_values} "
                        f"time offsets"
                    )

        spec = signal.spec
        batches.signals.add(
            {
                "signal_id": signal.id,
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
                "metadata": _json(signal.metadata),
                "modality": spec.modality.value,
            }
        )

    def _write_axis(self, signal: Signal, axis_id: EntityId, batches: _HierarchyBatches) -> _OffsetSignature | None:
        """Buffer an axis, and pack the time offsets of an irregular one under its axis id.

        Returns:
            A bounded signature of the irregular offsets, or ``None`` for other axis types.

        Raises:
            TimeFValidationError: If an irregular signal has no time offsets.
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
            time_offsets = signal._time_offsets
            if time_offsets is None:
                raise TimeFValidationError(f"irregular signal {signal.id!r} has no time offsets")
            if isinstance(time_offsets, StoredOffsets):
                # Stored time offsets keep their payload. Checking them needs no read: their location
                # records their length and digest, and the handle its axis's endpoints.
                _check_stored_offsets(signal, axis, time_offsets)
                self.external.pack(ObjectKind.AXIS, axis_id, PackedField.TIME_OFFSETS, time_offsets)
                return _OffsetSignature(time_offsets.length, time_offsets.digest)
            offsets = _validated_irregular_offsets(signal)
            signature = _offset_signature(offsets)
            self.external.pack(
                ObjectKind.AXIS,
                axis_id,
                PackedField.TIME_OFFSETS,
                pa.array(offsets, type=pa.int64()),
                digest=signature.digest,
            )
            return signature
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

    def _write_annotations(
        self,
        connection: duckdb.DuckDBPyConnection,
        annotations: Iterable[tuple[ObjectKind, int | None, Annotation]],
        annotation_refs: Iterable[_AnnotationRef],
    ) -> None:
        """Insert reusable content once, every occurrence separately, then the task references.

        The writer numbers occurrences in write order. So the reader gives back each object's
        annotations in the order they were attached. A task reference stores the id that this write
        gave to the same annotation object.

        Args:
            connection: The open write transaction.
            annotations: ``(object_type, object_id, annotation)`` triples. ``object_id`` is ``None``
                for the Dataset.
            annotation_refs: ``(task_id, field, position, annotation)`` task references.

        Raises:
            TimeFValidationError: If one occurrence is attached in more than one place, or a task
                refers to an annotation that is not attached in the dataset.
        """
        content_ids: dict[Hashable, EntityId] = {}
        occurrence_ids: dict[int, EntityId] = {}
        content_batch = _TableBatch(connection, TABLES["annotation_contents"])
        occurrence_batch = _TableBatch(connection, TABLES["annotation_occurrences"])
        for object_type, object_id, annotation in annotations:
            if id(annotation) in occurrence_ids:
                raise TimeFValidationError(
                    f"an occurrence of annotation {annotation.key!r} is attached in more than one place"
                )
            occurrence_id = occurrence_ids[id(annotation)] = next(self._storage_ids["occurrence"])
            content = annotation._content_key
            content_id = content_ids.get(content)
            if content_id is None:
                content_id = content_ids[content] = next(self._storage_ids["content"])
                content_batch.add(
                    {
                        "content_id": content_id,
                        "name": annotation.name,
                        **encode_annotation_value(
                            self._pack(ObjectKind.ANNOTATION_CONTENT, content_id, PackedField.VALUE, annotation.value),
                            label=f"annotation {annotation.key!r}",
                        ),
                        "unit": annotation.unit,
                        "description": annotation.description,
                        "metadata": _json(annotation.metadata),
                    }
                )
                self.external.write_locators(connection, at_least=_ROW_BATCH_SIZE)
            span = annotation.span
            occurrence_batch.add(
                {
                    "occurrence_id": occurrence_id,
                    "content_id": content_id,
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
        ref_batch = _TableBatch(connection, TABLES["task_annotation_refs"])
        for task_id, field, position, annotation in annotation_refs:
            occurrence_id = occurrence_ids.get(id(annotation))
            if occurrence_id is None:
                raise TimeFValidationError(
                    f"task {task_id!r} field {field.value!r} refers to an annotation that is not in the dataset"
                )
            ref_batch.add({"task_id": task_id, "field": field, "position": position, "occurrence_id": occurrence_id})
        ref_batch.flush()

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
        wrong_parent = connection.execute(CROSS_RECORD_PARENT).fetchone()
        if wrong_parent is not None:
            raise TimeFValidationError(f"source {wrong_parent[0]!r} and its parent belong to different records")

        cycle = connection.execute(SOURCE_CYCLE).fetchone()
        if cycle is not None:
            raise TimeFValidationError(f"source hierarchy contains a cycle at {cycle[0]!r}")
