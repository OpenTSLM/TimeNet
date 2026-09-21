"""Write the in-memory TimeF hierarchy into ``control.duckdb``."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, assert_never, cast

import duckdb
import pyarrow as pa

from timenet.dataset import IrregularAxis, OrdinalAxis, RegularAxis, Signal, TimeFDataset
from timenet.errors import TimeFValidationError
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


_TASK_BATCH_SIZE = 10_000
_TARGET_VALUE_COLUMNS = {
    "target_text_values": ("target_item_key", "value"),
    "target_integer_values": ("target_item_key", "value"),
    "target_float_values": ("target_item_key", "value"),
    "target_boolean_values": ("target_item_key", "value"),
    "target_record_values": ("target_item_key", "record_key"),
    "target_signal_values": ("target_item_key", "signal_key"),
    "target_span_values": ("target_item_key", "span_start", "span_end", "signal_keys"),
}


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


def _insert_rows(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    columns: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    """Insert one Arrow batch into a known control-schema table."""
    if not rows:
        return
    relation_name = f"_timenet_{table_name}_batch"
    names = ", ".join(columns)
    connection.register(relation_name, pa.Table.from_pylist(rows))
    try:
        connection.execute(
            f"INSERT INTO {table_name} ({names}) SELECT {names} FROM {relation_name}"  # noqa: S608
        )
    finally:
        connection.unregister(relation_name)


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
            TimeFValidationError: If the hierarchy contains a dangling relationship, conflicting
                shared identity, unbound annotation, or value that cannot be stored as JSON.
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
                    self._write_chunks(connection, placements or {})
                    self._validate_objects(connection, require_chunks=placements is not None)
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
        """
        axes: dict[str, tuple[object, int]] = {}
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
        for record in dataset.records:
            span = record.time_span
            record_key = _returned_key(
                connection.execute(
                    """INSERT INTO records (
                           record_id, start_time_us, time_span_start_us, time_span_end_us, metadata
                       ) VALUES (?, ?, ?, ?, ?) RETURNING record_key""",
                    [
                        record.record_id,
                        record.start_time,
                        None if span is None else span.start_us,
                        None if span is None else span.end_us,
                        _json(record.metadata),
                    ],
                )
            )
            record_keys[record.id] = record_key
            annotations.extend(("Record", record_key, annotation) for annotation in record.annotations)
            self._write_record_sources(connection, record, record_key, axes, signal_keys, annotations)
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
        placements: dict[tuple[str, int], ChunkPlacement],
    ) -> None:
        """Insert values-plane chunk locations."""
        if not placements:
            return
        signal_keys = dict(connection.execute("SELECT signal_id, signal_key FROM signals").fetchall())
        connection.executemany(
            """INSERT INTO signal_chunks (
                   signal_key, chunk_index, value_path, chunk_major_index, chunk_minor_index, n_values
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (
                    signal_keys[signal_id],
                    chunk_index,
                    placement.chunk_file,
                    placement.data_index.major_idx,
                    placement.data_index.minor_idx,
                    placement.n_values,
                )
                for (signal_id, chunk_index), placement in sorted(placements.items())
            ],
        )

    def _write_tasks(  # noqa: PLR0912, PLR0914 - one batch projects several normalized tables
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
            TimeFValidationError: If a task refers to an object outside the dataset.
        """
        task_ids: set[str] = set()
        dependency_rows: list[tuple[str, int, str]] = []
        annotation_refs: list[tuple[int, str, int, str]] = []
        task_keys: dict[str, int] = {}
        task_counts: dict[str, int] = {}
        for batch in _task_batches(tasks_source):
            batch_task_keys = _reserve_keys(connection, "object_key_sequence", len(batch))
            target_count = sum(len(task.targets or ()) for task in batch)
            batch_target_keys = iter(_reserve_keys(connection, "target_item_key_sequence", target_count))
            task_rows: list[dict[str, object]] = []
            target_rows: list[dict[str, object]] = []
            target_value_rows: dict[str, list[dict[str, object]]] = {table: [] for table in _TARGET_VALUE_COLUMNS}
            task_target_rows: list[dict[str, object]] = []
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
                        "scope": None if task.scope is None else _json(encode_span(task.scope)),
                        "has_inline_targets": task.targets is not None,
                        "payload": _json(encode_task_payload(task)),
                        "rationale": task.rationale,
                        "metadata": _json(task.metadata),
                    }
                )
                for position, target in enumerate(task.targets or ()):
                    target_item_key = next(batch_target_keys)
                    table, value_row, kind = self._target_value_row(
                        target_item_key=target_item_key,
                        target_label=f"task {task.id!r} target {position}",
                        target=target,
                        record_keys=record_keys,
                        signal_keys=signal_keys,
                    )
                    target_rows.append({"target_item_key": target_item_key, "target_kind": kind})
                    target_value_rows[table].append(value_row)
                    task_target_rows.append(
                        {
                            "task_key": task_key,
                            "position": position,
                            "target_item_key": target_item_key,
                        }
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
            _insert_rows(
                connection,
                "tasks",
                (
                    "task_key",
                    "task_id",
                    "task_type",
                    "prompt",
                    "scope",
                    "has_inline_targets",
                    "payload",
                    "rationale",
                    "metadata",
                ),
                task_rows,
            )
            _insert_rows(connection, "target_items", ("target_item_key", "target_kind"), target_rows)
            for table, rows in target_value_rows.items():
                _insert_rows(connection, table, _TARGET_VALUE_COLUMNS[table], rows)
            _insert_rows(
                connection,
                "task_targets",
                ("task_key", "position", "target_item_key"),
                task_target_rows,
            )
            _insert_rows(
                connection,
                "task_record_refs",
                ("task_key", "field", "position", "record_key"),
                record_ref_rows,
            )
        stored_dependencies: list[dict[str, object]] = []
        for task_id, position, parent_id in dependency_rows:
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
        _insert_rows(
            connection,
            "task_dependencies",
            ("task_key", "position", "parent_task_key"),
            stored_dependencies,
        )
        return annotation_refs, task_counts

    @staticmethod
    def _target_value_row(
        *,
        target_item_key: int,
        target_label: str,
        target: object,
        record_keys: dict[str, int],
        signal_keys: dict[str, int],
    ) -> tuple[str, dict[str, object], str]:
        """Validate and project one target into its typed value table.

        Returns:
            Target value table, row, and target kind.

        Raises:
            TimeFValidationError: If the target refers to an object outside the dataset.
        """
        row = encode_target(target)
        kind = cast("str", row["target_kind"])
        record_id = cast("str | None", row["record_id"])
        signal_id = cast("str | None", row["signal_id"])
        if record_id is not None and record_id not in record_keys:
            raise TimeFValidationError(f"{target_label} refers to unknown Record {record_id!r}")
        if signal_id is not None and signal_id not in signal_keys:
            raise TimeFValidationError(f"{target_label} refers to unknown Signal {signal_id!r}")
        span_signal_ids = cast("tuple[str, ...] | list[str]", row["span_signal_ids"] or ())
        missing_signals = [item for item in span_signal_ids if item not in signal_keys]
        if missing_signals:
            raise TimeFValidationError(f"{target_label} refers to unknown Signals {missing_signals}")

        values: dict[str, object]
        if kind == "text":
            table, values = "target_text_values", {"target_item_key": target_item_key, "value": row["text_value"]}
        elif kind == "integer":
            table, values = (
                "target_integer_values",
                {
                    "target_item_key": target_item_key,
                    "value": row["integer_value"],
                },
            )
        elif kind == "float":
            table, values = "target_float_values", {"target_item_key": target_item_key, "value": row["float_value"]}
        elif kind == "boolean":
            table, values = (
                "target_boolean_values",
                {
                    "target_item_key": target_item_key,
                    "value": row["boolean_value"],
                },
            )
        elif kind == "record":
            table = "target_record_values"
            values = {
                "target_item_key": target_item_key,
                "record_key": record_keys[cast("str", record_id)],
            }
        elif kind == "signal":
            table = "target_signal_values"
            values = {
                "target_item_key": target_item_key,
                "signal_key": signal_keys[cast("str", signal_id)],
            }
        else:
            table = "target_span_values"
            stored_signal_keys = (
                [signal_keys[signal_id]]
                if signal_id is not None
                else None
                if row["span_signal_ids"] is None
                else [signal_keys[item] for item in span_signal_ids]
            )
            values = {
                "target_item_key": target_item_key,
                "span_start": row["span_start"],
                "span_end": row["span_end"],
                "signal_keys": stored_signal_keys,
            }
        return table, values, kind

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
            _insert_rows(
                connection,
                "task_annotation_refs",
                ("task_key", "field", "position", "occurrence_key"),
                stored,
            )

    def _write_record_sources(  # noqa: PLR0913, PLR0917 - hierarchy maps stay explicit
        self,
        connection: duckdb.DuckDBPyConnection,
        record: Any,
        record_key: int,
        axes: dict[str, tuple[object, int]],
        signal_keys: dict[str, int],
        annotations: list[tuple[str, int, Annotation]],
    ) -> None:
        """Insert one record's source tree and signals.

        Raises:
            TimeFValidationError: If the record still uses the retired flat layout.
        """
        if not record.sources:
            raise TimeFValidationError(
                f"record {record.record_id!r} has no Source hierarchy; this TimeF version does not store flat time_series"
            )

        def write_source(source: Any, parent_key: int | None) -> None:
            source_key = _returned_key(
                connection.execute(
                    """INSERT INTO sources (source_id, record_key, parent_source_key, name, metadata)
                       VALUES (?, ?, ?, ?, ?) RETURNING source_key""",
                    [source.id, record_key, parent_key, source.name, _json(source.metadata)],
                )
            )
            annotations.extend(("Source", source_key, annotation) for annotation in source.annotations)
            for signal in source.signals:
                signal_key = self._write_signal(connection, signal, source_key, axes)
                signal_keys[signal.id] = signal_key
                annotations.extend(("Signal", signal_key, annotation) for annotation in signal.annotations)
            for child in source.sources:
                write_source(child, source_key)

        for source in record.sources:
            write_source(source, None)

    def _write_signal(
        self,
        connection: duckdb.DuckDBPyConnection,
        signal: Signal,
        source_key: int,
        axes: dict[str, tuple[object, int]],
    ) -> int:
        """Insert one signal and its shared axis.

        Returns:
            The generated Signal key.

        Raises:
            TimeFValidationError: If one axis ID identifies different axis definitions.
        """
        axis = signal.time_axis
        existing = axes.get(axis.axis_id)
        if existing is None:
            axis_key = self._write_axis(connection, signal)
            axes[axis.axis_id] = (axis, axis_key)
        elif existing[0] != axis:
            raise TimeFValidationError(f"axis id {axis.axis_id!r} is reused with different definitions")
        else:
            axis_key = existing[1]
        if existing is not None and isinstance(axis, IrregularAxis):
            if signal.time_offsets_loader is None:
                raise TimeFValidationError(f"irregular signal {signal.id!r} has no time-offset loader")
            stored_offsets = [
                row[0]
                for row in connection.execute(
                    "SELECT offset_us FROM axis_offsets WHERE axis_key = ? ORDER BY position",
                    [axis_key],
                ).fetchall()
            ]
            if signal.time_offsets_loader().to_pylist() != stored_offsets:
                raise TimeFValidationError(f"axis id {axis.axis_id!r} is shared by signals with different time offsets")

        spec = signal.spec
        return _returned_key(
            connection.execute(
                """INSERT INTO signals (
                       signal_id, source_key, name, axis_key, spec_type, spec_name, unit, dtype,
                       categories, value_shape, dimension_names, nullable, n_values, metadata
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING signal_key""",
                [
                    signal.id,
                    source_key,
                    signal.name,
                    axis_key,
                    spec.spec_type,
                    spec.name,
                    str(spec.unit_value),
                    spec.dtype,
                    _json(spec.categories),
                    _json(spec.value_shape),
                    _json(spec.dimension_names),
                    spec.nullable,
                    signal.n_values,
                    _json(signal.metadata),
                ],
            )
        )

    @staticmethod
    def _write_axis(connection: duckdb.DuckDBPyConnection, signal: Signal) -> int:
        """Insert an axis and any irregular offsets.

        Returns:
            The generated axis key.

        Raises:
            TimeFValidationError: If an irregular signal has no offsets loader.
        """
        axis = signal.time_axis
        if isinstance(axis, RegularAxis):
            return _returned_key(
                connection.execute(
                    """INSERT INTO axes (
                           axis_id, axis_type, period_numerator_us, period_denominator, origin_us,
                           first_us, last_us
                       ) VALUES (?, ?, ?, ?, ?, NULL, NULL) RETURNING axis_key""",
                    [
                        axis.axis_id,
                        str(axis.axis_type),
                        axis.period_us.numerator,
                        axis.period_us.denominator,
                        axis.start_index,
                    ],
                )
            )
        if isinstance(axis, IrregularAxis):
            axis_key = _returned_key(
                connection.execute(
                    """INSERT INTO axes (
                           axis_id, axis_type, period_numerator_us, period_denominator, origin_us,
                           first_us, last_us
                       ) VALUES (?, ?, NULL, NULL, NULL, ?, ?) RETURNING axis_key""",
                    [axis.axis_id, str(axis.axis_type), axis.first_us, axis.last_us],
                )
            )
            if signal.time_offsets_loader is None:
                raise TimeFValidationError(f"irregular signal {signal.id!r} has no time-offset loader")
            offsets = signal.time_offsets_loader().to_pylist()
            connection.executemany(
                "INSERT INTO axis_offsets VALUES (?, ?, ?)",
                [(axis_key, position, offset) for position, offset in enumerate(offsets)],
            )
            return axis_key
        if isinstance(axis, OrdinalAxis):
            return _returned_key(
                connection.execute(
                    """INSERT INTO axes (
                           axis_id, axis_type, period_numerator_us, period_denominator, origin_us,
                           first_us, last_us
                       ) VALUES (?, ?, NULL, NULL, NULL, NULL, NULL) RETURNING axis_key""",
                    [axis.axis_id, str(axis.axis_type)],
                )
            )
        assert_never(axis)

    @staticmethod
    def _write_annotations(
        connection: duckdb.DuckDBPyConnection,
        annotations: Iterable[tuple[str, int, Annotation]],
        signal_keys: dict[str, int],
    ) -> dict[str, int]:
        """Insert reusable content once and every occurrence separately.

        Returns:
            Internal occurrence keys indexed by public occurrence ID.

        Raises:
            TimeFValidationError: If an annotation is unbound or one content ID has two payloads.
        """
        contents: dict[str, tuple[object, ...]] = {}
        content_keys: dict[str, int] = {}
        occurrence_keys: dict[str, int] = {}
        for object_type, object_key, annotation in annotations:
            if annotation.occurrence_id is None:
                raise TimeFValidationError(
                    f"annotation {annotation.content_id!r} on {object_type} key {object_key} has no occurrence id; "
                    "attach it with annotate()"
                )
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
                content_keys[annotation.content_id] = _returned_key(
                    connection.execute(
                        """INSERT INTO annotation_contents (content_id, name, value, unit, metadata)
                           VALUES (?, ?, ?, ?, ?) RETURNING content_key""",
                        [
                            annotation.content_id,
                            annotation.name,
                            _json(annotation.value),
                            annotation.unit,
                            _json(metadata),
                        ],
                    )
                )
            span = annotation.span
            span_type = annotation_type_of(annotation)
            span_signal_ids = () if span is None else span.time_series_ids or ()
            missing_signals = [signal_id for signal_id in span_signal_ids if signal_id not in signal_keys]
            if missing_signals:
                raise TimeFValidationError(
                    f"annotation occurrence {annotation.occurrence_id!r} refers to unknown Signals {missing_signals}"
                )
            occurrence_key = _returned_key(
                connection.execute(
                    """INSERT INTO annotation_occurrences (
                           occurrence_id, content_key, object_type, object_key, span_type, start_us,
                           end_us, signal_keys, provenance, confidence, metadata
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING occurrence_key""",
                    [
                        annotation.occurrence_id,
                        content_keys[annotation.content_id],
                        object_type,
                        object_key,
                        str(span_type),
                        None if span is None else span.start_us,
                        None if span is None else span.exclusive_end,
                        None
                        if span is None or span.time_series_ids is None
                        else [signal_keys[s] for s in span_signal_ids],
                        None if annotation.source is None else _json(annotation.source),
                        annotation.confidence,
                        _json(annotation.occurrence_metadata),
                    ],
                )
            )
            occurrence_keys[annotation.occurrence_id] = occurrence_key
        return occurrence_keys

    @staticmethod
    def _validate_relational_integrity(connection: duckdb.DuckDBPyConnection) -> None:
        """Replace persisted database constraints with checks performed before publication.

        Raises:
            TimeFValidationError: If a logical key is repeated, a relationship is dangling, or a
                constrained value is invalid.
        """
        unique_keys = (
            ("control_metadata", ("key",)),
            ("datasets", ("dataset_key",)),
            ("datasets", ("dataset_id",)),
            ("records", ("record_key",)),
            ("records", ("record_id",)),
            ("sources", ("source_key",)),
            ("sources", ("source_id",)),
            ("axes", ("axis_key",)),
            ("axes", ("axis_id",)),
            ("axis_offsets", ("axis_key", "position")),
            ("signals", ("signal_key",)),
            ("signals", ("signal_id",)),
            ("signal_chunks", ("signal_key", "chunk_index")),
            ("annotation_contents", ("content_key",)),
            ("annotation_contents", ("content_id",)),
            ("annotation_occurrences", ("occurrence_key",)),
            ("annotation_occurrences", ("occurrence_id",)),
            ("tasks", ("task_key",)),
            ("tasks", ("task_id",)),
            ("target_items", ("target_item_key",)),
            ("target_text_values", ("target_item_key",)),
            ("target_integer_values", ("target_item_key",)),
            ("target_float_values", ("target_item_key",)),
            ("target_boolean_values", ("target_item_key",)),
            ("target_record_values", ("target_item_key",)),
            ("target_signal_values", ("target_item_key",)),
            ("target_span_values", ("target_item_key",)),
            ("task_targets", ("task_key", "position")),
            ("task_targets", ("target_item_key",)),
            ("task_record_refs", ("task_key", "field", "position")),
            ("task_annotation_refs", ("task_key", "field", "position")),
            ("task_dependencies", ("task_key", "position")),
        )
        for table, columns in unique_keys:
            names = ", ".join(columns)
            duplicate = connection.execute(
                f"SELECT {names} FROM {table} GROUP BY {names} HAVING count(*) > 1 LIMIT 1"  # noqa: S608
            ).fetchone()
            if duplicate is not None:
                raise TimeFValidationError(f"{table} contains duplicate values for {names}: {duplicate!r}")

        foreign_keys = (
            ("sources", "record_key", "records", "record_key"),
            ("sources", "parent_source_key", "sources", "source_key"),
            ("axis_offsets", "axis_key", "axes", "axis_key"),
            ("signals", "source_key", "sources", "source_key"),
            ("signals", "axis_key", "axes", "axis_key"),
            ("signal_chunks", "signal_key", "signals", "signal_key"),
            ("annotation_occurrences", "content_key", "annotation_contents", "content_key"),
            ("target_text_values", "target_item_key", "target_items", "target_item_key"),
            ("target_integer_values", "target_item_key", "target_items", "target_item_key"),
            ("target_float_values", "target_item_key", "target_items", "target_item_key"),
            ("target_boolean_values", "target_item_key", "target_items", "target_item_key"),
            ("target_record_values", "target_item_key", "target_items", "target_item_key"),
            ("target_record_values", "record_key", "records", "record_key"),
            ("target_signal_values", "target_item_key", "target_items", "target_item_key"),
            ("target_signal_values", "signal_key", "signals", "signal_key"),
            ("target_span_values", "target_item_key", "target_items", "target_item_key"),
            ("task_targets", "task_key", "tasks", "task_key"),
            ("task_targets", "target_item_key", "target_items", "target_item_key"),
            ("task_record_refs", "task_key", "tasks", "task_key"),
            ("task_record_refs", "record_key", "records", "record_key"),
            ("task_annotation_refs", "task_key", "tasks", "task_key"),
            ("task_annotation_refs", "occurrence_key", "annotation_occurrences", "occurrence_key"),
            ("task_dependencies", "task_key", "tasks", "task_key"),
            ("task_dependencies", "parent_task_key", "tasks", "task_key"),
        )
        for child_table, child_column, parent_table, parent_column in foreign_keys:
            orphan = connection.execute(
                f"""SELECT child.{child_column}
                    FROM {child_table} child
                    LEFT JOIN {parent_table} parent
                      ON parent.{parent_column} = child.{child_column}
                    WHERE child.{child_column} IS NOT NULL
                      AND parent.{parent_column} IS NULL
                    LIMIT 1"""  # noqa: S608
            ).fetchone()
            if orphan is not None:
                raise TimeFValidationError(
                    f"{child_table}.{child_column} refers to missing {parent_table}.{parent_column} value {orphan[0]!r}"
                )

        invalid = connection.execute(
            """SELECT 'sources.parent_source_key' AS field
               FROM sources WHERE parent_source_key = source_key
               UNION ALL
               SELECT 'signals.n_values' FROM signals WHERE n_values <= 0
               UNION ALL
               SELECT 'signal_chunks.n_values' FROM signal_chunks WHERE n_values <= 0
               UNION ALL
               SELECT 'annotation_occurrences.object_type' FROM annotation_occurrences
               WHERE object_type NOT IN ('Dataset', 'Task', 'Record', 'Source', 'Signal')
               UNION ALL
               SELECT 'annotation_occurrences.span_type' FROM annotation_occurrences
               WHERE span_type NOT IN ('static', 'point', 'interval')
               UNION ALL
               SELECT 'target_items.target_kind' FROM target_items
               WHERE target_kind NOT IN (
                   'text', 'integer', 'float', 'boolean', 'record', 'signal',
                   'time_point', 'time_interval', 'step_point', 'step_interval'
               )
               UNION ALL
               SELECT 'task_dependencies.parent_task_key' FROM task_dependencies
               WHERE task_key = parent_task_key
               LIMIT 1"""
        ).fetchone()
        if invalid is not None:
            raise TimeFValidationError(f"control database contains an invalid {invalid[0]} value")

    @staticmethod
    def _validate_objects(connection: duckdb.DuckDBPyConnection, *, require_chunks: bool) -> None:
        """Run relational checks that are clearer as bulk SQL queries.

        Raises:
            TimeFValidationError: If a parent crosses records, a cycle exists, or an axis length is wrong.
        """
        DuckDBControlWriter._validate_relational_integrity(connection)

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

        bad_axis = connection.execute(
            """SELECT signals.signal_id
               FROM signals
               JOIN axes USING (axis_key)
               LEFT JOIN (
                   SELECT axis_key, count(*) AS offset_count FROM axis_offsets GROUP BY axis_key
               ) offsets USING (axis_key)
               WHERE (axes.axis_type = 'irregular' AND coalesce(offset_count, 0) <> signals.n_values)
                  OR (axes.axis_type <> 'irregular' AND coalesce(offset_count, 0) <> 0)
               LIMIT 1"""
        ).fetchone()
        if bad_axis is not None:
            raise TimeFValidationError(f"signal {bad_axis[0]!r} has an axis length that does not match its values")
        if require_chunks:
            bad_chunks = connection.execute(
                """SELECT signals.signal_id
                   FROM signals
                   LEFT JOIN (
                       SELECT signal_key, sum(n_values) AS stored_values
                       FROM signal_chunks GROUP BY signal_key
                   ) chunks USING (signal_key)
                   WHERE coalesce(stored_values, 0) <> signals.n_values
                   LIMIT 1"""
            ).fetchone()
            if bad_chunks is not None:
                raise TimeFValidationError(
                    f"signal {bad_chunks[0]!r} has value chunks whose lengths do not match n_values"
                )
