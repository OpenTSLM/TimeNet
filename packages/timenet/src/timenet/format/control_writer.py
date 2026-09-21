"""Write the in-memory TimeF hierarchy into ``control.duckdb``."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, assert_never, cast

import duckdb

from timenet.dataset import IrregularAxis, OrdinalAxis, RegularAxis, Signal, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.format.duckdb import connect_control, create_control_schema, transaction
from timenet.format.task_codec import TARGET_VALUE_COLUMNS, encode_span, encode_target, encode_task_payload
from timenet.types import (
    Annotation,
    Task,
    TSCorrespondenceTask,
    annotation_type_of,
)


if TYPE_CHECKING:
    from timenet.values_backends.writer import ChunkPlacement


def _json(value: object) -> str:
    """Return deterministic JSON and name unsupported values as validation failures.

    Raises:
        TimeFValidationError: If ``value`` is not JSON-compatible.
    """
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TimeFValidationError(f"value is not JSON-compatible: {value!r}") from exc


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
        axes: dict[str, object] = {}
        object_types: dict[str, set[str]] = {dataset.metadata.dataset_id: {"Dataset"}}
        annotations: list[tuple[str, Annotation]] = [
            (dataset.metadata.dataset_id, annotation) for annotation in dataset.annotations
        ]
        for record in dataset.records:
            object_types.setdefault(record.record_id, set()).add("Record")
            span = record.time_span
            connection.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
                [
                    record.record_id,
                    record.start_time,
                    None if span is None else span.start_us,
                    None if span is None else span.end_us,
                    _json(record.metadata),
                ],
            )
            annotations.extend((record.record_id, annotation) for annotation in record.annotations)
            self._write_record_sources(connection, record, axes, annotations, object_types)
        annotation_refs, task_counts = self._write_tasks(connection, dataset, annotations, object_types, tasks)
        self._write_annotations(connection, annotations, object_types)
        self._write_task_annotation_refs(connection, annotation_refs)
        return task_counts

    @staticmethod
    def _write_chunks(
        connection: duckdb.DuckDBPyConnection,
        placements: dict[tuple[str, int], ChunkPlacement],
    ) -> None:
        """Insert values-plane chunk locations."""
        if not placements:
            return
        connection.executemany(
            "INSERT INTO signal_chunks VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    signal_id,
                    chunk_index,
                    placement.chunk_file,
                    placement.data_index.major_idx,
                    placement.data_index.minor_idx,
                    placement.n_values,
                )
                for (signal_id, chunk_index), placement in sorted(placements.items())
            ],
        )

    def _write_tasks(  # noqa: PLR0912 - each normalized task relationship is validated explicitly
        self,
        connection: duckdb.DuckDBPyConnection,
        dataset: TimeFDataset,
        annotations: list[tuple[str, Annotation]],
        object_types: dict[str, set[str]],
        tasks_source: Iterable[Task],
    ) -> tuple[list[tuple[str, str, int, str]], dict[str, int]]:
        """Insert tasks and normalized object relationships.

        Returns:
            Deferred task-to-annotation rows and counts keyed by concrete task type.

        Raises:
            TimeFValidationError: If a task refers to an object outside the dataset.
        """
        known_records = {record.id for record in dataset.records}
        known_signals = {signal.id for record in dataset.records for signal in record.signals}
        task_ids: set[str] = set()
        dependency_rows: list[tuple[str, int, str]] = []
        annotation_refs: list[tuple[str, str, int, str]] = []
        task_counts: dict[str, int] = {}
        for task in tasks_source:
            if task.id in task_ids:
                raise TimeFValidationError(f"task id {task.id!r} is not unique")
            task_ids.add(task.id)
            task_type = str(task.task_type)
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
            object_types.setdefault(task.id, set()).add("Task")
            connection.execute(
                "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    task.id,
                    str(task.task_type),
                    task.prompt,
                    None if task.scope is None else _json(encode_span(task.scope)),
                    task.targets is not None,
                    _json(encode_task_payload(task)),
                    task.rationale,
                    _json(task.metadata),
                ],
            )
            for position, target in enumerate(task.targets or ()):
                row = encode_target(target)
                record_id = row["record_id"]
                signal_id = row["signal_id"]
                if record_id is not None and record_id not in known_records:
                    raise TimeFValidationError(
                        f"task {task.id!r} target {position} refers to unknown Record {record_id!r}"
                    )
                if signal_id is not None and signal_id not in known_signals:
                    raise TimeFValidationError(
                        f"task {task.id!r} target {position} refers to unknown Signal {signal_id!r}"
                    )
                span_signal_ids = cast("tuple[str, ...] | list[str]", row["span_signal_ids"] or ())
                missing_signals = [signal for signal in span_signal_ids if signal not in known_signals]
                if missing_signals:
                    raise TimeFValidationError(
                        f"task {task.id!r} target {position} refers to unknown Signals {missing_signals}"
                    )
                values = [
                    _json(row[name]) if name == "span_signal_ids" and row[name] is not None else row[name]
                    for name in TARGET_VALUE_COLUMNS
                ]
                connection.execute(
                    """INSERT INTO task_targets (
                           task_id, position, target_kind, text_value, integer_value, float_value,
                           boolean_value, record_id, signal_id, span_start, span_end, span_signal_ids
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [task.id, position, row["target_kind"], *values],
                )
            annotations.extend((task.id, annotation) for annotation in task.annotations)
            record_refs = self._task_record_refs(task)
            for field, records in record_refs:
                for position, record_id in enumerate(records):
                    if record_id not in known_records:
                        raise TimeFValidationError(
                            f"task {task.id!r} field {field!r} refers to unknown record {record_id!r}"
                        )
                    connection.execute(
                        "INSERT INTO task_record_refs VALUES (?, ?, ?, ?)",
                        [task.id, field, position, record_id],
                    )
            for field, values in (
                ("input_annotations", task.input_annotations),
                ("target_annotations", task.target_annotations),
            ):
                for position, annotation in enumerate(values):
                    if annotation.occurrence_id is None:
                        raise TimeFValidationError(f"task {task.id!r} field {field!r} refers to an unbound annotation")
                    annotation_refs.append((task.id, field, position, annotation.occurrence_id))
            for position, parent in enumerate(task.from_tasks):
                dependency_rows.append((task.id, position, parent.id))
        for task_id, position, parent_id in dependency_rows:
            if parent_id not in task_ids:
                raise TimeFValidationError(
                    f"task {task_id!r} depends on task {parent_id!r}, which is not in the dataset"
                )
            connection.execute(
                "INSERT INTO task_dependencies VALUES (?, ?, ?)",
                [task_id, position, parent_id],
            )
        return annotation_refs, task_counts

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
        rows: list[tuple[str, str, int, str]],
    ) -> None:
        """Insert task annotation references after occurrence rows exist.

        Raises:
            TimeFValidationError: If a referenced occurrence is not attached to a stored object.
        """
        if not rows:
            return
        occurrence_ids = {row[3] for row in rows}
        stored_ids = {
            row[0]
            for row in connection.execute(
                "SELECT occurrence_id FROM annotation_occurrences WHERE occurrence_id IN (SELECT unnest(?))",
                [list(occurrence_ids)],
            ).fetchall()
        }
        missing = occurrence_ids - stored_ids
        if missing:
            task_id, field, _, occurrence_id = next(row for row in rows if row[3] in missing)
            raise TimeFValidationError(
                f"task {task_id!r} field {field!r} refers to unknown annotation occurrence {occurrence_id!r}"
            )
        connection.executemany("INSERT INTO task_annotation_refs VALUES (?, ?, ?, ?)", rows)

    def _write_record_sources(
        self,
        connection: duckdb.DuckDBPyConnection,
        record: Any,
        axes: dict[str, object],
        annotations: list[tuple[str, Annotation]],
        object_types: dict[str, set[str]],
    ) -> None:
        """Insert one record's source tree and signals.

        Raises:
            TimeFValidationError: If the record still uses the retired flat layout.
        """
        if not record.sources:
            raise TimeFValidationError(
                f"record {record.record_id!r} has no Source hierarchy; TimeF v2 does not store flat time_series"
            )

        def write_source(source: Any, parent_id: str | None) -> None:
            object_types.setdefault(source.id, set()).add("Source")
            connection.execute(
                "INSERT INTO sources VALUES (?, ?, ?, ?, ?)",
                [source.id, record.record_id, parent_id, source.name, _json(source.metadata)],
            )
            annotations.extend((source.id, annotation) for annotation in source.annotations)
            for signal in source.signals:
                object_types.setdefault(signal.id, set()).add("Signal")
                self._write_signal(connection, signal, source.id, axes)
                annotations.extend((signal.id, annotation) for annotation in signal.annotations)
            for child in source.sources:
                write_source(child, source.id)

        for source in record.sources:
            write_source(source, None)

    def _write_signal(
        self,
        connection: duckdb.DuckDBPyConnection,
        signal: Signal,
        source_id: str,
        axes: dict[str, object],
    ) -> None:
        """Insert one signal and its shared axis.

        Raises:
            TimeFValidationError: If one axis ID identifies different axis definitions.
        """
        axis = signal.time_axis
        existing = axes.get(axis.axis_id)
        if existing is None:
            axes[axis.axis_id] = axis
            self._write_axis(connection, signal)
        elif existing != axis:
            raise TimeFValidationError(f"axis id {axis.axis_id!r} is reused with different definitions")
        elif isinstance(axis, IrregularAxis):
            if signal.time_offsets_loader is None:
                raise TimeFValidationError(f"irregular signal {signal.id!r} has no time-offset loader")
            stored_offsets = [
                row[0]
                for row in connection.execute(
                    "SELECT offset_us FROM axis_offsets WHERE axis_id = ? ORDER BY position",
                    [axis.axis_id],
                ).fetchall()
            ]
            if signal.time_offsets_loader().to_pylist() != stored_offsets:
                raise TimeFValidationError(f"axis id {axis.axis_id!r} is shared by signals with different time offsets")

        spec = signal.spec
        connection.execute(
            """INSERT INTO signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                signal.id,
                source_id,
                signal.name,
                axis.axis_id,
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

    @staticmethod
    def _write_axis(connection: duckdb.DuckDBPyConnection, signal: Signal) -> None:
        """Insert an axis and any irregular offsets.

        Raises:
            TimeFValidationError: If an irregular signal has no offsets loader.
        """
        axis = signal.time_axis
        if isinstance(axis, RegularAxis):
            connection.execute(
                "INSERT INTO axes VALUES (?, ?, ?, ?, ?, NULL, NULL)",
                [
                    axis.axis_id,
                    str(axis.axis_type),
                    axis.period_us.numerator,
                    axis.period_us.denominator,
                    axis.start_index,
                ],
            )
            return
        if isinstance(axis, IrregularAxis):
            connection.execute(
                "INSERT INTO axes VALUES (?, ?, NULL, NULL, NULL, ?, ?)",
                [axis.axis_id, str(axis.axis_type), axis.first_us, axis.last_us],
            )
            if signal.time_offsets_loader is None:
                raise TimeFValidationError(f"irregular signal {signal.id!r} has no time-offset loader")
            offsets = signal.time_offsets_loader().to_pylist()
            connection.executemany(
                "INSERT INTO axis_offsets VALUES (?, ?, ?)",
                [(axis.axis_id, position, offset) for position, offset in enumerate(offsets)],
            )
            return
        if isinstance(axis, OrdinalAxis):
            connection.execute(
                "INSERT INTO axes VALUES (?, ?, NULL, NULL, NULL, NULL, NULL)",
                [axis.axis_id, str(axis.axis_type)],
            )
            return
        assert_never(axis)

    @staticmethod
    def _write_annotations(
        connection: duckdb.DuckDBPyConnection,
        annotations: Iterable[tuple[str, Annotation]],
        object_types: dict[str, set[str]],
    ) -> None:
        """Insert reusable content once and every occurrence separately.

        Raises:
            TimeFValidationError: If an annotation is unbound or one content ID has two payloads.
        """
        contents: dict[str, tuple[object, ...]] = {}
        for object_id, annotation in annotations:
            if annotation.occurrence_id is None:
                raise TimeFValidationError(
                    f"annotation {annotation.content_id!r} on object {object_id!r} has no occurrence id; "
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
                connection.execute(
                    "INSERT INTO annotation_contents VALUES (?, ?, ?, ?, ?)",
                    [
                        annotation.content_id,
                        annotation.name,
                        _json(annotation.value),
                        annotation.unit,
                        _json(metadata),
                    ],
                )
            span = annotation.span
            span_type = annotation_type_of(annotation)
            connection.execute(
                """INSERT INTO annotation_occurrences
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    annotation.occurrence_id,
                    annotation.content_id,
                    DuckDBControlWriter._object_type(object_types, object_id),
                    object_id,
                    str(span_type),
                    None if span is None else span.start_us,
                    None if span is None else span.exclusive_end,
                    None if span is None else _json(span.time_series_ids),
                    None if annotation.source is None else _json(annotation.source),
                    annotation.confidence,
                    _json(annotation.occurrence_metadata),
                ],
            )

    @staticmethod
    def _object_type(object_types: dict[str, set[str]], object_id: str) -> str:
        """Resolve an annotation target from the hierarchy traversal index.

        Returns:
            The stored polymorphic object tag.

        Raises:
            TimeFValidationError: If an ID is ambiguous across object tables.
        """
        found = object_types.get(object_id, set())
        if not found:
            raise TimeFValidationError(f"annotation target id {object_id!r} does not identify a stored object")
        if len(found) != 1:
            raise TimeFValidationError(f"annotation target id {object_id!r} is ambiguous across object types")
        return next(iter(found))

    @staticmethod
    def _validate_objects(connection: duckdb.DuckDBPyConnection, *, require_chunks: bool) -> None:
        """Run relational checks that are clearer as bulk SQL queries.

        Raises:
            TimeFValidationError: If a parent crosses records, a cycle exists, or an axis length is wrong.
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

        bad_axis = connection.execute(
            """SELECT signals.signal_id
               FROM signals
               JOIN axes USING (axis_id)
               LEFT JOIN (
                   SELECT axis_id, count(*) AS offset_count FROM axis_offsets GROUP BY axis_id
               ) offsets USING (axis_id)
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
                       SELECT signal_id, sum(n_values) AS stored_values
                       FROM signal_chunks GROUP BY signal_id
                   ) chunks USING (signal_id)
                   WHERE coalesce(stored_values, 0) <> signals.n_values
                   LIMIT 1"""
            ).fetchone()
            if bad_chunks is not None:
                raise TimeFValidationError(
                    f"signal {bad_chunks[0]!r} has value chunks whose lengths do not match n_values"
                )
