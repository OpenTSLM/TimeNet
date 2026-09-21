"""Write the in-memory TimeF hierarchy into ``control.duckdb``."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, assert_never

import duckdb

from timenet.dataset import IrregularAxis, OrdinalAxis, RegularAxis, Signal, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.format.duckdb import connect_control, create_control_schema, transaction
from timenet.format.task_codec import encode_span, encode_task_payload
from timenet.types import (
    Annotation,
    ForecastingTask,
    Task,
    TSCorrespondenceTask,
    TSEditingTask,
    TSGenerationTask,
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
    ) -> None:
        """Write records, sources, signals, axes, and their annotations atomically.

        Args:
            dataset: The complete in-memory hierarchy.
            placements: Values-plane chunks keyed by ``(signal_id, chunk_index)``.
            tasks: A previously validated task stream, or ``None`` to use the dataset's tasks.

        Raises:
            TimeFValidationError: If the hierarchy contains a dangling relationship, conflicting
                shared identity, unbound annotation, or value that cannot be stored as JSON.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with connect_control(self.path) as connection:
                create_control_schema(connection)
                with transaction(connection):
                    self._write_objects(connection, dataset, tuple(dataset.iter_tasks()) if tasks is None else tasks)
                    self._write_chunks(connection, placements or {})
                    self._validate_objects(connection, require_chunks=placements is not None)
                connection.execute("CHECKPOINT")
        except duckdb.Error as exc:
            self.path.unlink(missing_ok=True)
            raise TimeFValidationError(f"could not write TimeF control database: {exc}") from exc
        except BaseException:
            self.path.unlink(missing_ok=True)
            raise

    def _write_objects(
        self,
        connection: duckdb.DuckDBPyConnection,
        dataset: TimeFDataset,
        tasks: Iterable[Task],
    ) -> None:
        """Insert hierarchy rows into an open transaction."""
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
        annotation_refs = self._write_tasks(connection, dataset, annotations, object_types, tasks)
        self._write_annotations(connection, annotations, object_types)
        self._write_task_annotation_refs(connection, annotation_refs)

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
    ) -> list[tuple[str, str, int, str]]:
        """Insert tasks and normalized object relationships.

        Returns:
            Deferred task-to-annotation rows, inserted after annotation occurrences.

        Raises:
            TimeFValidationError: If a task refers to an object outside the dataset.
        """
        tasks = tuple(tasks_source)
        known_records = {record.id for record in dataset.records}
        known_signals = {signal.id for record in dataset.records for signal in record.signals}
        task_ids = {task.id for task in tasks}
        if len(task_ids) != len(tasks):
            raise TimeFValidationError("tasks must have unique IDs")

        for task in tasks:
            object_types.setdefault(task.id, set()).add("Task")
            connection.execute(
                "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    task.id,
                    str(task.task_type),
                    task.prompt,
                    None if task.scope is None else _json(encode_span(task.scope)),
                    _json(encode_task_payload(task)),
                    task.rationale,
                    _json(task.metadata),
                ],
            )
            annotations.extend((task.id, annotation) for annotation in task.annotations)

        annotation_refs: list[tuple[str, str, int, str]] = []
        for task in tasks:
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
            for field, signal_ids in self._task_signal_refs(task):
                for position, signal_id in enumerate(signal_ids):
                    if signal_id not in known_signals:
                        raise TimeFValidationError(
                            f"task {task.id!r} field {field!r} refers to unknown signal {signal_id!r}"
                        )
                    connection.execute(
                        "INSERT INTO task_signal_refs VALUES (?, ?, ?, ?)",
                        [task.id, field, position, signal_id],
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
                if parent.id not in task_ids:
                    raise TimeFValidationError(
                        f"task {task.id!r} depends on task {parent.id!r}, which is not in the dataset"
                    )
                connection.execute(
                    "INSERT INTO task_dependencies VALUES (?, ?, ?)",
                    [task.id, position, parent.id],
                )
        return annotation_refs

    @staticmethod
    def _task_record_refs(task: Task) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Project task record objects into normalized field rows.

        Returns:
            Pairs of public field name and ordered record IDs.
        """
        refs: list[tuple[str, tuple[str, ...]]] = [("inputs", tuple(record.id for record in task.inputs))]
        if isinstance(task, ForecastingTask):
            refs.append(("context_records", task.context_record_ids))
            if task.target_record_id is not None:
                refs.append(("target_record", (task.target_record_id,)))
        elif isinstance(task, TSEditingTask):
            if task.source_record_id is not None:
                refs.append(("source_record", (task.source_record_id,)))
            if task.target_record_id is not None:
                refs.append(("target_record", (task.target_record_id,)))
        elif isinstance(task, TSGenerationTask):
            if task.target_record_id is not None:
                refs.append(("target_record", (task.target_record_id,)))
        elif isinstance(task, TSCorrespondenceTask):
            refs.extend(
                (
                    ("candidate_records", task.candidate_record_ids),
                    ("target_records", task.target or ()),
                )
            )
        return tuple(refs)

    @staticmethod
    def _task_signal_refs(task: Task) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Project task signal objects into normalized field rows.

        Returns:
            Pairs of public field name and ordered signal IDs.
        """
        if isinstance(task, TSCorrespondenceTask) and task.target_time_series_ids is not None:
            return (("target_signals", task.target_time_series_ids),)
        return ()

    @staticmethod
    def _write_task_annotation_refs(
        connection: duckdb.DuckDBPyConnection,
        rows: list[tuple[str, str, int, str]],
    ) -> None:
        """Insert task annotation references after occurrence rows exist.

        Raises:
            TimeFValidationError: If a referenced occurrence is not attached to a stored object.
        """
        for task_id, field, position, occurrence_id in rows:
            exists = connection.execute(
                "SELECT 1 FROM annotation_occurrences WHERE occurrence_id = ?", [occurrence_id]
            ).fetchone()
            if exists is None:
                raise TimeFValidationError(
                    f"task {task_id!r} field {field!r} refers to unknown annotation occurrence {occurrence_id!r}"
                )
            connection.execute(
                "INSERT INTO task_annotation_refs VALUES (?, ?, ?, ?)",
                [task_id, field, position, occurrence_id],
            )

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
