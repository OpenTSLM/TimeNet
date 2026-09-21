"""Hydrate the TimeF object hierarchy from ``control.duckdb``."""

from collections import defaultdict
from collections.abc import Callable, Iterable
from fractions import Fraction
from functools import partial
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa

from timenet.dataset import IrregularAxis, OrdinalAxis, Record, RegularAxis, Signal, Source
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.duckdb import check_control_schema, connect_control
from timenet.format.task_codec import decode_span, decode_target, decode_task_payload
from timenet.types import (
    TASKS,
    Annotation,
    Task,
    TaskType,
    TimeInterval,
    TimePoint,
    TimeSeriesSpec,
    ureg,
)


ValueLoader = Callable[[str, TimeSeriesSpec], pa.Array]
ValueLoaderFactory = Callable[[str, TimeSeriesSpec], Callable[[], pa.Array]]
OffsetsLoader = Callable[[str], pa.Array]

_HIERARCHY_KEYS = {
    "records": ("record_id", "record_id"),
    "sources": ("record_id", "source_id"),
    "signals": ("source_id", "signal_id"),
}


def _decode_json(value: str | None, *, default: Any = None) -> Any:
    """Decode one DuckDB JSON value.

    Returns:
        The decoded value, or ``default`` for SQL ``NULL``.

    Raises:
        TimeFFormatError: If stored JSON is malformed.
    """
    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise TimeFFormatError(f"control.duckdb contains invalid JSON: {value!r}") from exc


def _missing_values(signal_id: str, _spec: TimeSeriesSpec) -> pa.Array:
    """Fail when metadata-only hydration is asked to load values.

    Raises:
        TimeFFormatError: Always, because no values loader was configured.
    """
    raise TimeFFormatError(f"no values loader is configured for signal {signal_id!r}")


class DuckDBControlReader:
    """Keep one read-only DuckDB connection and hydrate requested records from it."""

    def __init__(
        self,
        path: Path,
        *,
        value_loader: ValueLoader | None = None,
        value_loader_factory: ValueLoaderFactory | None = None,
        offsets_loader: OffsetsLoader | None = None,
    ) -> None:
        """Open and validate an immutable control database.

        Args:
            path: Local path to ``control.duckdb``.
            value_loader: Lazy values-plane resolver keyed by signal ID.
            value_loader_factory: Factory for range-aware per-Signal loaders.
            offsets_loader: Lazy irregular-axis resolver keyed by axis ID.

        Raises:
            TimeFFormatError: If the database cannot be opened or has an unsupported schema.
        """
        self.path = Path(path)
        try:
            self.connection = connect_control(self.path, read_only=True)
            check_control_schema(self.connection)
        except duckdb.Error as exc:
            raise TimeFFormatError(f"could not open control database {self.path}: {exc}") from exc
        self._value_loader = value_loader or _missing_values
        self._value_loader_factory = value_loader_factory
        self._offsets_loader = offsets_loader

    def close(self) -> None:
        """Close the reader's DuckDB connection."""
        self.connection.close()

    def __enter__(self) -> "DuckDBControlReader":
        """Return this open reader.

        Returns:
            This reader.
        """
        return self

    def __exit__(self, *_: object) -> None:
        """Close the connection when leaving a context manager."""
        self.close()

    def read_records(  # noqa: PLR0914 - hydration keeps related row sets together
        self,
        record_ids: Iterable[str] | None = None,
        *,
        with_annotations: bool = True,
    ) -> tuple[Record, ...]:
        """Hydrate complete recursive records while leaving signal values lazy.

        Args:
            record_ids: Requested IDs in result order, or ``None`` for every record.
            with_annotations: Whether to hydrate annotations on records, sources, and signals.

        Returns:
            The hydrated records.

        Raises:
            TimeFValidationError: If a requested record is absent.
            TimeFFormatError: If the stored hierarchy is inconsistent.
        """
        requested = None if record_ids is None else tuple(record_ids)
        read_all = requested is None
        rows = self._read_rows("records", requested)
        by_id = self._index_rows(rows, "record")
        order = tuple(by_id) if requested is None else requested
        missing = [record_id for record_id in order if record_id not in by_id]
        if missing:
            raise TimeFValidationError(f"no such record(s) in control.duckdb: {missing}")

        record_keys = list(by_id)
        source_rows = self._read_rows("sources", None if read_all else record_keys)
        source_data = self._index_rows(source_rows, "source")

        source_keys = list(source_data)
        signal_rows = self._read_rows("signals", None if read_all else source_keys)
        signal_keys = set(self._index_rows(signal_rows, "signal"))
        for signal_id, source_id, *_ in signal_rows:
            if source_id not in source_data:
                raise TimeFFormatError(f"signal {signal_id!r} refers to missing source {source_id!r}")

        annotations = (
            self._read_annotations(
                {"Record": set(record_keys), "Source": set(source_keys), "Signal": signal_keys},
                read_all=read_all,
            )
            if with_annotations
            else {}
        )
        axes = self._read_axes({row[3] for row in signal_rows})

        signals_by_source: dict[str, list[Signal]] = defaultdict(list)
        for row in signal_rows:
            signal_id, source_id, name, axis_id = row[:4]
            axis = axes.get(axis_id)
            if axis is None:
                raise TimeFFormatError(f"signal {signal_id!r} refers to missing axis {axis_id!r}")
            spec = self._spec_from_row(row)
            offsets_loader = None
            if isinstance(axis, IrregularAxis):
                offsets_loader = (
                    partial(self._offsets_loader, axis_id)
                    if self._offsets_loader is not None
                    else partial(self.load_axis_offsets, axis_id)
                )
            signals_by_source[source_id].append(
                Signal.from_loader(
                    id=signal_id,
                    name=name,
                    spec=spec,
                    time_axis=axis,
                    n_values=row[12],
                    loader=(
                        self._value_loader_factory(signal_id, spec)
                        if self._value_loader_factory is not None
                        else partial(self._value_loader, signal_id, spec)
                    ),
                    time_offsets_loader=offsets_loader,
                    annotations=annotations.get(("Signal", signal_id), ()),
                    metadata=_decode_json(row[13], default={}),
                )
            )

        children, roots_by_record = self._group_sources(source_rows)

        hydrated_sources: set[str] = set()

        def hydrate_source(source_id: str, record_id: str) -> Source:
            row = source_data[source_id]
            if row[1] != record_id:
                raise TimeFFormatError(f"source {source_id!r} crosses record boundaries")
            hydrated_sources.add(source_id)
            return Source(
                id=source_id,
                name=row[3],
                sources=tuple(hydrate_source(child, record_id) for child in children[source_id]),
                signals=tuple(signals_by_source[source_id]),
                annotations=annotations.get(("Source", source_id), ()),
                metadata=_decode_json(row[4], default={}),
            )

        records: list[Record] = []
        for record_id in order:
            row = by_id[record_id]
            span = None
            if row[2] is not None:
                span = TimeInterval(start_us=row[2], end_us=row[3])
            root_sources = tuple(hydrate_source(source_id, record_id) for source_id in roots_by_record[record_id])
            records.append(
                Record(
                    record_id=record_id,
                    sources=root_sources,
                    time_series=tuple(signal for source in root_sources for signal in source.walk_signals()),
                    annotations=annotations.get(("Record", record_id), ()),
                    start_time=row[1],
                    time_span=span,
                    metadata=_decode_json(row[4], default={}),
                )
            )
        unreachable = source_data.keys() - hydrated_sources
        if unreachable:
            raise TimeFFormatError(
                f"source hierarchy contains a missing record, missing parent, or cycle: {sorted(unreachable)}"
            )
        return tuple(records)

    def _read_rows(
        self,
        table: str,
        related_ids: Iterable[str] | None,
    ) -> list[tuple[Any, ...]]:
        """Read all rows or only rows related to the supplied IDs.

        Returns:
            Rows from the requested hierarchy table.
        """
        relation_column, id_column = _HIERARCHY_KEYS[table]
        query = (
            f"SELECT * FROM {table} WHERE ? OR {relation_column} "  # noqa: S608 - fixed identifiers
            f"IN (SELECT unnest(?)) ORDER BY {id_column}"
        )
        return self.connection.execute(
            query,
            [related_ids is None, list(related_ids or ())],
        ).fetchall()

    @staticmethod
    def _index_rows(rows: list[tuple[Any, ...]], kind: str) -> dict[str, tuple[Any, ...]]:
        """Return rows by public ID and reject duplicates.

        Raises:
            TimeFFormatError: If an ID occurs more than once.
        """
        indexed = {row[0]: row for row in rows}
        if len(indexed) != len(rows):
            raise TimeFFormatError(f"control.duckdb contains duplicate {kind} IDs")
        return indexed

    @staticmethod
    def _group_sources(
        rows: list[tuple[Any, ...]],
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Group child and root Source IDs.

        Returns:
            Child IDs by parent ID and root IDs by Record ID.
        """
        children: dict[str, list[str]] = defaultdict(list)
        roots: dict[str, list[str]] = defaultdict(list)
        for source_id, record_id, parent_id, *_ in rows:
            if parent_id is None:
                roots[record_id].append(source_id)
            else:
                children[parent_id].append(source_id)
        return children, roots

    def read_dataset_annotations(self, dataset_id: str) -> tuple[Annotation, ...]:
        """Return annotations attached directly to a dataset.

        Returns:
            Dataset-level annotation occurrences in stable occurrence-ID order.
        """
        return self._read_annotations().get(("Dataset", dataset_id), ())

    def chunk_rows(self, signal_id: str) -> list[dict[str, Any]]:
        """Return values-backend chunk locators for one signal.

        Returns:
            Chunk rows in chunk-index order.

        Raises:
            TimeFFormatError: If the signal has no stored chunks.
        """
        rows = self.connection.execute(
            """SELECT chunk_index, value_path, chunk_major_index, chunk_minor_index, n_values
               FROM signal_chunks WHERE signal_id = ? ORDER BY chunk_index""",
            [signal_id],
        ).fetchall()
        if not rows:
            raise TimeFFormatError(f"signal {signal_id!r} has no stored value chunks")
        return [
            {
                "chunk_idx": row[0],
                "chunk_file": row[1],
                "chunk_major_idx": row[2],
                "chunk_minor_idx": row[3],
                "n_values": row[4],
            }
            for row in rows
        ]

    def read_tasks(self, records: Iterable[Record] | None = None) -> tuple[Task, ...]:  # noqa: PLR0914
        """Hydrate concrete tasks and restore all in-memory object references.

        Args:
            records: Records already hydrated by this reader, or ``None`` to load all records.

        Returns:
            Concrete tasks in stable ID order.

        Raises:
            TimeFFormatError: If a relationship refers to a missing object or has an invalid field.
        """
        hydrated_records = self.read_records() if records is None else tuple(records)
        records_by_id = {record.id: record for record in hydrated_records}
        signals_by_id = {signal.id: signal for record in hydrated_records for signal in record.signals}
        annotations = self._read_annotations(
            {
                "Record": set(records_by_id),
                "Source": {source.id for record in hydrated_records for source in record.walk_sources()},
                "Signal": set(signals_by_id),
            },
            read_all=True,
        )
        annotations_by_occurrence = {
            annotation.occurrence_id: annotation
            for values in annotations.values()
            for annotation in values
            if annotation.occurrence_id is not None
        }
        record_refs = self._relationship_rows("task_record_refs", "record_id")
        annotation_refs = self._relationship_rows("task_annotation_refs", "occurrence_id")

        targets_by_task: dict[str, list[object]] = defaultdict(list)
        target_rows = self.connection.execute(
            """SELECT task_id, target_kind, text_value, integer_value, float_value,
                      boolean_value, record_id, signal_id, span_start, span_end, span_signal_ids
               FROM task_targets ORDER BY task_id, position"""
        ).fetchall()
        target_columns = (
            "target_kind",
            "text_value",
            "integer_value",
            "float_value",
            "boolean_value",
            "record_id",
            "signal_id",
            "span_start",
            "span_end",
            "span_signal_ids",
        )
        for task_id, *values in target_rows:
            row = dict(zip(target_columns, values, strict=True))
            row["span_signal_ids"] = _decode_json(row["span_signal_ids"])
            targets_by_task[task_id].append(decode_target(row, records=records_by_id, signals=signals_by_id))

        task_rows = self.connection.execute(
            """SELECT task_id, task_type, prompt, scope, has_inline_targets, payload, rationale, metadata
               FROM tasks ORDER BY task_id"""
        ).fetchall()
        tasks: dict[str, Task] = {}
        for (
            task_id,
            stored_type,
            prompt,
            scope_data,
            has_inline_targets,
            payload_data,
            rationale,
            metadata,
        ) in task_rows:
            try:
                task_type = TaskType(stored_type)
                cls = TASKS[task_type]
            except (ValueError, KeyError) as exc:
                raise TimeFFormatError(f"task {task_id!r} has unknown type {stored_type!r}") from exc
            refs = record_refs.get(task_id, {})
            inputs = self._resolve_records(task_id, "inputs", refs, records_by_id)
            kwargs: dict[str, Any] = {
                "id": task_id,
                "inputs": inputs,
                "targets": tuple(targets_by_task.get(task_id, ())) if has_inline_targets else None,
                "prompt": prompt,
                "scope": decode_span(_decode_json(scope_data)),
                "rationale": rationale,
                "annotations": annotations.get(("Task", task_id), ()),
                "metadata": _decode_json(metadata, default={}),
                **decode_task_payload(task_type, _decode_json(payload_data, default={})),
            }
            if "candidate_records" in cls.__dataclass_fields__:
                kwargs["candidate_records"] = self._resolve_records(
                    task_id,
                    "candidate_records",
                    refs,
                    records_by_id,
                )
            input_annotations = self._resolve_annotations(
                task_id,
                "input_annotations",
                annotation_refs.get(task_id, {}),
                annotations_by_occurrence,
            )
            target_annotations = self._resolve_annotations(
                task_id,
                "target_annotations",
                annotation_refs.get(task_id, {}),
                annotations_by_occurrence,
            )
            kwargs.update(
                input_annotations=input_annotations,
                target_annotations=target_annotations,
            )
            tasks[task_id] = cls(**kwargs)

        dependency_rows = self.connection.execute(
            "SELECT task_id, parent_task_id FROM task_dependencies ORDER BY task_id, position"
        ).fetchall()
        dependencies: dict[str, list[Task]] = defaultdict(list)
        for task_id, parent_id in dependency_rows:
            if task_id not in tasks or parent_id not in tasks:
                raise TimeFFormatError(f"task dependency {task_id!r} -> {parent_id!r} refers to a missing task")
            dependencies[task_id].append(tasks[parent_id])
        for task_id, parents in dependencies.items():
            tasks[task_id].from_tasks = tuple(parents)
        return tuple(tasks.values())

    def _relationship_rows(self, table: str, value_column: str) -> dict[str, dict[str, tuple[str, ...]]]:
        """Read one normalized task relationship table in stored order.

        Returns:
            ``task_id -> field -> ordered object IDs``.

        Raises:
            AssertionError: If the caller requests a table outside the fixed internal allowlist.
        """
        allowed = {
            ("task_record_refs", "record_id"),
            ("task_signal_refs", "signal_id"),
            ("task_annotation_refs", "occurrence_id"),
        }
        if (table, value_column) not in allowed:
            raise AssertionError(f"unsupported relationship table {table!r}")
        rows = self.connection.execute(
            f"SELECT task_id, field, {value_column} FROM {table} ORDER BY task_id, field, position"  # noqa: S608
        ).fetchall()
        grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for task_id, field, value in rows:
            grouped[task_id][field].append(value)
        return {
            task_id: {field: tuple(values) for field, values in fields.items()} for task_id, fields in grouped.items()
        }

    @staticmethod
    def _resolve_records(
        task_id: str,
        field: str,
        refs: dict[str, tuple[str, ...]],
        records: dict[str, Record],
    ) -> tuple[Record, ...]:
        """Resolve one ordered record-reference field.

        Returns:
            The referenced records.

        Raises:
            TimeFFormatError: If an ID is missing.
        """
        try:
            return tuple(records[record_id] for record_id in refs.get(field, ()))
        except KeyError as exc:
            raise TimeFFormatError(
                f"task {task_id!r} field {field!r} refers to missing record {exc.args[0]!r}"
            ) from exc

    @staticmethod
    def _resolve_annotations(
        task_id: str,
        field: str,
        refs: dict[str, tuple[str, ...]],
        annotations: dict[str, Annotation],
    ) -> tuple[Annotation, ...]:
        """Resolve one ordered annotation-reference field.

        Returns:
            The referenced annotation occurrences.

        Raises:
            TimeFFormatError: If an occurrence is missing.
        """
        try:
            return tuple(annotations[occurrence_id] for occurrence_id in refs.get(field, ()))
        except KeyError as exc:
            raise TimeFFormatError(
                f"task {task_id!r} field {field!r} refers to missing annotation occurrence {exc.args[0]!r}"
            ) from exc

    @staticmethod
    def _add_task_record_kwargs(
        task_id: str,
        cls: type[Task],
        refs: dict[str, tuple[str, ...]],
        records: dict[str, Record],
        kwargs: dict[str, Any],
    ) -> None:
        """Add concrete task record relationships to constructor arguments."""
        resolve = lambda field: DuckDBControlReader._resolve_records(task_id, field, refs, records)  # noqa: E731
        if issubclass(cls, ForecastingTask):
            context = resolve("context_records")
            target = resolve("target_record")
            kwargs.update(context_records=context, target_record=target[0] if target else None)
        elif issubclass(cls, TSEditingTask):
            source, target = resolve("source_record"), resolve("target_record")
            kwargs.update(
                source_record=source[0] if source else None,
                target_record=target[0] if target else None,
            )
        elif issubclass(cls, TSGenerationTask):
            target = resolve("target_record")
            kwargs["target_record"] = target[0] if target else None
        elif issubclass(cls, TSCorrespondenceTask):
            candidates, targets = resolve("candidate_records"), resolve("target_records")
            kwargs.update(candidate_records=candidates, target_records=targets or None)

    @staticmethod
    def _add_task_signal_kwargs(
        task_id: str,
        cls: type[Task],
        refs: dict[str, tuple[str, ...]],
        signals: dict[str, Signal],
        kwargs: dict[str, Any],
    ) -> None:
        """Add concrete task signal relationships to constructor arguments.

        Raises:
            TimeFFormatError: If a signal ID is missing.
        """
        if not issubclass(cls, TSCorrespondenceTask):
            return
        try:
            resolved = tuple(signals[signal_id] for signal_id in refs.get("target_signals", ()))
        except KeyError as exc:
            raise TimeFFormatError(
                f"task {task_id!r} field 'target_signals' refers to missing signal {exc.args[0]!r}"
            ) from exc
        kwargs["target_signals"] = resolved or None

    def _read_axes(self, axis_ids: set[str]) -> dict[str, RegularAxis | IrregularAxis | OrdinalAxis]:
        """Hydrate each shared axis exactly once.

        Returns:
            Axes keyed by their stored IDs.

        Raises:
            TimeFFormatError: If an axis has an unknown type.
        """
        axes: dict[str, RegularAxis | IrregularAxis | OrdinalAxis] = {}
        rows = self.connection.execute(
            """SELECT axis_id, axis_type, period_numerator_us, period_denominator,
                      origin_us, first_us, last_us
               FROM axes
               WHERE axis_id IN (SELECT unnest(?))""",
            [list(axis_ids)],
        ).fetchall()
        for axis_id, axis_type, numerator, denominator, origin, first, last in rows:
            if axis_id in axes:
                raise TimeFFormatError("control.duckdb contains duplicate axis IDs")
            if axis_type == "regular":
                axes[axis_id] = RegularAxis(
                    axis_id=axis_id,
                    period_us=Fraction(numerator, denominator),
                    start_index=origin,
                )
            elif axis_type == "irregular":
                axes[axis_id] = IrregularAxis(axis_id=axis_id, first_us=first, last_us=last)
            elif axis_type == "ordinal":
                axes[axis_id] = OrdinalAxis(axis_id=axis_id)
            else:
                raise TimeFFormatError(f"axis {axis_id!r} has unknown type {axis_type!r}")
        return axes

    def load_axis_offsets(self, axis_id: str) -> pa.Array:
        """Load one shared irregular axis.

        Returns:
            The axis offsets as Arrow int64 values.

        Raises:
            TimeFFormatError: If the offsets disagree with the axis or its Signals.
        """
        rows = self.connection.execute(
            "SELECT offset_us FROM axis_offsets WHERE axis_id = ? ORDER BY position", [axis_id]
        ).fetchall()
        offsets = np.asarray([row[0] for row in rows], dtype=np.int64)
        axis = self.connection.execute(
            "SELECT first_us, last_us FROM axes WHERE axis_id = ? AND axis_type = 'irregular'",
            [axis_id],
        ).fetchone()
        if axis is None:
            raise TimeFFormatError(f"axis {axis_id!r} has offsets but is missing or not irregular")
        lengths = {
            row[0]
            for row in self.connection.execute(
                "SELECT DISTINCT n_values FROM signals WHERE axis_id = ?", [axis_id]
            ).fetchall()
        }
        if len(lengths) != 1 or len(offsets) not in lengths:
            raise TimeFFormatError(
                f"axis {axis_id!r} stores {len(offsets)} offsets but its Signals declare lengths {sorted(lengths)}"
            )
        if len(offsets) == 0 or (int(offsets[0]), int(offsets[-1])) != axis:
            raise TimeFFormatError(f"axis {axis_id!r} has offsets disagreeing with its axis endpoints {axis}")
        if np.any(np.diff(offsets) < 0):
            raise TimeFFormatError(f"axis {axis_id!r} has decreasing offsets")
        return pa.array(offsets, type=pa.int64())

    @staticmethod
    def _spec_from_row(row: tuple[Any, ...]) -> TimeSeriesSpec:
        """Reconstruct an inline signal specification.

        Returns:
            The typed specification.
        """
        return TimeSeriesSpec(
            spec_type=row[4],
            name=row[5],
            unit_value=ureg.Unit(row[6]),
            dtype=row[7],
            categories=tuple(_decode_json(row[8], default=[])),
            value_shape=tuple(_decode_json(row[9], default=[])),
            dimension_names=tuple(_decode_json(row[10], default=[])),
            nullable=row[11],
        )

    def _read_annotations(
        self,
        object_ids: dict[str, set[str]] | None = None,
        *,
        read_all: bool = True,
    ) -> dict[tuple[str, str], tuple[Annotation, ...]]:
        """Hydrate annotation content and occurrences for hierarchy objects.

        Returns:
            Attached occurrences keyed by object type and object ID.

        Raises:
            TimeFFormatError: If an occurrence has an unknown span type.
        """
        grouped: dict[tuple[str, str], list[Annotation]] = defaultdict(list)
        targets = object_ids or {}
        rows = self.connection.execute(
            """SELECT o.occurrence_id, o.content_id, o.object_type, o.object_id, o.span_type,
                      o.start_us, o.end_us, o.signal_ids, o.provenance, o.confidence,
                      o.metadata, c.content_id, c.name, c.value, c.unit, c.metadata
               FROM annotation_occurrences o
               LEFT JOIN annotation_contents c USING (content_id)
               WHERE ?
                  OR (o.object_type = 'Record' AND o.object_id IN (SELECT unnest(?)))
                  OR (o.object_type = 'Source' AND o.object_id IN (SELECT unnest(?)))
                  OR (o.object_type = 'Signal' AND o.object_id IN (SELECT unnest(?)))
               ORDER BY o.object_type, o.object_id, o.content_id, o.span_type,
                        o.start_us NULLS FIRST, o.end_us NULLS FIRST, o.occurrence_id""",
            [
                read_all,
                list(targets.get("Record", set())),
                list(targets.get("Source", set())),
                list(targets.get("Signal", set())),
            ],
        ).fetchall()
        for row in rows:
            occurrence_id, content_id, object_type, object_id = row[:4]
            if row[11] is None:
                raise TimeFFormatError(
                    f"annotation occurrence {occurrence_id!r} refers to missing content {content_id!r}"
                )
            if object_type not in {"Dataset", "Record", "Source", "Signal", "Task"}:
                raise TimeFFormatError(
                    f"annotation occurrence {occurrence_id!r} has unknown object type {object_type!r}"
                )
            target_ids = targets.get(object_type)
            if target_ids is not None and object_id not in target_ids:
                raise TimeFFormatError(
                    f"annotation occurrence {occurrence_id!r} refers to missing {object_type} {object_id!r}"
                )

            signal_ids = _decode_json(row[7])
            scope = None if signal_ids is None else tuple(signal_ids)
            span = None
            if row[4] == "point":
                span = TimePoint(start_us=row[5], time_series_ids=scope)
            elif row[4] == "interval":
                span = TimeInterval(start_us=row[5], end_us=row[6], time_series_ids=scope)
            elif row[4] != "static":
                raise TimeFFormatError(f"annotation occurrence {row[0]!r} has unknown span type {row[4]!r}")
            content_metadata = _decode_json(row[15], default={})
            description = content_metadata.pop("description", None)
            grouped[object_type, object_id].append(
                Annotation(
                    occurrence_id=occurrence_id,
                    id=row[11],
                    key=row[12],
                    value=_decode_json(row[13]),
                    unit=row[14],
                    description=description,
                    metadata=content_metadata,
                    span=span,
                    source=_decode_json(row[8]),
                    confidence=row[9],
                    occurrence_metadata=_decode_json(row[10], default={}),
                )
            )
        return {key: tuple(value) for key, value in grouped.items()}
