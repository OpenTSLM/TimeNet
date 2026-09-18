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
ValueLoaderFactory = Callable[[int, str, TimeSeriesSpec], Callable[[], pa.Array]]
OffsetsLoader = Callable[[int, str], pa.Array]


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

    def read_records(  # noqa: PLR0912, PLR0914, PLR0915 - hydration keeps related row sets together
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
        if requested == ():
            return ()
        record_query = """SELECT record_key, record_id, start_time_us, time_span_start_us,
                      time_span_end_us, metadata
               FROM records"""
        if requested is None:
            rows = self.connection.execute(f"{record_query} ORDER BY record_id").fetchall()
        else:
            rows = self.connection.execute(
                f"{record_query} WHERE record_id IN (SELECT unnest(?)) ORDER BY record_id",
                [list(requested)],
            ).fetchall()
        by_id = {row[1]: row for row in rows}
        order = tuple(by_id) if requested is None else requested
        missing = [record_id for record_id in order if record_id not in by_id]
        if missing:
            raise TimeFValidationError(f"no such record(s) in control.duckdb: {missing}")

        record_keys = [row[0] for row in rows]
        source_rows = self.connection.execute(
            """SELECT source_key, source_id, record_key, parent_source_key, name, metadata
               FROM sources
               WHERE record_key IN (SELECT unnest(?))
               ORDER BY source_id""",
            [record_keys],
        ).fetchall()
        source_keys = [row[0] for row in source_rows]
        signal_rows = self.connection.execute(
            """SELECT signal_key, signal_id, source_key, name, axis_key, spec_type, spec_name, unit, dtype,
                      categories, value_shape, dimension_names, nullable, n_values, metadata
               FROM signals
               WHERE source_key IN (SELECT unnest(?))
               ORDER BY signal_id""",
            [source_keys],
        ).fetchall()
        signal_keys = [row[0] for row in signal_rows]
        object_keys = [*record_keys, *source_keys, *signal_keys]
        annotations = self._read_annotations(object_keys) if with_annotations else {}
        axes = self._read_axes({row[4] for row in signal_rows})

        signals_by_source: dict[int, list[Signal]] = defaultdict(list)
        specs: dict[tuple[Any, ...], TimeSeriesSpec] = {}
        for row in signal_rows:
            signal_key, signal_id, source_key, name, axis_key = row[:5]
            axis = axes.get(axis_key)
            if axis is None:
                raise TimeFFormatError(f"signal {signal_id!r} refers to missing axis key {axis_key!r}")
            spec_key = row[5:13]
            spec = specs.get(spec_key)
            if spec is None:
                spec = self._spec_from_row(row)
                specs[spec_key] = spec
            offsets_loader = None
            if isinstance(axis, IrregularAxis):
                offsets_loader = (
                    partial(self._offsets_loader, axis_key, axis.axis_id)
                    if self._offsets_loader is not None
                    else partial(self.load_axis_offsets_by_key, axis_key, axis.axis_id)
                )
            signals_by_source[source_key].append(
                Signal(
                    time_series_id=signal_id,
                    signal=name,
                    spec=spec,
                    time_axis=axis,
                    n_values=row[13],
                    loader=(
                        self._value_loader_factory(signal_key, signal_id, spec)
                        if self._value_loader_factory is not None
                        else partial(self._value_loader, signal_id, spec)
                    ),
                    time_offsets_loader=offsets_loader,
                    annotations=annotations.get(("Signal", signal_id), ()),
                    metadata=_decode_json(row[14], default={}),
                )
            )

        source_data = {row[0]: row for row in source_rows}
        children: dict[int, list[int]] = defaultdict(list)
        roots_by_record: dict[int, list[int]] = defaultdict(list)
        for source_key, _, record_key, parent_key, *_ in source_rows:
            if parent_key is None:
                roots_by_record[record_key].append(source_key)
            else:
                children[parent_key].append(source_key)

        active: set[int] = set()

        def hydrate_source(source_key: int, record_key: int) -> Source:
            if source_key in active:
                raise TimeFFormatError(f"source hierarchy contains a cycle at key {source_key!r}")
            row = source_data.get(source_key)
            if row is None:
                raise TimeFFormatError(f"source hierarchy refers to missing source key {source_key!r}")
            if row[2] != record_key:
                raise TimeFFormatError(f"source {row[1]!r} crosses record boundaries")
            active.add(source_key)
            source = Source(
                id=row[1],
                name=row[4],
                sources=tuple(hydrate_source(child, record_key) for child in children[source_key]),
                signals=tuple(signals_by_source[source_key]),
                annotations=annotations.get(("Source", row[1]), ()),
                metadata=_decode_json(row[5], default={}),
            )
            active.remove(source_key)
            return source

        records: list[Record] = []
        for record_id in order:
            row = by_id[record_id]
            span = None
            if row[3] is not None:
                span = TimeInterval(start_us=row[3], end_us=row[4])
            root_sources = tuple(hydrate_source(source_key, row[0]) for source_key in roots_by_record[row[0]])
            records.append(
                Record(
                    record_id=record_id,
                    sources=root_sources,
                    time_series=tuple(signal for source in root_sources for signal in source.walk_signals()),
                    annotations=annotations.get(("Record", record_id), ()),
                    start_time=row[2],
                    time_span=span,
                    metadata=_decode_json(row[5], default={}),
                )
            )
        return tuple(records)

    def read_dataset_annotations(self, dataset_id: str) -> tuple[Annotation, ...]:
        """Return annotations attached directly to a dataset.

        Returns:
            Dataset-level annotation occurrences in stable occurrence-ID order.
        """
        row = self.connection.execute(
            "SELECT dataset_key FROM datasets WHERE dataset_id = ?",
            [dataset_id],
        ).fetchone()
        if row is None:
            return ()
        return self._read_annotations((row[0],)).get(("Dataset", dataset_id), ())

    def chunk_rows(self, signal_id: str) -> list[dict[str, Any]]:
        """Return values-backend chunk locators for one signal.

        Returns:
            Chunk rows in chunk-index order.
        """
        rows = self.connection.execute(
            """SELECT c.chunk_index, c.value_path, c.chunk_major_index, c.chunk_minor_index, c.n_values
               FROM signal_chunks c
               JOIN signals s USING (signal_key)
               WHERE s.signal_id = ? ORDER BY c.chunk_index""",
            [signal_id],
        ).fetchall()
        return self._chunk_dicts(rows, signal_id)

    def chunk_rows_by_key(self, signal_key: int, signal_id: str) -> list[dict[str, Any]]:
        """Return values-backend chunk locators through an internal integer key.

        Returns:
            Chunk rows in chunk-index order.
        """
        rows = self.connection.execute(
            """SELECT chunk_index, value_path, chunk_major_index, chunk_minor_index, n_values
               FROM signal_chunks
               WHERE signal_key = ? ORDER BY chunk_index""",
            [signal_key],
        ).fetchall()
        return self._chunk_dicts(rows, signal_id)

    @staticmethod
    def _chunk_dicts(rows: list[tuple[Any, ...]], signal_id: str) -> list[dict[str, Any]]:
        """Convert stored chunk tuples to the values-backend row contract.

        Returns:
            Chunk dictionaries in their query order.

        Raises:
            TimeFFormatError: If the signal has no stored chunks.
        """
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
        records_by_key = {
            key: records_by_id[record_id]
            for key, record_id in self.connection.execute("SELECT record_key, record_id FROM records").fetchall()
            if record_id in records_by_id
        }
        annotations_by_occurrence: dict[int, Annotation] = {}
        annotations = self._read_annotations(annotations_by_occurrence=annotations_by_occurrence)
        record_refs = self._task_record_relationships()
        annotation_refs = self._task_annotation_relationships()

        targets_by_task: dict[int, list[object]] = defaultdict(list)
        target_rows = self.connection.execute(
            """SELECT links.task_key, items.target_kind,
                      text_values.value, integer_values.value, float_values.value,
                      boolean_values.value, record_values.record_key, signal_values.signal_key,
                      span_values.span_start, span_values.span_end, span_values.signal_keys
               FROM task_targets links
               JOIN target_items items USING (target_item_key)
               LEFT JOIN target_text_values text_values USING (target_item_key)
               LEFT JOIN target_integer_values integer_values USING (target_item_key)
               LEFT JOIN target_float_values float_values USING (target_item_key)
               LEFT JOIN target_boolean_values boolean_values USING (target_item_key)
               LEFT JOIN target_record_values record_values USING (target_item_key)
               LEFT JOIN target_signal_values signal_values USING (target_item_key)
               LEFT JOIN target_span_values span_values USING (target_item_key)
               ORDER BY links.task_key, links.position"""
        ).fetchall()
        needs_signals = any(row[7] is not None or row[10] for row in target_rows)
        signals_by_id = (
            {signal.id: signal for record in hydrated_records for signal in record.signals} if needs_signals else {}
        )
        signals_by_key = (
            {
                key: signals_by_id[signal_id]
                for key, signal_id in self.connection.execute("SELECT signal_key, signal_id FROM signals").fetchall()
                if signal_id in signals_by_id
            }
            if needs_signals
            else {}
        )
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
        for task_key, *values in target_rows:
            row = dict(zip(target_columns, values, strict=True))
            record_key = row["record_id"]
            signal_key = row["signal_id"]
            span_signal_keys = row["span_signal_ids"]
            row["record_id"] = None if record_key is None else records_by_key[record_key].id
            row["signal_id"] = None if signal_key is None else signals_by_key[signal_key].id
            row["span_signal_ids"] = (
                None if span_signal_keys is None else tuple(signals_by_key[key].id for key in span_signal_keys)
            )
            if row["target_kind"] in {"step_point", "step_interval"}:
                span_signal_ids = row["span_signal_ids"]
                row["signal_id"] = None if not span_signal_ids else span_signal_ids[0]
            targets_by_task[task_key].append(decode_target(row, records=records_by_id, signals=signals_by_id))

        task_rows = self.connection.execute(
            """SELECT task_key, task_id, task_type, prompt, scope, has_inline_targets, payload, rationale, metadata
               FROM tasks ORDER BY task_id"""
        ).fetchall()
        tasks: dict[int, Task] = {}
        for (
            task_key,
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
            refs = record_refs.get(task_key, {})
            inputs = self._resolve_records(task_id, "inputs", refs, records_by_key)
            kwargs: dict[str, Any] = {
                "id": task_id,
                "inputs": inputs,
                "targets": tuple(targets_by_task.get(task_key, ())) if has_inline_targets else None,
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
                    records_by_key,
                )
            input_annotations = self._resolve_annotations(
                task_id,
                "input_annotations",
                annotation_refs.get(task_key, {}),
                annotations_by_occurrence,
            )
            target_annotations = self._resolve_annotations(
                task_id,
                "target_annotations",
                annotation_refs.get(task_key, {}),
                annotations_by_occurrence,
            )
            kwargs.update(
                input_annotations=input_annotations,
                target_annotations=target_annotations,
            )
            tasks[task_key] = cls(**kwargs)

        dependency_rows = self.connection.execute(
            "SELECT task_key, parent_task_key FROM task_dependencies ORDER BY task_key, position"
        ).fetchall()
        dependencies: dict[int, list[Task]] = defaultdict(list)
        for task_key, parent_key in dependency_rows:
            if task_key not in tasks or parent_key not in tasks:
                raise TimeFFormatError(f"task dependency key {task_key!r} -> {parent_key!r} refers to a missing task")
            dependencies[task_key].append(tasks[parent_key])
        for task_key, parents in dependencies.items():
            tasks[task_key].from_tasks = tuple(parents)
        return tuple(tasks.values())

    def _task_record_relationships(self) -> dict[int, dict[str, tuple[int, ...]]]:
        """Read ordered task-to-Record relationships through integer keys.

        Returns:
            ``task_key -> field -> ordered internal Record keys``.
        """
        rows = self.connection.execute(
            """SELECT task_key, field, record_key
               FROM task_record_refs
               ORDER BY task_key, field, position"""
        ).fetchall()
        return self._group_relationships(rows)

    def _task_annotation_relationships(self) -> dict[int, dict[str, tuple[int, ...]]]:
        """Read ordered task-to-annotation relationships through integer keys.

        Returns:
            ``task_key -> field -> ordered internal occurrence keys``.
        """
        rows = self.connection.execute(
            """SELECT task_key, field, occurrence_key
               FROM task_annotation_refs
               ORDER BY task_key, field, position"""
        ).fetchall()
        return self._group_relationships(rows)

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
    def _resolve_records(
        task_id: str,
        field: str,
        refs: dict[str, tuple[int, ...]],
        records: dict[int, Record],
    ) -> tuple[Record, ...]:
        """Resolve one ordered record-reference field.

        Returns:
            The referenced records.

        Raises:
            TimeFFormatError: If an ID is missing.
        """
        try:
            return tuple(records[record_key] for record_key in refs.get(field, ()))
        except KeyError as exc:
            raise TimeFFormatError(
                f"task {task_id!r} field {field!r} refers to missing record key {exc.args[0]!r}"
            ) from exc

    @staticmethod
    def _resolve_annotations(
        task_id: str,
        field: str,
        refs: dict[str, tuple[int, ...]],
        annotations: dict[int, Annotation],
    ) -> tuple[Annotation, ...]:
        """Resolve one ordered annotation-reference field.

        Returns:
            The referenced annotation occurrences.

        Raises:
            TimeFFormatError: If an occurrence is missing.
        """
        try:
            return tuple(annotations[occurrence_key] for occurrence_key in refs.get(field, ()))
        except KeyError as exc:
            raise TimeFFormatError(
                f"task {task_id!r} field {field!r} refers to missing annotation occurrence key {exc.args[0]!r}"
            ) from exc

    def _read_axes(
        self,
        axis_keys: Iterable[int] | None = None,
    ) -> dict[int, RegularAxis | IrregularAxis | OrdinalAxis]:
        """Hydrate each shared axis exactly once.

        Returns:
            Axes keyed by their stored IDs.

        Raises:
            TimeFFormatError: If an axis has an unknown type.
        """
        requested = None if axis_keys is None else tuple(axis_keys)
        if requested == ():
            return {}
        axes: dict[int, RegularAxis | IrregularAxis | OrdinalAxis] = {}
        query = """SELECT axis_key, axis_id, axis_type, period_numerator_us, period_denominator,
                          origin_us, first_us, last_us FROM axes"""
        rows = (
            self.connection.execute(query).fetchall()
            if requested is None
            else self.connection.execute(
                f"{query} WHERE axis_key IN (SELECT unnest(?))",
                [list(requested)],
            ).fetchall()
        )
        for axis_key, axis_id, axis_type, numerator, denominator, origin, first, last in rows:
            if axis_type == "regular":
                axes[axis_key] = RegularAxis(
                    axis_id=axis_id,
                    period_us=Fraction(numerator, denominator),
                    start_index=origin,
                )
            elif axis_type == "irregular":
                axes[axis_key] = IrregularAxis(axis_id=axis_id, first_us=first, last_us=last)
            elif axis_type == "ordinal":
                axes[axis_key] = OrdinalAxis(axis_id=axis_id)
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
        axis = self.connection.execute(
            "SELECT axis_key FROM axes WHERE axis_id = ? AND axis_type = 'irregular'",
            [axis_id],
        ).fetchone()
        if axis is None:
            raise TimeFFormatError(f"axis {axis_id!r} has offsets but is missing or not irregular")
        return self.load_axis_offsets_by_key(axis[0], axis_id)

    def load_axis_offsets_by_key(self, axis_key: int, axis_id: str) -> pa.Array:
        """Load one shared irregular axis through its internal integer key.

        Returns:
            The axis offsets as Arrow int64 values.

        Raises:
            TimeFFormatError: If the offsets disagree with the axis or its Signals.
        """
        rows = self.connection.execute(
            """SELECT offsets.offset_us
               FROM axis_offsets offsets
               WHERE axis_key = ? ORDER BY offsets.position""",
            [axis_key],
        ).fetchall()
        offsets = np.asarray([row[0] for row in rows], dtype=np.int64)
        axis = self.connection.execute(
            "SELECT first_us, last_us FROM axes WHERE axis_key = ? AND axis_type = 'irregular'",
            [axis_key],
        ).fetchone()
        if axis is None:
            raise TimeFFormatError(f"axis {axis_id!r} has offsets but is missing or not irregular")
        lengths = {
            row[0]
            for row in self.connection.execute(
                "SELECT DISTINCT n_values FROM signals WHERE axis_key = ?", [axis_key]
            ).fetchall()
        }
        if len(lengths) != 1 or len(offsets) not in lengths:
            raise TimeFFormatError(
                f"axis {axis_id!r} stores {len(offsets)} offsets but its Signals declare lengths {sorted(lengths)}"
            )
        endpoints = axis[:2]
        if len(offsets) == 0 or (int(offsets[0]), int(offsets[-1])) != endpoints:
            raise TimeFFormatError(f"axis {axis_id!r} has offsets disagreeing with its axis endpoints {endpoints}")
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
            spec_type=row[5],
            name=row[6],
            unit_value=ureg.Unit(row[7]),
            dtype=row[8],
            categories=tuple(_decode_json(row[9], default=[])),
            value_shape=tuple(_decode_json(row[10], default=[])),
            dimension_names=tuple(_decode_json(row[11], default=[])),
            nullable=row[12],
        )

    def _read_annotations(
        self,
        object_keys: Iterable[int] | None = None,
        *,
        annotations_by_occurrence: dict[int, Annotation] | None = None,
    ) -> dict[tuple[str, str], tuple[Annotation, ...]]:
        """Hydrate annotation content and occurrences for hierarchy objects.

        Returns:
            Attached occurrences keyed by object type and object ID.

        Raises:
            TimeFFormatError: If an occurrence has an unknown span type.
        """
        requested = None if object_keys is None else tuple(object_keys)
        if requested == ():
            return {}
        grouped: dict[tuple[str, str], list[Annotation]] = defaultdict(list)
        query = """WITH object_ids AS (
                   SELECT dataset_key AS object_key, 'Dataset' AS object_type, dataset_id AS object_id
                   FROM datasets
                   UNION ALL
                   SELECT record_key, 'Record', record_id FROM records
                   UNION ALL
                   SELECT source_key, 'Source', source_id FROM sources
                   UNION ALL
                   SELECT signal_key, 'Signal', signal_id FROM signals
                   UNION ALL
                   SELECT task_key, 'Task', task_id FROM tasks
               )
               SELECT o.occurrence_key, o.occurrence_id, o.object_type, objects.object_id, o.span_type,
                      o.start_us, o.end_us, o.signal_keys, o.provenance, o.confidence,
                      o.metadata, c.content_id, c.name, c.value, c.unit, c.metadata
               FROM annotation_occurrences o
               JOIN annotation_contents c USING (content_key)
               JOIN object_ids objects
                 ON objects.object_key = o.object_key AND objects.object_type = o.object_type
               {predicate}
               ORDER BY o.object_type, objects.object_id, c.content_id, o.span_type,
                        o.start_us NULLS FIRST, o.end_us NULLS FIRST, o.occurrence_id"""
        predicate = "" if requested is None else "WHERE o.object_key IN (SELECT unnest(?))"
        rows = self.connection.execute(
            query.format(predicate=predicate),
            [] if requested is None else [list(requested)],
        ).fetchall()
        referenced_signal_keys = sorted({signal_key for row in rows for signal_key in (row[7] or ())})
        signal_ids_by_key = (
            dict(
                self.connection.execute(
                    """SELECT signal_key, signal_id FROM signals
                       WHERE signal_key IN (SELECT unnest(?))""",
                    [referenced_signal_keys],
                ).fetchall()
            )
            if referenced_signal_keys
            else {}
        )
        for row in rows:
            signal_keys = row[7]
            scope = None if signal_keys is None else tuple(signal_ids_by_key[signal_key] for signal_key in signal_keys)
            span = None
            if row[4] == "point":
                span = TimePoint(start_us=row[5], time_series_ids=scope)
            elif row[4] == "interval":
                span = TimeInterval(start_us=row[5], end_us=row[6], time_series_ids=scope)
            elif row[4] != "static":
                raise TimeFFormatError(f"annotation occurrence {row[1]!r} has unknown span type {row[4]!r}")
            content_metadata = _decode_json(row[15], default={})
            description = content_metadata.pop("description", None)
            annotation = Annotation(
                occurrence_id=row[1],
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
            grouped[row[2], row[3]].append(annotation)
            if annotations_by_occurrence is not None:
                annotations_by_occurrence[row[0]] = annotation
        return {key: tuple(value) for key, value in grouped.items()}
