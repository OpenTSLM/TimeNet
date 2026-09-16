"""``TimeFReader`` reads a TimeF version directory and rebuilds it as a :class:`TimeFDataset`.

Opening a version reads only ``manifest.json``, and the reader never runs connector code. Tasks,
records, annotations, and each series' values resolve on first use. The reader rebuilds types from
the manifest's flat descriptors and the version's control database. It never creates classes at
runtime, so read-back objects pickle and match the original objects field for field.

Records come back a batch at a time, with one query per table for the whole batch.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import json
import types as _types
from typing import TYPE_CHECKING, Any, cast

import duckdb
import numpy as np
import pyarrow as pa

from timenet.control_plane.payload import PayloadKind, task_payload
from timenet.control_plane.reader import ControlPlaneReader
from timenet.control_plane.spans import span_from_row
from timenet.dataset import Record, TimeFDataset, TimeSeries
from timenet.dataset.axis import AxisType, IrregularAxis, OrdinalAxis, RegularAxis, TimeAxis
from timenet.dataset.record import check_span_within_window
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.checksums import stream_checksum
from timenet.types import (
    TASKS,
    Annotation,
    DatasetMetadata,
    DatasetSchema,
    Task,
    TaskType,
    TimeInterval,
    annotation_type_of,
    value_type_of,
)
from timenet.values_backends.reader import BaseValuesReader, make_values_reader


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion


_RECORD_BATCH_ROWS = 512
"""How many records one round of control-plane queries rebuilds.

Each query in the round answers for the whole batch. A larger batch costs less per record and holds
more decoded rows in memory.
"""

_INDEX_ROWS_CACHE_RECORDS = 8
"""How many records keep their chunk locators. A read walks one record at a time, so this is small."""

_AXIS_CACHE_SIZE = 1024
"""Maximum number of stored time axes. Series with the same timing can reuse an axis.
After this limit, the reader builds other axes without keeping them."""


class TimeFReader:
    """Reads a committed TimeF version directory. Use as a context manager to close its handles."""

    def __init__(self, version: DatasetVersion) -> None:
        """Open a committed dataset version through a storage handle.

        This constructor reads nothing. The handle already carries the parsed manifest, the
        filesystem, and the root that every later read uses. Tasks, records, annotations, and each
        series' values resolve on first use.

        A corrupt or missing file fails on its first access, not here. A first access is ``.tasks``,
        the first record, or the first value read. Call :meth:`verify` to check the version's
        integrity up front. Build the handle with
        :meth:`~timenet.registry.BaseRegistry.open_version` or with
        :meth:`~timenet.registry.version.DatasetVersion.open_local`.

        Args:
            version: The opened version handle: a manifest plus a filesystem-rooted view of its files.
        """
        self._version = version
        # Aliases of the handle's fields, for the read sites below. All three pickle, so a
        # DataLoader worker can rebuild the reader and its lazy loaders without reopening the
        # registry.
        self._fs = version.filesystem
        self._root = version.root
        self._manifest = version.manifest

        self._spec_by_type = {spec.spec_type: spec for spec in self._manifest.schema.time_series_specs}
        self._annotation_descriptors = {d.key: d for d in self._manifest.schema.annotations}
        # Per-process scratch space, built on demand and dropped on pickle. Keep ``__getstate__`` in
        # step with these fields.
        self._control: ControlPlaneReader | None = None
        self._tasks: tuple[Task, ...] | None = None
        self._values: BaseValuesReader | None = None
        self._axis_cache: dict[tuple, TimeAxis] = {}
        self._index_rows_cache: OrderedDict[str, dict[str, list[dict]]] = OrderedDict()

    # ---- pickling ------------------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Drop every on-demand cache so the reader (and its loaders) pickle small and safely.

        A DuckDB connection does not cross a process boundary, so a worker opens its own on first
        use.

        Returns:
            The reader's state with every on-demand cache emptied.
        """
        state = self.__dict__.copy()
        state["_control"] = None
        state["_values"] = None
        state["_tasks"] = None
        state["_axis_cache"] = {}
        state["_index_rows_cache"] = OrderedDict()
        return state

    # ---- context manager -----------------------------------------------------------------------

    def __enter__(self) -> TimeFReader:
        """Return this reader."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: _types.TracebackType | None,
    ) -> None:
        """Close the control database and all cached shard handles."""
        self.close()

    def close(self) -> None:
        """Close the values backend and the control database, releasing their caches."""
        if self._values is not None:
            self._values.close()
            self._values = None
        if self._control is not None:
            self._control.close()
            self._control = None
        self._index_rows_cache.clear()

    # ---- public API ----------------------------------------------------------------------------

    def verify(self) -> None:
        """Check every file the manifest lists against its recorded ``sha256:`` checksum.

        Opening a version does not run this check. It hashes every listed file, so it reads the
        whole dataset. Call it when integrity matters more than speed: after a download, before a
        long training run, or from a fsck-style command. It reopens each file through the version's
        filesystem, so a missing file surfaces here.

        Raises:
            TimeFFormatError: If a listed file is missing, or its contents do not match the manifest.
        """
        for part in sorted(self._manifest.files.all_files(), key=lambda p: p.path):
            try:
                handle = self._fs.open_input_file(self._version.path(part.path))
            except FileNotFoundError as exc:
                raise TimeFFormatError(f"manifest lists a missing file: {part.path}") from exc
            with handle:
                actual_size = handle.size()
                if actual_size != part.size:
                    raise TimeFFormatError(
                        f"size mismatch for {part.path}: manifest says {part.size}, file is {actual_size}"
                    )
                actual = stream_checksum(handle)
            if actual != part.checksum:
                raise TimeFFormatError(
                    f"checksum mismatch for {part.path}: manifest says {part.checksum}, file is {actual}"
                )

    @property
    def metadata(self) -> DatasetMetadata:
        """The dataset's descriptive identity."""
        return self._manifest.metadata

    @property
    def schema(self) -> DatasetSchema:
        """The dataset's type declaration, reconstructed from the manifest."""
        return self._manifest.schema

    @property
    def tasks(self) -> tuple[Task, ...]:
        """All tasks, with ``from_tasks`` resolved. Decoded on first access and cached."""
        if self._tasks is None:
            with self._as_format_error():
                self._tasks = self._load_tasks()
        return self._tasks

    @property
    def values_backend(self) -> str:
        """The manifest's values-plane backend tag (``"parquet"`` or ``"zarr"``)."""
        return self._manifest.values_backend

    def read(self) -> TimeFDataset:
        """Materialize the full dataset.

        Returns:
            A :class:`TimeFDataset` with lazy per-series loaders and the reconstructed schema/tasks.
        """
        records = list(self.iter_records())
        tasks = self.tasks
        # A streamed-written dataset stores empty record.task_ids, because its tasks were never held
        # in memory. read() materializes every task, so rebuild the reverse map here, or tasks_for()
        # and the torch view return no tasks. A materialized dataset already round-trips its
        # task_ids through the record rows, so this changes nothing for it.
        by_id = {record.record_id: record for record in records}
        for task in tasks:
            for record_id in task.record_ids:
                record = by_id.get(record_id)
                if record is not None and task.id not in record.task_ids:
                    record.task_ids = (*record.task_ids, task.id)
        return TimeFDataset.from_parts(
            metadata=self._manifest.metadata,
            records=records,
            tasks=tasks,
            schema=self._manifest.schema,
            registered_annotations=self._read_registered_annotations(),
        )

    def iter_records(
        self,
        record_ids: Iterable[str] | None = None,
        *,
        with_annotations: bool = True,
    ) -> Iterator[Record]:
        """Yield records lazily without materializing a :class:`TimeFDataset`.

        Records are rebuilt a batch at a time, with one query per table for the whole batch. They
        come back in stored order (sorted by ``record_id``), not in the order you asked for them.

        Args:
            record_ids: The records to yield, or ``None`` to yield every record.
            with_annotations: Whether to resolve the annotations of each record. ``False`` leaves
                :attr:`Record.annotations` empty, and skips their query, their JSON parse, and the
                check of each stored span against the record window.

        Yields:
            Each reconstructed :class:`Record`.

        Raises:
            TimeFValidationError: If ``record_ids`` names an id the dataset does not contain. The
                error raises once the iterator is fully consumed, not at the offending id, so a
                consumer that stops early never reaches the check.
        """
        with self._as_format_error():
            control = self._control_plane()
            missing: list[str] = []
            if record_ids is None:
                batches: Iterable[list[int]] = control.record_id_batches(_RECORD_BATCH_ROWS)
            else:
                found, missing = control.resolve_record_ids(list(dict.fromkeys(record_ids)))
                batches = [found[at : at + _RECORD_BATCH_ROWS] for at in range(0, len(found), _RECORD_BATCH_ROWS)]
            for batch in batches:
                yield from self._build_batch(batch, with_annotations=with_annotations)
        if missing:
            raise TimeFValidationError(f"no such record(s) in {self._manifest.dataset_id}: {', '.join(missing)}")

    # ---- loading -------------------------------------------------------------------------------

    def _control_plane(self) -> ControlPlaneReader:
        """Return the open control-plane view, built on first use.

        Returns:
            The cached view.
        """
        if self._control is None:
            self._control = ControlPlaneReader(self._version)
        return self._control

    @contextmanager
    def _as_format_error(self) -> Iterator[None]:
        """Re-raise a decode failure against the manifest's schema as a :class:`TimeFFormatError`.

        The loaders rebuild typed objects from the control database and the manifest's schema.
        Anything they raise means the artifact is corrupt, or that it disagrees with its manifest.
        The reader's contract is that every such failure reaches the caller as a
        :class:`TimeFFormatError`, so a bare ``ValueError`` from ``TaskType()``, or a
        ``duckdb.Error`` from a truncated database, must not escape as-is.

        Yields:
            Nothing. This context manager only rewrites the exception type.

        Raises:
            TimeFFormatError: If the wrapped code fails to decode what the manifest describes.
        """
        try:
            yield
        except TimeFFormatError:
            raise
        except (ValueError, KeyError, IndexError, TypeError, AttributeError, OSError, duckdb.Error) as exc:
            raise TimeFFormatError(f"corrupt or inconsistent TimeF artifact at {self._root}: {exc}") from exc

    def _build_batch(self, record_ids: list[int], *, with_annotations: bool) -> Iterator[Record]:
        """Rebuild one batch of records, with one query per table for the whole batch.

        Args:
            record_ids: The surrogate ids of the records to rebuild, in stored order.
            with_annotations: Whether to resolve each record's annotations.

        Yields:
            Each rebuilt record, in stored order.
        """
        if not record_ids:
            return
        control = self._control_plane()
        series = _by_record(control.record_series(record_ids))
        annotations = _by_record(control.record_annotations(record_ids)) if with_annotations else {}
        for row in control.records(record_ids):
            key = row["record_id"]
            yield self._build_record(row, series.get(key, ()), annotations.get(key, ()))

    def _load_tasks(self) -> tuple[Task, ...]:
        """Rebuild every task, one query per payload table for the whole version.

        Returns:
            The tasks, with ``from_tasks`` resolved to the rebuilt instances.

        Raises:
            TimeFFormatError: If a task derives from an id no task carries.
        """
        rows = self._control_plane().tasks()
        items = _grouped(rows["items"], "task_id")
        fields = _grouped(rows["fields"], "task_id")
        refs = _grouped(rows["refs"], "task_id")
        spans = _grouped(rows["spans"], "task_id")
        attached = _grouped(rows["annotations"], "task_id")

        by_id: dict[str, Task] = {}
        pending: dict[str, tuple[str, ...]] = {}
        for row in rows["tasks"]:
            key = row["task_id"]
            task_type = TaskType(row["task_type"])
            cls = TASKS[task_type]
            own_spans = _grouped(spans.get(key, ()), "field")
            scope_rows = own_spans.get("scope", ())
            task = cls(
                id=row["external_id"],
                record_ids=_items(items.get(key, ()), "record"),
                prompt=row["prompt"],
                scope=_span(scope_rows[0]) if scope_rows else None,
                input_annotation_ids=_items(attached.get(key, ()), "input", column="role"),
                target_annotation_ids=_items(attached.get(key, ()), "target", column="role"),
                rationale=row["rationale"],
                **_task_payload(task_type, fields.get(key, ()), refs.get(key, ()), own_spans),  # ty: ignore[invalid-argument-type]
            )
            by_id[task.id] = task
            pending[task.id] = _items(items.get(key, ()), "from_task")
        for task_id, from_ids in pending.items():
            resolved = []
            for from_id in from_ids:
                if from_id not in by_id:
                    raise TimeFFormatError(f"task {task_id!r} references unknown from_task_id {from_id!r}")
                resolved.append(by_id[from_id])
            by_id[task_id].from_tasks = tuple(resolved)
        return tuple(by_id.values())

    def _decode_annotation(self, row: dict) -> Annotation:
        """Rebuild one annotation from its stored row, checked against its manifest descriptor.

        Args:
            row: The stored annotation row.

        Returns:
            The rebuilt annotation.

        Raises:
            TimeFFormatError: If the row's key is not in the schema, or the decoded annotation's shape
                or value type disagrees with its descriptor.
        """
        key = row["key"]
        if key not in self._annotation_descriptors:
            raise TimeFFormatError(f"annotation row references unknown key {key!r} (not in schema)")
        descriptor = self._annotation_descriptors[key]
        fields: dict = {
            "id": row["external_id"],
            "key": key,
            "value": None if row["value"] is None else json.loads(row["value"]),
            "source": row["source"],
            "unit": descriptor.unit,
            "description": descriptor.description,
        }
        if row["span_start_us"] is not None:
            fields["span"] = span_from_row(
                "seconds", row["span_start_us"], row["span_end_us"], row["span_time_series_ids"]
            )
        annotation = Annotation(**fields)
        # The descriptor is what a registry query filters on. A decoded annotation whose shape or
        # value type disagrees with the descriptor answers those queries wrongly, and that is corruption.
        derived_type = annotation_type_of(annotation)
        if derived_type != descriptor.annotation_type:
            raise TimeFFormatError(
                f"annotation {key!r} decodes to shape {derived_type.value!r} but its descriptor "
                f"says {descriptor.annotation_type.value!r}"
            )
        derived_value_type = value_type_of(annotation.value)
        if derived_value_type != descriptor.value_type:
            raise TimeFFormatError(
                f"annotation {key!r} decodes to value type {derived_value_type!r} but its "
                f"descriptor says {descriptor.value_type!r}"
            )
        return annotation

    def _read_registered_annotations(self) -> tuple[Annotation, ...]:
        """Rebuild annotations that no record carries: they exist only for tasks to reference.

        No record lists them, so :meth:`iter_records` never reaches them and :meth:`read` pulls them
        here. The manifest count gates the query, so a dataset with none runs none.

        Returns:
            The registered annotations, in registration order.
        """
        if self._manifest.counts.registered_annotations == 0:
            return ()
        with self._as_format_error():
            return tuple(self._decode_annotation(row) for row in self._control_plane().registered_annotations())

    def _index_rows(self, record_id: str, time_series_id: str) -> list[dict]:
        """Return one series' chunk locators, in ``chunk_idx`` order.

        One query returns the locators of every series of a record. The result is held for the few
        records a read has in flight.

        Args:
            record_id: The owning record's id.
            time_series_id: The series id.

        Returns:
            The series' chunk locator rows, empty if it has none.
        """
        with self._as_format_error():
            by_series = self._index_rows_cache.get(record_id)
            if by_series is None:
                by_series = {}
                for row in self._control_plane().record_chunks(record_id):
                    by_series.setdefault(row["time_series_id"], []).append(row)
                self._index_rows_cache[record_id] = by_series
                if len(self._index_rows_cache) > _INDEX_ROWS_CACHE_RECORDS:
                    self._index_rows_cache.popitem(last=False)
            return by_series.get(time_series_id, [])

    # ---- record construction -------------------------------------------------------------------

    def _build_record(self, row: dict, series_rows: Iterable[dict], annotation_rows: Iterable[dict]) -> Record:
        """Rebuild one record from its stored row and the rows that hang off it.

        Args:
            row: The record's own stored row.
            series_rows: Its series rows, in the record's own series order.
            annotation_rows: Its annotation rows, in the record's own annotation order.

        Returns:
            The rebuilt record.

        Raises:
            TimeFFormatError: If a stored span or ``time_span`` no longer fits the record, which is a
                corrupt artifact rather than a caller mistake.
        """
        record_id = row["external_id"]
        series = tuple(self._build_series(record_id, struct) for struct in series_rows)
        annotations = tuple(self._decode_annotation(annotation) for annotation in annotation_rows)
        # The record is decoded and built inside the try block so ``Record.__post_init__`` can reject
        # a malformed time_span before it is used to recheck the stored annotation spans.
        try:
            start, end = row["time_span_start_us"], row["time_span_end_us"]
            time_span = None if start is None else cast("TimeInterval", span_from_row("seconds", start, end, None))
            record = Record(
                record_id=record_id,
                time_series=series,
                subject_ids=tuple(row["subject_ids"]),
                task_ids=tuple(row["task_ids"]),
                annotations=annotations,
                start_time=row["start_time_us"],
                time_span=time_span,
            )
            for annotation in annotations:
                if annotation.span is not None:
                    # The span is already on disk. This warns rather than refuses the load, so the
                    # caller gets the dataset the writer accepted and can judge the span itself.
                    check_span_within_window(
                        f"annotation {annotation.key!r}", annotation.span, series, record_id, time_span
                    )
            return record
        except TimeFValidationError as exc:
            raise TimeFFormatError(str(exc)) from exc

    def _build_series(self, record_id: str, struct: dict) -> TimeSeries:
        """Rebuild one series from its stored row, with lazy loaders for its values.

        Args:
            record_id: The owning record's id.
            struct: The stored series row, with its spec and axis columns joined in.

        Returns:
            The rebuilt series.

        Raises:
            TimeFFormatError: If the row names a spec type the schema does not declare, or the series
                cannot be built from what the row says.
        """
        spec_type = struct["spec_type"]
        if spec_type not in self._spec_by_type:
            raise TimeFFormatError(f"record {record_id!r} references unknown spec_type {spec_type!r}")
        time_series_id = struct["external_id"]
        axis = self._axis(struct, record_id)
        n_values = struct["n_values"]
        time_offsets_loader = None
        if isinstance(axis, IrregularAxis):
            time_offsets_loader = _TimeOffsetsLoader(
                self, record_id, time_series_id, axis.first_us, axis.last_us, n_values
            )
        try:
            return TimeSeries(
                spec=self._spec_by_type[spec_type],
                signal=struct["signal"],
                time_axis=axis,
                loader=_SeriesLoader(self, record_id, time_series_id),
                time_offsets_loader=time_offsets_loader,
                source_id=struct["source_id"],
                time_series_id=time_series_id,
                n_values=n_values,
            )
        except TimeFValidationError as exc:
            # TimeSeries raises the same type for a corrupt row (bad n_values, a time offsets/axis
            # mismatch) and for a caller mistake. A row read from disk is a format failure.
            raise TimeFFormatError(f"record {record_id!r} has an unbuildable series {time_series_id!r}: {exc}") from exc

    def _axis(self, struct: dict, record_id: str) -> TimeAxis:
        """Return a time axis and reuse an existing axis when its stored columns match.

        Axis objects cannot change, so series can share them. Reuse saves repeated ``Fraction``
        arithmetic and validation. The cache holds at most :data:`_AXIS_CACHE_SIZE` distinct axes.
        After that limit, the reader builds other axes without keeping them.

        Args:
            struct: The stored series row.
            record_id: The owning record, for the error message.

        Returns:
            The axis.
        """
        key = (
            struct["axis_type"],
            struct["period_numerator_us"],
            struct["period_denominator"],
            struct["start_index"],
            struct["first_us"],
            struct["last_us"],
        )
        axis = self._axis_cache.get(key)
        if axis is None:
            axis = self._build_axis(struct, record_id)
            if len(self._axis_cache) < _AXIS_CACHE_SIZE:
                self._axis_cache[key] = axis
        return axis

    @staticmethod
    def _build_axis(struct: dict, record_id: str) -> TimeAxis:
        """Rebuild a series' time axis from the stored discriminator.

        This method reads the tag before any shape-specific column, so a corrupt row raises here
        rather than building an axis out of null columns.

        Args:
            struct: The stored series row.
            record_id: The owning record, for the error message.

        Returns:
            The axis.

        Raises:
            TimeFFormatError: If the tag is missing, unknown, or disagrees with the columns beside it.
        """
        kind = struct["axis_type"]
        if kind == AxisType.ORDINAL:
            # An ordinal series has no cadence and no per-value time offsets, so every shape column
            # must be null. A populated one means the tag and the columns disagree, which is the
            # corruption this method catches.
            shape_cols = ("period_numerator_us", "period_denominator", "start_index", "first_us", "last_us")
            if any(struct[c] is not None for c in shape_cols):
                raise TimeFFormatError(
                    f"record {record_id!r} has a series tagged {kind!r} but carries regular- or "
                    f"irregular-axis columns; an ordinal series has neither"
                )
            return OrdinalAxis()
        if kind == AxisType.REGULAR:
            numerator, denominator = struct["period_numerator_us"], struct["period_denominator"]
            start_index = struct["start_index"]
            if numerator is None or denominator is None or start_index is None:
                raise TimeFFormatError(
                    f"record {record_id!r} has a series tagged {kind!r} with no period or start "
                    f"index; a regular axis needs both"
                )
            try:
                return RegularAxis(period_us=Fraction(numerator, denominator), start_index=start_index)
            except (ZeroDivisionError, TypeError, TimeFValidationError) as exc:
                raise TimeFFormatError(
                    f"record {record_id!r} has a series with an unbuildable regular axis "
                    f"(period {numerator}/{denominator}, start_index {start_index}): {exc}"
                ) from exc
        if kind == AxisType.IRREGULAR:
            first, last = struct["first_us"], struct["last_us"]
            if first is None or last is None:
                raise TimeFFormatError(
                    f"record {record_id!r} has a series tagged {kind!r} with no endpoints; an irregular axis needs both"
                )
            try:
                return IrregularAxis(first_us=first, last_us=last)
            except TimeFValidationError as exc:
                raise TimeFFormatError(
                    f"record {record_id!r} has a series with unbuildable irregular endpoints ({first}, {last}): {exc}"
                ) from exc
        raise TimeFFormatError(
            f"record {record_id!r} has a series with axis_type {kind!r}; expected one of {[t.value for t in AxisType]}"
        )

    # ---- values --------------------------------------------------------------------------------

    def _load_time_offsets(self, record_id: str, time_series_id: str) -> pa.Array:
        """Read an irregular series' per-value time offsets through the values backend.

        Args:
            record_id: The owning record's id.
            time_series_id: The series id to read.

        Returns:
            One int64 microsecond time offset per value.

        Raises:
            TimeFFormatError: If the series has no chunk locator or its time offsets cannot be read.
        """
        rows = self._index_rows(record_id, time_series_id)
        if not rows:
            raise TimeFFormatError(f"no index entry for record {record_id!r} series {time_series_id!r}")
        if self._values is None:
            self._values = make_values_reader(self._manifest.values_backend)
        try:
            return self._values.load_time_offsets(self._version, rows)
        except (KeyError, OSError, IndexError, ValueError) as exc:
            raise TimeFFormatError(
                f"failed to read time offsets for series {time_series_id!r} for record {record_id!r}: {exc}"
            ) from exc

    def _load_values(self, record_id: str, time_series_id: str) -> pa.Array:
        """Read and concatenate a series' chunk values through the values backend.

        Args:
            record_id: The owning record's id.
            time_series_id: The series id to read.

        Returns:
            The series values in the spec's canonical Arrow representation.

        Raises:
            TimeFFormatError: If the series has no chunk locator or a chunk cannot be read.
        """
        rows = self._index_rows(record_id, time_series_id)
        if not rows:
            raise TimeFFormatError(f"no index entry for record {record_id!r} series {time_series_id!r}")
        if self._values is None:
            self._values = make_values_reader(self._manifest.values_backend)
        try:
            spec_type = rows[0]["spec_type"]
            return self._values.load(self._version, rows, self._spec_by_type[spec_type])
        except (KeyError, OSError, IndexError, ValueError) as exc:
            raise TimeFFormatError(f"failed to read series {time_series_id!r} for record {record_id!r}: {exc}") from exc


def _by_record(rows: Iterable[dict]) -> dict[int, list[dict]]:
    """Group a batch's rows by the record they belong to, keeping their stored order.

    Args:
        rows: The rows to group, already ordered by record and then by position.

    Returns:
        Each record's rows, keyed by its surrogate id.
    """
    return _grouped(rows, "record_id")


def _grouped(rows: Iterable[dict], column: str) -> dict[Any, list[dict]]:
    """Group rows by one column, keeping their stored order within each group.

    Args:
        rows: The rows to group.
        column: The column to group on.

    Returns:
        Each group's rows, keyed by the column's value.
    """
    groups: dict[Any, list[dict]] = {}
    for row in rows:
        groups.setdefault(row[column], []).append(row)
    return groups


def _items(rows: Iterable[dict], role: str, *, column: str = "role") -> tuple[str, ...]:
    """Return the external ids of one role's rows, in position order.

    Args:
        rows: The task's item or attachment rows.
        role: The role to keep.
        column: The column holding the role.

    Returns:
        The external ids.
    """
    return tuple(row["external_id"] for row in rows if row[column] == role)


def _span(row: dict):
    """Rebuild one span from a ``task_spans`` row.

    Args:
        row: The stored row.

    Returns:
        The span.
    """
    return span_from_row(row["frame"], row["start_us"], row["end_us"], row["time_series_ids"])


def _task_payload(
    task_type: TaskType,
    field_rows: Iterable[dict],
    ref_rows: Iterable[dict],
    span_rows: dict[Any, list[dict]],
) -> dict[str, object]:
    """Rebuild one task's type-specific payload from its three payload tables.

    A field with no ``task_fields`` row was ``None`` when it was written, so it is left out and the
    dataclass default applies. That is what keeps an empty tuple distinguishable from an absent one.

    Args:
        task_type: The task's type, which declares what its payload holds.
        field_rows: The task's ``task_fields`` rows.
        ref_rows: The task's ``task_refs`` rows.
        span_rows: The task's ``task_spans`` rows, grouped by field.

    Returns:
        The payload, as keyword arguments for the task's constructor.
    """
    present = {row["field"]: row for row in field_rows}
    refs = _grouped(ref_rows, "field")
    payload: dict[str, object] = {}
    for declared in task_payload(task_type):
        stored = present.get(declared.name)
        if stored is None:
            continue
        if declared.kind is PayloadKind.TEXT:
            payload[declared.name] = stored["text_value"]
        elif declared.kind is PayloadKind.NUMBER:
            payload[declared.name] = stored["double_value"]
        elif declared.kind is PayloadKind.SPAN:
            spans = tuple(_span(row) for row in span_rows.get(declared.name, ()))
            payload[declared.name] = spans if declared.is_list else spans[0]
        else:
            ids = tuple(row["external_id"] for row in refs.get(declared.name, ()))
            payload[declared.name] = ids if declared.is_list else ids[0]
    return payload


@dataclass(frozen=True)
class _SeriesLoader:
    """A picklable lazy loader for one series' values.

    A nested closure cannot pickle, so this loader holds the reader and the series' identity
    instead. A read-back dataset therefore pickles, and a multi-worker torch ``DataLoader`` can use
    it. The loader reads the values only when you call it.
    """

    reader: TimeFReader
    """The reader that reads and decodes the series' values."""
    record_id: str
    """The owning record's id."""
    time_series_id: str
    """The id of the series to read."""

    def __call__(self) -> pa.Array:
        """Read the series' values.

        Returns:
            The series values in the spec's canonical Arrow representation.
        """
        return self.reader._load_values(self.record_id, self.time_series_id)

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Read a temporal subsection through the selected values backend.

        Returns:
            The requested steps in their canonical Arrow representation.

        Raises:
            TimeFFormatError: If this series has no chunk locator.
        """
        rows = self.reader._index_rows(self.record_id, self.time_series_id)
        if not rows:
            raise TimeFFormatError(f"no index entry for record {self.record_id!r} series {self.time_series_id!r}")
        if self.reader._values is None:
            self.reader._values = make_values_reader(self.reader._manifest.values_backend)
        spec = self.reader._spec_by_type[rows[0]["spec_type"]]
        return self.reader._values.load_range(self.reader._version, rows, start, stop, spec)


@dataclass(frozen=True)
class _TimeOffsetsLoader:
    """A picklable lazy loader for an irregular series' per-value time offsets.

    A nested closure cannot pickle, so this loader is a class, not a lambda. It mirrors
    :class:`_SeriesLoader`, because time offsets and values are separate columns and each one reads
    on its own.
    """

    reader: TimeFReader
    """The reader that reads and decodes the series' time offsets."""
    record_id: str
    """The owning record's id."""
    time_series_id: str
    """The id of the series to read."""
    first_us: int
    """The axis' first time offset, checked against the stored stream."""
    last_us: int
    """The axis' last time offset, checked against the stored stream."""
    n_values: int
    """The declared value count, checked against the stored stream's length."""

    def __call__(self) -> pa.Array:
        """Read the series' time offsets, checking them against the axis and value count.

        The writer checks order, count, and endpoints. Rechecking them here stops a corrupt shard
        from handing back a decreasing, wrong-length, or off-endpoint stream.

        Returns:
            One int64 microsecond time offset per value.

        Raises:
            TimeFFormatError: If the stored time offsets disagree with the axis endpoints or the
                value count, or if they decrease at any point.
        """
        time_offsets = self.reader._load_time_offsets(self.record_id, self.time_series_id)
        values = time_offsets.to_numpy(zero_copy_only=False)
        where = f"series {self.time_series_id!r} on record {self.record_id!r}"
        if len(values) != self.n_values:
            raise TimeFFormatError(f"{where} stores {len(values)} time offsets but declares n_values={self.n_values}")
        if len(values) and (int(values[0]) != self.first_us or int(values[-1]) != self.last_us):
            raise TimeFFormatError(
                f"{where} has time offsets [{values[0]}, {values[-1]}] disagreeing with its axis "
                f"endpoints ({self.first_us}, {self.last_us})"
            )
        if np.any(np.diff(values) < 0):
            raise TimeFFormatError(f"{where} has non-decreasing time offsets that decrease on disk")
        return time_offsets
