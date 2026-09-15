"""``TimeFReader`` reads a TimeF version directory and rebuilds it as a :class:`TimeFDataset`.

The file ``manifest.json`` controls the reader. The reader does not run connector code. When you open
a version, the reader reads only the manifest. It resolves the schema, tasks, records, annotations,
and each series' values only on first use. The reader rebuilds types from the control database's own
declaration, and it does not create classes at runtime. As a result, read-back objects can pickle,
and they match the original objects field for field.

The rebuild is batched. Records come back a batch at a time, and each batch costs one query per
table for the whole batch rather than one query per record. Measured on a 618,508-record corpus:
34.49 ms per record one at a time, 0.111 ms per record in batches of a thousand.
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

from timenet.control_plane import schema as ddl
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
    AnnotationDescriptor,
    AnnotationType,
    DatasetMetadata,
    DatasetSchema,
    DataSource,
    Task,
    TaskType,
    TimeInterval,
    TimeSeriesSpec,
    annotation_type_of,
    ureg,
    value_type_of,
)
from timenet.values_backends.reader import BaseValuesReader, make_values_reader


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion


_RECORD_BATCH_ROWS = 512
"""How many records one round of control-plane queries rebuilds.

Each query in the round scans its table once and answers for the whole batch, so the per-record cost
falls with the batch size. A few hundred is where the curve flattens and the decoded rows still sit
comfortably in memory.
"""

_INDEX_ROWS_CACHE_RECORDS = 2 * _RECORD_BATCH_ROWS
"""How many records keep their chunk locators.

The locators are fetched a batch at a time, so the memo holds the batch a read is walking and the
one before it. A memo smaller than a batch would drop rows the same batch is about to ask for, and
the next record would pay another statement for them.
"""

_AXIS_CACHE_SIZE = 1024
"""Maximum number of stored time axes. Series with the same timing can reuse an axis.
After this limit, the reader builds other axes without keeping them."""


class TimeFReader:
    """Reads a committed TimeF version directory. Use as a context manager to close its handles."""

    def __init__(self, version: DatasetVersion) -> None:
        """Open a committed dataset version through a storage handle.

        This constructor reads nothing. The handle already carries the parsed manifest. The handle
        also wraps the filesystem and root that every later read uses. The reader resolves tasks,
        records, annotations, and each series' values only on first use. As a result, the cost to
        open a version is the same for three records or three million.

        A structurally corrupt or missing file fails on its first access, not here. Examples of a
        first access are ``.tasks``, the first record, or the first value read. The failure reaches
        the caller as :class:`~timenet.errors.TimeFFormatError`, which is what :meth:`verify` raises
        for the same file, and not as the ``FileNotFoundError`` the filesystem raised under it. One
        unreadable artifact reported with two types would make a caller catch both to cover one
        condition. Call :meth:`verify` for a check of the version's integrity at construction time.
        Build the handle with
        :meth:`~timenet.registry.BaseRegistry.open_version` or with
        :meth:`~timenet.registry.version.DatasetVersion.open_local`.

        Args:
            version: The opened version handle: a manifest plus a filesystem-rooted view of its files.
        """
        self._version = version
        # These are aliases of the handle's fields, for the read sites below. All three pickle, so a
        # DataLoader worker can rebuild the reader (and its lazy loaders) from them without reopening
        # the registry.
        self._fs = version.filesystem
        self._root = version.root
        self._manifest = version.manifest

        # The schema types the rows, and the rows are in the control database, so that is where the
        # reader reads it from. The manifest carries the same block as a projection a registry can
        # filter on without downloading the database.
        self._schema: DatasetSchema | None = None
        self._spec_by_type: dict[str, TimeSeriesSpec] = {}
        self._annotation_descriptors: dict[str, AnnotationDescriptor] = {}
        # This is per-process scratch space. The reader builds it on demand and drops it on pickle.
        # ``__getstate__`` must stay in step with these fields.
        self._control: ControlPlaneReader | None = None
        self._tasks: tuple[Task, ...] | None = None
        self._values: BaseValuesReader | None = None
        self._axis_cache: dict[tuple, TimeAxis] = {}
        self._index_rows_cache: OrderedDict[int, dict[str, list[dict]]] = OrderedDict()

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

        This method does not run automatically when you open a version. It reads and hashes every
        file, so it reads the whole dataset. The lazy read design of this class avoids that cost on
        open.

        When integrity matters more than speed, call this method explicitly. Examples are after a
        download, before a long training run, or inside a fsck-style command. This method reopens
        each file through the version's filesystem. As a result, a missing file surfaces here,
        because ``__init__`` does not stat-sweep the files.

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
        """The dataset's type declaration, rebuilt from the control database on first use.

        Returns:
            The specs and annotation descriptors the database declares, with the task classes the
            manifest names. A task class is code rather than data, so the version declares which ones
            it holds and the reader resolves them against the built-in registry.
        """
        return self._declaration()

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
        # A streamed-written dataset stores empty record.task_ids: its tasks were never held in memory
        # to populate them. read() materializes every task, so rebuild the reverse map here, or
        # tasks_for() and the torch view would return no tasks. This is idempotent for a materialized
        # dataset, whose task_ids already round-trip through the record rows.
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
            schema=self.schema,
            registered_annotations=self._read_registered_annotations(),
        )

    def iter_records(
        self,
        record_ids: Iterable[str] | None = None,
        *,
        with_annotations: bool = True,
    ) -> Iterator[Record]:
        """Yield records lazily without materializing a :class:`TimeFDataset`.

        Records are rebuilt a batch at a time, so the control plane is queried once per batch per
        table rather than once per record. Records come back in stored order (sorted by
        ``record_id``), not in the order you asked for them. An id that the dataset does not contain
        raises ``TimeFValidationError``.

        Args:
            record_ids: The records to yield, or ``None`` to yield every record.
            with_annotations: Whether to resolve the annotations of each record. ``False`` leaves
                :attr:`Record.annotations` empty and skips the query and the JSON parse that each
                one costs. The stored spans are then not checked against the record window.

        Yields:
            Each reconstructed :class:`Record`.

        Raises:
            TimeFValidationError: If ``record_ids`` names an id that the dataset does not contain,
                the error raises after the iterator is fully consumed, not at the offending id. An
                early-stopping consumer is served what exists and never reaches the check.
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

    def _declaration(self) -> DatasetSchema:
        """Rebuild the version's type declaration from the control database, once per reader.

        Returns:
            The rebuilt schema, cached with the by-type lookups the record and annotation builders use.

        Raises:
            TimeFFormatError: If the database declares a spec or a descriptor that cannot be rebuilt.
        """  # noqa: DOC502 - raised by _as_format_error
        if self._schema is None:
            with self._as_format_error():
                control = self._control_plane()
                schema = DatasetSchema(
                    time_series_specs=tuple(_spec(row) for row in control.specs()),
                    annotations=tuple(_descriptor(row) for row in control.annotation_descriptors()),
                    tasks=self._manifest.schema.tasks,
                )
            self._schema = schema
            self._spec_by_type = {spec.spec_type: spec for spec in schema.time_series_specs}
            self._annotation_descriptors = {d.key: d for d in schema.annotations}
        return self._schema

    def _specs(self) -> dict[str, TimeSeriesSpec]:
        """Return each declared spec, keyed by its spec type.

        Returns:
            The lookup the series builder resolves a row's ``spec_type`` against.
        """
        self._declaration()
        return self._spec_by_type

    def _descriptors(self) -> dict[str, AnnotationDescriptor]:
        """Return each declared annotation descriptor, keyed by its annotation key.

        Returns:
            The lookup the annotation builder resolves a row's ``key`` against.
        """
        self._declaration()
        return self._annotation_descriptors

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
        Anything they raise means that the artifact is corrupt, or that it disagrees with its
        manifest. Both cases are a format failure. This context manager stops a bare ``ValueError``
        from ``TaskType()``, or a ``duckdb.Error`` from a truncated database, from escaping as-is.
        The reader's contract requires every failure to reach the caller as a
        :class:`TimeFFormatError`.

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
        tasks = _by_record(control.record_tasks(record_ids))
        annotations = _by_record(control.record_annotations(record_ids)) if with_annotations else {}
        # The batch travels with the records it built, so the first series that reads its values
        # fetches the locators of the whole batch instead of its own record's.
        batch_keys = tuple(record_ids)
        for row in control.records(record_ids):
            key = row["record_id"]
            yield self._build_record(row, series.get(key, ()), tasks.get(key, ()), annotations.get(key, ()), batch_keys)

    def _load_tasks(self) -> tuple[Task, ...]:
        """Rebuild every task, one query per payload table for the whole version.

        Returns:
            The tasks, with ``from_tasks`` resolved to the rebuilt instances.

        Raises:
            TimeFFormatError: If a task derives from an id no task carries.
        """
        rows = self._control_plane().tasks()
        grouped = {
            name: _grouped(rows[name], "task_id") for name in ("items", "fields", "refs", "spans", "annotations")
        }
        derivations = _grouped(rows["from_tasks"], "task_id")

        by_id: dict[str, Task] = {}
        pending: dict[str, tuple[str, ...]] = {}
        for row in rows["tasks"]:
            key = row["task_id"]
            task = _build_task(row, {name: group.get(key, []) for name, group in grouped.items()})
            by_id[task.id] = task
            pending[task.id] = tuple(item["external_id"] for item in derivations.get(key, ()))
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
        descriptors = self._descriptors()
        if key not in descriptors:
            raise TimeFFormatError(f"annotation row references unknown key {key!r} (not in schema)")
        descriptor = descriptors[key]
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

        :meth:`iter_records` never reaches them, since no record lists them, so :meth:`read` pulls
        them here. The manifest count gates the query, so a dataset with none (the common case) pays
        nothing.

        Returns:
            The registered annotations, in registration order.
        """
        if self._manifest.counts.registered_annotations == 0:
            return ()
        with self._as_format_error():
            return tuple(self._decode_annotation(row) for row in self._control_plane().registered_annotations())

    def _index_rows(
        self, record_key: int, time_series_id: str, batch_keys: tuple[int, ...] | None = None
    ) -> list[dict]:
        """Return one series' chunk locators, in ``chunk_idx`` order.

        One query returns every series of a whole batch of records, because a read walks a batch and
        then asks each of its records for its series' values. A record built by :meth:`_build_batch`
        carries the batch it belongs to, so the first series that wants its locators fetches the
        batch's. The rows are held for the batch a read is walking and the one before it.

        Args:
            record_key: The owning record's surrogate id, which the control plane joins on.
            time_series_id: The series id.
            batch_keys: The surrogate ids of the batch ``record_key`` was built in. ``None`` fetches
                that record alone, which is what a caller outside the batched walk gets.

        Returns:
            The series' chunk locator rows, empty if it has none.
        """
        with self._as_format_error():
            by_series = self._index_rows_cache.get(record_key)
            if by_series is None:
                by_series = self._fetch_locators(record_key, batch_keys or (record_key,))
            return by_series.get(time_series_id, [])

    def _fetch_locators(self, record_key: int, batch_keys: tuple[int, ...]) -> dict[str, list[dict]]:
        """Read one batch's chunk locators with a single statement and memo them per record.

        Args:
            record_key: The record whose locators the caller asked for.
            batch_keys: The batch to fetch, which contains ``record_key``.

        Returns:
            ``record_key``'s locators, keyed by series id, empty if the record has no series.
        """
        wanted = [key for key in batch_keys if key not in self._index_rows_cache]
        fetched: dict[int, dict[str, list[dict]]] = {key: {} for key in wanted}
        for row in self._control_plane().record_chunks(wanted):
            fetched[row["record_id"]].setdefault(row["time_series_id"], []).append(row)
        self._index_rows_cache.update(fetched)
        while len(self._index_rows_cache) > _INDEX_ROWS_CACHE_RECORDS:
            self._index_rows_cache.popitem(last=False)
        return fetched.get(record_key, {})

    # ---- record construction -------------------------------------------------------------------

    def _build_record(
        self,
        row: dict,
        series_rows: Iterable[dict],
        task_rows: Iterable[dict],
        annotation_rows: Iterable[dict],
        batch_keys: tuple[int, ...],
    ) -> Record:
        """Rebuild one record from its stored row and the rows that hang off it.

        Args:
            row: The record's own stored row.
            series_rows: Its series rows, in the record's own series order.
            task_rows: Its task links, in task order.
            annotation_rows: Its annotation rows, in the record's own annotation order.
            batch_keys: The surrogate ids of the batch this record was read in, which its series'
                loaders fetch their chunk locators with.

        Returns:
            The rebuilt record.

        Raises:
            TimeFFormatError: If a stored span or ``time_span`` no longer fits the record, which is a
                corrupt artifact rather than a caller mistake.
        """
        record_id = row["external_id"]
        series = tuple(self._build_series(row["record_id"], record_id, struct, batch_keys) for struct in series_rows)
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
                task_ids=tuple(task["external_id"] for task in task_rows),
                annotations=annotations,
                start_time=row["start_time_us"],
                time_span=time_span,
            )
            for annotation in annotations:
                if annotation.span is not None:
                    # A stored span is already written. Refusing to load it hides a dataset that a
                    # writer accepted on purpose, and the span is still there for a caller to judge.
                    check_span_within_window(
                        f"annotation {annotation.key!r}", annotation.span, series, record_id, time_span
                    )
            return record
        except TimeFValidationError as exc:
            raise TimeFFormatError(str(exc)) from exc

    def _build_series(self, record_key: int, record_id: str, struct: dict, batch_keys: tuple[int, ...]) -> TimeSeries:
        """Rebuild one series from its stored row, with lazy loaders for its values.

        Args:
            record_key: The owning record's surrogate id, which the loaders join on.
            record_id: The owning record's id, which the loaders report errors against.
            struct: The stored series row, with its spec and axis columns joined in.
            batch_keys: The surrogate ids of the batch the owning record was read in, which the
                loaders fetch their chunk locators with.

        Returns:
            The rebuilt series.

        Raises:
            TimeFFormatError: If the row names a spec type the schema does not declare, or the series
                cannot be built from what the row says.
        """
        specs = self._specs()
        spec_type = struct["spec_type"]
        if spec_type not in specs:
            raise TimeFFormatError(f"record {record_id!r} references unknown spec_type {spec_type!r}")
        time_series_id = struct["external_id"]
        axis = self._axis(struct, record_id)
        n_values = struct["n_values"]
        time_offsets_loader = None
        if isinstance(axis, IrregularAxis):
            time_offsets_loader = _TimeOffsetsLoader(
                self, record_key, record_id, time_series_id, batch_keys, axis.first_us, axis.last_us, n_values
            )
        try:
            return TimeSeries(
                spec=specs[spec_type],
                signal=struct["signal"],
                time_axis=axis,
                loader=_SeriesLoader(self, record_key, record_id, time_series_id, batch_keys),
                time_offsets_loader=time_offsets_loader,
                source_id=struct["source_id"],
                time_series_id=time_series_id,
                n_values=n_values,
            )
        except TimeFValidationError as exc:
            # A series rebuilt from a corrupt struct (bad n_values, a time offsets/axis mismatch) is a
            # format failure, not a caller mistake, even though TimeSeries raises the same type for both.
            raise TimeFFormatError(f"record {record_id!r} has an unbuildable series {time_series_id!r}: {exc}") from exc

    def _axis(self, struct: dict, record_id: str) -> TimeAxis:
        """Return a time axis and reuse an existing axis when its stored columns match.

        Axis objects cannot change, so series can share them. Reuse avoids repeated ``Fraction``
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

        This method reads the tag before any shape-specific column. As a result, a corrupt row
        raises an error here, not later from an axis inferred from null columns.

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
            # An ordinal series has no cadence and no per-value time offsets, so every shape column must
            # be null. A populated column means that the tag and the columns disagree. This disagreement
            # is the corruption that this method catches.
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

    def _load_time_offsets(
        self, record_key: int, record_id: str, time_series_id: str, batch_keys: tuple[int, ...]
    ) -> pa.Array:
        """Read an irregular series' per-value time offsets through the values backend.

        Args:
            record_key: The owning record's surrogate id.
            record_id: The owning record's id, for the error message.
            time_series_id: The series id to read.
            batch_keys: The batch the owning record was read in, which the locators are fetched for.

        Returns:
            One int64 microsecond time offset per value.

        Raises:
            TimeFFormatError: If the series has no chunk locator or its time offsets cannot be read.
        """
        rows = self._index_rows(record_key, time_series_id, batch_keys)
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

    def _load_values(
        self, record_key: int, record_id: str, time_series_id: str, batch_keys: tuple[int, ...]
    ) -> pa.Array:
        """Read and concatenate a series' chunk values through the values backend.

        Args:
            record_key: The owning record's surrogate id.
            record_id: The owning record's id, for the error message.
            time_series_id: The series id to read.
            batch_keys: The batch the owning record was read in, which the locators are fetched for.

        Returns:
            The series values in the spec's canonical Arrow representation.

        Raises:
            TimeFFormatError: If the series has no chunk locator or a chunk cannot be read.
        """
        rows = self._index_rows(record_key, time_series_id, batch_keys)
        if not rows:
            raise TimeFFormatError(f"no index entry for record {record_id!r} series {time_series_id!r}")
        if self._values is None:
            self._values = make_values_reader(self._manifest.values_backend)
        try:
            spec_type = rows[0]["spec_type"]
            return self._values.load(self._version, rows, self._specs()[spec_type])
        except (KeyError, OSError, IndexError, ValueError) as exc:
            raise TimeFFormatError(f"failed to read series {time_series_id!r} for record {record_id!r}: {exc}") from exc


def _spec(row: dict) -> TimeSeriesSpec:
    """Rebuild one time-series spec from its stored row.

    Args:
        row: The stored ``specs`` row.

    Returns:
        The spec, with its unit resolved against the shared registry.
    """
    source = row["data_source_type"]
    return TimeSeriesSpec(
        spec_type=row["spec_type"],
        name=row["name"],
        unit_value=ureg.Unit(row["unit_value"]),
        data_source=(
            None
            if source is None
            else DataSource(data_source_type=source, name=row["data_source_name"], provider=row["data_source_provider"])
        ),
        dtype=row["dtype"],
        categories=tuple(row["categories"]),
        value_shape=tuple(row["value_shape"]),
        dimension_names=tuple(row["dimension_names"]),
        nullable=row["nullable"],
    )


def _descriptor(row: dict) -> AnnotationDescriptor:
    """Rebuild one annotation descriptor from its stored row.

    Args:
        row: The stored ``annotation_descriptors`` row.

    Returns:
        The descriptor.
    """
    return AnnotationDescriptor(
        key=row["key"],
        annotation_type=AnnotationType(row["annotation_type"]),
        value_type=row["value_type"],
        unit=row["unit"],
        description=row["description"],
    )


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


def _build_task(row: dict, own: dict[str, list[dict]]) -> Task:
    """Rebuild one task from its own row and the rows that hang off it.

    Args:
        row: The task's own stored row.
        own: Its ``items``, ``fields``, ``refs``, ``spans`` and ``annotations`` rows, each in
            position order.

    Returns:
        The rebuilt task, without its ``from_tasks``, which need every task to be built first.
    """
    task_type = TaskType(row["task_type"])
    spans = _grouped(own["spans"], "field")
    scope_rows = spans.get("scope", ())
    return TASKS[task_type](
        id=row["external_id"],
        record_ids=_items(own["items"], "input", "record"),
        prompt=row["prompt"],
        scope=_span(scope_rows[0]) if scope_rows else None,
        input_annotation_ids=_annotation_ids(own["annotations"], "input"),
        target_annotation_ids=_annotation_ids(own["annotations"], "target"),
        rationale=row["rationale"],
        **_task_payload(task_type, own["fields"], own["refs"], spans, own["items"]),  # ty: ignore[invalid-argument-type]
    )


def _items(rows: Iterable[dict], role: str, item_type: str, *, column: str = "external_id") -> tuple[str, ...]:
    """Return one kind of task item, in position order.

    Args:
        rows: The task's ``task_items`` rows.
        role: ``"input"`` or ``"target"``.
        item_type: ``"record"`` or ``"text"``.
        column: The column holding the item's value.

    Returns:
        The values of the items that match.
    """
    return tuple(row[column] for row in rows if row["role"] == role and row["item_type"] == item_type)


def _annotation_ids(rows: Iterable[dict], role: str) -> tuple[str, ...]:
    """Return the ids of the annotations a task holds in one role, in position order.

    Args:
        rows: The task's attachment rows.
        role: ``"input"`` or ``"target"``.

    Returns:
        The annotation ids.
    """
    return tuple(row["external_id"] for row in rows if row["role"] == role)


def _span(row: dict):
    """Rebuild one span from a ``task_spans`` row.

    Args:
        row: The stored row.

    Returns:
        The span.
    """
    return span_from_row(row["frame"], row["start_at"], row["end_at"], row["time_series_ids"])


def _task_payload(
    task_type: TaskType,
    field_rows: Iterable[dict],
    ref_rows: Iterable[dict],
    span_rows: dict[Any, list[dict]],
    item_rows: Iterable[dict],
) -> dict[str, object]:
    """Rebuild one task's type-specific payload from its payload tables and its items.

    A field with no ``task_fields`` row was ``None`` when it was written, so it is left out and the
    dataclass default applies. That is what keeps an empty tuple distinguishable from an absent one.
    A free-text answer is an item of the task rather than a named field, so it comes from
    ``task_items`` instead.

    Args:
        task_type: The task's type, which declares what its payload holds.
        field_rows: The task's ``task_fields`` rows.
        ref_rows: The task's ``task_refs`` rows.
        span_rows: The task's ``task_spans`` rows, grouped by field.
        item_rows: The task's ``task_items`` rows.

    Returns:
        The payload, as keyword arguments for the task's constructor.
    """
    present = {row["field"]: row for row in field_rows}
    refs = _grouped(ref_rows, "field")
    answer = ddl.text_answer(task_type)
    payload: dict[str, object] = {}
    answers = _items(item_rows, "target", "text", column="text_value")
    if answer is not None and answers:
        payload[answer.name] = answers[0]
    for declared in ddl.task_payload(task_type):
        stored = present.get(declared.name)
        if stored is None or declared is answer:
            continue
        if declared.kind is ddl.PayloadKind.TEXT:
            payload[declared.name] = stored["text_value"]
        elif declared.kind is ddl.PayloadKind.NUMBER:
            payload[declared.name] = stored["double_value"]
        elif declared.kind is ddl.PayloadKind.SPAN:
            spans = tuple(_span(row) for row in span_rows.get(declared.name, ()))
            payload[declared.name] = spans if declared.is_list else spans[0]
        else:
            ids = tuple(row["external_id"] for row in refs.get(declared.name, ()))
            payload[declared.name] = ids if declared.is_list else ids[0]
    return payload


@dataclass(frozen=True)
class _SeriesLoader:
    """A picklable lazy loader for one series' values (replaces a per-series closure).

    A nested closure cannot pickle. This class holds the reader and the series' identity instead of
    a closure. As a result, a read-back dataset can pickle, and a multi-worker torch ``DataLoader``
    can use it. The class reads the series' values only when you call it.
    """

    reader: TimeFReader
    """The reader that reads and decodes the series' values."""
    record_key: int
    """The owning record's surrogate id, which the control plane joins on."""
    record_id: str
    """The owning record's id, which errors are reported against."""
    time_series_id: str
    """The id of the series to read."""
    batch_keys: tuple[int, ...]
    """The batch the owning record was read in, so one statement locates the whole batch's chunks."""

    def __call__(self) -> pa.Array:
        """Read the series' values.

        Returns:
            The series values in the spec's canonical Arrow representation.
        """
        return self.reader._load_values(self.record_key, self.record_id, self.time_series_id, self.batch_keys)

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Read a temporal subsection through the selected values backend.

        Returns:
            The requested steps in their canonical Arrow representation.

        Raises:
            TimeFFormatError: If this series has no chunk locator.
        """
        rows = self.reader._index_rows(self.record_key, self.time_series_id, self.batch_keys)
        if not rows:
            raise TimeFFormatError(f"no index entry for record {self.record_id!r} series {self.time_series_id!r}")
        if self.reader._values is None:
            self.reader._values = make_values_reader(self.reader._manifest.values_backend)
        spec = self.reader._specs()[rows[0]["spec_type"]]
        return self.reader._values.load_range(self.reader._version, rows, start, stop, spec)


@dataclass(frozen=True)
class _TimeOffsetsLoader:
    """A picklable lazy loader for an irregular series' per-value time offsets.

    A nested closure cannot pickle, so this loader is a class, not a lambda. A multi-worker torch
    ``DataLoader`` pickles the series to send it to its workers, and a closure cannot pickle. This
    class mirrors :class:`_SeriesLoader`, because time offsets and values are separate columns, and
    each column reads on its own.
    """

    reader: TimeFReader
    """The reader that reads and decodes the series' time offsets."""
    record_key: int
    """The owning record's surrogate id, which the control plane joins on."""
    record_id: str
    """The owning record's id, which errors are reported against."""
    time_series_id: str
    """The id of the series to read."""
    batch_keys: tuple[int, ...]
    """The batch the owning record was read in, so one statement locates the whole batch's chunks."""
    first_us: int
    """The axis' first time offset, checked against the stored stream."""
    last_us: int
    """The axis' last time offset, checked against the stored stream."""
    n_values: int
    """The declared value count, checked against the stored stream's length."""

    def __call__(self) -> pa.Array:
        """Read the series' time offsets, checking them against the axis and value count.

        The writer checks ordering, count, and endpoints, but nothing rechecks them on read. As a
        result, a corrupt shard can otherwise hand back a decreasing, wrong-length, or off-endpoint
        stream.

        Returns:
            One int64 microsecond time offset per value.

        Raises:
            TimeFFormatError: If the stored time offsets disagree with the axis endpoints or the
                value count, or if they decrease at any point.
        """
        time_offsets = self.reader._load_time_offsets(
            self.record_key, self.record_id, self.time_series_id, self.batch_keys
        )
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
