"""``TimeFReader`` reads a TimeF version directory and rebuilds it as a :class:`TimeFDataset`.

The file ``manifest.json`` controls the reader. The reader does not run connector code. When you open a
version, the reader reads only the manifest. The reader resolves tasks, annotations, the time-series
index, and each series' values only on first use. The reader rebuilds types from the manifest's flat
descriptors, and it does not create classes at runtime. As a result, read-back objects can pickle, and
they match the original objects field for field.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import types as _types
from typing import TYPE_CHECKING, cast

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from timenet.dataset import Sample, TimeFDataset, TimeSeries
from timenet.dataset.axis import AxisType, IrregularAxis, OrdinalAxis, RegularAxis, TimeAxis
from timenet.dataset.sample import check_span_within_window
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.checksums import stream_checksum
from timenet.format.constants import ANNOTATIONS_SORT_KEY, INDEX_SORT_KEY
from timenet.format.schemas import TASK_COMMON_NAMES, IdCodec, task_schema
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
    import pyarrow.fs as pafs

    from timenet.registry.version import DatasetVersion


#: The number of rows in each batch when the reader streams ``samples.parquet``.
_SAMPLE_BATCH_ROWS = 4096

#: The maximum number of decoded annotations that the cache keeps for reuse. This lets an
#: annotation shared across samples decode only once.
_ANNOTATION_CACHE_SIZE = 4096

#: The byte budget for decoded control-table row groups. The budget counts total bytes, not the
#: number of groups. This design stops a shuffled read from filling and clearing a small cache too
#: often.
_CONTROL_TABLE_CACHE_MAX_BYTES = 64 * 2**20

#: An id in its stored form: the id string, or the 16 raw bytes of a ``uuid16`` column.
_StoredId = str | bytes


@dataclass(frozen=True)
class _RowGroupStats:
    """One row group of a sorted control table, with the stored-key range that its statistics guarantee."""

    part: str
    """The table part's manifest-relative path."""
    ordinal: int
    """The row group's position within that part."""
    min_key: _StoredId
    """The lowest stored key in the group."""
    max_key: _StoredId
    """The highest stored key in the group."""

    def may_hold(self, key: _StoredId) -> bool:
        """Return whether this row group can hold a stored key.

        Args:
            key: The key in its stored form (a string, or 16 bytes for a ``uuid16`` column).

        Returns:
            ``False`` only when the statistics prove that the key is outside the group.
        """
        return self.min_key <= key <= self.max_key  # ty: ignore[unsupported-operator]


class _PrunedControlTable:
    """A control-plane table sorted by one key column, read one row group at a time, with pruning.

    The reader never holds the whole table in memory. It builds a row-group directory from the
    Parquet footers. It bisects the directory on the sorted key column's statistics, and this
    prunes the groups that a lookup cannot match. Then it decodes the matching groups into a
    byte-bounded LRU cache. Inside a decoded group, it bisects a materialized sorted list of the
    lookup-key column to select the matching rows. The time-series index (keyed by ``sample_id``,
    many rows per key) and the annotations table (keyed by ``id``, one row per key) both use this
    method. The key column is the first entry of ``lookup_columns``.
    """

    def __init__(
        self,
        filesystem: pafs.FileSystem,
        root: str,
        parts: tuple[str, ...],
        lookup_columns: tuple[str, ...],
        *,
        columns: list[str] | None = None,
    ) -> None:
        """Configure a pruned view over a sorted control table.

        Args:
            filesystem: The filesystem the version's files live on.
            root: The version's root prefix on ``filesystem``.
            parts: The table's manifest-relative part paths.
            lookup_columns: The columns whose stored values a lookup matches, given as a tuple. The
                first column is the sorted key column. The footer statistics use it to prune row groups.
            columns: The columns to decode per row group, or ``None`` for all of them.

        """
        self._fs = filesystem
        self._root = root
        self._parts = parts
        self._lookup_columns = lookup_columns
        self._key_column = lookup_columns[0]
        self._columns = columns
        self._files: dict[str, pq.ParquetFile] = {}
        self._directory: list[_RowGroupStats] | None = None
        self._cache: OrderedDict[tuple[str, int], tuple[pa.Table, list[tuple]]] = OrderedDict()
        self._cache_bytes = 0

    def close(self) -> None:
        """Close the open Parquet handles and drop the directory and decoded-row-group cache."""
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        self._directory = None
        self._cache.clear()
        self._cache_bytes = 0

    def groups(self) -> list[_RowGroupStats]:
        """Return the row-group directory: where each group lives and the stored keys it can hold.

        This method builds the directory from the Parquet footers on the first lookup. As a result,
        the cost is O(row groups), never O(rows). The writer emits key-column statistics on every row
        group, so a lookup always bisects on the maxima.

        Returns:
            One entry per row group, in file order.
        """
        if self._directory is None:
            directory: list[_RowGroupStats] = []
            for rel in self._parts:
                metadata = self._file(rel).metadata
                column = metadata.schema.names.index(self._key_column)
                for ordinal in range(metadata.num_row_groups):
                    stats = metadata.row_group(ordinal).column(column).statistics
                    directory.append(_RowGroupStats(part=rel, ordinal=ordinal, min_key=stats.min, max_key=stats.max))
            self._directory = directory
        return self._directory

    def rows_for(self, lookup: tuple[_StoredId, ...]) -> list[dict]:
        """Return the rows that match a stored lookup key, and prune the row groups that cannot hold it.

        The prune key is ``lookup[0]`` (the sorted key column). The search bisects to the first group
        that can hold the key, and stops at the first group that cannot. The rows of one key are
        contiguous within a group. The reader visits the groups in file order, so the writer's sort
        order carries through to the result.

        Args:
            lookup: The stored values to match, one per ``lookup_columns`` entry, in that order.

        Returns:
            The matching rows as dicts, in stored order, or empty if the key is absent.
        """
        key = lookup[0]
        groups = self.groups()
        first = bisect.bisect_left(groups, key, key=lambda group: group.max_key)
        rows: list[dict] = []
        for position in range(first, len(groups)):
            group = groups[position]
            if not group.may_hold(key):
                break  # group maxima are sorted, so the first that cannot hold the key ends the search
            table, keys = self._group_rows(group)
            # The group is sorted by its lookup columns, so the matches form the half-open range
            # [bisect_left, bisect_right). One key's rows are contiguous. A unique key is a run of one row.
            lo = bisect.bisect_left(keys, lookup)
            hi = bisect.bisect_right(keys, lookup)
            if hi > lo:
                rows.extend(table.slice(lo, hi - lo).to_pylist())
        return rows

    def _file(self, rel: str) -> pq.ParquetFile:
        """Return the open Parquet handle for one part, opening it on first use.

        Args:
            rel: The part's manifest-relative path.

        Returns:
            The cached handle.
        """
        handle = self._files.get(rel)
        if handle is None:
            handle = pq.ParquetFile(f"{self._root}/{rel}", filesystem=self._fs)
            self._files[rel] = handle
        return handle

    def _group_rows(self, group: _RowGroupStats) -> tuple[pa.Table, list[tuple]]:
        """Return one row group as Arrow, plus its lookup-key column as a sorted list to bisect.

        This method zips the group's lookup columns into one tuple per row, in stored (sorted)
        order, so :meth:`rows_for` can bisect it with the stdlib.

        Args:
            group: The row group to decode.

        Returns:
            The row group's table and its per-row lookup-key tuples, in sorted order.
        """
        key = (group.part, group.ordinal)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        table = self._file(group.part).read_row_group(group.ordinal, columns=self._columns)
        columns = [table.column(name).to_pylist() for name in self._lookup_columns]
        keys = list(zip(*columns, strict=True))
        entry = (table, keys)
        if table.nbytes > _CONTROL_TABLE_CACHE_MAX_BYTES:
            return entry  # this skips the cache for a group over budget, so it does not evict the whole cache
        self._cache[key] = entry
        self._cache_bytes += table.nbytes
        while self._cache_bytes > _CONTROL_TABLE_CACHE_MAX_BYTES:
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= evicted[0].nbytes
        return entry


class TimeFReader:
    """Reads a committed TimeF version directory. Use as a context manager to close shard handles."""

    def __init__(self, version: DatasetVersion) -> None:
        """Open a committed dataset version through a storage handle.

        This constructor reads nothing. The handle already carries the parsed manifest. The handle
        also wraps the filesystem and root that every later read uses. The reader resolves tasks,
        annotations, the time-series index, and each series' values only on first use. As a result,
        the cost to open a version is the same for three samples or three million.

        This constructor no longer decodes the control plane up front, and it no longer stat-sweeps
        every file on open. As a result, a structurally corrupt or missing file now fails on its
        first access, not here. Examples of a first access are ``.tasks``, the first annotation, or
        the first value read. Call :meth:`verify` for a check of the version's integrity at
        construction time. Build the handle with :meth:`~timenet.registry.BaseRegistry.open_version`
        or with :meth:`~timenet.registry.version.DatasetVersion.open_local`.

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

        self._codec = IdCodec.from_encoding(self._manifest.id_encoding)
        self._spec_by_type = {spec.spec_type: spec for spec in self._manifest.schema.time_series_specs}
        self._annotation_descriptors = {d.key: d for d in self._manifest.schema.annotations}
        # This is per-process scratch space. The reader builds it on demand and drops it on pickle.
        # ``__getstate__`` must stay in step with these fields.
        self._tasks: tuple[Task, ...] | None = None
        self._values: BaseValuesReader | None = None
        self._samples_data: pads.Dataset | None = None
        self._index: _PrunedControlTable | None = None
        self._annotations: _PrunedControlTable | None = None
        self._annotation_cache: OrderedDict[str, Annotation] = OrderedDict()

    # ---- pickling ------------------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Drop every on-demand cache so the reader (and its loaders) pickle small and safely.

        Returns:
            The reader's state with every on-demand cache emptied.
        """
        state = self.__dict__.copy()
        state["_values"] = None
        state["_tasks"] = None
        state["_samples_data"] = None
        state["_index"] = None
        state["_annotations"] = None
        state["_annotation_cache"] = OrderedDict()
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
        """Close all cached shard handles."""
        self.close()

    def close(self) -> None:
        """Close the values backend and the control-table handles, releasing their decoded caches."""
        if self._values is not None:
            self._values.close()
            self._values = None
        if self._index is not None:
            self._index.close()
            self._index = None
        if self._annotations is not None:
            self._annotations.close()
            self._annotations = None
        self._annotation_cache.clear()

    # ---- public API ----------------------------------------------------------------------------

    def verify(self) -> None:
        """Check every file the manifest lists against its recorded ``sha256:`` checksum.

        This method does not run automatically when you open a version. It reads and hashes every
        shard, so it reads the whole dataset. The lazy read design of this class avoids that cost on
        open.

        When integrity matters more than speed, call this method explicitly. Examples are after a
        download, before a long training run, or inside a fsck-style command. This method reopens
        each file through the version's filesystem. As a result, a missing file now surfaces here,
        because ``__init__`` no longer stat-sweeps the files.

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
        samples = list(self.iter_samples())
        tasks = self.tasks
        # A streamed-written dataset stores empty sample.task_ids: its tasks were never held in memory
        # to populate them. read() materializes every task, so rebuild the reverse map here, or
        # tasks_for() and the torch view would return no tasks. This is idempotent for a materialized
        # dataset, whose task_ids already round-trip through the sample rows.
        by_id = {sample.sample_id: sample for sample in samples}
        for task in tasks:
            for sample_id in task.sample_ids:
                sample = by_id.get(sample_id)
                if sample is not None and task.id not in sample.task_ids:
                    sample.task_ids = (*sample.task_ids, task.id)
        return TimeFDataset.from_parts(
            metadata=self._manifest.metadata,
            samples=samples,
            tasks=tasks,
            schema=self._manifest.schema,
            registered_annotations=self._read_registered_annotations(),
        )

    def iter_samples(self, sample_ids: Iterable[str] | None = None) -> Iterator[Sample]:
        """Yield samples lazily without materializing a :class:`TimeFDataset`.

        When you pass ``sample_ids``, this method filters the read on the stored id column. This
        prunes the row groups that cannot hold a requested id, and it does not scan the whole file.
        Samples come back in stored order (sorted by ``sample_id``), not in the order you asked for
        them. An id that the dataset does not contain raises ``TimeFValidationError``.

        Args:
            sample_ids: The samples to yield, or ``None`` to yield every sample.

        Yields:
            Each reconstructed :class:`Sample`.
        """
        for row in self._iter_sample_rows(sample_ids):
            yield self._build_sample(row)

    # ---- loading -------------------------------------------------------------------------------

    @contextmanager
    def _as_format_error(self) -> Iterator[None]:
        """Re-raise a decode failure against the manifest's schema as a :class:`TimeFFormatError`.

        The loaders parse on-disk control tables against the manifest's schema. Anything they raise
        means that the artifact is corrupt, or that it disagrees with its manifest. Both cases are a
        format failure. This context manager stops a bare ``ValueError`` from ``TaskType()``, or an
        ``OSError`` from a corrupt data page, from escaping as-is. The reader's contract requires
        every failure to reach the caller as a :class:`TimeFFormatError`.

        Yields:
            Nothing. This context manager only rewrites the exception type.

        Raises:
            TimeFFormatError: If the wrapped code fails to decode what the manifest describes.
        """
        try:
            yield
        except TimeFFormatError:
            raise
        except (ValueError, KeyError, TypeError, AttributeError, OSError, pa.ArrowException) as exc:
            raise TimeFFormatError(f"corrupt or inconsistent TimeF artifact at {self._root}: {exc}") from exc

    def _iter_sample_rows(self, sample_ids: Iterable[str] | None) -> Iterator[dict]:
        """Stream ``samples.parquet`` in batches, optionally restricted to a set of ids.

        Args:
            sample_ids: The samples to read, or ``None`` for all of them.

        Yields:
            Each sample row as a dict.

        Raises:
            TimeFValidationError: If ``sample_ids`` names an id that the dataset does not contain,
                the error raises after the iterator is fully consumed, not at the offending id. A
                lazy reader cannot know that an id is absent until it has read every part.
        """
        if self._samples_data is None:
            parts = [self._version.path(part.path) for part in self._manifest.files.samples]
            self._samples_data = pads.dataset(parts, filesystem=self._fs, format="parquet")
        data = self._samples_data
        if sample_ids is None:
            for batch in data.to_batches(batch_size=_SAMPLE_BATCH_ROWS, use_threads=False):
                yield from batch.to_pylist()
            return
        stored = {self._codec.encode("sample_id", sid): sid for sid in dict.fromkeys(sample_ids)}
        stored_type = data.schema.field("sample_id").type
        expression = pads.field("sample_id").isin(pa.array(list(stored), type=stored_type))
        for batch in data.to_batches(filter=expression, batch_size=_SAMPLE_BATCH_ROWS, use_threads=False):
            for row in batch.to_pylist():
                stored.pop(row["sample_id"], None)
                yield row
        if stored:
            raise TimeFValidationError(
                f"no such sample(s) in {self._manifest.dataset_id}: {', '.join(stored.values())}"
            )

    def _load_tasks(self) -> tuple[Task, ...]:
        by_id: dict[str, Task] = {}
        pending: dict[str, tuple[str, ...]] = {}
        for part in self._manifest.files.tasks:
            rel = part.path
            task_type = TaskType(Path(rel).parent.name.split("=", 1)[1])
            cls = TASKS[task_type]
            payload_cols = [name for name in task_schema(task_type).names if name not in TASK_COMMON_NAMES]
            for row in pq.read_table(self._version.path(rel), filesystem=self._fs).to_pylist():
                payload = {
                    name: self._codec.decode_payload(cls.refs, name, _as_tuple_if_list(row[name]))
                    for name in payload_cols
                }
                task = cls(
                    id=self._codec.decode("task_id", row["id"]),
                    sample_ids=tuple(self._codec.decode_list("sample_id", row["sample_ids"])),
                    prompt=row["prompt"],
                    scope=self._codec.decode_span(row["scope"]),
                    input_annotation_ids=tuple(self._codec.decode_list("annotation_id", row["input_annotation_ids"])),
                    target_annotation_ids=tuple(self._codec.decode_list("annotation_id", row["target_annotation_ids"])),
                    rationale=row["rationale"],
                    **payload,  # ty: ignore[invalid-argument-type]
                )
                by_id[task.id] = task
                pending[task.id] = tuple(self._codec.decode_list("task_id", row["from_task_ids"]))
        for task_id, from_ids in pending.items():
            resolved = []
            for from_id in from_ids:
                if from_id not in by_id:
                    raise TimeFFormatError(f"task {task_id!r} references unknown from_task_id {from_id!r}")
                resolved.append(by_id[from_id])
            by_id[task_id].from_tasks = tuple(resolved)
        return tuple(by_id.values())

    def _annotations_table(self) -> _PrunedControlTable:
        """Return the pruned view over the annotations table, built on first annotation access.

        A decode reads only four columns, and this method pulls only those columns. It skips
        ``sample_ids``, the widest column.

        Returns:
            The cached pruned annotations view.
        """
        if self._annotations is None:
            key_column = ANNOTATIONS_SORT_KEY
            self._annotations = _PrunedControlTable(
                self._fs,
                self._root,
                tuple(part.path for part in self._manifest.files.annotations),
                lookup_columns=(key_column,),
                columns=["id", "key", "value", "span"],
            )
        return self._annotations

    def _decode_annotation(self, row: dict) -> Annotation:
        """Rebuild one annotation from its stored row, checked against its manifest descriptor.

        Args:
            row: The stored annotation row (``id``, ``key``, ``value``, ``span``).

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
            "id": self._codec.decode("annotation_id", row["id"]),
            "key": key,
            "value": None if row["value"] is None else json.loads(row["value"]),
            "unit": descriptor.unit,
            "description": descriptor.description,
        }
        span = self._codec.decode_span(row["span"])
        if span is not None:
            fields["span"] = span
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
        """Rebuild annotations that no sample carries: they exist only for tasks to reference.

        These rows store an empty ``sample_ids``. :meth:`iter_samples` never reaches them, since no
        sample lists them, so :meth:`read` pulls them here to restore the dataset's registered
        annotations. The manifest count gates the scan, so a dataset with none (the common case) pays
        nothing.

        Returns:
            The registered annotations, in stored order.
        """
        if self._manifest.counts.registered_annotations == 0:
            return ()
        registered: list[Annotation] = []
        with self._as_format_error():
            for part in self._manifest.files.annotations:
                table = pq.read_table(
                    self._version.path(part.path),
                    filesystem=self._fs,
                    columns=["id", "key", "value", "span", "sample_ids"],
                )
                for row in table.to_pylist():
                    if not row["sample_ids"]:
                        registered.append(self._decode_annotation(row))
        return tuple(registered)

    def _index_table(self) -> _PrunedControlTable:
        """Return the pruned view over the time-series index, built on first lookup.

        Returns:
            The cached pruned index view.
        """
        if self._index is None:
            key_column = INDEX_SORT_KEY
            self._index = _PrunedControlTable(
                self._fs,
                self._root,
                tuple(part.path for part in self._manifest.files.time_series_index),
                lookup_columns=(key_column, "time_series_id"),
            )
        return self._index

    def _index_rows(self, sample_id: str, time_series_id: str) -> list[dict]:
        """Return one series' index rows, in ``chunk_idx`` order.

        Args:
            sample_id: The owning sample's id.
            time_series_id: The series id.

        Returns:
            The series' index rows, empty if it has none.
        """
        with self._as_format_error():
            probe: tuple[_StoredId, ...] = (
                cast(_StoredId, self._codec.encode(INDEX_SORT_KEY, sample_id)),
                cast(_StoredId, self._codec.encode("time_series_id", time_series_id)),
            )
            return self._index_table().rows_for(probe)

    # ---- sample construction -------------------------------------------------------------------

    def _build_sample(self, row: dict) -> Sample:
        sample_id = self._codec.decode("sample_id", row["sample_id"])
        series = tuple(self._build_series(sample_id, struct) for struct in row["time_series"])
        annotations = tuple(
            self._resolve_annotation(sample_id, aid)
            for aid in self._codec.decode_list("annotation_id", row["annotation_ids"])
        )
        # A stored span or time_span that no longer fits the sample is a corrupt artifact, not a caller
        # mistake. As a result, the sample's own invariants surface as a format error, not a ``ValueError``.
        # This code decodes and builds the sample inside the try block below. This lets
        # ``Sample.__post_init__`` reject a malformed time_span (bad bounds, or point-shaped) before the
        # code uses it to recheck the stored annotation spans.
        try:
            time_span = cast("TimeInterval | None", self._codec.decode_span(row.get("time_span")))
            sample = Sample(
                sample_id=sample_id,
                time_series=series,
                subject_ids=tuple(self._codec.decode_list("subject_id", row["subject_ids"])),
                task_ids=tuple(self._codec.decode_list("task_id", row["task_ids"])),
                annotations=annotations,
                start_time=row.get("start_time_us"),
                time_span=time_span,
            )
            for annotation in annotations:
                if annotation.span is not None:
                    check_span_within_window(
                        f"annotation {annotation.key!r}", annotation.span, series, sample_id, time_span
                    )
            return sample
        except TimeFValidationError as exc:
            raise TimeFFormatError(str(exc)) from exc

    def _build_series(self, sample_id: str, struct: dict) -> TimeSeries:
        spec_type = struct["spec_type"]
        if spec_type not in self._spec_by_type:
            raise TimeFFormatError(f"sample {sample_id!r} references unknown spec_type {spec_type!r}")
        time_series_id = self._codec.decode("time_series_id", struct["time_series_id"])
        axis = self._axis(struct, sample_id)
        n_values = struct["n_values"]
        time_offsets_loader = None
        if isinstance(axis, IrregularAxis):
            time_offsets_loader = _TimeOffsetsLoader(
                self, sample_id, time_series_id, axis.first_us, axis.last_us, n_values
            )
        try:
            return TimeSeries(
                spec=self._spec_by_type[spec_type],
                channel=struct["channel"],
                time_axis=axis,
                loader=_SeriesLoader(self, sample_id, time_series_id),
                time_offsets_loader=time_offsets_loader,
                source_id=self._codec.decode_opt("source_id", struct["source_id"]),
                time_series_id=time_series_id,
                n_values=n_values,
            )
        except TimeFValidationError as exc:
            # A series rebuilt from a corrupt struct (bad n_values, a time offsets/axis mismatch) is a
            # format failure, not a caller mistake, even though TimeSeries raises the same type for both.
            raise TimeFFormatError(f"sample {sample_id!r} has an unbuildable series {time_series_id!r}: {exc}") from exc

    @staticmethod
    def _axis(struct: dict, sample_id: str) -> TimeAxis:
        """Rebuild a series' time axis from the stored discriminator.

        This method reads the tag before any shape-specific column. As a result, a corrupt row
        raises an error here, not later from an axis inferred from null columns.

        Args:
            struct: The stored time-series struct.
            sample_id: The owning sample, for the error message.

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
            shape_cols = (
                "period_numerator_us",
                "period_denominator",
                "start_index",
                "first_time_offset_us",
                "last_time_offset_us",
            )
            if any(struct[c] is not None for c in shape_cols):
                raise TimeFFormatError(
                    f"sample {sample_id!r} has a series tagged {kind!r} but carries regular- or "
                    f"irregular-axis columns; an ordinal series has neither"
                )
            return OrdinalAxis()
        if kind == AxisType.REGULAR:
            numerator, denominator = struct["period_numerator_us"], struct["period_denominator"]
            start_index = struct["start_index"]
            if numerator is None or denominator is None or start_index is None:
                raise TimeFFormatError(
                    f"sample {sample_id!r} has a series tagged {kind!r} with no period or start "
                    f"index; a regular axis needs both. pyarrow only enforces non-null when the file "
                    f"is written, so this is the check that catches a corrupt row"
                )
            try:
                return RegularAxis(period_us=Fraction(numerator, denominator), start_index=start_index)
            except (ZeroDivisionError, TypeError, TimeFValidationError) as exc:
                raise TimeFFormatError(
                    f"sample {sample_id!r} has a series with an unbuildable regular axis "
                    f"(period {numerator}/{denominator}, start_index {start_index}): {exc}"
                ) from exc
        if kind == AxisType.IRREGULAR:
            first, last = struct["first_time_offset_us"], struct["last_time_offset_us"]
            if first is None or last is None:
                raise TimeFFormatError(
                    f"sample {sample_id!r} has a series tagged {kind!r} with no endpoints; an "
                    f"irregular axis needs both. pyarrow only enforces non-null when the file is "
                    f"written, so this check catches a corrupt row"
                )
            try:
                return IrregularAxis(first_us=first, last_us=last)
            except TimeFValidationError as exc:
                raise TimeFFormatError(
                    f"sample {sample_id!r} has a series with unbuildable irregular endpoints ({first}, {last}): {exc}"
                ) from exc
        raise TimeFFormatError(
            f"sample {sample_id!r} has a series with axis_type {kind!r}; expected one of {[t.value for t in AxisType]}"
        )

    def _load_time_offsets(self, sample_id: str, time_series_id: str) -> pa.Array:
        """Read an irregular series' per-value time offsets through the values backend.

        Args:
            sample_id: The owning sample's id.
            time_series_id: The series id to read.

        Returns:
            One int64 microsecond time offset per value.

        Raises:
            TimeFFormatError: If the series has no index entry or its time offsets cannot be read.
        """
        rows = self._index_rows(sample_id, time_series_id)
        if not rows:
            raise TimeFFormatError(f"no index entry for sample {sample_id!r} series {time_series_id!r}")
        if self._values is None:
            self._values = make_values_reader(self._manifest.values_backend)
        try:
            return self._values.load_time_offsets(self._version, rows)
        except (KeyError, OSError, IndexError, ValueError) as exc:
            raise TimeFFormatError(
                f"failed to read time offsets for series {time_series_id!r} for sample {sample_id!r}: {exc}"
            ) from exc

    def _resolve_annotation(self, sample_id: str, annotation_id: str) -> Annotation:
        """Return one annotation, decoding it on first use and caching it in a bounded LRU.

        This method looks up the id through the pruned annotations table. As a result, it decodes
        only the row group that can hold the id. A shared annotation decodes once, and the LRU
        serves it after that.

        Args:
            sample_id: The referencing sample, for the error message.
            annotation_id: The annotation to resolve.

        Returns:
            The decoded annotation.

        Raises:
            TimeFFormatError: If the dataset has no annotation with that id.
        """
        cached = self._annotation_cache.get(annotation_id)
        if cached is not None:
            self._annotation_cache.move_to_end(annotation_id)
            return cached
        with self._as_format_error():
            probe = cast(_StoredId, self._codec.encode("annotation_id", annotation_id))
            rows = self._annotations_table().rows_for((probe,))
            if not rows:
                raise TimeFFormatError(f"sample {sample_id!r} references unknown annotation {annotation_id!r}")
            annotation = self._decode_annotation(rows[0])
        self._annotation_cache[annotation_id] = annotation
        if len(self._annotation_cache) > _ANNOTATION_CACHE_SIZE:
            self._annotation_cache.popitem(last=False)
        return annotation

    # ---- id decoding ---------------------------------------------------------------------------

    def _load_values(self, sample_id: str, time_series_id: str) -> pa.Array:
        """Read and concatenate a series' chunk values through the values backend.

        Args:
            sample_id: The owning sample's id.
            time_series_id: The series id to read.

        Returns:
            The series values in the spec's canonical Arrow representation.

        Raises:
            TimeFFormatError: If the series has no index entry or a chunk cannot be read.
        """
        rows = self._index_rows(sample_id, time_series_id)
        if not rows:
            raise TimeFFormatError(f"no index entry for sample {sample_id!r} series {time_series_id!r}")
        if self._values is None:
            self._values = make_values_reader(self._manifest.values_backend)
        try:
            spec_type = rows[0]["spec_type"]
            return self._values.load(self._version, rows, self._spec_by_type[spec_type])
        except (KeyError, OSError, IndexError, ValueError) as exc:
            raise TimeFFormatError(f"failed to read series {time_series_id!r} for sample {sample_id!r}: {exc}") from exc


@dataclass(frozen=True)
class _SeriesLoader:
    """A picklable lazy loader for one series' values (replaces a per-series closure).

    A nested closure cannot pickle. This class holds the reader and the series' identity instead of
    a closure. As a result, a read-back dataset can pickle, and a multi-worker torch ``DataLoader``
    can use it. The class reads the series' values only when you call it.
    """

    reader: TimeFReader
    """The reader that reads and decodes the series' values."""
    sample_id: str
    """The owning sample's id."""
    time_series_id: str
    """The id of the series to read."""

    def __call__(self) -> pa.Array:
        """Read the series' values.

        Returns:
            The series values in the spec's canonical Arrow representation.
        """
        return self.reader._load_values(self.sample_id, self.time_series_id)

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Read a temporal subsection through the selected values backend.

        Returns:
            The requested steps in their canonical Arrow representation.

        Raises:
            TimeFFormatError: If this series has no index entry.
        """
        rows = self.reader._index_rows(self.sample_id, self.time_series_id)
        if not rows:
            raise TimeFFormatError(f"no index entry for sample {self.sample_id!r} series {self.time_series_id!r}")
        if self.reader._values is None:
            self.reader._values = make_values_reader(self.reader._manifest.values_backend)
        spec = self.reader._spec_by_type[rows[0]["spec_type"]]
        return self.reader._values.load_range(self.reader._version, rows, start, stop, spec)


def _as_tuple_if_list(value: object) -> object:
    """Convert list payloads (and nested lists) to tuples. Pass scalars and structs through unchanged.

    Span structs stay as dicts, so :meth:`~timenet.format.schemas.IdCodec.decode_span` can rebuild
    them. This function normalizes only the list nesting around them, because tasks store tuples.

    Args:
        value: A cell value read from a task partition.

    Returns:
        The value with any list (recursively) converted to a tuple.
    """
    if isinstance(value, list):
        return tuple(_as_tuple_if_list(item) for item in value)
    return value


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
    sample_id: str
    """The owning sample's id."""
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

        The writer checks ordering, count, and endpoints, but nothing rechecks them on read. As a
        result, a corrupt shard can otherwise hand back a decreasing, wrong-length, or off-endpoint
        stream.

        Returns:
            One int64 microsecond time offset per value.

        Raises:
            TimeFFormatError: If the stored time offsets disagree with the axis endpoints or the
                value count, or if they decrease at any point.
        """
        time_offsets = self.reader._load_time_offsets(self.sample_id, self.time_series_id)
        values = time_offsets.to_numpy(zero_copy_only=False)
        where = f"series {self.time_series_id!r} on sample {self.sample_id!r}"
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
