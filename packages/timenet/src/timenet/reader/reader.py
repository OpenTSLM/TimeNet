"""``TimeFReader``: deserialize a TimeF version directory back into a :class:`TimeFDataset`.

Driven entirely by ``manifest.json`` — the reader never runs connector code. Opening a version reads
only the manifest; tasks, annotations, the time-series index, and per-series values are all resolved on
first use. Types are reconstructed from the manifest's flat descriptors (no runtime class synthesis), so
read-back objects pickle and match the originals field-for-field.
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


#: Rows per batch when streaming ``samples.parquet``.
_SAMPLE_BATCH_ROWS = 4096

#: Decoded annotations kept for reuse, so an annotation shared across samples is decoded once.
_ANNOTATION_CACHE_SIZE = 4096

#: Byte budget for decoded control-table row groups. Sized by total bytes, not group count, so shuffled
#: reads do not thrash a small cache.
_CONTROL_TABLE_CACHE_MAX_BYTES = 64 * 2**20

#: An id in its stored form: the id string, or the 16 raw bytes of a ``uuid16`` column.
_StoredId = str | bytes


@dataclass(frozen=True)
class _RowGroupStats:
    """One row group of a sorted control table, and the stored-key range its statistics promise."""

    part: str
    """The table part's manifest-relative path."""
    ordinal: int
    """The row group's position within that part."""
    min_key: _StoredId
    """The lowest stored key in the group."""
    max_key: _StoredId
    """The highest stored key in the group."""

    def may_hold(self, key: _StoredId) -> bool:
        """Report whether this row group can contain a stored key.

        Args:
            key: The key in its stored form (a string, or 16 bytes for a ``uuid16`` column).

        Returns:
            ``False`` only when the statistics prove the key is outside the group.
        """
        return self.min_key <= key <= self.max_key  # ty: ignore[unsupported-operator]


class _PrunedControlTable:
    """A control-plane table sorted by one key column, read a row group at a time with pruning.

    The reader never holds the whole table. It builds a row-group directory from the Parquet footers,
    bisects it on the sorted key column's statistics to prune the groups a lookup cannot match, and
    decodes the matching groups into a byte-bounded LRU. Within a decoded group it bisects a materialized
    sorted list of the lookup-key column to slice the matching run. The time-series index (keyed by
    ``sample_id``, many rows per key) and the annotations table (keyed by ``id``, one row per key) are
    both served this one way; the key column is the leading entry of ``lookup_columns``.
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
            lookup_columns: The columns whose stored values a lookup matches on, as a tuple; the first is
                the sorted key column the footer statistics prune on.
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

        Built from the Parquet footers on first lookup, so it is O(row groups), never O(rows). The
        writer emits key-column statistics on every row group, so a lookup always bisects on the maxima.

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
        """Return the rows matching a stored lookup key, pruning the row groups that cannot hold it.

        The prune key is ``lookup[0]`` (the sorted key column). The search bisects to the first group
        that can hold it and stops at the first that cannot; the rows of one key are contiguous within a
        group, and groups are visited in file order, so the writer's sort carries through to the result.

        Args:
            lookup: The stored values to match, one per ``lookup_columns`` entry, in that order.

        Returns:
            The matching rows as dicts, in stored order; empty if the key is absent.
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
            # The group is sorted by its lookup columns, so the matches are the half-open range
            # [bisect_left, bisect_right). One key's rows are contiguous; a unique key is a run of one.
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

        The list is the group's lookup columns zipped into one tuple per row, in stored (sorted) order,
        so :meth:`rows_for` bisects it with the stdlib.

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
            return entry  # a single group over budget is served without displacing the whole cache
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

        Nothing is read here: the handle already carries the parsed manifest, and the filesystem and root
        it wraps are the seam every read flows through. Tasks, annotations, the time-series index, and
        per-series values all resolve on first use, so opening a version costs the same whether it has
        three samples or three million.

        Because the control plane is no longer decoded up front, and no file is stat-swept on open, a
        structurally corrupt (or missing) file now fails on the first access that needs it (``.tasks``,
        the first annotation, the first value read) rather than here. Call :meth:`verify` when you want a
        construction-time integrity check. Build the handle with
        :meth:`~timenet.registry.BaseRegistry.open_version` or
        :meth:`~timenet.registry.version.DatasetVersion.open_local`.

        Args:
            version: The opened version handle: a manifest plus a filesystem-rooted view of its files.
        """
        self._version = version
        # Aliases of the handle's fields for the read sites below; all three pickle, so a DataLoader
        # worker rebuilds the reader (and its lazy loaders) from them without re-opening the registry.
        self._fs = version.filesystem
        self._root = version.root
        self._manifest = version.manifest

        self._codec = IdCodec.from_encoding(self._manifest.id_encoding)
        self._spec_by_type = {spec.spec_type: spec for spec in self._manifest.schema.time_series_specs}
        self._annotation_descriptors = {d.key: d for d in self._manifest.schema.annotations}
        # Per-process scratch, built on demand and dropped on pickle; __getstate__ must stay in step.
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

        Not done on open: hashing every shard would read the whole dataset and defeat the lazy read
        path this class exists to provide. Call it explicitly when integrity matters more than
        latency (after a download, before a long training run, in a fsck-style command). Each file is
        reopened through the version's filesystem, so this is also where a missing file surfaces now that
        ``__init__`` no longer stat-sweeps.

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
        return TimeFDataset.from_parts(
            metadata=self._manifest.metadata,
            samples=list(self.iter_samples()),
            tasks=self.tasks,
            schema=self._manifest.schema,
        )

    def iter_samples(self, sample_ids: Iterable[str] | None = None) -> Iterator[Sample]:
        """Yield samples lazily without materializing a :class:`TimeFDataset`.

        With ``sample_ids``, the read is filtered on the stored id column, which prunes the row groups
        that cannot contain a requested id instead of scanning the file. Samples come back in stored
        order (sorted by ``sample_id``), not in the order they were asked for, and an id the dataset
        does not contain raises ``TimeFValidationError``.

        Args:
            sample_ids: The samples to yield; ``None`` yields every sample.

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
        means the artifact is corrupt or disagrees with its manifest, which is a format failure; letting
        a bare ``ValueError`` from ``TaskType()`` or an ``OSError`` from a corrupt data page escape would
        contradict the reader's contract.

        Yields:
            Nothing; this only rewrites the exception type.

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
            TimeFValidationError: If ``sample_ids`` names an id the dataset does not contain, raised
                once the iterator is fully consumed rather than on the offending id: a lazy reader
                cannot know an id is absent until it has read every part.
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

        Only the four columns a decode reads are pulled; ``sample_ids``, the widest column, is skipped.

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
        # The descriptor is what a registry query filters on, so a decoded annotation whose shape
        # or value type disagrees with it would answer those queries wrongly. That is corruption.
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
        # A stored span or time_span that no longer fits is a corrupt artifact, not a caller mistake, so
        # the sample's own invariants surface as a format error, not a ValueError. Decode and build the
        # sample inside the seam so a malformed time_span (bad bounds, or point-shaped) is rejected by
        # Sample.__post_init__ before it is used to re-check the stored annotation spans.
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

        The tag is read before any shape-specific column, so a corrupt row raises here rather than
        producing an axis inferred from which columns happen to be null.

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
            # be null. A populated one means the tag and the columns disagree, the corruption this
            # method catches.
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

        The id is looked up through the pruned annotations table, so only the row group that can hold it
        is decoded; a shared annotation is decoded once and served from the LRU thereafter.

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

    A nested closure can't be pickled, which would make every read-back dataset unpicklable and break
    a multi-worker torch ``DataLoader``. This holds the reader and the series' identity instead, and
    reads on call.
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
    """Convert list payloads (and nested lists) to tuples; pass scalars and structs through.

    Span structs stay dicts for :meth:`~timenet.format.schemas.IdCodec.decode_span` to rebuild; only the
    list nesting around them is normalized, since tasks store tuples.


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

    A nested closure can't be pickled, so this is a class rather than a lambda: a multi-worker torch
    DataLoader sends the series to its workers, and a closure would fail there. It mirrors
    :class:`_SeriesLoader` because time offsets and values are separate columns and each is read on its
    own.
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

        The writer verifies ordering, count, and endpoints, but nothing re-checks them on read, so a
        corrupt shard could otherwise hand back a decreasing, wrong-length, or off-endpoint stream.

        Returns:
            One int64 microsecond time offset per value.

        Raises:
            TimeFFormatError: If the stored time offsets disagree with the axis endpoints or value count,
                or are not non-decreasing.
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
