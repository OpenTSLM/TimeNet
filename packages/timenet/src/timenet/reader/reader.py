"""``TimeFReader``: deserialize a TimeF version directory back into a :class:`TimeFDataset`.

Driven entirely by ``manifest.json`` — the reader never runs connector code. Opening a version reads
only the manifest; tasks, annotations, the time-series index, and per-series values are all resolved on
first use. Types are reconstructed from the manifest's flat descriptors (no runtime class synthesis),
so read-back objects pickle and match the originals field-for-field.
"""

import bisect
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import types as _types
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from timenet.dataset import Sample, TimeFDataset, TimeSeries
from timenet.dataset.axis import AxisType, IrregularAxis, OrdinalAxis, RegularAxis, TimeAxis
from timenet.dataset.sample import check_span_within_window
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.checksums import file_checksum
from timenet.format.constants import MANIFEST_FILE
from timenet.format.schemas import TASK_COMMON_NAMES, IdCodec, task_schema
from timenet.manifest import Manifest
from timenet.types import (
    TASKS,
    Annotation,
    DatasetMetadata,
    DatasetSchema,
    Task,
    TaskType,
    annotation_type_of,
    value_type_of,
)
from timenet.values_backends.reader import BaseValuesReader, make_values_reader


#: Rows per batch when streaming ``samples.parquet``. Small enough that one batch's Python rows stay
#: bounded, large enough that per-batch overhead stays off the critical path.
_SAMPLE_BATCH_ROWS = 4096

#: Decoded annotations kept for reuse. Annotations shared across samples are decoded once; the bound
#: keeps a dataset with millions of them from re-growing the eager dict this replaced.
_ANNOTATION_CACHE_SIZE = 4096

#: Byte budget for decoded index row groups, mirroring the zarr reader's chunk cache. Bounded by size
#: rather than by count because the two access patterns differ sharply: a sample-ordered scan reuses one
#: group thousands of times, while shuffled reads (the sleep-staging pattern) touch groups at random and
#: thrash any small count. Measured at 24 row groups, a 2-entry cache made shuffled access 6.1x slower
#: than in-order; a byte budget keeps both within noise of each other.
_INDEX_CACHE_MAX_BYTES = 64 * 2**20

#: An id in its stored form: the id string, or the 16 raw bytes of a ``uuid16`` column.
_StoredId = str | bytes


@dataclass(frozen=True)
class _IndexRowGroup:
    """One row group of the time-series index, and the ``sample_id`` range its statistics promise."""

    part: str
    """The index part's manifest-relative path."""
    ordinal: int
    """The row group's position within that part."""
    min_sample_id: _StoredId | None
    """The lowest stored ``sample_id`` in the group, or ``None`` if the file carries no statistics."""
    max_sample_id: _StoredId | None
    """The highest stored ``sample_id`` in the group, or ``None`` if the file carries no statistics."""

    def may_hold(self, sample_id: _StoredId) -> bool:
        """Report whether this row group can contain a stored ``sample_id``.

        Args:
            sample_id: The id in its stored form (a string, or 16 bytes for a ``uuid16`` column).

        Returns:
            ``False`` only when the statistics prove the id is outside the group.
        """
        if self.min_sample_id is None or self.max_sample_id is None:
            return True
        return self.min_sample_id <= sample_id <= self.max_sample_id  # ty: ignore[unsupported-operator]


class TimeFReader:
    """Reads a committed TimeF version directory. Use as a context manager to close shard handles."""

    def __init__(self, root: Path) -> None:
        """Open a dataset version directory and load its manifest.

        Only the manifest is read here. Tasks, annotations, the time-series index, and per-series values
        are resolved on first use, so opening a version costs the same whether it has three samples or
        three million. A malformed or unsupported-version manifest raises ``InvalidManifestError`` (a
        ``TimeFFormatError``) while parsing.

        Because the control plane is no longer decoded up front, a structurally corrupt but
        checksum-valid control-plane file now fails on the first access that needs it (``.tasks``, the
        first annotation, the first value read) rather than here. Call :meth:`verify` when you want
        construction-time integrity checking.

        Args:
            root: The version directory written by :class:`~timenet.writer.TimeFWriter`.

        Raises:
            FileNotFoundError: If ``root``, its ``manifest.json``, or any file the manifest lists is
                missing.
        """
        self._root = Path(root)
        if not self._root.exists():
            raise FileNotFoundError(f"dataset directory does not exist: {self._root}")
        manifest_path = self._root / MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest at {manifest_path}")
        self._manifest = Manifest.from_json(manifest_path.read_text())
        self._check_files_exist()

        self._codec = IdCodec.from_encoding(self._manifest.id_encoding)
        self._spec_by_type = {spec.spec_type: spec for spec in self._manifest.schema.time_series_specs}
        self._annotation_descriptors = {d.key: d for d in self._manifest.schema.annotations}
        # Everything below is per-process scratch: built on demand, dropped on pickle, rebuilt in the
        # unpickling process. __getstate__ must stay in step with this list.
        self._values: BaseValuesReader | None = None
        self._tasks: tuple[Task, ...] | None = None
        self._annotation_table: pa.Table | None = None
        self._annotation_rows: dict[object, int] | None = None
        self._annotation_cache: OrderedDict[str, Annotation] = OrderedDict()
        self._samples_data: pads.Dataset | None = None
        self._index_directory: list[_IndexRowGroup] | None = None
        self._index_bisectable = False
        self._index_files: dict[str, pq.ParquetFile] = {}
        self._index_cache: OrderedDict[tuple[str, int], tuple[pa.Table, dict[tuple, tuple[int, int]]]] = OrderedDict()
        self._index_cache_bytes = 0

    # ---- pickling ------------------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Drop every on-demand cache so the reader (and its loaders) pickle small and safely.

        The lazy loaders returned by :meth:`read` reference this reader, so a read-back dataset is
        only picklable (e.g. for a multi-worker torch ``DataLoader``) if the reader is. The values
        backend holds per-process scratch (file handles, decode caches) and the control-plane caches
        hold decoded tables; sending either to a worker would copy megabytes per worker to rebuild
        state each one can rebuild itself.

        Returns:
            The reader's state with every cache emptied.
        """
        state = self.__dict__.copy()
        state["_values"] = None
        state["_tasks"] = None
        state["_annotation_table"] = None
        state["_annotation_rows"] = None
        state["_annotation_cache"] = OrderedDict()
        state["_samples_data"] = None
        state["_index_directory"] = None
        state["_index_bisectable"] = False
        state["_index_files"] = {}
        state["_index_cache"] = OrderedDict()
        state["_index_cache_bytes"] = 0
        return state

    # ---- context manager -----------------------------------------------------------------------

    def __enter__(self) -> "TimeFReader":
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
        """Close the values backend and the index handles, releasing their decoded caches."""
        if self._values is not None:
            self._values.close()
            self._values = None
        for handle in self._index_files.values():
            handle.close()
        self._index_files.clear()
        self._index_directory = None
        self._index_bisectable = False
        self._index_cache.clear()
        self._index_cache_bytes = 0

    # ---- public API ----------------------------------------------------------------------------

    def verify(self) -> None:
        """Check every file the manifest lists against its recorded ``sha256:`` checksum.

        Not done on open: hashing every shard would read the whole dataset and defeat the lazy read
        path this class exists to provide. Call it explicitly when integrity matters more than
        latency (after a download, before a long training run, in a fsck-style command).

        Raises:
            TimeFFormatError: If a listed file is missing, or its contents do not match the manifest.
        """
        for rel, expected in sorted(self._manifest.checksums.items()):
            path = self._root / rel
            if not path.exists():
                raise TimeFFormatError(f"manifest lists a missing file: {rel}")
            actual = file_checksum(path)
            if actual != expected:
                raise TimeFFormatError(f"checksum mismatch for {rel}: manifest says {expected}, file is {actual}")

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

        The on-demand loaders parse on-disk control tables against the manifest's schema. Anything they
        raise means the artifact is corrupt or disagrees with its manifest, which is a format failure;
        letting a bare ``ValueError`` from ``TaskType()`` or pyarrow escape would contradict the
        reader's contract.

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
            parts = [str(self._root / rel) for rel in self._manifest.files.samples]
            self._samples_data = pads.dataset(parts, format="parquet")
        data = self._samples_data
        if sample_ids is None:
            for batch in data.to_batches(batch_size=_SAMPLE_BATCH_ROWS, use_threads=False):
                yield from batch.to_pylist()
            return
        wanted = list(dict.fromkeys(sample_ids))
        stored_type = data.schema.field("sample_id").type
        stored = {self._codec.encode("sample_id", sid): sid for sid in wanted}
        expression = pads.field("sample_id").isin(pa.array(list(stored), type=stored_type))
        for batch in data.to_batches(filter=expression, batch_size=_SAMPLE_BATCH_ROWS, use_threads=False):
            for row in batch.to_pylist():
                stored.pop(row["sample_id"], None)
                yield row
        # Enforced only once the caller drains the iterator: a lazy reader cannot know an id is absent
        # until it has looked everywhere, and an early-stopping consumer never asks it to.
        if stored:
            raise TimeFValidationError(
                f"no such sample(s) in {self._manifest.dataset_id}: {', '.join(stored.values())}"
            )

    def _check_files_exist(self) -> None:
        for rel in self._manifest.files.all_parts():
            if not (self._root / rel).exists():
                raise FileNotFoundError(f"manifest lists a missing file: {rel}")

    def _load_tasks(self) -> tuple[Task, ...]:
        by_id: dict[str, Task] = {}
        pending: dict[str, tuple[str, ...]] = {}
        for rel in self._manifest.files.tasks:
            task_type = TaskType(Path(rel).parent.name.split("=", 1)[1])
            cls = TASKS[task_type]
            payload_cols = [name for name in task_schema(task_type).names if name not in TASK_COMMON_NAMES]
            for row in pq.read_table(self._root / rel).to_pylist():
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

    def _annotations(self) -> tuple[pa.Table, dict[object, int]]:
        """Return the annotation columns needed to decode a row, plus the id-to-row map.

        Read on first annotation access, not on open, and only the four columns a decode reads:
        ``sample_ids`` is the widest column in the file and nothing on the read path uses it. The rows
        stay in Arrow and are decoded one at a time, so what is resident is the file's own columns
        rather than one :class:`Annotation` (with its JSON-decoded value and rebuilt span) per row.

        Annotations are stored sorted by ``(key, id)``, not by id alone, so a filtered read cannot
        prune to a single row group the way the index can; the id map is what turns a lookup into a
        direct row offset.

        Returns:
            The annotation table and a mapping from each stored id to its row offset.
        """
        if self._annotation_table is None or self._annotation_rows is None:
            columns = ["id", "key", "value", "span"]
            parts = [pq.read_table(self._root / rel, columns=columns) for rel in self._manifest.files.annotations]
            table = parts[0] if len(parts) == 1 else pa.concat_tables(parts)
            self._annotation_table = table
            self._annotation_rows = {stored: row for row, stored in enumerate(table.column("id").to_pylist())}
        return self._annotation_table, self._annotation_rows

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

    def _index_groups(self) -> list["_IndexRowGroup"]:
        """Return the index's row-group directory: where each lives and the ids it can hold.

        Read from the Parquet footers on first lookup, so it is O(row groups), not O(rows): a 139M-row
        index is a few thousand entries here whatever its row count. This is what a lookup prunes
        against, and the reason the reader never holds the index itself.

        A lookup bisects the directory on each group's maximum id. That is a sorted key only when every
        group carries statistics, so ``_index_bisectable`` records whether it does; when it does not,
        every group must be considered.

        Returns:
            One entry per row group, in file order.
        """
        if self._index_directory is None:
            directory: list[_IndexRowGroup] = []
            for rel in self._manifest.files.time_series_index:
                metadata = self._index_file(rel).metadata
                column = metadata.schema.names.index("sample_id")
                for ordinal in range(metadata.num_row_groups):
                    stats = metadata.row_group(ordinal).column(column).statistics
                    ranged = stats is not None and stats.has_min_max
                    directory.append(
                        _IndexRowGroup(
                            part=rel,
                            ordinal=ordinal,
                            min_sample_id=stats.min if ranged else None,
                            max_sample_id=stats.max if ranged else None,
                        )
                    )
            self._index_directory = directory
            self._index_bisectable = all(group.max_sample_id is not None for group in directory)
        return self._index_directory

    def _index_file(self, rel: str) -> pq.ParquetFile:
        """Return the open Parquet handle for one index part, opening it on first use.

        Args:
            rel: The part's manifest-relative path.

        Returns:
            The cached handle.
        """
        handle = self._index_files.get(rel)
        if handle is None:
            handle = pq.ParquetFile(self._root / rel)
            self._index_files[rel] = handle
        return handle

    def _index_group_rows(self, group: "_IndexRowGroup") -> tuple[pa.Table, dict[tuple, tuple[int, int]]]:
        """Return one index row group as Arrow, plus where each series' rows start and how many there are.

        The offsets are built once per decoded row group and thrown away with it, so what is resident is
        bounded by the cache size rather than by the index's row count. A sample-ordered read walks the
        index in order, so the cache is enough to serve a whole scan from a handful of decodes.

        Args:
            group: The row group to decode.

        Returns:
            The row group's table and its ``(sample_id, time_series_id) -> (offset, count)`` map.
        """
        key = (group.part, group.ordinal)
        cached = self._index_cache.get(key)
        if cached is not None:
            self._index_cache.move_to_end(key)
            return cached
        table = self._index_file(group.part).read_row_group(group.ordinal)
        offsets: dict[tuple, tuple[int, int]] = {}
        keys = zip(table.column("sample_id").to_pylist(), table.column("time_series_id").to_pylist(), strict=True)
        for row, series_key in enumerate(keys):
            offset, count = offsets.get(series_key, (row, 0))
            offsets[series_key] = (offset, count + 1)
        entry = (table, offsets)
        if table.nbytes > _INDEX_CACHE_MAX_BYTES:
            return entry  # a single group over budget is served without displacing the whole cache
        self._index_cache[key] = entry
        self._index_cache_bytes += table.nbytes
        while self._index_cache_bytes > _INDEX_CACHE_MAX_BYTES:
            _, evicted = self._index_cache.popitem(last=False)
            self._index_cache_bytes -= evicted[0].nbytes
        return entry

    def _index_rows(self, sample_id: str, time_series_id: str) -> list[dict]:
        """Return one series' index rows, in ``chunk_idx`` order.

        A per-series filtered read over the already ``sample_id``-sorted table: the row-group statistics
        rule out every row group whose id range excludes the probe, and only the survivors are decoded.
        Opening a dataset therefore costs nothing per index row, where holding the index as one sorted
        key list per row cost about 180 bytes each — 23 GB at the 139M-row index this is sized for.

        The groups are in ``sample_id`` order, so the search bisects to the first one that can hold the
        probe and stops at the first that cannot; without that a lookup would still be linear in the
        number of row groups, which is thousands at that scale. Rows for one series are contiguous
        within a group and groups are visited in file order, so the writer's ``(sample_id,
        time_series_id, chunk_idx)`` sort carries through to the result.

        Args:
            sample_id: The owning sample's id.
            time_series_id: The series id.

        Returns:
            The series' index rows, empty if it has none.
        """
        with self._as_format_error():
            stored_sample_id = cast(_StoredId, self._codec.encode("sample_id", sample_id))
            probe = (stored_sample_id, self._codec.encode("time_series_id", time_series_id))
            groups = self._index_groups()
            first = (
                bisect.bisect_left(groups, stored_sample_id, key=lambda group: group.max_sample_id)
                if self._index_bisectable
                else 0
            )
            rows: list[dict] = []
            for position in range(first, len(groups)):
                group = groups[position]
                if not group.may_hold(stored_sample_id):
                    break
                table, offsets = self._index_group_rows(group)
                span = offsets.get(probe)
                if span is not None:
                    rows.extend(table.slice(span[0], span[1]).to_pylist())
            return rows

    # ---- sample construction -------------------------------------------------------------------

    def _build_sample(self, row: dict) -> Sample:
        sample_id = self._codec.decode("sample_id", row["sample_id"])
        series = tuple(self._build_series(sample_id, struct) for struct in row["time_series"])
        annotations = tuple(
            self._resolve_annotation(sample_id, aid)
            for aid in self._codec.decode_list("annotation_id", row["annotation_ids"])
        )
        # A span is stored on the annotation but resolved against the sample; a stored span that no
        # longer fits the series it lands on is a corrupt artifact, not a caller mistake.
        for annotation in annotations:
            if annotation.span is not None:
                try:
                    check_span_within_window(f"annotation {annotation.key!r}", annotation.span, series, sample_id)
                except TimeFValidationError as exc:
                    raise TimeFFormatError(str(exc)) from exc
        return Sample(
            sample_id=sample_id,
            time_series=series,
            subject_ids=tuple(self._codec.decode_list("subject_id", row["subject_ids"])),
            task_ids=tuple(self._codec.decode_list("task_id", row["task_ids"])),
            annotations=annotations,
            start_time=row.get("start_time_us"),
        )

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
            return self._values.load_time_offsets(self._root, rows)
        except (KeyError, OSError, IndexError, ValueError) as exc:
            raise TimeFFormatError(
                f"failed to read time offsets for series {time_series_id!r} for sample {sample_id!r}: {exc}"
            ) from exc

    def _resolve_annotation(self, sample_id: str, annotation_id: str) -> Annotation:
        """Return one annotation, decoding it on first use and caching it in a bounded LRU.

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
            table, rows = self._annotations()
            row_idx = rows.get(self._codec.encode("annotation_id", annotation_id))
            if row_idx is None:
                raise TimeFFormatError(f"sample {sample_id!r} references unknown annotation {annotation_id!r}")
            annotation = self._decode_annotation(table.slice(row_idx, 1).to_pylist()[0])
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
            return self._values.load(self._root, rows, self._spec_by_type[spec_type])
        except (KeyError, OSError, IndexError, ValueError) as exc:
            raise TimeFFormatError(f"failed to read series {time_series_id!r} for sample {sample_id!r}: {exc}") from exc


@dataclass(frozen=True)
class _SeriesLoader:
    """A picklable lazy loader for one series' values (replaces a per-series closure).

    A nested closure can't be pickled, which would make every read-back dataset unpicklable and break
    a multi-worker torch ``DataLoader``. This holds the reader and the series' identity instead, and
    reads on call.
    """

    reader: "TimeFReader"
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
        return self.reader._values.load_range(self.reader._root, rows, start, stop, spec)


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

    reader: "TimeFReader"
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
