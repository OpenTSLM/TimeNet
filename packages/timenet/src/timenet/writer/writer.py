"""``TimeFWriter``: serialize a :class:`~timenet.dataset.TimeFDataset` to the TimeF layout on disk.

The writer keeps the Parquet control plane fixed and sends series values to a selected Parquet or
Zarr backend. The configured chunk, row-group, and shard targets bound the backend buffers.
The writer stages everything in a temporary directory and publishes it with a single atomic rename.
A ``manifest.json`` in the version directory marks a committed version.
"""

from collections.abc import Callable, Iterable, Iterator
from enum import StrEnum
import itertools
import json
from pathlib import Path
import shutil
import types as _types
from typing import Any, assert_never
import uuid

import numpy as np
import pyarrow as pa

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import IrregularAxis, OrdinalAxis, RegularAxis, TimeAxis, to_time_offsets_us
from timenet.dataset.time_series import _validate_enum_values
from timenet.errors import TimeFValidationError
from timenet.format.checksums import file_checksum
from timenet.format.constants import (
    ANNOTATIONS_TEMPLATE,
    DEFAULT_CHUNK_MAX_BYTES,
    DEFAULT_COMPRESSION,
    DEFAULT_CONTROL_SHARD_TARGET_BYTES,
    DEFAULT_ROW_GROUP_TARGET_BYTES,
    DEFAULT_SHARD_TARGET_BYTES,
    INDEX_TEMPLATE,
    MANIFEST_FILE,
    SAMPLES_TEMPLATE,
    TASK_PART_TEMPLATE,
    part_path,
)
from timenet.format.schemas import (
    LOGICAL_IDS,
    TASK_COMMON_NAMES,
    UUID16,
    IdCodec,
    IdTypes,
    annotations_schema,
    index_schema,
    samples_schema,
    task_schema,
)
from timenet.manifest import FilePart, Manifest, ManifestCounts, ManifestFiles
from timenet.provenance import build_env
from timenet.types import Annotation, Task
from timenet.types.ids import is_canonical_uuid
from timenet.values_backends import SUPPORTED_VALUES_BACKENDS, ValuesBackend
from timenet.values_backends.writer import (
    ChunkPlacement,
    ParquetValuesConfig,
    ZarrValuesConfig,
    make_values_backend,
)
from timenet.writer import encodings
from timenet.writer.progress import ProgressStage, WriteProgressEvent
from timenet.writer.sharded import ShardedTableWriter
from timenet.writer.value_encoding import AUTO, SUPPORTED_VALUE_ENCODINGS, ValueEncoding


class TimeFWriter:
    """Context manager that serializes a dataset into the TimeF format and commits it atomically."""

    def __init__(  # noqa: PLR0913
        self,
        root: Path,
        dataset: TimeFDataset,
        *,
        shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
        control_shard_target_bytes: int = DEFAULT_CONTROL_SHARD_TARGET_BYTES,
        row_group_target_bytes: int = DEFAULT_ROW_GROUP_TARGET_BYTES,
        chunk_max_bytes: int = DEFAULT_CHUNK_MAX_BYTES,
        compression: str = DEFAULT_COMPRESSION,
        compression_level: int | None = None,
        data_page_size: int | None = None,
        values_backend: str = ValuesBackend.PARQUET,
        value_encoding: str = AUTO,
        progress_cb: Callable[[WriteProgressEvent], None] | None = None,
    ) -> None:
        """Configure the writer.

        Args:
            root: Parent directory. The writer creates ``<root>/<dataset_id>/<version>/``.
            dataset: The populated dataset. The writer derives the schema automatically if needed.
            shard_target_bytes: Rotate to a new shard once a shard's buffered values exceed this.
            control_shard_target_bytes: Split a control table (samples, annotations, index, tasks) into
                a new part once the in-memory Arrow size of the emitted rows exceeds this.
            row_group_target_bytes: Flush a row group once buffered values exceed this.
            chunk_max_bytes: Split a series into chunks no larger than this.
            compression: Values codec (Parquet codec or Zarr Blosc inner codec).
            compression_level: Pinned compression level, or ``None`` for the backend default.
            data_page_size: Target uncompressed bytes per Parquet data page, or ``None`` for
                the backend default. Ignored by the Zarr backend.
            values_backend: Storage backend for the values plane.
            value_encoding: ``"auto"`` (the default) selects the values-column encoding per
                ``spec_type`` from the data. ``"dictionary"``, ``"byte_stream_split"``, or ``"plain"``
                forces one for every modality. Only the Parquet backend applies an encoding, so the
                writer rejects forcing one on another backend.
            progress_cb: Optional callback invoked with each :class:`WriteProgressEvent`.

        Raises:
            TimeFValidationError: If ``dataset.metadata.dataset_id`` is empty, ``values_backend`` or
                ``value_encoding`` is unsupported, or a forced ``value_encoding`` targets a backend
                that cannot apply it.
        """
        if not dataset.metadata.dataset_id:
            raise TimeFValidationError("dataset_id must be non-empty")
        if values_backend not in SUPPORTED_VALUES_BACKENDS:
            raise TimeFValidationError(
                f"unknown values_backend {values_backend!r}; supported: {', '.join(sorted(SUPPORTED_VALUES_BACKENDS))}"
            )
        if value_encoding != AUTO and value_encoding not in SUPPORTED_VALUE_ENCODINGS:
            raise TimeFValidationError(
                f"unknown value_encoding {value_encoding!r}; "
                f"supported: {AUTO}, {', '.join(sorted(SUPPORTED_VALUE_ENCODINGS))}"
            )
        if value_encoding != AUTO and values_backend != ValuesBackend.PARQUET:
            raise TimeFValidationError(
                f"value_encoding {value_encoding!r} is only applied by the "
                f"{ValuesBackend.PARQUET.value!r} values backend, not {values_backend!r}"
            )
        if data_page_size is not None and data_page_size <= 0:
            raise TimeFValidationError(f"data_page_size must be positive, got {data_page_size}")
        self._root = Path(root)
        self._dataset = dataset
        self._shard_target_bytes = shard_target_bytes
        self._control_shard_target_bytes = control_shard_target_bytes
        self._row_group_target_bytes = row_group_target_bytes
        self._chunk_max_bytes = chunk_max_bytes
        self._compression = compression
        self._compression_level = compression_level
        self._data_page_size = data_page_size
        self._values_backend_name = values_backend
        self._forced_value_encoding = None if value_encoding == AUTO else ValueEncoding(value_encoding)
        self._value_encoding: dict[str, str] = {}
        self._progress_cb = progress_cb

        version = str(dataset.metadata.dataset_version)
        self._final_dir = self._root / dataset.metadata.dataset_id / version
        self._staging_dir = self._root / dataset.metadata.dataset_id / f"{version}.tmp-{uuid.uuid4().hex}"

        self._written = False
        # A streamed dataset's task iterator, peeked once for id types and reused for the write.
        self._task_iter: Iterator[Task] = iter(())
        self._first_task: Task | None = None

    # ---- context manager -----------------------------------------------------------------------

    def __enter__(self) -> "TimeFWriter":
        """Create the staging directory, refusing to overwrite a committed version.

        Returns:
            This writer.

        Raises:
            FileExistsError: If a committed ``manifest.json`` already exists at the version directory.
        """
        if (self._final_dir / MANIFEST_FILE).exists():
            raise FileExistsError(f"a committed version already exists at {self._final_dir}")
        self._sweep_stale_staging()
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        return self

    def _sweep_stale_staging(self) -> None:
        """Remove abandoned ``<version>.tmp-*`` staging dirs left by a crashed build.

        A hard kill (SIGKILL or OOM) never reaches :meth:`abort`, so its staging directory lingers.
        Clear any such sibling for this version before you write a fresh one. The writer does not
        support concurrent writes of the same version.
        """
        parent = self._final_dir.parent
        if not parent.is_dir():
            return
        for entry in parent.glob(f"{self._final_dir.name}.tmp-*"):
            if entry.is_dir() and entry != self._staging_dir:
                shutil.rmtree(entry, ignore_errors=True)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: _types.TracebackType | None,
    ) -> None:
        """Commit on success, abort on any failure."""
        if exc is not None:
            self.abort()
            return
        try:
            self.close()
        except BaseException:
            self.abort()
            raise

    # ---- lifecycle -----------------------------------------------------------------------------

    def write(self) -> None:
        """Serialize every artifact except the manifest into the staging directory."""
        if self._dataset.schema is None:
            self._dataset.derive_schema()
        self._validate_shared_annotations()
        self._resolve_id_types()

        unique_series, series_to_samples = self._dedupe_series()
        placements = self._write_values(unique_series)
        self._write_samples()
        self._write_annotations()
        self._write_tasks()
        self._write_index(placements, series_to_samples)
        self._counts = self._build_counts(unique_series, placements)
        self._written = True

    def close(self) -> None:
        """Write the manifest and atomically publish the staging directory.

        Raises:
            RuntimeError: If :meth:`write` has not run successfully.
        """
        if not self._written:
            raise RuntimeError("write() must run successfully before close()")
        self._write_manifest()
        if self._final_dir.exists():
            # Only reachable when the caller pre-created the target, or a previous run died between
            # this rmtree and the replace() below. run_pipeline returns early on a committed version
            # and drops it itself on --force, so it never deletes a committed dataset here.
            shutil.rmtree(self._final_dir)
        self._final_dir.parent.mkdir(parents=True, exist_ok=True)
        self._staging_dir.replace(self._final_dir)
        self._emit(ProgressStage.COMMIT, 1, 1)

    def abort(self) -> None:
        """Delete the staging directory. Safe to call more than once."""
        shutil.rmtree(self._staging_dir, ignore_errors=True)

    # ---- id storage ----------------------------------------------------------------------------

    def _resolve_id_types(self) -> None:
        """Pick per-logical-id storage: ``binary(16)`` when every value is a canonical UUID, else string."""
        values: dict[str, list[str]] = {name: [] for name in LOGICAL_IDS}
        for sample in self._dataset.samples:
            values["sample_id"].append(sample.sample_id)
            values["subject_id"].extend(sample.subject_ids)
            for ts in sample.time_series:
                values["time_series_id"].append(ts.time_series_id)
                if ts.source_id is not None:
                    values["source_id"].append(ts.source_id)
            for ann in sample.annotations:
                values["annotation_id"].append(ann.id)
        for ann in self._dataset.registered_annotations:  # task-referenced, carried by no sample
            values["annotation_id"].append(ann.id)
        if self._dataset.has_task_stream:
            # Streamed tasks are not materialized; one peeked id decides binary(16)-vs-string storage
            # without draining millions of tasks. Keep the validating iterator so _write_tasks reuses
            # it: the source is consumed (and validated) exactly once, even a one-shot generator.
            self._task_iter = self._dataset.iter_streamed_tasks_validated()
            self._first_task = next(self._task_iter, None)
            if self._first_task is not None:
                values["task_id"].append(self._first_task.id)
        else:
            for task in self._dataset.tasks:
                values["task_id"].append(task.id)

        id_types: IdTypes = {}
        for name in LOGICAL_IDS:
            vals = values[name]
            is_uuid16 = bool(vals) and all(is_canonical_uuid(v) for v in vals)
            id_types[name] = UUID16 if is_uuid16 else pa.string()
        self._id_types = id_types
        self._uuid16 = {name for name in LOGICAL_IDS if id_types[name] == UUID16}
        self._codec = IdCodec.from_uuid16(self._uuid16)

    # ---- values --------------------------------------------------------------------------------

    def _dedupe_series(self) -> tuple[list[TimeSeries], dict[str, list[str]]]:
        """Return unique series (sorted for stable output) and the series-id -> sample-ids map.

        Sharing one series across samples is the supported dedupe path. Two different series that
        claim one ``time_series_id`` is a contradiction. The writer can write only one of them, so
        the other's samples read back the wrong data. Two series that share an id must describe the
        same channel. The writer rejects a disagreement instead of keeping the first series.

        Returns:
            The sorted unique series and a mapping from ``time_series_id`` to the ids of the samples
            that reference it (first-seen order).

        Raises:
            TimeFValidationError: If two series share a ``time_series_id`` but describe different
                channels.
        """
        unique: dict[str, TimeSeries] = {}
        series_to_samples: dict[str, list[str]] = {}
        seen_pairs: set[tuple[str, str]] = set()
        for sample in self._dataset.samples:
            for ts in sample.time_series:
                existing = unique.get(ts.time_series_id)
                if existing is None:
                    unique[ts.time_series_id] = ts
                elif existing is not ts and _series_identity(existing) != _series_identity(ts):
                    raise TimeFValidationError(
                        f"time_series_id {ts.time_series_id!r} is claimed by two different series: "
                        f"{_series_identity(existing)} and {_series_identity(ts)}; ids must be unique "
                        f"per channel, or reuse the same series instance to share it across samples"
                    )
                pair = (ts.time_series_id, sample.sample_id)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    series_to_samples.setdefault(ts.time_series_id, []).append(sample.sample_id)
        # Group a recording's series together (source_id) before splitting by channel, so all leads of
        # one record are contiguous: the writer reads the record's source once, and a reader pulls a
        # sample's series from one place instead of scattered across channel-ordered shards. Falls back
        # to channel order when source_id is unset (one series per sample), matching the prior layout.
        ordered = sorted(
            unique.values(), key=lambda ts: (ts.spec.spec_type, ts.source_id or "", ts.channel, ts.time_series_id)
        )
        return ordered, series_to_samples

    def _write_values(self, unique_series: list[TimeSeries]) -> dict[tuple[str, int], ChunkPlacement]:
        """Write all series' values through the configured backend.

        Args:
            unique_series: The deduped, sorted series to serialize.

        Returns:
            A mapping from ``(time_series_id, chunk_idx)`` to its on-disk placement.
        """
        if self._values_backend_name == ValuesBackend.PARQUET:
            parquet_kwargs: dict[str, Any] = {
                "staging_dir": self._staging_dir,
                "id_types": self._id_types,
                "codec": self._codec,
                "shard_target_bytes": self._shard_target_bytes,
                "row_group_target_bytes": self._row_group_target_bytes,
                "chunk_max_bytes": self._chunk_max_bytes,
                "compression": self._compression,
                "data_page_size": self._data_page_size,
                "value_encoding": self._forced_value_encoding,
            }
            if self._compression_level is not None:
                parquet_kwargs["compression_level"] = self._compression_level
            config = ParquetValuesConfig(**parquet_kwargs)
        else:
            zarr_kwargs: dict[str, Any] = {
                "staging_dir": self._staging_dir,
                "shard_target_bytes": self._shard_target_bytes,
                "chunk_max_bytes": self._chunk_max_bytes,
                "compression": self._compression,
            }
            if self._compression_level is not None:
                zarr_kwargs["compression_level"] = self._compression_level
            config = ZarrValuesConfig(**zarr_kwargs)
        values_backend = make_values_backend(config)
        result = values_backend.write_series(
            unique_series,
            read_and_validate=self._read_and_validate,
            read_time_offsets=self._read_time_offsets,
            on_series_done=lambda completed, total: self._emit(ProgressStage.TIME_SERIES, completed, total),
            on_file_done=lambda count: self._emit(ProgressStage.SHARD_FINALIZED, count, None),
        )
        self._value_files = result.files
        self._value_encoding = dict(result.value_encoding)
        return result.placements

    def _read_and_validate(self, ts: TimeSeries) -> pa.Array:  # noqa: PLR6301
        """Read a series' values and enforce the per-series array contract.

        Args:
            ts: The series to read.

        Returns:
            The validated values in the spec's canonical Arrow representation.

        Raises:
            TimeFValidationError: If the array disagrees with the spec's dtype/shape, has
                non-finite inexact values, or its length disagrees with ``n_values``.
        """
        values = ts.to_arrow()
        if ts.spec.dtype == "enum":
            valid_type = isinstance(values, pa.DictionaryArray) and values.type.value_type == pa.string()
        else:
            expected_type = pa.string() if ts.spec.dtype == "str" else pa.from_numpy_dtype(np.dtype(ts.spec.dtype))
            if ts.spec.value_shape:
                valid_type = (
                    isinstance(values, pa.FixedShapeTensorArray)
                    and values.type.value_type == expected_type
                    and tuple(values.type.shape) == ts.spec.value_shape
                )
            else:
                valid_type = (
                    isinstance(values, pa.Array)
                    and not isinstance(values, pa.ExtensionArray)
                    and values.type == expected_type
                )
        if not valid_type:
            raise TimeFValidationError(
                f"series {ts.time_series_id!r} must load dtype={ts.spec.dtype}, "
                f"value_shape={ts.spec.value_shape} as Arrow, got "
                f"{values.type if isinstance(values, pa.Array) else type(values)!r}"
            )
        if ts.spec.dtype == "enum":
            _validate_enum_values(ts.spec, values.to_pylist())
        if ts.spec.dtype not in {"str", "enum"}:
            as_numpy = (
                values.to_numpy_ndarray()
                if isinstance(values, pa.FixedShapeTensorArray)
                else values.to_numpy(zero_copy_only=False)
            )
            if np.issubdtype(as_numpy.dtype, np.inexact) and not np.isfinite(as_numpy).all():
                raise TimeFValidationError(f"series {ts.time_series_id!r} has non-finite values")
        if len(values) != ts.n_values:
            raise TimeFValidationError(
                f"series {ts.time_series_id!r}: its loader returned {len(values)} values but it "
                f"declares n_values={ts.n_values}"
            )
        return values

    def _read_time_offsets(self, ts: TimeSeries) -> pa.Array | None:  # noqa: PLR6301
        """Read an irregular series' time offsets and check them against what it declares.

        The method returns ``None`` for every other axis shape. The backend writes that as a null cell.
        These checks make ``first_time_offset_us`` and ``last_time_offset_us`` verified metadata. An
        axis cannot claim endpoints that its own stream does not have.

        Args:
            ts: The series to read.

        Returns:
            The validated int64 time offsets, or ``None`` if the series stores none.

        Raises:
            TimeFValidationError: If the stream is unusable, its length disagrees with ``n_values``, or
                its endpoints disagree with the axis.
        """
        if ts.time_offsets_loader is None:
            return None
        axis = ts.time_axis
        if not isinstance(axis, IrregularAxis):  # pragma: no cover - TimeSeries.__post_init__ pairs these
            raise TimeFValidationError(
                f"series {ts.time_series_id!r} carries time offsets but has {type(axis).__name__}"
            )
        time_offsets = to_time_offsets_us(ts.time_offsets_loader().to_numpy(zero_copy_only=False))
        if len(time_offsets) != ts.n_values:
            raise TimeFValidationError(
                f"series {ts.time_series_id!r}: its time_offsets_loader returned {len(time_offsets)} time offsets "
                f"but it declares n_values={ts.n_values}"
            )
        if int(time_offsets[0]) != axis.first_us or int(time_offsets[-1]) != axis.last_us:
            raise TimeFValidationError(
                f"series {ts.time_series_id!r}: its axis claims the stream runs "
                f"{axis.first_us}..{axis.last_us} us but the stream runs "
                f"{int(time_offsets[0])}..{int(time_offsets[-1])} us"
            )
        return pa.array(time_offsets)

    # ---- metadata tables -----------------------------------------------------------------------

    def _control_sink(
        self,
        schema: pa.Schema,
        part_path: Callable[[int], str],
        *,
        dictionary_columns: list[str],
        column_encoding: dict[str, str] | None = None,
    ) -> ShardedTableWriter:
        """Open a control-table sink wired to this writer's staging dir, byte budgets, and compression.

        One place owns that wiring, so the materialized tables and the streamed task write share it.

        Args:
            schema: The Arrow schema for the table.
            part_path: Maps a part index to its relative path.
            dictionary_columns: Columns to dictionary-encode.
            column_encoding: Optional per-column encoding overrides.

        Returns:
            A sink ready to accept rows via :meth:`ShardedTableWriter.add`.
        """
        encoding_kwargs: dict[str, Any] = {
            "dictionary_columns": dictionary_columns,
            "column_encoding": column_encoding,
            "compression": self._compression,
            "data_page_size": self._data_page_size,
        }
        if self._compression_level is not None:
            encoding_kwargs["compression_level"] = self._compression_level
        return ShardedTableWriter(
            schema,
            part_path,
            staging_dir=self._staging_dir,
            control_target_bytes=self._control_shard_target_bytes,
            row_group_target_bytes=self._row_group_target_bytes,
            encoding=encodings.ParquetEncoding(**encoding_kwargs),
        )

    def _write_control_table(
        self,
        rows: Iterable[dict],
        schema: pa.Schema,
        part_path: Callable[[int], str],
        *,
        dictionary_columns: list[str],
        column_encoding: dict[str, str] | None = None,
    ) -> list[str]:
        sink = self._control_sink(
            schema, part_path, dictionary_columns=dictionary_columns, column_encoding=column_encoding
        )
        for row in rows:
            sink.add(row)
        return sink.finish(write_empty_part=True)

    def _write_samples(self) -> None:
        codec = self._codec
        rows: list[dict] = [
            {
                "sample_id": codec.encode("sample_id", sample.sample_id),
                "start_time_us": sample.start_time,
                "time_span": codec.encode_span(sample.time_span),
                "subject_ids": codec.encode_list("subject_id", sample.subject_ids),
                "time_series": [_time_series_struct(ts, codec) for ts in sample.time_series],
                "task_ids": codec.encode_list("task_id", sample.task_ids),
                "annotation_ids": codec.encode_list("annotation_id", [ann.id for ann in sample.annotations]),
            }
            for sample in self._dataset.samples
        ]
        rows.sort(key=lambda r: r["sample_id"])
        self._sample_parts = self._write_control_table(
            iter(rows),
            samples_schema(self._id_types),
            lambda index: part_path(SAMPLES_TEMPLATE, index),
            dictionary_columns=encodings.SAMPLES_DICTIONARY,
        )

    def _annotation_row(self, ann: Annotation) -> dict:
        """Build an annotation's stored row with an empty ``sample_ids`` for the caller to fill.

        Args:
            ann: The annotation to encode.

        Returns:
            The row dict; a sample-carried annotation appends its sample ids, a registered one leaves
            the list empty.
        """
        codec = self._codec
        return {
            "id": codec.encode("annotation_id", ann.id),
            "key": ann.key,
            "value": None if ann.value is None else json.dumps(ann.value),
            "source": ann.source,
            "span": codec.encode_span(ann.span),
            "sample_ids": [],
        }

    def _write_annotations(self) -> None:
        codec = self._codec
        by_id: dict[str, dict] = {}
        for sample in self._dataset.samples:
            for ann in sample.annotations:
                row = by_id.setdefault(ann.id, self._annotation_row(ann))
                row["sample_ids"].append(codec.encode("sample_id", sample.sample_id))
        for ann in self._dataset.registered_annotations:
            # A registered annotation that no sample carries writes with an empty sample_ids: tasks
            # reference it by id, and the reader resolves it from the table, not through a sample.
            by_id.setdefault(ann.id, self._annotation_row(ann))
        rows = sorted(by_id.values(), key=lambda r: r["id"])
        self._annotation_parts = self._write_control_table(
            iter(rows),
            annotations_schema(self._id_types),
            lambda index: part_path(ANNOTATIONS_TEMPLATE, index),
            dictionary_columns=encodings.ANNOTATIONS_DICTIONARY,
        )

    def _write_tasks(self) -> None:
        # Route tasks to a byte-budgeted sink per type, created on first sight of a type, so they reach
        # disk in row-group batches without the whole list ever living in memory. iter_tasks() yields a
        # materialized dataset's tasks and a streamed one's identically, so one path serves both.
        # _task_type_counts feeds the manifest counts, tallied here so the tasks are never iterated twice.
        self._task_files: list[str] = []
        self._task_type_counts: dict[str, int] = {}
        sinks: dict[str, ShardedTableWriter] = {}
        schemas: dict[str, pa.Schema] = {}
        if self._dataset.has_task_stream:
            # Reuse the iterator the id-type peek started: the source is consumed and validated once.
            # The peek already pulled the first task, so chain it back ahead of the rest.
            if self._first_task is None:
                tasks: Iterable[Task] = ()
            else:
                tasks = itertools.chain((self._first_task,), self._task_iter)
        else:
            tasks = self._dataset.iter_tasks()
        for task in tasks:
            task_type_str = str(task.task_type)
            schema = schemas.get(task_type_str)
            if schema is None:
                schema = task_schema(task.task_type, self._id_types)
                schemas[task_type_str] = schema
                sinks[task_type_str] = self._control_sink(
                    schema,
                    lambda index, tt=task_type_str: part_path(TASK_PART_TEMPLATE, index, task_type=tt),
                    dictionary_columns=encodings.task_dictionary(schema),
                )
            sinks[task_type_str].add(_task_row(task, schema, self._codec))
            self._task_type_counts[task_type_str] = self._task_type_counts.get(task_type_str, 0) + 1
        for task_type_str in sorted(sinks):  # stable file order regardless of the type interleaving
            self._task_files.extend(sinks[task_type_str].finish(write_empty_part=True))

    def _write_index(
        self, placements: dict[tuple[str, int], ChunkPlacement], series_to_samples: dict[str, list[str]]
    ) -> None:
        # Emit rows in encoded (sample_id, time_series_id, chunk_idx) order through nested iteration.
        # This keeps the parts globally sorted and avoids materializing the whole index in memory.
        # The reader selects the parts where a probe lands, then bisects each part. This approach
        # requires global order across the split.
        codec = self._codec
        sample_to_series: dict[str, list[str]] = {}
        for time_series_id, sample_ids in series_to_samples.items():
            for sample_id in sample_ids:
                sample_to_series.setdefault(sample_id, []).append(time_series_id)
        chunks_by_series: dict[str, list[int]] = {}
        for time_series_id, chunk_idx in placements:  # noqa: PLE1141 - keys are (id, chunk) tuples
            chunks_by_series.setdefault(time_series_id, []).append(chunk_idx)
        for chunk_idxs in chunks_by_series.values():
            chunk_idxs.sort()

        enc_sid: dict[str, Any] = {sid: codec.encode("sample_id", sid) for sid in sample_to_series}
        enc_tid: dict[str, Any] = {tid: codec.encode("time_series_id", tid) for tid in chunks_by_series}

        def rows() -> Iterable[dict]:
            for sample_id in sorted(sample_to_series, key=lambda s: enc_sid[s]):
                series = sample_to_series[sample_id]
                for time_series_id in sorted(series, key=lambda t: enc_tid[t]):
                    for chunk_idx in chunks_by_series[time_series_id]:
                        placement = placements[time_series_id, chunk_idx]
                        yield {
                            "sample_id": enc_sid[sample_id],
                            "time_series_id": enc_tid[time_series_id],
                            "spec_type": placement.spec_type,
                            "channel": placement.channel,
                            "chunk_idx": chunk_idx,
                            "chunk_file": placement.chunk_file,
                            "chunk_major_idx": placement.data_index.major_idx,
                            "chunk_minor_idx": placement.data_index.minor_idx,
                            "n_values": placement.n_values,
                        }

        self._index_rows = sum(len(chunks_by_series[t]) for series in sample_to_series.values() for t in series)
        self._index_parts = self._write_control_table(
            rows(),
            index_schema(self._id_types),
            lambda index: part_path(INDEX_TEMPLATE, index),
            dictionary_columns=encodings.INDEX_DICTIONARY,
            column_encoding=encodings.INDEX_ENCODING,
        )

    def _write_manifest(self) -> None:
        schema = self._dataset.schema
        if schema is None:  # unreachable: write() already checked, but keeps the type non-optional
            raise RuntimeError("schema was not derived")
        manifest = Manifest(
            dataset_id=self._dataset.metadata.dataset_id,
            metadata=self._dataset.metadata,
            schema=schema,
            counts=self._counts,
            files=ManifestFiles(
                samples=self._file_parts(self._sample_parts),
                annotations=self._file_parts(self._annotation_parts),
                time_series_index=self._file_parts(self._index_parts),
                tasks=self._file_parts(self._task_files),
                time_series=self._file_parts(self._value_files),
            ),
            values_backend=self._values_backend_name,
            value_encoding=self._value_encoding,
            build_env=build_env(),
        )
        (self._staging_dir / MANIFEST_FILE).write_text(manifest.to_json())

    # ---- helpers -------------------------------------------------------------------------------

    def _validate_shared_annotations(self) -> None:
        """Check annotations sharing an id across samples are field-equal, and registered ids are distinct.

        Raises:
            TimeFValidationError: If two annotations share an id but are not equal, or an id is both
                registered and carried by a sample. The reader restores a registered annotation from its
                empty ``sample_ids``, so an id that is also sample-carried writes non-empty and is lost on
                read; reject it here instead of silently dropping it.
        """
        sample_ann_ids = {ann.id for sample in self._dataset.samples for ann in sample.annotations}
        overlap = sorted(sample_ann_ids & {ann.id for ann in self._dataset.registered_annotations})
        if overlap:
            raise TimeFValidationError(
                f"annotation id(s) {overlap} are both registered and carried by a sample; a registered "
                f"annotation must be one no sample carries"
            )
        seen: dict[str, object] = {}
        sample_annotations = (ann for sample in self._dataset.samples for ann in sample.annotations)
        for ann in (*sample_annotations, *self._dataset.registered_annotations):
            if ann.id in seen and seen[ann.id] != ann:
                raise TimeFValidationError(
                    f"annotation id {ann.id!r} is shared across samples but instances are not equal"
                )
            seen[ann.id] = ann

    def _build_counts(
        self,
        unique_series: list[TimeSeries],
        placements: dict[tuple[str, int], ChunkPlacement],
    ) -> ManifestCounts:
        tasks_by_type = self._task_type_counts  # tallied while streaming/writing the tasks
        specs_by_type: dict[str, int] = {}
        for ts in unique_series:
            specs_by_type[ts.spec.spec_type] = specs_by_type.get(ts.spec.spec_type, 0) + 1
        annotation_ids = {ann.id for sample in self._dataset.samples for ann in sample.annotations}
        annotation_ids |= {ann.id for ann in self._dataset.registered_annotations}
        return ManifestCounts(
            samples=len(self._dataset.samples),
            annotations=len(annotation_ids),
            registered_annotations=len(self._dataset.registered_annotations),
            tasks=tasks_by_type,
            time_series_chunks=len(placements),
            time_series_index_rows=self._index_rows,
            time_series_specs=specs_by_type,
        )

    def _file_parts(self, rels: Iterable[str]) -> tuple[FilePart, ...]:
        """Describe each staged artifact by its path, checksum, and size.

        Args:
            rels: The version-relative paths of the artifacts to describe.

        Returns:
            One :class:`FilePart` per path, in the given order.
        """
        return tuple(self._file_part(rel) for rel in rels)

    def _file_part(self, rel: str) -> FilePart:
        """Describe one staged file: its path, ``sha256:`` checksum, and byte size.

        Args:
            rel: The file's version-relative path.

        Returns:
            The file's descriptor.
        """
        path = self._staging_dir / rel
        return FilePart(path=rel, checksum=file_checksum(path), size=path.stat().st_size)

    def _emit(self, stage: ProgressStage, completed: int, total: int | None) -> None:
        if self._progress_cb is not None:
            self._progress_cb(WriteProgressEvent(stage=stage, completed=completed, total=total))


def _series_identity(ts: TimeSeries) -> tuple:
    """Return the fields that must agree for two series to be the same channel.

    This compares the descriptive fields that the writer persists, not the values. The ``loader`` is a
    callable, so two equal series built separately compare unequal. Reading every shared series
    only to compare it defeats the lazy read path on the largest datasets.

    Args:
        ts: The series to describe.

    Returns:
        The identifying fields, suitable for equality comparison and for error messages.
    """
    return (ts.spec.spec_type, ts.channel, ts.time_axis, ts.source_id, ts.n_values)


def _axis_columns(axis: TimeAxis) -> dict:
    """Return the shape-specific axis columns. Set every column the shape does not use to null.

    The dispatch is positive and ends in :func:`~typing.assert_never`. A new axis shape fails here at
    type-check time instead of writing a row of nulls under another shape's tag.

    Args:
        axis: The series' time axis.

    Returns:
        The axis columns of the series struct.
    """
    empty = {
        "period_numerator_us": None,
        "period_denominator": None,
        "start_index": None,
        "first_time_offset_us": None,
        "last_time_offset_us": None,
    }
    if isinstance(axis, RegularAxis):
        return empty | {
            "period_numerator_us": axis.period_us.numerator,
            "period_denominator": axis.period_us.denominator,
            "start_index": axis.start_index,
        }
    if isinstance(axis, IrregularAxis):
        return empty | {"first_time_offset_us": axis.first_us, "last_time_offset_us": axis.last_us}
    if isinstance(axis, OrdinalAxis):
        return empty
    assert_never(axis)


def _time_series_struct(ts: TimeSeries, codec: IdCodec) -> dict:
    return {
        "spec_type": ts.spec.spec_type,
        "channel": ts.channel,
        "source_id": codec.encode("source_id", ts.source_id),
        "time_series_id": codec.encode("time_series_id", ts.time_series_id),
        "axis_type": str(ts.time_axis.axis_type),
        **_axis_columns(ts.time_axis),
        "n_values": ts.n_values,
    }


def _task_row(task: Task, schema: pa.Schema, codec: IdCodec) -> dict:
    refs = type(task).refs
    row: dict = {
        "id": codec.encode("task_id", task.id),
        "sample_ids": codec.encode_list("sample_id", task.sample_ids),
        "from_task_ids": codec.encode_list("task_id", task.from_task_ids),
        "prompt": task.prompt,
        "scope": codec.encode_span(task.scope),
        "input_annotation_ids": codec.encode_list("annotation_id", task.input_annotation_ids),
        "target_annotation_ids": codec.encode_list("annotation_id", task.target_annotation_ids),
        "rationale": task.rationale,
    }
    for name in schema.names:
        if name in TASK_COMMON_NAMES:
            continue
        value = getattr(task, name)
        if isinstance(value, StrEnum):  # a StrEnum payload (for example localization mode) stores as its value
            value = str(value)
        value = list(value) if isinstance(value, tuple) else value
        row[name] = codec.encode_payload(refs, name, value)
    return row


def _ordered_unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))
