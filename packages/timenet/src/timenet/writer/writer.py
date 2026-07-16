"""``TimeFWriter``: serialize a :class:`~timenet.dataset.TimeFDataset` to the TimeF layout on disk.

The writer streams shards (one row group per ``row_group_target_bytes`` of values, splitting series at
``chunk_max_bytes`` and rotating shards at ``shard_target_bytes``) so peak memory stays near one row
group. Everything is staged in a temporary directory and published with a single atomic rename;
``manifest.json`` present in the version directory marks a committed version.
"""

from collections.abc import Callable, Iterable
import hashlib
import json
from pathlib import Path
import shutil
import types as _types
from typing import cast
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.errors import TimeFValidationError
from timenet.manifest import Manifest, ManifestCounts, ManifestFiles
from timenet.types import Task, annotation_type_of
from timenet.types.ids import id_to_bytes, is_canonical_uuid
from timenet.values_backends import PARQUET_VALUES_BACKEND, SUPPORTED_VALUES_BACKENDS
from timenet.writer import encodings
from timenet.writer.constants import (
    ANNOTATIONS_FILE,
    DEFAULT_CHUNK_MAX_BYTES,
    DEFAULT_COMPRESSION,
    DEFAULT_COMPRESSION_LEVEL,
    DEFAULT_ROW_GROUP_TARGET_BYTES,
    DEFAULT_SHARD_TARGET_BYTES,
    INDEX_FILE,
    MANIFEST_FILE,
    SAMPLES_FILE,
    TASK_PART_TEMPLATE,
)
from timenet.writer.progress import ProgressStage, WriteProgressEvent
from timenet.writer.schemas import (
    LOGICAL_IDS,
    TASK_COMMON_NAMES,
    TASK_PAYLOAD_ID_COLUMNS,
    UUID16,
    IdTypes,
    annotations_schema,
    index_schema,
    samples_schema,
    task_schema,
)
from timenet.writer.values import ChunkPlacement, ParquetValuesConfig, ZarrValuesConfig, make_values_backend


_CHECKSUM_BLOCK_BYTES = 1 << 20  # hash files a block at a time, not all-in-memory


class TimeFWriter:
    """Context manager that serializes a dataset into the TimeF format and commits it atomically."""

    def __init__(
        self,
        root: Path,
        dataset: TimeFDataset,
        *,
        shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
        row_group_target_bytes: int = DEFAULT_ROW_GROUP_TARGET_BYTES,
        chunk_max_bytes: int = DEFAULT_CHUNK_MAX_BYTES,
        compression: str = DEFAULT_COMPRESSION,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
        values_backend: str = PARQUET_VALUES_BACKEND,
        progress_cb: Callable[[WriteProgressEvent], None] | None = None,
        derived_from: dict[str, str] | None = None,
    ) -> None:
        """Configure the writer.

        Args:
            root: Parent directory; the writer creates ``<root>/<dataset_id>/<version>/``.
            dataset: The populated dataset (its ``schema`` must be derived before writing).
            shard_target_bytes: Rotate to a new shard once a shard's buffered values exceed this.
            row_group_target_bytes: Flush a row group once buffered values exceed this.
            chunk_max_bytes: Split a series into chunks no larger than this.
            compression: Parquet values codec.
            compression_level: Pinned level (applied for zstd) for reproducible output.
            values_backend: Storage backend for the values plane.
            progress_cb: Optional callback invoked with each :class:`WriteProgressEvent`.
            derived_from: Lineage recorded in the manifest when this version is a copy-on-write edit of
                another (e.g. ``{"dataset_version": "1.0.0", "op": "remove_samples"}``).

        Raises:
            TimeFValidationError: If ``dataset.metadata.dataset_id`` is empty.
            ValueError: If ``values_backend`` is unsupported.
        """
        if not dataset.metadata.dataset_id:
            raise TimeFValidationError("dataset_id must be non-empty")
        if values_backend not in SUPPORTED_VALUES_BACKENDS:
            raise ValueError(
                f"unknown values_backend {values_backend!r}; supported: {sorted(SUPPORTED_VALUES_BACKENDS)}"
            )
        self._root = Path(root)
        self._dataset = dataset
        self._derived_from = derived_from
        self._shard_target_bytes = shard_target_bytes
        self._row_group_target_bytes = row_group_target_bytes
        self._chunk_max_bytes = chunk_max_bytes
        self._compression = compression
        self._compression_level = compression_level
        self._values_backend_name = values_backend
        self._progress_cb = progress_cb

        version = str(dataset.metadata.dataset_version)
        self._final_dir = self._root / dataset.metadata.dataset_id / version
        self._staging_dir = self._root / dataset.metadata.dataset_id / f"{version}.tmp-{uuid.uuid4().hex}"

        self._written = False

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

        A hard kill (SIGKILL/OOM) never reaches :meth:`abort`, so its staging directory lingers. Clear
        any such sibling for this version before writing a fresh one. Concurrent writes of the same
        version are not supported.
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
        """Serialize every artifact except the manifest into the staging directory.

        Raises:
            TimeFValidationError: If the schema was not derived, a shared annotation id is not
                field-equal across samples, or a series array violates the per-series contract.
        """
        if self._dataset.schema is None:
            raise TimeFValidationError("call dataset.derive_schema() before writing")
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
            shutil.rmtree(self._final_dir)  # a manifest-less partial dir; readers ignore it anyway
        self._final_dir.parent.mkdir(parents=True, exist_ok=True)
        self._staging_dir.replace(self._final_dir)
        self._emit(ProgressStage.COMMIT, 1, 1)

    def abort(self) -> None:
        """Delete the staging directory. Safe to call more than once."""
        shutil.rmtree(self._staging_dir, ignore_errors=True)

    # ---- id storage ----------------------------------------------------------------------------

    def _resolve_id_types(self) -> None:
        """Pick per-logical-id storage: ``binary(16)`` when every value is a canonical UUID, else string.

        Stores the resolved Arrow types, the set of ``uuid16`` logical ids, the manifest ``id_encoding``
        map (only the uuid16 entries; an absent entry means string), and the shard schema.
        """
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
        for task in self._dataset.tasks:
            values["task_id"].append(task.id)

        id_types: IdTypes = {}
        for name in LOGICAL_IDS:
            vals = values[name]
            is_uuid16 = bool(vals) and all(is_canonical_uuid(v) for v in vals)
            id_types[name] = UUID16 if is_uuid16 else pa.string()
        self._id_types = id_types
        self._uuid16 = {name for name in LOGICAL_IDS if id_types[name] == UUID16}
        self._id_encoding = dict.fromkeys(self._uuid16, "uuid16")

    # ---- values ---------------------------------------------------------------------------------

    def _dedupe_series(self) -> tuple[list[TimeSeries], dict[str, list[str]]]:
        """Return unique series (sorted for stable output) and the series-id -> sample-ids map.

        Returns:
            The sorted unique series and a mapping from ``time_series_id`` to the ids of the samples
            that reference it (first-seen order).
        """
        unique: dict[str, TimeSeries] = {}
        series_to_samples: dict[str, list[str]] = {}
        seen_pairs: set[tuple[str, str]] = set()
        for sample in self._dataset.samples:
            for ts in sample.time_series:
                unique.setdefault(ts.time_series_id, ts)
                pair = (ts.time_series_id, sample.sample_id)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    series_to_samples.setdefault(ts.time_series_id, []).append(sample.sample_id)
        ordered = sorted(unique.values(), key=lambda ts: (ts.spec.spec_type, ts.channel, ts.time_series_id))
        return ordered, series_to_samples

    def _write_values(self, unique_series: list[TimeSeries]) -> dict[tuple[str, int], ChunkPlacement]:
        """Write all series' values through the configured backend and return chunk placements.

        Args:
            unique_series: The deduped, sorted series to serialize.

        Returns:
            A mapping from ``(time_series_id, chunk_idx)`` to its on-disk :class:`ChunkPlacement`.
        """
        if self._values_backend_name == PARQUET_VALUES_BACKEND:
            config = ParquetValuesConfig(
                staging_dir=self._staging_dir,
                id_types=self._id_types,
                tsid_uuid16="time_series_id" in self._uuid16,
                shard_target_bytes=self._shard_target_bytes,
                row_group_target_bytes=self._row_group_target_bytes,
                chunk_max_bytes=self._chunk_max_bytes,
                compression=self._compression,
                compression_level=self._compression_level,
            )
        else:
            config = ZarrValuesConfig(
                staging_dir=self._staging_dir,
                shard_target_bytes=self._shard_target_bytes,
                chunk_max_bytes=self._chunk_max_bytes,
                compression=self._compression,
                compression_level=self._compression_level,
            )
        values_backend = make_values_backend(config)
        result = values_backend.write_series(
            unique_series,
            read_and_validate=self._read_and_validate,
            on_series_done=lambda completed, total: self._emit(ProgressStage.TIME_SERIES, completed, total),
            on_file_done=lambda count: self._emit(ProgressStage.SHARD_FINALIZED, count, None),
        )
        self._value_files = result.files
        return result.placements

    def _read_and_validate(self, ts: TimeSeries) -> pa.Array:
        """Read a series' values and enforce the per-series array contract.

        Args:
            ts: The series to read.

        Returns:
            The validated float32 values array.

        Raises:
            TimeFValidationError: If the array is not a non-empty, finite float32 array, or its length
                disagrees with a set window.
        """
        values = ts.to_arrow()
        expected_type = pa.from_numpy_dtype(np.dtype(ts.spec.dtype))
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
        if len(values) == 0:
            raise TimeFValidationError(f"series {ts.time_series_id!r} loaded an empty array")
        as_numpy = (
            values.to_numpy_ndarray()
            if isinstance(values, pa.FixedShapeTensorArray)
            else values.to_numpy(zero_copy_only=False)
        )
        if np.issubdtype(as_numpy.dtype, np.inexact) and not np.isfinite(as_numpy).all():
            raise TimeFValidationError(f"series {ts.time_series_id!r} has non-finite values")
        if ts.t_end_s is not None:
            expected = round((ts.t_end_s - ts.t_start_s) * ts.sampling_rate_hz)
            if len(values) != expected:
                raise TimeFValidationError(
                    f"series {ts.time_series_id!r}: len(values)={len(values)} != expected {expected} "
                    f"from window/sampling rate"
                )
        return values

    # ---- metadata tables -----------------------------------------------------------------------

    def _write_samples(self) -> None:
        u = self._uuid16
        rows = []
        for sample in sorted(self._dataset.samples, key=lambda s: s.sample_id):
            rows.append(
                {
                    "sample_id": _enc_id(sample.sample_id, "sample_id" in u),
                    "view": str(sample.view),
                    "subject_ids": _enc_ids(list(sample.subject_ids), "subject_id" in u),
                    "source_ids": _enc_ids(
                        _ordered_unique(ts.source_id for ts in sample.time_series if ts.source_id),
                        "source_id" in u,
                    ),
                    "time_series": [_time_series_struct(ts, u) for ts in sample.time_series],
                    "task_ids": _enc_ids(list(sample.task_ids), "task_id" in u),
                    "annotation_ids": _enc_ids([ann.id for ann in sample.annotations], "annotation_id" in u),
                }
            )
        self._write_table(
            rows, samples_schema(self._id_types), SAMPLES_FILE, dictionary_columns=encodings.SAMPLES_DICTIONARY
        )

    def _write_annotations(self) -> None:
        u = self._uuid16
        aid16, tsid16, sid16 = "annotation_id" in u, "time_series_id" in u, "sample_id" in u
        by_id: dict[str, dict] = {}
        for sample in self._dataset.samples:
            for ann in sample.annotations:
                series_ids = getattr(ann, "time_series_ids", None)
                row = by_id.setdefault(
                    ann.id,
                    {
                        "id": _enc_id(ann.id, aid16),
                        "key": ann.key,
                        "annotation_type": str(annotation_type_of(ann)),
                        "value": None if ann.value is None else json.dumps(ann.value),
                        "start_time_s": getattr(ann, "start_time_s", None),
                        "end_time_s": getattr(ann, "end_time_s", None),
                        "time_series_ids": _enc_ids(list(series_ids), tsid16) if series_ids is not None else None,
                        "sample_ids": [],
                    },
                )
                row["sample_ids"].append(_enc_id(sample.sample_id, sid16))
        rows = sorted(by_id.values(), key=lambda r: (r["annotation_type"], r["key"], r["id"]))
        self._write_table(
            rows,
            annotations_schema(self._id_types),
            ANNOTATIONS_FILE,
            dictionary_columns=encodings.ANNOTATIONS_DICTIONARY,
        )

    def _write_tasks(self) -> None:
        self._task_files: list[str] = []
        by_type: dict[str, list[Task]] = {}
        for task in self._dataset.tasks:
            by_type.setdefault(str(task.task_type), []).append(task)
        for task_type_str, tasks in sorted(by_type.items()):
            schema = task_schema(tasks[0].task_type, self._id_types)
            rows = [_task_row(task, schema, self._uuid16) for task in tasks]
            rel = TASK_PART_TEMPLATE.format(task_type=task_type_str)
            (self._staging_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            self._write_table(rows, schema, rel, dictionary_columns=encodings.task_dictionary(schema))
            self._task_files.append(rel)

    def _write_index(
        self, placements: dict[tuple[str, int], ChunkPlacement], series_to_samples: dict[str, list[str]]
    ) -> None:
        sid16, tsid16 = "sample_id" in self._uuid16, "time_series_id" in self._uuid16
        rows = []
        for (time_series_id, chunk_idx), placement in placements.items():
            for sample_id in series_to_samples.get(time_series_id, ()):
                rows.append(
                    {
                        "sample_id": _enc_id(sample_id, sid16),
                        "time_series_id": _enc_id(time_series_id, tsid16),
                        "spec_type": placement.spec_type,
                        "channel": placement.channel,
                        "chunk_idx": chunk_idx,
                        "chunk_file": placement.chunk_file,
                        "chunk_offset0": placement.chunk_offset0,
                        "chunk_offset1": placement.chunk_offset1,
                        "t_start_s": placement.t_start_s,
                        "t_end_s": placement.t_start_s + placement.n_values / placement.sampling_rate_hz,
                        "n_values": placement.n_values,
                    }
                )
        rows.sort(key=lambda r: (r["sample_id"], r["time_series_id"], r["chunk_idx"]))
        self._index_rows = len(rows)
        self._write_table(
            rows,
            index_schema(self._id_types),
            INDEX_FILE,
            dictionary_columns=encodings.INDEX_DICTIONARY,
            column_encoding=encodings.INDEX_ENCODING,
        )

    def _write_manifest(self) -> None:
        schema = self._dataset.schema
        if schema is None:  # unreachable: write() already checked, but keeps the type non-optional
            raise RuntimeError("schema was not derived")
        checksums = self._checksums()
        manifest = Manifest(
            dataset_id=self._dataset.metadata.dataset_id,
            metadata=self._dataset.metadata,
            schema=schema,
            counts=self._counts,
            files=ManifestFiles(
                samples=(SAMPLES_FILE,),
                annotations=(ANNOTATIONS_FILE,),
                time_series_index=(INDEX_FILE,),
                tasks=tuple(self._task_files),
                time_series=tuple(self._value_files),
            ),
            checksums=checksums,
            id_encoding=self._id_encoding,
            values_backend=self._values_backend_name,
            derived_from=self._derived_from,
        )
        (self._staging_dir / MANIFEST_FILE).write_text(manifest.to_json())

    # ---- helpers -------------------------------------------------------------------------------

    def _validate_shared_annotations(self) -> None:
        """Check annotations sharing an id across samples are field-equal.

        Raises:
            TimeFValidationError: If two annotations share an id but are not equal.
        """
        seen: dict[str, object] = {}
        for sample in self._dataset.samples:
            for ann in sample.annotations:
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
        tasks_by_type: dict[str, int] = {}
        for task in self._dataset.tasks:
            tasks_by_type[str(task.task_type)] = tasks_by_type.get(str(task.task_type), 0) + 1
        specs_by_type: dict[str, int] = {}
        for ts in unique_series:
            specs_by_type[ts.spec.spec_type] = specs_by_type.get(ts.spec.spec_type, 0) + 1
        annotation_ids = {ann.id for sample in self._dataset.samples for ann in sample.annotations}
        return ManifestCounts(
            samples=len(self._dataset.samples),
            annotations=len(annotation_ids),
            tasks=tasks_by_type,
            time_series_chunks=len(placements),
            time_series_index_rows=self._index_rows,
            time_series_specs=specs_by_type,
        )

    def _checksums(self) -> dict[str, str]:
        """Checksum every artifact staged so far (metadata parquet plus the backend's value files).

        Hashes all files under the staging directory except the not-yet-written manifest, so the map
        covers Parquet shards and a Zarr store's chunk/metadata files alike.

        Returns:
            A ``relpath -> "sha256:..."`` map.
        """
        checksums: dict[str, str] = {}
        for path in sorted(self._staging_dir.rglob("*")):
            if not path.is_file() or path.name == MANIFEST_FILE:
                continue
            rel = path.relative_to(self._staging_dir).as_posix()
            checksums[rel] = "sha256:" + _sha256_hex(path)
        return checksums

    def _write_table(
        self,
        rows: list[dict],
        schema: pa.Schema,
        rel_path: str,
        *,
        dictionary_columns: list[str],
        column_encoding: dict[str, str] | None = None,
    ) -> None:
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(
            table,
            self._staging_dir / rel_path,
            **encodings.parquet_kwargs(
                dictionary_columns=dictionary_columns,
                column_encoding=column_encoding,
                compression=self._compression,
                compression_level=self._compression_level,
            ),
        )

    def _emit(self, stage: ProgressStage, completed: int, total: int | None) -> None:
        if self._progress_cb is not None:
            self._progress_cb(WriteProgressEvent(stage=stage, completed=completed, total=total))


def _sha256_hex(path: Path) -> str:
    """Return the SHA-256 of a file, hashing it a block at a time.

    Args:
        path: The file to hash.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHECKSUM_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _time_series_struct(ts: TimeSeries, uuid16: set[str]) -> dict:
    return {
        "spec_type": ts.spec.spec_type,
        "channel": ts.channel,
        "source_id": _enc_id(ts.source_id, "source_id" in uuid16),
        "time_series_id": _enc_id(ts.time_series_id, "time_series_id" in uuid16),
        "sampling_rate_hz": ts.sampling_rate_hz,
        "t_start_s": ts.t_start_s,
        "t_end_s": ts.t_end_s,
    }


def _task_row(task: Task, schema: pa.Schema, uuid16: set[str]) -> dict:
    row: dict = {
        "id": _enc_id(task.id, "task_id" in uuid16),
        "sample_ids": _enc_ids(list(task.sample_ids), "sample_id" in uuid16),
        "from_task_ids": _enc_ids(list(task.from_task_ids), "task_id" in uuid16),
    }
    for name in schema.names:
        if name in TASK_COMMON_NAMES:
            continue
        value = getattr(task, name)
        value = list(value) if isinstance(value, tuple) else value
        logical = TASK_PAYLOAD_ID_COLUMNS.get(name)
        if logical is not None and logical in uuid16 and value is not None:
            value = _enc_ids(value, True) if isinstance(value, list) else _enc_id(value, True)
        row[name] = value
    return row


def _enc_id(value: object, uuid16: bool) -> object:
    """Return a scalar id as 16 raw bytes when its column is uuid16, else unchanged.

    Args:
        value: The id string (or ``None``).
        uuid16: Whether the id's column is stored as ``binary(16)``.

    Returns:
        The 16-byte form for a uuid16 column, else the value unchanged.
    """
    if uuid16 and value is not None:
        return id_to_bytes(cast(str, value))
    return value


def _enc_ids(values: Iterable[object], uuid16: bool) -> list:
    """Return a list of ids encoded element-wise via :func:`_enc_id`.

    Args:
        values: The id strings.
        uuid16: Whether the ids' column is stored as ``binary(16)``.

    Returns:
        The encoded list.
    """
    return [_enc_id(value, uuid16) for value in values]


def _ordered_unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))
