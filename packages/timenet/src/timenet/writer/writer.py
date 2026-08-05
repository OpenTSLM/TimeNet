"""``TimeFWriter``: serialize a :class:`~timenet.dataset.TimeFDataset` to the TimeF layout on disk.

The writer keeps the Parquet control plane fixed and delegates series values to a selected Parquet or
Zarr backend. Backend buffers are bounded by the configured chunk, row-group, and shard targets.
Everything is staged in a temporary directory and published with a single atomic rename;
``manifest.json`` present in the version directory marks a committed version.
"""

from collections.abc import Callable, Iterable
from enum import StrEnum
import json
from pathlib import Path
import shutil
import types as _types
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.errors import TimeFValidationError
from timenet.format.checksums import file_checksum
from timenet.format.constants import (
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
from timenet.manifest import Manifest, ManifestCounts, ManifestFiles
from timenet.types import Task
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


class TimeFWriter:
    """Context manager that serializes a dataset into the TimeF format and commits it atomically."""

    def __init__(  # noqa: PLR0913
        self,
        root: Path,
        dataset: TimeFDataset,
        *,
        shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
        row_group_target_bytes: int = DEFAULT_ROW_GROUP_TARGET_BYTES,
        chunk_max_bytes: int = DEFAULT_CHUNK_MAX_BYTES,
        compression: str = DEFAULT_COMPRESSION,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
        values_backend: str = ValuesBackend.PARQUET,
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
            compression: Values codec (Parquet codec or Zarr Blosc inner codec).
            compression_level: Pinned level (applied for zstd) for reproducible output.
            values_backend: Storage backend for the values plane.
            progress_cb: Optional callback invoked with each :class:`WriteProgressEvent`.
            derived_from: Lineage recorded in the manifest when this version is a copy-on-write edit of
                another (e.g. ``{"dataset_version": "1.0.0", "op": "remove_samples"}``).

        Raises:
            TimeFValidationError: If ``dataset.metadata.dataset_id`` is empty or ``values_backend`` is
                unsupported.
        """
        if not dataset.metadata.dataset_id:
            raise TimeFValidationError("dataset_id must be non-empty")
        if values_backend not in SUPPORTED_VALUES_BACKENDS:
            raise TimeFValidationError(
                f"unknown values_backend {values_backend!r}; supported: {', '.join(sorted(SUPPORTED_VALUES_BACKENDS))}"
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
            # Only reachable when the caller pre-created the target, or a previous run died between
            # this rmtree and the replace() below: run_pipeline returns early on a committed version
            # and drops it itself on --force, so a *committed* dataset is never deleted here.
            shutil.rmtree(self._final_dir)
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
        map (only the uuid16 entries; an absent entry means string), and the shared codec.
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
        self._codec = IdCodec.from_uuid16(self._uuid16)

    # ---- values --------------------------------------------------------------------------------

    def _dedupe_series(self) -> tuple[list[TimeSeries], dict[str, list[str]]]:
        """Return unique series (sorted for stable output) and the series-id -> sample-ids map.

        Sharing one series across samples is the supported dedupe path, but two *different* series
        claiming one ``time_series_id`` is a contradiction: only one can be written, so the other's
        samples would silently read back the wrong data. Series reached under the same id must
        therefore describe the same channel, and disagreement is rejected rather than resolved by
        first-wins.

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
        ordered = sorted(unique.values(), key=lambda ts: (ts.spec.spec_type, ts.channel, ts.time_series_id))
        return ordered, series_to_samples

    def _write_values(self, unique_series: list[TimeSeries]) -> dict[tuple[str, int], ChunkPlacement]:
        """Write all series' values through the configured backend.

        Args:
            unique_series: The deduped, sorted series to serialize.

        Returns:
            A mapping from ``(time_series_id, chunk_idx)`` to its on-disk placement.
        """
        if self._values_backend_name == ValuesBackend.PARQUET:
            config = ParquetValuesConfig(
                staging_dir=self._staging_dir,
                id_types=self._id_types,
                codec=self._codec,
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

    def _read_and_validate(self, ts: TimeSeries) -> pa.Array:  # noqa: PLR6301
        """Read a series' values and enforce the per-series array contract.

        Args:
            ts: The series to read.

        Returns:
            The validated values in the spec's canonical Arrow representation.

        Raises:
            TimeFValidationError: If the array is empty, disagrees with the spec's dtype/shape, has
                non-finite inexact values, or its length disagrees with a set window.
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
        codec = self._codec
        rows = []
        for sample in sorted(self._dataset.samples, key=lambda s: s.sample_id):
            rows.append(
                {
                    "sample_id": codec.encode("sample_id", sample.sample_id),
                    "view": str(sample.view),
                    "start_time_us": sample.start_time,
                    "subject_ids": codec.encode_list("subject_id", sample.subject_ids),
                    "time_series": [_time_series_struct(ts, codec) for ts in sample.time_series],
                    "task_ids": codec.encode_list("task_id", sample.task_ids),
                    "annotation_ids": codec.encode_list("annotation_id", [ann.id for ann in sample.annotations]),
                }
            )
        self._write_table(
            rows, samples_schema(self._id_types), SAMPLES_FILE, dictionary_columns=encodings.SAMPLES_DICTIONARY
        )

    def _write_annotations(self) -> None:
        codec = self._codec
        by_id: dict[str, dict] = {}
        for sample in self._dataset.samples:
            for ann in sample.annotations:
                row = by_id.setdefault(
                    ann.id,
                    {
                        "id": codec.encode("annotation_id", ann.id),
                        "key": ann.key,
                        "value": None if ann.value is None else json.dumps(ann.value),
                        "span": codec.encode_span(ann.span),
                        "sample_ids": [],
                    },
                )
                row["sample_ids"].append(codec.encode("sample_id", sample.sample_id))
        rows = sorted(by_id.values(), key=lambda r: (r["key"], r["id"]))
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
            rows = [_task_row(task, schema, self._codec) for task in tasks]
            rel = TASK_PART_TEMPLATE.format(task_type=task_type_str)
            (self._staging_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            self._write_table(rows, schema, rel, dictionary_columns=encodings.task_dictionary(schema))
            self._task_files.append(rel)

    def _write_index(
        self, placements: dict[tuple[str, int], ChunkPlacement], series_to_samples: dict[str, list[str]]
    ) -> None:
        codec = self._codec
        rows = []
        for (time_series_id, chunk_idx), placement in placements.items():
            for sample_id in series_to_samples.get(time_series_id, ()):
                rows.append(
                    {
                        "sample_id": codec.encode("sample_id", sample_id),
                        "time_series_id": codec.encode("time_series_id", time_series_id),
                        "spec_type": placement.spec_type,
                        "channel": placement.channel,
                        "chunk_idx": chunk_idx,
                        "chunk_file": placement.chunk_file,
                        "chunk_major_idx": placement.data_index.major_idx,
                        "chunk_minor_idx": placement.data_index.minor_idx,
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
        """Checksum every staged artifact except the not-yet-written manifest.

        Returns:
            A mapping from version-relative file paths to prefixed SHA-256 digests.
        """
        checksums: dict[str, str] = {}
        for path in sorted(self._staging_dir.rglob("*")):
            if not path.is_file() or path.name == MANIFEST_FILE:
                continue
            rel = path.relative_to(self._staging_dir).as_posix()
            checksums[rel] = file_checksum(path)
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


def _series_identity(ts: TimeSeries) -> tuple:
    """Return the fields that must agree for two series to be the same channel.

    Compares the descriptive fields the writer persists, not the values: ``loader`` is a callable
    (so two equal series built separately would compare unequal), and materializing every shared
    series purely to compare it would defeat the lazy read path on exactly the largest datasets.

    Args:
        ts: The series to describe.

    Returns:
        The identifying fields, suitable for equality comparison and for error messages.
    """
    return (ts.spec.spec_type, ts.channel, ts.sampling_rate_hz, ts.source_id, ts.t_start_s, ts.t_end_s)


def _time_series_struct(ts: TimeSeries, codec: IdCodec) -> dict:
    return {
        "spec_type": ts.spec.spec_type,
        "channel": ts.channel,
        "source_id": codec.encode("source_id", ts.source_id),
        "time_series_id": codec.encode("time_series_id", ts.time_series_id),
        "sampling_rate_hz": ts.sampling_rate_hz,
        "t_start_s": ts.t_start_s,
        "t_end_s": ts.t_end_s,
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
        if isinstance(value, StrEnum):  # a StrEnum payload (e.g. localization mode) stores as its value
            value = str(value)
        value = list(value) if isinstance(value, tuple) else value
        row[name] = codec.encode_payload(refs, name, value)
    return row


def _ordered_unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))
