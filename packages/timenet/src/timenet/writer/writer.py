"""Write a TimeF DuckDB control plane and a sharded Parquet or Zarr values plane."""

from collections.abc import Callable, Iterable
from pathlib import Path
import shutil
import types as _types
from typing import Any
import uuid

import numpy as np
import pyarrow as pa

from timenet.dataset import Signal, TimeFDataset
from timenet.dataset.axis import IrregularAxis, to_time_offsets_us
from timenet.dataset.time_series import _validate_enum_values
from timenet.errors import TimeFValidationError
from timenet.format.checksums import file_checksum
from timenet.format.constants import (
    DEFAULT_CHUNK_MAX_BYTES,
    DEFAULT_COMPRESSION,
    DEFAULT_ROW_GROUP_TARGET_BYTES,
    DEFAULT_SHARD_TARGET_BYTES,
    MANIFEST_FILE,
)
from timenet.format.control_writer import DuckDBControlWriter
from timenet.format.duckdb import CONTROL_FILE, connect_control
from timenet.format.schemas import (
    LOGICAL_IDS,
    UUID16,
    IdCodec,
    IdTypes,
)
from timenet.manifest import FileGroup, FileKind, FilePart, Manifest, ManifestCounts, ManifestFiles
from timenet.manifest.files import CONTROL_BACKEND
from timenet.provenance import build_env
from timenet.types.ids import is_canonical_uuid
from timenet.values_backends import SUPPORTED_VALUES_BACKENDS, ValuesBackend
from timenet.values_backends.writer import (
    ChunkPlacement,
    ParquetValuesConfig,
    ZarrValuesConfig,
    make_values_backend,
)
from timenet.writer.progress import ProgressStage, WriteProgressEvent
from timenet.writer.value_encoding import AUTO, SUPPORTED_VALUE_ENCODINGS, ValueEncoding


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

        unique_series, _series_to_records = self._dedupe_series()
        placements = self._write_values(unique_series)
        tasks = (
            self._dataset.iter_streamed_tasks_validated()
            if self._dataset.has_task_stream
            else self._dataset.iter_tasks()
        )
        self._task_type_counts = DuckDBControlWriter(self._staging_dir / CONTROL_FILE).write_hierarchy(
            self._dataset,
            placements,
            tasks=tasks,
        )
        self._control_file = CONTROL_FILE
        self._counts = self._build_counts()
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
        for record in self._dataset.records:
            values["record_id"].append(record.record_id)
            values["subject_id"].extend(record.subject_ids)
            for ts in record.signals:
                values["time_series_id"].append(ts.id)
                if ts.source_id is not None:
                    values["source_id"].append(ts.source_id)
            for ann in record.annotations:
                values["annotation_id"].append(ann.id)
        for ann in self._dataset.registered_annotations:  # task-referenced, carried by no record
            values["annotation_id"].append(ann.id)
        if not self._dataset.has_task_stream:
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

    def _dedupe_series(self) -> tuple[list[Signal], dict[str, list[str]]]:
        """Return unique series (sorted for stable output) and the series-id -> record-ids map.

        Sharing one series across records is the supported dedupe path. Two different series that
        claim one ``time_series_id`` is a contradiction. The writer can write only one of them, so
        the other's records read back the wrong data. Two series that share an id must describe the
        same signal. The writer rejects a disagreement instead of keeping the first series.

        Returns:
            The sorted unique series and a mapping from ``time_series_id`` to the ids of the records
            that reference it (first-seen order).

        Raises:
            TimeFValidationError: If two series share a ``time_series_id`` but describe different
                signals.
        """
        unique: dict[str, Signal] = {}
        series_to_records: dict[str, list[str]] = {}
        seen_pairs: set[tuple[str, str]] = set()
        for record in self._dataset.records:
            for ts in record.signals:
                existing = unique.get(ts.id)
                if existing is None:
                    unique[ts.id] = ts
                elif existing is not ts and _series_identity(existing) != _series_identity(ts):
                    raise TimeFValidationError(
                        f"time_series_id {ts.id!r} is claimed by two different series: "
                        f"{_series_identity(existing)} and {_series_identity(ts)}; ids must be unique "
                        f"per signal, or reuse the same series instance to share it across records"
                    )
                pair = (ts.id, record.record_id)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    series_to_records.setdefault(ts.id, []).append(record.record_id)
        # Group a recording's series together (source_id) before splitting by signal, so all leads of
        # one record are contiguous: the writer reads the record's source once, and a reader pulls a
        # record's series from one place instead of scattered across signal-ordered shards. Falls back
        # to signal order when source_id is unset (one series per record), matching the prior layout.
        ordered = sorted(unique.values(), key=lambda ts: (ts.spec.spec_type, ts.source_id or "", ts.name, ts.id))
        return ordered, series_to_records

    def _write_values(self, unique_series: list[Signal]) -> dict[tuple[str, int], ChunkPlacement]:
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

    def _read_and_validate(self, ts: Signal) -> pa.Array:  # noqa: PLR6301
        """Read a series' values and enforce the per-series array contract.

        Args:
            ts: The series to read.

        Returns:
            The validated values in the spec's canonical Arrow representation.

            The method accepts ``NaN``, ``+Infinity``, and ``-Infinity`` as floating-point values.
            They are distinct from missing values. The ``nullable`` flag controls Arrow nulls only.

        Raises:
            TimeFValidationError: If the array disagrees with the spec's dtype, shape, or
                nullability, has partial tensor nulls, or its length disagrees with ``n_values``.
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
                f"series {ts.id!r} must load dtype={ts.spec.dtype}, "
                f"value_shape={ts.spec.value_shape} as Arrow, got "
                f"{values.type if isinstance(values, pa.Array) else type(values)!r}"
            )
        if values.null_count and not ts.spec.nullable:
            raise TimeFValidationError(f"series {ts.id!r} has null values but nullable=False")
        if isinstance(values, pa.FixedShapeTensorArray) and values.storage.flatten().null_count:
            raise TimeFValidationError(f"series {ts.id!r} may have nulls only for whole timesteps")
        if ts.spec.dtype == "enum":
            # Compare each distinct label with the allowed categories.
            # This avoids creating a Python object for every value in the series.
            _validate_enum_values(ts.spec, values.dictionary.to_pylist())
        if len(values) != ts.n_values:
            raise TimeFValidationError(
                f"series {ts.id!r}: its loader returned {len(values)} values but it declares n_values={ts.n_values}"
            )
        return values

    def _read_time_offsets(self, ts: Signal) -> pa.Array | None:  # noqa: PLR6301
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
        if not isinstance(axis, IrregularAxis):  # pragma: no cover - Signal.__post_init__ pairs these
            raise TimeFValidationError(f"series {ts.id!r} carries time offsets but has {type(axis).__name__}")
        time_offsets = to_time_offsets_us(ts.time_offsets_loader().to_numpy(zero_copy_only=False))
        if len(time_offsets) != ts.n_values:
            raise TimeFValidationError(
                f"series {ts.id!r}: its time_offsets_loader returned {len(time_offsets)} time offsets "
                f"but it declares n_values={ts.n_values}"
            )
        if int(time_offsets[0]) != axis.first_us or int(time_offsets[-1]) != axis.last_us:
            raise TimeFValidationError(
                f"series {ts.id!r}: its axis claims the stream runs "
                f"{axis.first_us}..{axis.last_us} us but the stream runs "
                f"{int(time_offsets[0])}..{int(time_offsets[-1])} us"
            )
        return pa.array(time_offsets)

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
                groups=(
                    FileGroup(
                        kind=FileKind.CONTROL,
                        backend=CONTROL_BACKEND,
                        parts=(self._file_part(self._control_file),),
                    ),
                    FileGroup(
                        kind=FileKind.TIME_SERIES,
                        backend=self._values_backend_name,
                        parts=self._file_parts(self._value_files),
                        encoding=self._value_encoding,
                    ),
                )
            ),
            build_env=build_env(),
        )
        (self._staging_dir / MANIFEST_FILE).write_text(manifest.to_json())

    # ---- helpers -------------------------------------------------------------------------------

    def _validate_shared_annotations(self) -> None:
        """Check annotations sharing an id across records are field-equal, and registered ids are distinct.

        Raises:
            TimeFValidationError: If two annotations share an id but are not equal, or an id is both
                registered and carried by a record. The reader restores a registered annotation from its
                empty ``record_ids``, so an id that is also record-carried writes non-empty and is lost on
                read; reject it here instead of silently dropping it.
        """
        record_ann_ids = {ann.id for record in self._dataset.records for ann in record.annotations}
        overlap = sorted(record_ann_ids & {ann.id for ann in self._dataset.registered_annotations})
        if overlap:
            raise TimeFValidationError(
                f"annotation id(s) {overlap} are both registered and carried by a record; a registered "
                f"annotation must be one no record carries"
            )
        seen: dict[str, object] = {}
        record_annotations = (ann for record in self._dataset.records for ann in record.annotations)
        for ann in (*record_annotations, *self._dataset.registered_annotations):
            if ann.id in seen and seen[ann.id] != ann:
                raise TimeFValidationError(
                    f"annotation id {ann.id!r} is shared across records but instances are not equal"
                )
            seen[ann.id] = ann

    def _build_counts(self) -> ManifestCounts:
        """Read exact entity counts from the completed control database.

        Returns:
            Counts for the manifest.
        """
        with connect_control(self._staging_dir / CONTROL_FILE, read_only=True) as connection:

            def count(table: str) -> int:
                row = connection.execute(f"SELECT count(*) FROM {table}").fetchone()  # noqa: S608
                if row is None:
                    raise TimeFValidationError(f"could not count rows in control table {table!r}")
                return int(row[0])

            signals_by_spec = dict(
                connection.execute(
                    "SELECT spec_type, count(*) FROM signals GROUP BY spec_type ORDER BY spec_type"
                ).fetchall()
            )
            return ManifestCounts(
                records=count("records"),
                sources=count("sources"),
                signals=count("signals"),
                axes=count("axes"),
                annotation_contents=count("annotation_contents"),
                annotation_occurrences=count("annotation_occurrences"),
                tasks=self._task_type_counts,
                signal_chunks=count("signal_chunks"),
                signals_by_spec=signals_by_spec,
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


def _series_identity(ts: Signal) -> tuple:
    """Return the fields that must agree for two series to be the same signal.

    This compares the descriptive fields that the writer persists, not the values. The ``loader`` is a
    callable, so two equal series built separately compare unequal. Reading every shared series
    only to compare it defeats the lazy read path on the largest datasets.

    Args:
        ts: The series to describe.

    Returns:
        The identifying fields, suitable for equality comparison and for error messages.
    """
    return (ts.spec.spec_type, ts.name, ts.time_axis, ts.source_id, ts.n_values)


def _ordered_unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))
