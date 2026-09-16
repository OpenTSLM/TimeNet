"""``TimeFWriter``: serialize a :class:`~timenet.dataset.TimeFDataset` to the TimeF layout on disk.

The writer loads the version's structure into one embedded DuckDB control database and sends series
values to a selected Parquet or Zarr backend. The configured chunk, row-group, and shard targets
bound the backend buffers. The writer stages everything in a temporary directory and publishes it
with a single atomic rename. A ``manifest.json`` in the version directory marks a committed version.
"""

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
import shutil
import types as _types
from typing import Any
import uuid

import numpy as np
import pyarrow as pa

from timenet.control_plane.writer import ValuesPlane, write_control_plane
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import IrregularAxis, to_time_offsets_us
from timenet.dataset.time_series import _validate_enum_values
from timenet.errors import TimeFValidationError
from timenet.format.checksums import file_checksum
from timenet.format.constants import (
    CONTROL_DB_FILE,
    DEFAULT_COMPRESSION,
    DEFAULT_SHARD_TARGET_BYTES,
    MANIFEST_FILE,
)
from timenet.format.layout import DEFAULT_VALUES_LAYOUT, ValuesLayout
from timenet.format.schemas import LOGICAL_IDS, UUID16, IdCodec, IdTypes
from timenet.manifest import FilePart, Manifest, ManifestCounts, ManifestFiles
from timenet.provenance import build_env
from timenet.types import Task
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
        values_layout: ValuesLayout = DEFAULT_VALUES_LAYOUT,
        row_group_target_bytes: int | None = None,
        chunk_max_bytes: int | None = None,
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
            dataset: The populated dataset. The writer derives the schema when the dataset has none.
            shard_target_bytes: Rotate to a new shard once a shard's buffered values exceed this.
            values_layout: Chunk and row-group byte targets for the values plane.
            row_group_target_bytes: Flush a row group once buffered values exceed this, or ``None``
                to take the target from ``values_layout``.
            chunk_max_bytes: Split a series into chunks no larger than this, or ``None`` to take the
                target from ``values_layout``.
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
        self._row_group_target_bytes = (
            values_layout.row_group_target_bytes if row_group_target_bytes is None else row_group_target_bytes
        )
        self._chunk_max_bytes = values_layout.chunk_max_bytes if chunk_max_bytes is None else chunk_max_bytes
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
        """Create the staging directory, and refuse to overwrite a committed version.

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
        This method removes any such sibling of this version before the writer stages a fresh one.
        The writer does not support concurrent writes of the same version.
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

        unique_series, series_to_records = self._dedupe_series()
        placements = self._write_values(unique_series)
        self._counts = self._write_control_db(unique_series, series_to_records, placements)
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
        """Pick the shard's id storage: ``binary(16)`` when every id is a canonical UUID, else string.

        ``time_series_id`` is the one id a shard carries, so it is the one this resolves. Every other
        id lives in the control database as the caller's own string.
        """
        ids = [ts.time_series_id for record in self._dataset.records for ts in record.time_series]
        is_uuid16 = bool(ids) and all(is_canonical_uuid(value) for value in ids)
        id_types: IdTypes = {name: pa.string() for name in LOGICAL_IDS}
        id_types["time_series_id"] = UUID16 if is_uuid16 else pa.string()
        self._id_types = id_types
        self._codec = IdCodec.from_id_types(id_types)

    # ---- values --------------------------------------------------------------------------------

    def _dedupe_series(self) -> tuple[list[TimeSeries], dict[str, list[str]]]:
        """Return unique series (sorted for stable output) and the series-id -> record-ids map.

        One series shared across records is the supported dedupe path. Two different series that
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
        unique: dict[str, TimeSeries] = {}
        series_to_records: dict[str, list[str]] = {}
        seen_pairs: set[tuple[str, str]] = set()
        for record in self._dataset.records:
            for ts in record.time_series:
                existing = unique.get(ts.time_series_id)
                if existing is None:
                    unique[ts.time_series_id] = ts
                elif existing is not ts and _series_identity(existing) != _series_identity(ts):
                    raise TimeFValidationError(
                        f"time_series_id {ts.time_series_id!r} is claimed by two different series: "
                        f"{_series_identity(existing)} and {_series_identity(ts)}; ids must be unique "
                        f"per signal, or reuse the same series instance to share it across records"
                    )
                pair = (ts.time_series_id, record.record_id)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    series_to_records.setdefault(ts.time_series_id, []).append(record.record_id)
        # Group a recording's series together (source_id) before splitting by signal, so all leads of
        # one record are contiguous: the writer reads the record's source once, and a reader pulls a
        # record's series from one place instead of scattered across signal-ordered shards. Falls back
        # to signal order when source_id is unset (one series per record), matching the prior layout.
        ordered = sorted(
            unique.values(), key=lambda ts: (ts.spec.spec_type, ts.source_id or "", ts.signal, ts.time_series_id)
        )
        return ordered, series_to_records

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
                f"series {ts.time_series_id!r} must load dtype={ts.spec.dtype}, "
                f"value_shape={ts.spec.value_shape} as Arrow, got "
                f"{values.type if isinstance(values, pa.Array) else type(values)!r}"
            )
        if values.null_count and not ts.spec.nullable:
            raise TimeFValidationError(f"series {ts.time_series_id!r} has null values but nullable=False")
        if isinstance(values, pa.FixedShapeTensorArray) and values.storage.flatten().null_count:
            raise TimeFValidationError(f"series {ts.time_series_id!r} may have nulls only for whole timesteps")
        if ts.spec.dtype == "enum":
            # Compare each distinct label with the allowed categories.
            # This avoids creating a Python object for every value in the series.
            _validate_enum_values(ts.spec, values.dictionary.to_pylist())
        if len(values) != ts.n_values:
            raise TimeFValidationError(
                f"series {ts.time_series_id!r}: its loader returned {len(values)} values but it "
                f"declares n_values={ts.n_values}"
            )
        return values

    def _read_time_offsets(self, ts: TimeSeries) -> pa.Array | None:  # noqa: PLR6301
        """Read an irregular series' time offsets and check them against what it declares.

        The method returns ``None`` for every other axis shape. The backend writes that as a null cell.
        These checks make ``first_time_offset_us`` and ``last_time_offset_us`` checked metadata. An
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

    # ---- control plane -------------------------------------------------------------------------

    def _tasks_to_write(self) -> Iterator[Task]:
        """Yield the dataset's tasks, and validate a streamed one as it passes.

        ``iter_tasks()`` yields a materialized dataset's tasks and a streamed one's identically, so
        one path serves both. A stream is read exactly once, so a one-shot source can still be
        written.

        Yields:
            Each task to store.
        """
        if self._dataset.has_task_stream:
            yield from self._dataset.iter_streamed_tasks_validated()
        else:
            yield from self._dataset.iter_tasks()

    def _write_control_db(
        self,
        unique_series: list[TimeSeries],
        series_to_records: dict[str, list[str]],
        placements: dict[tuple[str, int], ChunkPlacement],
    ) -> ManifestCounts:
        """Load the version's structure into ``control.duckdb`` and return the manifest's counts.

        Args:
            unique_series: The deduped series, in the order the values plane wrote them.
            series_to_records: Each series id mapped to the records that reference it.
            placements: Where the values plane put every chunk.

        Returns:
            The counts block for the manifest.
        """
        counts = write_control_plane(
            self._staging_dir / CONTROL_DB_FILE,
            self._dataset,
            series=unique_series,
            series_to_records=series_to_records,
            values=ValuesPlane(placements=placements, backend=self._values_backend_name),
            tasks=self._tasks_to_write(),
        )
        return ManifestCounts(
            records=counts.records,
            annotations=counts.annotations,
            registered_annotations=counts.registered_annotations,
            tasks=counts.tasks,
            time_series_chunks=counts.chunks,
            time_series_index_rows=counts.record_series_chunks,
            time_series_specs=counts.specs,
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
                control_db=self._file_part(CONTROL_DB_FILE),
                time_series=self._file_parts(self._value_files),
            ),
            values_backend=self._values_backend_name,
            value_encoding=self._value_encoding,
            build_env=build_env(),
        )
        (self._staging_dir / MANIFEST_FILE).write_text(manifest.to_json())

    # ---- helpers -------------------------------------------------------------------------------

    def _validate_shared_annotations(self) -> None:
        """Check that annotations that share an id are field-equal, and that registered ids are distinct.

        Raises:
            TimeFValidationError: If two annotations share an id but are not equal, or an id is both
                registered and carried by a record. The reader restores a registered annotation from
                the dataset attachment table. An id that a record also carries comes back attached
                to that record instead. The writer rejects it here and does not move it silently.
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
    """Return the fields that must agree for two series to be the same signal.

    This compares the descriptive fields that the writer persists, not the values. The ``loader`` is a
    callable, so two equal series built separately compare unequal. A read of every shared series
    only for that comparison defeats the lazy read path on the largest datasets.

    Args:
        ts: The series to describe.

    Returns:
        The identifying fields, suitable for equality comparison and for error messages.
    """
    return (ts.spec.spec_type, ts.signal, ts.time_axis, ts.source_id, ts.n_values)
