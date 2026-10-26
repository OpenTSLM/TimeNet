"""Write a TimeF DuckDB control plane and a sharded Parquet or Zarr values plane."""

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
import shutil
import types as _types
from typing import TYPE_CHECKING, cast
import uuid

import numpy as np
import pyarrow as pa

from timenet.dataset import Signal, TimeFDataset
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
from timenet.format.storage import StorageSources
from timenet.format.stored_array import StoredArray
from timenet.format.stored_values import PackedValueWriter
from timenet.manifest import FilePart, Manifest, ManifestCounts, ManifestFiles, PayloadSource
from timenet.provenance import build_env
from timenet.types.ids import EntityId
from timenet.types.reference_keys import SourceKey
from timenet.types.references import DatasetRef
from timenet.types.wire import ValueEncoding as ManifestValueEncoding
from timenet.values_backends import SUPPORTED_VALUES_BACKENDS, ValuesBackend
from timenet.values_backends.parquet.config import DEFAULT_PARQUET_COMPRESSION_LEVEL
from timenet.values_backends.writer import (
    ChunkPlacement,
    ParquetValuesConfig,
    ZarrValuesConfig,
    make_values_backend,
)
from timenet.values_backends.zarr.config import DEFAULT_ZARR_COMPRESSION_LEVEL
from timenet.writer.progress import ProgressStage, WriteProgressEvent
from timenet.writer.value_encoding import AUTO, SUPPORTED_VALUE_ENCODINGS, ValueEncoding


if TYPE_CHECKING:
    from timenet.registry.base import BaseRegistry


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
        replace: bool = False,
        registry: "BaseRegistry | None" = None,
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
            replace: Replace the version if it is already committed. The old release stays in place
                until the new one is published, so a failed write keeps it. A replace is not isolated
                from readers that already opened the old release, or from releases that reuse its
                payloads.
            registry: The registry the release is published to. Each payload source of a reused
                value is checked there at its first reuse. Defaults to a local registry over
                ``root``. A remote store passes itself while it stages the release locally.

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
        self._values_backend_name = ValuesBackend(values_backend)
        self._forced_value_encoding = None if value_encoding == AUTO else ValueEncoding(value_encoding)
        self._value_encoding: dict[str, ManifestValueEncoding] = {}
        self._progress_cb = progress_cb
        self._replace = replace

        version = str(dataset.metadata.dataset_version)
        self._final_dir = self._root / dataset.metadata.dataset_id / version
        self._staging_dir = self._root / dataset.metadata.dataset_id / f"{version}.tmp-{uuid.uuid4().hex}"

        self._written = False
        if registry is None:
            # The registry package imports the writer, so the default is resolved here.
            from timenet.registry.local import LocalRegistry  # noqa: PLC0415

            registry = LocalRegistry(self._root)
        self._storage_sources = StorageSources(
            DatasetRef(dataset_id=dataset.metadata.dataset_id, version=dataset.metadata.dataset_version), registry
        )
        self._external = PackedValueWriter(self._staging_dir, self._storage_sources)

    # ---- context manager -----------------------------------------------------------------------

    def __enter__(self) -> "TimeFWriter":
        """Create the staging directory, refusing a committed version unless ``replace`` is set.

        Returns:
            This writer.

        Raises:
            FileExistsError: If a committed ``manifest.json`` already exists at the version directory
                and ``replace`` is not set.
        """
        if not self._replace and (self._final_dir / MANIFEST_FILE).exists():
            raise FileExistsError(f"a committed version already exists at {self._final_dir}")
        self._sweep_stale_staging()
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        return self

    def _sweep_stale_staging(self) -> None:
        """Remove abandoned ``<version>.tmp-*`` staging dirs left by a crashed build.

        A hard kill (SIGKILL or OOM) never reaches :meth:`abort`, so its staging directory lingers.
        Clear any such sibling for this version before you write a fresh one. The writer does not
        support concurrent writes of the same version.

        The sweep never removes a ``<version>.old-*`` directory. A hard kill while :meth:`close`
        replaces a committed version can leave the old release there, and renaming it back recovers
        it.
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
        del exc_type, tb
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

        unique_series = self._dedupe_series()
        placements = self._write_values(unique_series)
        tasks = (
            self._dataset.iter_streamed_tasks_validated()
            if self._dataset.has_task_stream
            else self._dataset.iter_tasks()
        )
        self._task_type_counts = DuckDBControlWriter(
            self._staging_dir / CONTROL_FILE, external=self._external
        ).write_hierarchy(
            self._dataset,
            placements,
            tasks=tasks,
        )
        self._control_file = CONTROL_FILE
        self._counts = self._build_counts()
        self._written = True

    def close(self) -> None:
        """Write the manifest and atomically publish the staging directory.

        An existing version directory is renamed aside to a sibling ``<version>.old-<uuid>`` first.
        The staging directory then takes its place, and the old directory is deleted. If that
        rename fails, the old directory moves back. A hard kill between the two renames leaves the
        old release in ``<version>.old-<uuid>``, which no sweep deletes, so it can be renamed back.
        The manifest pins the payload source files that :meth:`write` checked.

        Raises:
            RuntimeError: If :meth:`write` has not run successfully.
            OSError: If publishing fails and the old release cannot be moved back either. The
                error names the ``<version>.old-<uuid>`` directory that holds it.
        """
        if not self._written:
            raise RuntimeError("write() must run successfully before close()")
        self._write_manifest()
        self._final_dir.parent.mkdir(parents=True, exist_ok=True)
        old = None
        if self._final_dir.exists():
            # A committed version reaches here only with replace=True, which LocalRegistry.store()
            # and run_pipeline() pass for force. Otherwise the caller pre-created the directory.
            old = self._final_dir.with_name(f"{self._final_dir.name}.old-{uuid.uuid4().hex}")
            self._final_dir.replace(old)
        try:
            self._staging_dir.replace(self._final_dir)
        except BaseException as error:
            if old is not None:
                try:
                    old.replace(self._final_dir)
                except OSError as restore_error:
                    raise OSError(
                        f"could not publish {self._final_dir} ({error}) or move the old release back "
                        f"({restore_error}); it is in {old}, so rename that to {self._final_dir} to recover it"
                    ) from error
            raise
        if old is not None:
            # The new release is published, so failing to delete the old one must not fail the write.
            shutil.rmtree(old, ignore_errors=True)
        self._emit(ProgressStage.COMMIT, 1, 1)

    def abort(self) -> None:
        """Delete the staging directory. Safe to call more than once."""
        shutil.rmtree(self._staging_dir, ignore_errors=True)

    @property
    def payload_sources(self) -> Mapping[SourceKey, PayloadSource]:
        """Return the releases that store the payloads this release reuses, pinned to their files.

        :meth:`write` fills it, and the manifest records it as ``payload_sources``.
        """
        return self._storage_sources.sources

    # ---- values --------------------------------------------------------------------------------

    def _dedupe_series(self) -> list[Signal]:
        """Return every series once, sorted for stable output.

        A dataset gives each Signal its own id. Only one record owns each Signal. Thus the writer does
        not merge series.

        Returns:
            The sorted series.
        """
        owned = [(record.id, ts) for record in self._dataset.records for ts in record.signals]
        # Group the series of a record together before the split by signal. Then all leads of one record
        # are contiguous. The writer reads the source of the record once. A reader gets the series of a
        # record from one place, not scattered across signal-ordered shards.
        owned.sort(key=lambda pair: (pair[1].spec.spec_type, pair[0], pair[1].name, pair[1].id))
        return [ts for _, ts in owned]

    def _write_values(self, unique_series: list[Signal]) -> dict[tuple[EntityId, int], ChunkPlacement]:
        """Write all series' values through the configured backend.

        Args:
            unique_series: The deduped, sorted series to serialize.

        Returns:
            A mapping from ``(time_series_id, chunk_idx)`` to its on-disk placement.
        """
        if self._values_backend_name == ValuesBackend.PARQUET:
            config = ParquetValuesConfig(
                staging_dir=self._staging_dir,
                shard_target_bytes=self._shard_target_bytes,
                row_group_target_bytes=self._row_group_target_bytes,
                chunk_max_bytes=self._chunk_max_bytes,
                compression=self._compression,
                data_page_size=self._data_page_size,
                value_encoding=self._forced_value_encoding,
                compression_level=DEFAULT_PARQUET_COMPRESSION_LEVEL
                if self._compression_level is None
                else self._compression_level,
            )
        else:
            config = ZarrValuesConfig(
                staging_dir=self._staging_dir,
                shard_target_bytes=self._shard_target_bytes,
                chunk_max_bytes=self._chunk_max_bytes,
                compression=self._compression,
                compression_level=DEFAULT_ZARR_COMPRESSION_LEVEL
                if self._compression_level is None
                else self._compression_level,
            )
        shared, owned = self._split_shared_and_owned_signals(unique_series)
        values_backend = make_values_backend(config)
        result = values_backend.write_series(
            owned,
            read_and_validate=self._read_and_validate,
            on_series_done=lambda completed, total: self._emit(ProgressStage.TIME_SERIES, completed, total),
            on_file_done=lambda count: self._emit(ProgressStage.SHARD_FINALIZED, count, None),
        )
        self._value_files = result.files
        self._value_encoding = cast("dict[str, ManifestValueEncoding]", dict(result.value_encoding))
        return {**result.placements, **shared}

    def _split_shared_and_owned_signals(
        self, unique_series: list[Signal]
    ) -> tuple[dict[tuple[EntityId, int], ChunkPlacement], list[Signal]]:
        """Separate signals whose stored chunks the new release reuses from signals it writes out.

        A signal read from a release holds a ``StoredArray``, whose location names its chunks and
        the release that stores them, so the new catalog names the same chunks. Values built in
        memory or read through a connector's function are written to new value files.

        Args:
            unique_series: The deduplicated signals to serialize.

        Returns:
            Shared chunk placements and the signals whose values must be written.

        Raises:
            TimeFValidationError: If a shared signal no longer matches its stored values.
        """
        shared: dict[tuple[EntityId, int], ChunkPlacement] = {}
        owned: list[Signal] = []
        for signal in unique_series:
            values = signal._values
            if not isinstance(values, StoredArray):
                owned.append(signal)
                continue
            location = values._location
            if signal.spec != values.spec or signal.n_values != sum(chunk.n_values for chunk in location.chunks):
                raise TimeFValidationError(f"shared signal {signal.id!r} changed its storage shape")
            source = self._storage_sources.add(values.reader._version, location, nullable=values.spec.nullable)
            for chunk in location.chunks:
                shared[signal.id, chunk.chunk_idx] = ChunkPlacement(
                    chunk.chunk_file,
                    chunk.data_index,
                    signal.spec.spec_type,
                    signal.name,
                    chunk.n_values,
                    source,
                )
        return shared, owned

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

    def _write_manifest(self) -> None:
        schema = self._dataset.schema
        if schema is None:  # unreachable: write() already checked, but keeps the type non-optional
            raise RuntimeError("schema was not derived")
        manifest = Manifest(
            dataset_id=self._dataset.metadata.dataset_id,
            metadata=self._dataset.metadata,
            dataset_schema=schema,
            counts=self._counts,
            files=ManifestFiles(
                control=(self._file_part(self._control_file),),
                time_series=self._file_parts(self._value_files),
                payloads=self._file_parts(self._external.files),
            ),
            values_backend=self._values_backend_name,
            value_encoding=self._value_encoding,
            payload_sources=self._storage_sources.sources,
            build_env=build_env(),
            timef_format_version=1,
        )
        (self._staging_dir / MANIFEST_FILE).write_text(manifest.to_json())

    # ---- helpers -------------------------------------------------------------------------------

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
