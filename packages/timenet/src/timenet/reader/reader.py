"""Read TimeF datasets from a DuckDB control plane and a sharded values plane."""

from __future__ import annotations

from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from functools import partial
import types as _types
from typing import TYPE_CHECKING

import pyarrow as pa

from timenet.dataset import Record, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.checksums import stream_checksum
from timenet.format.control_cache import materialize_control
from timenet.format.control_reader import TASK_BATCH_SIZE, DuckDBControlReader
from timenet.format.stored_array import ChunkLocation, StoredArray
from timenet.format.stored_values import (
    ANNOTATION_COLUMNS,
    TARGET_COLUMNS,
    TASK_COLUMNS,
    PackedValueReader,
    StoredValueColumns,
    materialize_table,
)
from timenet.types import (
    Annotation,
    DatasetMetadata,
    DatasetSchema,
    EntityId,
    InputModality,
    ObjectKind,
    Task,
    TimeSeriesSpec,
)
from timenet.types.reference_keys import SourceKey
from timenet.types.splits import Split
from timenet.values_backends.reader import BaseValuesReader, make_values_reader


if TYPE_CHECKING:
    from timenet.manifest import FilePart
    from timenet.registry.version import DatasetVersion
    from timenet.types.references import DatasetRef


_CHUNK_LOCATOR_BATCH_SIZE = 1_024


class TimeFReader:
    """Read a committed TimeF dataset through its DuckDB control plane."""

    def __init__(self, version: DatasetVersion) -> None:
        """Configure a lazy reader for an opened dataset version.

        Opening a reader does not download or query ``control.duckdb``. The first structural read
        materializes the control database locally, verifies it, and opens it read-only.

        Args:
            version: Opened dataset version containing a parsed manifest and filesystem handle.
        """
        self._version = version
        # A release that neither packs nor reuses payload files needs no locator lookups.
        manifest = version.manifest
        self._packed_value_reader = (
            PackedValueReader(version) if manifest.files.payloads or manifest.payload_sources else None
        )
        self._fs = version.filesystem
        self._root = version.root
        self._manifest = version.manifest
        self._control: DuckDBControlReader | None = None
        self._values: BaseValuesReader | None = None
        self._source_readers: dict[SourceKey, BaseValuesReader] = {}
        self._chunk_locator_batches: list[list[int]] = []
        self._chunk_locator_batch_by_id: dict[int, int] = {}
        self._chunk_locations: dict[int, ChunkLocation | None] = {}

    def __getstate__(self) -> dict:
        """Return picklable state without open connections or decoded caches."""
        state = self.__dict__.copy()
        state["_control"] = None
        state["_values"] = None
        state["_source_readers"] = {}
        state["_chunk_locations"] = {}
        return state

    def __enter__(self) -> TimeFReader:
        """Return this open reader."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: _types.TracebackType | None,
    ) -> None:
        """Close cached control and values handles."""
        self.close()

    def close(self) -> None:
        """Close cached handles while keeping the reader reusable."""
        if self._values is not None:
            self._values.close()
            self._values = None
        if self._control is not None:
            # The control reader stays, so it reopens its verified local catalog without checking it again.
            self._control.close()
        for backend in self._source_readers.values():
            backend.close()
        self._source_readers.clear()
        if self._packed_value_reader is not None:
            self._packed_value_reader.close()
        self._chunk_locations.clear()

    def verify(self) -> None:
        """Verify the size and checksum of every file the release reads.

        These are the artifacts its manifest lists, and the files it pins on each payload source.

        Raises:
            TimeFFormatError: If a file is missing or does not match its manifest entry or pin.
        """  # noqa: DOC502 - raised by _verify_files
        _verify_files(self._version, self._manifest.files.all_files())
        for key, pinned in self._manifest.payload_sources.items():
            _verify_files(self._version.resolve_payload_source(key), pinned.files, source=pinned.release)

    @property
    def metadata(self) -> DatasetMetadata:
        """Return the dataset's descriptive metadata."""
        return self._manifest.metadata

    @property
    def schema(self) -> DatasetSchema:
        """Return the dataset's type declaration."""
        return self._manifest.dataset_schema

    def record_ids(self) -> tuple[EntityId, ...]:
        """Return every Record ID without building a Record.

        Returns:
            The IDs in the order :meth:`iter_records` yields them when called without arguments.
        """
        with self._as_format_error():
            return self._control_reader().record_ids()

    def read_tasks(self, records: Iterable[Record] | None = None) -> tuple[Task, ...]:
        """Build the tasks attached to ``records`` in one pass.

        Every call builds new tasks. A record a task uses that is not among ``records`` is read
        once and shared by every task of the call.

        Args:
            records: Input Records to select, or ``None`` for every task. Tasks reuse these objects.

        Returns:
            Concrete tasks in stable ID order.
        """
        with self._as_format_error(preserve_validation=True):
            return self._control_reader().read_tasks(records)

    def iter_tasks(
        self,
        records: Iterable[Record] | None = None,
        *,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
        batch_size: int = TASK_BATCH_SIZE,
    ) -> Iterator[Task]:
        """Yield tasks a bounded batch at a time instead of building the whole table.

        Each batch is built on its own. Its tasks share the objects it builds: a record
        that two of them use is one object, and the signals on one irregular axis share one
        time-offsets handle. Batches share nothing but ``records``, so the stream keeps none of a
        batch's objects once it has yielded them: it holds one batch of objects and that batch's
        task IDs at a time, and selects the next batch's IDs after the last one, while the reader
        keeps compact chunk locators for the signals it has resolved until it closes. A record or
        parent task that tasks of several batches use is built once per batch, as separate objects
        with the same ID.

        Args:
            records: Records from :meth:`iter_records` whose input tasks to yield, or ``None`` for
                every task of the dataset. Tasks reference these ``Record`` objects directly.
            split: Keep only tasks of this split, or ``None`` for every task. The filter runs in
                DuckDB, so a task outside the split is never built.
            batch_size: The most tasks a batch holds. By default, a batch holds up to 5,000 tasks
                (``TASK_BATCH_SIZE``). Pass another value for larger or smaller batches. The
                stream holds one batch at a time.

        Yields:
            Concrete tasks in stable ID order.
        """
        with self._as_format_error(preserve_validation=True):
            for batch in self._control_reader().iter_task_batches(
                records,
                required_modalities=required_modalities,
                supported_modalities=supported_modalities,
                split=split,
                batch_size=batch_size,
            ):
                yield from batch
                # Hold no batch while the next one is built.
                del batch

    def read_task_table(
        self,
        record_ids: Iterable[EntityId] | None = None,
        *,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
    ) -> pa.Table:
        """Return the tasks as an Arrow table without building Task objects.

        Args:
            record_ids: Keep only tasks whose inputs include one of these Records, or ``None`` for all.
            split: Keep only tasks of this split, or ``None`` for all.

        Returns:
            One row per task with public IDs in place of object references. Targets are in
            :meth:`read_target_table`.
        """
        with self._as_format_error(preserve_validation=True):
            table = self._control_reader().read_task_table(
                record_ids,
                required_modalities=required_modalities,
                supported_modalities=supported_modalities,
                split=split,
            )

            return self._materialized(table, TASK_COLUMNS)

    def read_target_table(
        self,
        record_ids: Iterable[EntityId] | None = None,
        *,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
    ) -> pa.Table:
        """Return every inline task target as one typed Arrow row.

        Args:
            record_ids: Keep only targets of tasks whose inputs include one of these Records.
            split: Keep only targets of tasks of this split, or ``None`` for all.

        Returns:
            Targets in task ID and position order.
        """
        with self._as_format_error(preserve_validation=True):
            table = self._control_reader().read_target_table(
                record_ids,
                required_modalities=required_modalities,
                supported_modalities=supported_modalities,
                split=split,
            )

            return self._materialized(table, TARGET_COLUMNS)

    def read_annotation_table(self, object_type: ObjectKind | None = None) -> pa.Table:
        """Return every annotation occurrence with its content as an Arrow table.

        Args:
            object_type: Keep occurrences on this object kind, such as ``ObjectKind.RECORD``,
                or ``None`` for all annotated objects.

        Returns:
            One row per occurrence.
        """
        with self._as_format_error():
            table = self._control_reader().read_annotation_table(object_type)

            return self._materialized(table, ANNOTATION_COLUMNS)

    def _materialized(self, table: pa.Table, columns: StoredValueColumns) -> pa.Table:
        """Resolve packed values in a public Arrow table, or keep it for a release that packs nothing.

        Returns:
            The table with native values.
        """
        packed = self._packed_value_reader
        if packed is None:
            return table
        return self._control_reader().run(lambda connection: materialize_table(table, packed, connection, columns))

    @property
    def values_backend(self) -> str:
        """Return the values-plane backend named by the manifest."""
        return self._manifest.values_backend

    def read(self) -> TimeFDataset:
        """Build the dataset while keeping every Signal's values lazy.

        Every call builds new objects.

        Returns:
            The reconstructed dataset.
        """
        control = self._control_reader()
        records = control.read_records()
        occurrences: dict[int, Annotation] = {}
        annotations = control.read_dataset_annotations(self._manifest.dataset_id, by_occurrence=occurrences)
        with self._as_format_error():
            tasks = control.read_tasks(cached_records=records, dataset_annotations=occurrences)
        return TimeFDataset.from_parts(
            metadata=self.metadata,
            records=records,
            tasks=tasks,
            schema=self.schema,
            annotations=annotations,
            last_ids=control.read_last_ids(),
        )

    def iter_records(
        self,
        record_ids: Iterable[EntityId] | None = None,
        *,
        with_annotations: bool = True,
    ) -> Iterator[Record]:
        """Yield complete records, optionally in a requested ID order.

        Args:
            record_ids: IDs to read in result order, or ``None`` for every record.
            with_annotations: Whether to build annotations on the records and descendants.

        Yields:
            Records whose Signal values remain lazy.
        """
        with self._as_format_error(preserve_validation=True):
            yield from self._control_reader().read_records(
                record_ids,
                with_annotations=with_annotations,
            )

    def _control_reader(self) -> DuckDBControlReader:
        """Open and return the cached read-only control database.

        A remote catalog is fetched and verified once, and again only if its cached copy is gone
        when the connection reopens. The connection opens again on demand after :meth:`close` or a
        fork.

        Returns:
            The cached control reader.
        """
        if self._control is None:
            self._control = DuckDBControlReader(
                materialize_control(self._version),
                value_loader_factory=self._make_signal_loader,
                external=self._packed_value_reader,
                fetch=partial(materialize_control, self._version),
            )
        return self._control

    def _values_reader(self) -> BaseValuesReader:
        """Return the lazily opened values-plane reader."""
        if self._values is None:
            self._values = make_values_reader(self._manifest.values_backend)
        return self._values

    def _read_array(self, location: ChunkLocation, spec: TimeSeriesSpec) -> pa.Array:
        """Read every chunk of one Signal from the release that stores it.

        Returns:
            The complete canonical Arrow array.
        """
        version, backend = self._storage_reader(location.source)
        return backend.load(version, location.chunks, spec)

    def _read_array_steps(self, location: ChunkLocation, spec: TimeSeriesSpec, start: int, stop: int) -> pa.Array:
        """Read a half-open step range of one Signal from the release that stores it.

        Returns:
            The requested canonical Arrow values.
        """
        version, backend = self._storage_reader(location.source)
        return backend.load_range(version, location.chunks, start, stop, spec)

    def _storage_reader(self, source: SourceKey | None) -> tuple[DatasetVersion, BaseValuesReader]:
        """Return the payload source that a key names, or this release for ``None``, and its values reader.

        Returns:
            The version handle and its backend.
        """
        if source is None:
            return self._version, self._values_reader()
        version = self._version.resolve_payload_source(source)
        if source not in self._source_readers:
            self._source_readers[source] = make_values_reader(version.manifest.values_backend)
        return version, self._source_readers[source]

    def _chunk_location(self, signal_id: EntityId) -> ChunkLocation:
        """Return where one Signal's chunks are stored, finding them for its batch on first use.

        Raises:
            TimeFFormatError: If the Signal has no stored chunks.
        """
        if signal_id not in self._chunk_locations:
            batch_index = self._chunk_locator_batch_by_id.get(signal_id)
            if batch_index is None:
                return self._control_reader().chunk_location(signal_id)
            batch = self._chunk_locator_batches[batch_index]
            self._chunk_locations.update(self._control_reader().chunk_locations_by_ids(batch))
        location = self._chunk_locations[signal_id]
        if location is None:
            raise TimeFFormatError(f"signal {signal_id!r} has no stored value chunks")
        return location

    def _make_signal_loader(self, signal_id: EntityId, spec: TimeSeriesSpec) -> StoredArray:
        """Return a picklable, range-aware handle to one Signal's stored values."""
        if signal_id not in self._chunk_locator_batch_by_id:
            if not self._chunk_locator_batches or len(self._chunk_locator_batches[-1]) >= _CHUNK_LOCATOR_BATCH_SIZE:
                self._chunk_locator_batches.append([])
            batch_index = len(self._chunk_locator_batches) - 1
            self._chunk_locator_batches[batch_index].append(signal_id)
            self._chunk_locator_batch_by_id[signal_id] = batch_index
        return StoredArray(self, signal_id, spec)

    @contextmanager
    def _as_format_error(self, *, preserve_validation: bool = False) -> Generator[None]:
        """Add dataset context to unexpected decoding failures.

        Yields:
            Control to the wrapped read operation.

        Raises:
            TimeFValidationError: If ``preserve_validation`` is true and the wrapped read rejects
                caller input.
            TimeFFormatError: If the stored artifact cannot be decoded.
        """
        try:
            yield
        except TimeFFormatError:
            raise
        except TimeFValidationError as exc:
            if preserve_validation:
                raise
            raise TimeFFormatError(f"corrupt or inconsistent TimeF artifact at {self._root}: {exc}") from exc
        except (ValueError, KeyError, TypeError, AttributeError, OSError, pa.ArrowException) as exc:
            raise TimeFFormatError(f"corrupt or inconsistent TimeF artifact at {self._root}: {exc}") from exc


def _verify_files(version: DatasetVersion, parts: Iterable[FilePart], *, source: DatasetRef | None = None) -> None:
    """Hash each file of a release and compare it with its manifest record, or its pin.

    Args:
        version: The release that stores the files.
        parts: The records to check, which name each file relative to ``version``.
        source: The payload source whose files the records pin, or ``None`` for the release's own.

    Raises:
        TimeFFormatError: If a file is missing, or its size or checksum differs from its record.
    """
    of, record = ("", "manifest says") if source is None else (f" of payload source {source}", "pinned")
    for part in sorted(parts, key=lambda item: item.path):
        try:
            handle = version.filesystem.open_input_file(version.resolve_file_path(part.path))
        except FileNotFoundError as exc:
            raise TimeFFormatError(f"manifest lists a missing file{of}: {part.path}") from exc
        with handle:
            actual_size = handle.size()
            if actual_size != part.size:
                raise TimeFFormatError(
                    f"size mismatch for {part.path}{of}: {record} {part.size}, file is {actual_size}"
                )
            actual_checksum = stream_checksum(handle)
        if actual_checksum != part.checksum:
            raise TimeFFormatError(
                f"checksum mismatch for {part.path}{of}: {record} {part.checksum}, file is {actual_checksum}"
            )
