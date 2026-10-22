"""Read TimeF datasets from a DuckDB control plane and a sharded values plane."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import types as _types
from typing import TYPE_CHECKING

import pyarrow as pa

from timenet.dataset import Record, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.checksums import stream_checksum
from timenet.format.control_cache import materialize_control
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.signal_loader import StoredSignalLoader
from timenet.format.stored_values import PackedValueReader, materialize_table
from timenet.reader.handles import DatasetHandle, _materialize_record
from timenet.types import DatasetMetadata, DatasetSchema, InputModality, Task, TimeSeriesSpec
from timenet.types.object_kind import ObjectKind
from timenet.types.reference_keys import OwnerKey
from timenet.types.splits import Split
from timenet.values_backends.reader import BaseValuesReader, make_values_reader


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion
    from timenet.values_backends.types import StoredChunk


_CHUNK_LOCATOR_BATCH_SIZE = 1_024


@dataclass(frozen=True)
class _ReaderState:
    """Keep only prepared sources and loader registration across worker processes."""

    version: DatasetVersion
    packed_values: PackedValueReader
    """Preserve the shared reader referenced by text and timestamp handles; its cache resets on pickle."""
    chunk_batches: tuple[tuple[int, ...], ...]


class TimeFReader:
    """Read a committed TimeF dataset through its DuckDB control plane."""

    def __init__(self, version: DatasetVersion) -> None:
        """Configure a lazy reader for an opened dataset version.

        The version is already prepared, including its dependencies. Opening a reader does not
        query ``control.duckdb``. The first structural read opens the local catalog read-only.

        Args:
            version: Opened dataset version containing a parsed manifest and filesystem handle.
        """
        self._version = version
        self._packed_value_reader = PackedValueReader(version)
        self._fs = version.filesystem
        self._root = version.root
        self._manifest = version.manifest
        self._tasks: tuple[Task, ...] | None = None
        self._control: DuckDBControlReader | None = None
        self._records: tuple[Record, ...] | None = None
        self._dataset: TimeFDataset | None = None
        self._values: BaseValuesReader | None = None
        self._owner_readers: dict[OwnerKey, BaseValuesReader] = {}
        self._chunk_locator_batches: list[list[int]] = []
        self._chunk_locator_batch_by_key: dict[int, int] = {}
        self._chunk_rows_cache: dict[int, list[StoredChunk]] = {}

    def __getstate__(self) -> _ReaderState:
        """Return picklable state without open connections or decoded object caches."""
        return _ReaderState(
            self._version, self._packed_value_reader, tuple(tuple(batch) for batch in self._chunk_locator_batches)
        )

    def __setstate__(self, state: _ReaderState) -> None:
        """Restore prepared sources with fresh connections and decoded caches."""
        self.__init__(state.version)
        self._packed_value_reader = state.packed_values
        self._chunk_locator_batches = [list(batch) for batch in state.chunk_batches]
        self._chunk_locator_batch_by_key = {
            key: index for index, batch in enumerate(self._chunk_locator_batches) for key in batch
        }

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
            self._control.close()
            self._control = None
        for backend in self._owner_readers.values():
            backend.close()
        self._owner_readers.clear()
        self._records = None
        self._dataset = None
        self._tasks = None
        self._packed_value_reader.close()
        self._chunk_rows_cache.clear()

    def verify(self) -> None:
        """Verify the size and checksum of every artifact in the manifest.

        Raises:
            TimeFFormatError: If a listed artifact is missing or does not match its manifest entry.
        """
        for part in sorted(self._manifest.files.all_files(), key=lambda item: item.path):
            try:
                handle = self._fs.open_input_file(self._version.resolve_file_path(part.path))
            except FileNotFoundError as exc:
                raise TimeFFormatError(f"manifest lists a missing file: {part.path}") from exc
            with handle:
                actual_size = handle.size()
                if actual_size != part.size:
                    raise TimeFFormatError(
                        f"size mismatch for {part.path}: manifest says {part.size}, file is {actual_size}"
                    )
                actual_checksum = stream_checksum(handle)
            if actual_checksum != part.checksum:
                raise TimeFFormatError(
                    f"checksum mismatch for {part.path}: manifest says {part.checksum}, file is {actual_checksum}"
                )

    @property
    def metadata(self) -> DatasetMetadata:
        """Return the dataset's descriptive metadata."""
        return self._manifest.metadata

    @property
    def schema(self) -> DatasetSchema:
        """Return the dataset's type declaration."""
        return self._manifest.dataset_schema

    @property
    def tasks(self) -> tuple[Task, ...]:
        """Return all tasks with their object references restored."""
        if self._tasks is None:
            with self._as_format_error():
                self._tasks = self._control_reader().read_tasks(cached_records=self._all_records())
        return self._tasks

    def record_ids(self) -> tuple[str, ...]:
        """Return every Record ID without hydrating a Record.

        Returns:
            The IDs in the order :meth:`iter_records` yields them when called without arguments.
        """
        with self._as_format_error():
            return self._control_reader().record_ids()

    def read_tasks(self, records: Iterable[Record] | None = None) -> tuple[Task, ...]:
        """Hydrate the tasks attached to ``records`` in one pass.

        Args:
            records: Input Records to select, or ``None`` for every task. Tasks reuse these objects.

        Returns:
            Concrete tasks in stable ID order.
        """
        with self._as_format_error(preserve_validation=True):
            selected = None if records is None else tuple(records)
            tasks = self._control_reader().read_tasks(selected, eager=True)
            for record in selected or ():
                _materialize_record(record)
            return tasks

    def iter_tasks(
        self,
        records: Iterable[Record] | None = None,
        *,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
    ) -> Iterator[Task]:
        """Yield tasks a bounded batch at a time instead of materializing the whole table.

        Args:
            records: Records from :meth:`iter_records` whose input tasks to yield, or ``None`` for
                every task of the dataset. Tasks reference these ``Record`` objects directly.
            split: Keep only tasks of this split, or ``None`` for every task. The filter runs in
                DuckDB, so a task outside the split is never built.

        Yields:
            Concrete tasks in stable ID order.
        """
        filtered = required_modalities is not None or supported_modalities is not None or split is not None
        cached_records = self._all_records() if records is None and not filtered else ()
        with self._as_format_error(preserve_validation=True):
            yield from self._control_reader().iter_tasks(
                records,
                required_modalities=required_modalities,
                supported_modalities=supported_modalities,
                split=split,
                cached_records=cached_records,
            )

    def create_task_table(
        self,
        record_ids: Iterable[str] | None = None,
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
            :meth:`create_target_table`.
        """
        with self._as_format_error():
            table = self._control_reader().create_task_table(
                record_ids,
                required_modalities=required_modalities,
                supported_modalities=supported_modalities,
                split=split,
            )

            return materialize_table(
                table, self._packed_value_reader, self._control_reader().connection, ObjectKind.TASK
            )

    def create_target_table(
        self,
        record_ids: Iterable[str] | None = None,
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
        with self._as_format_error():
            table = self._control_reader().create_target_table(
                record_ids,
                required_modalities=required_modalities,
                supported_modalities=supported_modalities,
                split=split,
            )

            return materialize_table(
                table, self._packed_value_reader, self._control_reader().connection, ObjectKind.TASK
            )

    def create_annotation_table(self, object_type: ObjectKind | None = None) -> pa.Table:
        """Return every annotation occurrence with its content as an Arrow table.

        Args:
            object_type: Keep occurrences on this object kind, such as ``ObjectKind.RECORD``,
                or ``None`` for all annotated objects.

        Returns:
            One row per occurrence.
        """
        with self._as_format_error():
            table = self._control_reader().create_annotation_table(object_type)

            return materialize_table(
                table, self._packed_value_reader, self._control_reader().connection, ObjectKind.ANNOTATION
            )

    @property
    def values_backend(self) -> str:
        """Return the values-plane backend named by the manifest."""
        return self._manifest.values_backend

    def get_handle(self) -> DatasetHandle:
        """Return a lazy dataset handle without reading the catalog or payloads."""
        return DatasetHandle(self)

    def read(self) -> DatasetHandle:
        """Return a lazy dataset handle. Call its ``read()`` method to load all values."""
        return self.get_handle()

    def _read_dataset(self) -> TimeFDataset:
        """Return the cached dataset with every value materialized in memory."""
        if self._dataset is None:
            self._dataset = self._read_catalog(eager=True)
        return self._dataset

    def _read_catalog(self, *, eager: bool = False) -> TimeFDataset:
        """Hydrate the dataset with native values for eager reads and handles otherwise.

        Returns:
            The reconstructed dataset.
        """
        control = self._control_reader()
        records = list(control.read_records(eager=True) if eager else self._all_records())
        tasks = control.read_tasks(cached_records=records, eager=True) if eager else self.tasks
        by_id = {record.record_id: record for record in records}
        for task in tasks:
            for referenced in task.inputs:
                record = by_id.get(referenced.id)
                if record is not None and task.id not in record.task_ids:
                    record.task_ids = (*record.task_ids, task.id)
        return TimeFDataset.from_parts(
            metadata=self.metadata,
            records=records,
            tasks=tasks,
            schema=self.schema,
            annotations=control.read_dataset_annotations(self._manifest.dataset_id, eager=eager),
        )

    def iter_records(
        self,
        record_ids: Iterable[str] | None = None,
        *,
        with_annotations: bool = True,
    ) -> Iterator[Record]:
        """Yield complete records, optionally in a requested ID order.

        Args:
            record_ids: IDs to read in result order, or ``None`` for every record.
            with_annotations: Whether to hydrate annotations on the records and descendants.

        Yields:
            Hydrated records whose Signal values remain lazy.
        """
        with self._as_format_error(preserve_validation=True):
            yield from self._control_reader().read_records(
                record_ids,
                with_annotations=with_annotations,
            )

    def _control_reader(self) -> DuckDBControlReader:
        """Open and return the cached read-only control database.

        Returns:
            The cached control reader.
        """
        if self._control is None:
            self._control = DuckDBControlReader(
                materialize_control(self._version),
                value_loader_factory=self._make_signal_loader,
                external=self._packed_value_reader,
            )
        return self._control

    def _all_records(self) -> tuple[Record, ...]:
        """Return cached records so task inputs share their object identity."""
        if self._records is None:
            self._records = self._control_reader().read_records()
        return self._records

    def _values_reader(self) -> BaseValuesReader:
        """Return the lazily opened values-plane reader."""
        if self._values is None:
            self._values = make_values_reader(self._manifest.values_backend)
        return self._values

    def _load_signal(self, signal_key: int, signal_id: str, spec: TimeSeriesSpec) -> pa.Array:
        """Load all values for one Signal.

        Returns:
            The complete canonical Arrow array.
        """
        rows = self._chunk_rows(signal_key, signal_id)
        version, backend = self._storage_reader(rows)
        return backend.load(version, rows, spec)

    def _load_signal_range(
        self,
        signal_key: int,
        signal_id: str,
        spec: TimeSeriesSpec,
        start: int,
        stop: int,
    ) -> pa.Array:
        """Load a half-open step range for one Signal.

        Returns:
            The requested canonical Arrow values.
        """
        rows = self._chunk_rows(signal_key, signal_id)
        version, backend = self._storage_reader(rows)
        return backend.load_range(version, rows, start, stop, spec)

    def _storage_reader(self, rows: list[StoredChunk]) -> tuple[DatasetVersion, BaseValuesReader]:
        """Select one direct physical owner and its cached backend.

        Returns:
            The version handle and its backend.

        Raises:
            TimeFFormatError: If a signal mixes physical owners.
        """
        key = rows[0].owner_key
        if any(row.owner_key != key for row in rows):
            raise TimeFFormatError("a signal has chunks from multiple physical owners")
        if key is None:
            return self._version, self._values_reader()
        version = self._version.resolve_payload_source(key)
        if key not in self._owner_readers:
            self._owner_readers[key] = make_values_reader(version.manifest.values_backend)
        return version, self._owner_readers[key]

    def _chunk_rows(self, signal_key: int, signal_id: str) -> list[StoredChunk]:
        """Return cached chunk locators after loading the Signal's bounded batch.

        Raises:
            TimeFFormatError: If the Signal has no stored chunks.
        """
        cached = self._chunk_rows_cache.get(signal_key)
        if cached is not None:
            if not cached:
                raise TimeFFormatError(f"signal {signal_id!r} has no stored value chunks")
            return cached
        batch_index = self._chunk_locator_batch_by_key.get(signal_key)
        if batch_index is None:
            return self._control_reader().chunk_rows_by_key(signal_key, signal_id)
        batch = self._chunk_locator_batches[batch_index]
        self._chunk_rows_cache.update(self._control_reader().chunk_rows_by_keys(batch))
        rows = self._chunk_rows_cache[signal_key]
        if not rows:
            raise TimeFFormatError(f"signal {signal_id!r} has no stored value chunks")
        return rows

    def _make_signal_loader(self, signal_key: int, signal_id: str, spec: TimeSeriesSpec) -> StoredSignalLoader:
        """Return a picklable, range-aware Signal loader."""
        if signal_key not in self._chunk_locator_batch_by_key:
            if not self._chunk_locator_batches or len(self._chunk_locator_batches[-1]) >= _CHUNK_LOCATOR_BATCH_SIZE:
                self._chunk_locator_batches.append([])
            batch_index = len(self._chunk_locator_batches) - 1
            self._chunk_locator_batches[batch_index].append(signal_key)
            self._chunk_locator_batch_by_key[signal_key] = batch_index
        return StoredSignalLoader(self, signal_key, signal_id, spec)

    @contextmanager
    def _as_format_error(self, *, preserve_validation: bool = False) -> Iterator[None]:
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
