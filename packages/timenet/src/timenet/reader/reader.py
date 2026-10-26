"""Read TimeF datasets from a DuckDB control plane and a sharded values plane."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import types as _types
from typing import TYPE_CHECKING

import pyarrow as pa

from timenet.dataset import Record, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.checksums import stream_checksum
from timenet.format.control_cache import materialize_control
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.stored_array import ChunkLocation, StoredArray
from timenet.format.stored_values import (
    ANNOTATION_COLUMNS,
    TARGET_COLUMNS,
    TASK_COLUMNS,
    PackedValueReader,
    materialize_table,
)
from timenet.reader.load_plan import LoadPlan, ValueLoad
from timenet.types import DatasetMetadata, DatasetSchema, InputModality, Task, TimeSeriesSpec
from timenet.types.object_kind import ObjectKind
from timenet.types.reference_keys import OwnerKey
from timenet.types.splits import Split
from timenet.values_backends.reader import BaseValuesReader, make_values_reader


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion


_CHUNK_LOCATOR_BATCH_SIZE = 1_024


@dataclass(frozen=True)
class _ReaderState:
    """Keep only prepared sources and the signals' chunk-location batches across worker processes."""

    version: DatasetVersion
    packed_values: PackedValueReader
    """Preserve the shared reader referenced by text and time-offset handles; its cache resets on pickle."""
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
        self._control: DuckDBControlReader | None = None
        self._values: BaseValuesReader | None = None
        self._owner_readers: dict[OwnerKey, BaseValuesReader] = {}
        self._chunk_locator_batches: list[list[int]] = []
        self._chunk_locator_batch_by_key: dict[int, int] = {}
        self._chunk_locations: dict[int, ChunkLocation | None] = {}

    def __getstate__(self) -> _ReaderState:
        """Return picklable state without open connections or decoded caches."""
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
            # The control reader stays, so it reopens its verified local catalog without checking it again.
            self._control.close()
        for backend in self._owner_readers.values():
            backend.close()
        self._owner_readers.clear()
        self._packed_value_reader.close()
        self._chunk_locations.clear()

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

    def record_ids(self) -> tuple[str, ...]:
        """Return every Record ID without hydrating a Record.

        Returns:
            The IDs in the order :meth:`iter_records` yields them when called without arguments.
        """
        with self._as_format_error():
            return self._control_reader().record_ids()

    def read_tasks(
        self, records: Iterable[Record] | None = None, *, load: LoadPlan = LoadPlan.ALL_LAZY
    ) -> tuple[Task, ...]:
        """Read the tasks attached to ``records`` in one pass.

        Every call builds new tasks. A record a task uses that is not among ``records`` is read
        once and shared by every task of the call.

        Args:
            records: Input Records to select, or ``None`` for every task. Tasks reuse these objects,
                which the read leaves as they are.
            load: Which kinds of stored values to load into the objects the read builds. The
                default loads none.

        Returns:
            Concrete tasks in stable ID order.
        """
        with self._as_format_error(preserve_validation=True):
            selected = None if records is None else tuple(records)
            tasks = self._control_reader().read_tasks(selected)
            values = ValueLoad(load, keep=selected or ())
            values.add(tasks)
            values.run()
            return tasks

    def iter_tasks(
        self,
        records: Iterable[Record] | None = None,
        *,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        split: Split | str | None = None,
        load: LoadPlan = LoadPlan.ALL_LAZY,
    ) -> Iterator[Task]:
        """Yield tasks a bounded batch at a time instead of building the whole table.

        Each batch is built and loaded on its own. Its tasks share the objects it builds: a record
        that two of them use is one object, and the signals on one irregular axis share one
        time-offsets handle. Batches share nothing but ``records``, so the stream keeps none of a
        batch's objects once it has yielded them: it holds one batch of objects and the selected
        task keys at a time, while the reader keeps compact chunk locators for the signals it has
        resolved until it closes. A record or parent task that tasks of several batches use is
        built once per batch, as separate objects with the same ID, each loaded as far as its own
        batch's plan reaches it.

        Args:
            records: Records from :meth:`iter_records` whose input tasks to yield, or ``None`` for
                every task of the dataset. Tasks reference these ``Record`` objects directly, and
                the read leaves them as they are.
            split: Keep only tasks of this split, or ``None`` for every task. The filter runs in
                DuckDB, so a task outside the split is never built.
            load: Which kinds of stored values to load into the objects each batch builds. The
                default loads none. A batch builds but does not load what it reaches only through
                its tasks' ``from_tasks`` parents, which is the parents and their records and
                annotations; a record or annotation that a task of the batch uses as well loads.

        Yields:
            Concrete tasks in stable ID order.
        """
        selected = None if records is None else tuple(records)
        values = ValueLoad(load, keep=selected or ())
        with self._as_format_error(preserve_validation=True):
            for batch in self._control_reader().iter_task_batches(
                selected,
                required_modalities=required_modalities,
                supported_modalities=supported_modalities,
                split=split,
            ):
                values.add(batch)
                values.run()
                yield from batch
                # Hold no batch while the next one is built.
                del batch

    def read_task_table(
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
            :meth:`read_target_table`.
        """
        with self._as_format_error():
            control = self._control_reader()
            table = control.read_task_table(
                record_ids,
                required_modalities=required_modalities,
                supported_modalities=supported_modalities,
                split=split,
            )

            return control.run(
                lambda connection: materialize_table(table, self._packed_value_reader, connection, TASK_COLUMNS)
            )

    def read_target_table(
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
            control = self._control_reader()
            table = control.read_target_table(
                record_ids,
                required_modalities=required_modalities,
                supported_modalities=supported_modalities,
                split=split,
            )

            return control.run(
                lambda connection: materialize_table(table, self._packed_value_reader, connection, TARGET_COLUMNS)
            )

    def read_annotation_table(self, object_type: ObjectKind | None = None) -> pa.Table:
        """Return every annotation occurrence with its content as an Arrow table.

        Args:
            object_type: Keep occurrences on this object kind, such as ``ObjectKind.RECORD``,
                or ``None`` for all annotated objects.

        Returns:
            One row per occurrence.
        """
        with self._as_format_error():
            control = self._control_reader()
            table = control.read_annotation_table(object_type)

            return control.run(
                lambda connection: materialize_table(table, self._packed_value_reader, connection, ANNOTATION_COLUMNS)
            )

    @property
    def values_backend(self) -> str:
        """Return the values-plane backend named by the manifest."""
        return self._manifest.values_backend

    def read(self, *, load: LoadPlan = LoadPlan.ALL_LAZY) -> TimeFDataset:
        """Read the whole dataset, loading the kinds of stored values that ``load`` names.

        Every call builds new objects. With the default, every stored value stays behind its value
        handle and reads through to storage on access. With ``LoadPlan.ALL_EAGER``, the dataset
        needs no storage after the reader closes.

        Args:
            load: Which kinds of stored values to load into the objects the read builds.

        Returns:
            The dataset.
        """
        control = self._control_reader()
        records = control.read_records()
        annotations = control.read_dataset_annotations(self._manifest.dataset_id)
        with self._as_format_error():
            tasks = control.read_tasks(cached_records=records, dataset_annotations=annotations)
        by_id = {record.record_id: record for record in records}
        for task in tasks:
            for referenced in task.inputs:
                record = by_id.get(referenced.id)
                if record is not None and task.id not in record.task_ids:
                    record.task_ids = (*record.task_ids, task.id)
        with self._as_format_error():
            values = ValueLoad(load)
            values.add((*records, *tasks, *annotations))
            values.run()
        return TimeFDataset.from_parts(
            metadata=self.metadata, records=records, tasks=tasks, schema=self.schema, annotations=annotations
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
            Hydrated records whose stored values stay behind their value handles.
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
        version, backend = self._storage_reader(location.owner)
        return backend.load(version, location.chunks, spec)

    def _read_array_steps(
        self,
        location: ChunkLocation,
        spec: TimeSeriesSpec,
        start: int,
        stop: int,
    ) -> pa.Array:
        """Read a half-open step range of one Signal from the release that stores it.

        Returns:
            The requested canonical Arrow values.
        """
        version, backend = self._storage_reader(location.owner)
        return backend.load_range(version, location.chunks, start, stop, spec)

    def _storage_reader(self, owner: OwnerKey | None) -> tuple[DatasetVersion, BaseValuesReader]:
        """Return the release that an owner key names, or this release for ``None``, and its values reader.

        Returns:
            The version handle and its backend.
        """
        if owner is None:
            return self._version, self._values_reader()
        version = self._version.resolve_payload_source(owner)
        if owner not in self._owner_readers:
            self._owner_readers[owner] = make_values_reader(version.manifest.values_backend)
        return version, self._owner_readers[owner]

    def _chunk_location(self, signal_key: int, signal_id: str) -> ChunkLocation:
        """Return where one Signal's chunks are stored, finding them for its batch on first use.

        Raises:
            TimeFFormatError: If the Signal has no stored chunks.
        """
        if signal_key not in self._chunk_locations:
            batch_index = self._chunk_locator_batch_by_key.get(signal_key)
            if batch_index is None:
                return self._control_reader().chunk_location_by_key(signal_key, signal_id)
            batch = self._chunk_locator_batches[batch_index]
            self._chunk_locations.update(self._control_reader().chunk_locations_by_keys(batch))
        location = self._chunk_locations[signal_key]
        if location is None:
            raise TimeFFormatError(f"signal {signal_id!r} has no stored value chunks")
        return location

    def _make_signal_loader(self, signal_key: int, signal_id: str, spec: TimeSeriesSpec) -> StoredArray:
        """Return a picklable, range-aware handle to one Signal's stored values, unloaded."""
        if signal_key not in self._chunk_locator_batch_by_key:
            if not self._chunk_locator_batches or len(self._chunk_locator_batches[-1]) >= _CHUNK_LOCATOR_BATCH_SIZE:
                self._chunk_locator_batches.append([])
            batch_index = len(self._chunk_locator_batches) - 1
            self._chunk_locator_batches[batch_index].append(signal_key)
            self._chunk_locator_batch_by_key[signal_key] = batch_index
        return StoredArray(self, signal_key, signal_id, spec)

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
