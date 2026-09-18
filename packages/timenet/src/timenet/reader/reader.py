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
from timenet.types import DatasetMetadata, DatasetSchema, Task, TimeSeriesSpec
from timenet.values_backends.reader import BaseValuesReader, make_values_reader


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion


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
        self._fs = version.filesystem
        self._root = version.root
        self._manifest = version.manifest
        self._tasks: tuple[Task, ...] | None = None
        self._control: DuckDBControlReader | None = None
        self._records: tuple[Record, ...] | None = None
        self._values: BaseValuesReader | None = None
        self._chunk_locator_batches: list[list[int]] = []
        self._chunk_locator_batch_by_key: dict[int, int] = {}
        self._chunk_rows_cache: dict[int, list[dict]] = {}

    def __getstate__(self) -> dict:
        """Return picklable state without open connections or decoded object caches."""
        state = self.__dict__.copy()
        state["_tasks"] = None
        state["_control"] = None
        state["_records"] = None
        state["_values"] = None
        state["_chunk_rows_cache"] = {}
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
            self._control.close()
            self._control = None
        self._records = None
        self._tasks = None
        self._chunk_rows_cache.clear()

    def verify(self) -> None:
        """Verify the size and checksum of every artifact in the manifest.

        Raises:
            TimeFFormatError: If a listed artifact is missing or does not match its manifest entry.
        """
        for part in sorted(self._manifest.files.all_files(), key=lambda item: item.path):
            try:
                handle = self._fs.open_input_file(self._version.path(part.path))
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
        return self._manifest.schema

    @property
    def tasks(self) -> tuple[Task, ...]:
        """Return all tasks with their object references restored."""
        if self._tasks is None:
            with self._as_format_error():
                self._tasks = self._control_reader().read_tasks(self._all_records())
        return self._tasks

    @property
    def values_backend(self) -> str:
        """Return the values-plane backend named by the manifest."""
        return self._manifest.values_backend

    def read(self) -> TimeFDataset:
        """Hydrate the dataset while keeping every Signal's values lazy.

        Returns:
            The reconstructed dataset.
        """
        records = list(self._all_records())
        tasks = self.tasks
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
            annotations=self._control_reader().read_dataset_annotations(self._manifest.dataset_id),
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
                offsets_loader=self._load_offsets,
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
        return self._values_reader().load(self._version, rows, spec)

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
        return self._values_reader().load_range(self._version, rows, start, stop, spec)

    def _chunk_rows(self, signal_key: int, signal_id: str) -> list[dict]:
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

    def _make_signal_loader(self, signal_key: int, signal_id: str, spec: TimeSeriesSpec) -> _SignalLoader:
        """Return a picklable, range-aware Signal loader."""
        if signal_key not in self._chunk_locator_batch_by_key:
            if not self._chunk_locator_batches or len(self._chunk_locator_batches[-1]) >= _CHUNK_LOCATOR_BATCH_SIZE:
                self._chunk_locator_batches.append([])
            batch_index = len(self._chunk_locator_batches) - 1
            self._chunk_locator_batches[batch_index].append(signal_key)
            self._chunk_locator_batch_by_key[signal_key] = batch_index
        return _SignalLoader(self, signal_key, signal_id, spec)

    def _load_offsets(self, axis_key: int, axis_id: str) -> pa.Array:
        """Load the stored offsets for one irregular TimeAxis.

        Returns:
            The axis offsets in microseconds.
        """
        return self._control_reader().load_axis_offsets_by_key(axis_key, axis_id)

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


@dataclass(frozen=True)
class _SignalLoader:
    """Picklable values loader for one DuckDB-indexed Signal."""

    reader: TimeFReader
    signal_key: int
    signal_id: str
    spec: TimeSeriesSpec

    def __call__(self) -> pa.Array:
        """Load all Signal values.

        Returns:
            The complete canonical Arrow array.
        """
        return self.reader._load_signal(self.signal_key, self.signal_id, self.spec)

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Load a half-open Signal step range.

        Returns:
            The requested canonical Arrow values.
        """
        return self.reader._load_signal_range(self.signal_key, self.signal_id, self.spec, start, stop)
