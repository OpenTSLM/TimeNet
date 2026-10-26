"""Read TimeF datasets from a DuckDB control plane and a sharded values plane."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import types as _types
from typing import TYPE_CHECKING, TypeAlias, TypeVar, cast

import pyarrow as pa

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFNotFoundError, TimeFValidationError
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
from timenet.types import Annotation, DatasetMetadata, DatasetSchema, InputModality, Task, TimeSeriesSpec
from timenet.types.object_kind import ObjectKind
from timenet.types.reference_keys import InputAlias, OwnerKey
from timenet.types.references import DatasetRef
from timenet.types.splits import Split
from timenet.values_backends.reader import BaseValuesReader, make_values_reader


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion


_CHUNK_LOCATOR_BATCH_SIZE = 1_024

Entity: TypeAlias = Record | Signal | Task
"""An object that :meth:`TimeFReader.get` reads by ID: a Record, a Signal, or a task.

Sources and annotations are read through the objects that hold them.
"""

E = TypeVar("E", bound=Entity)

_KINDS: tuple[tuple[type[Entity], ObjectKind], ...] = (
    (Record, ObjectKind.RECORD),
    (Signal, ObjectKind.SIGNAL),
    (Task, ObjectKind.TASK),
)
"""Each kind that :meth:`TimeFReader.get` reads, with its catalog tag."""


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
        self,
        *,
        inputs: Iterable[Record] | None = None,
        split: Split | str | None = None,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        load: LoadPlan = LoadPlan.ALL_LAZY,
    ) -> tuple[Task, ...]:
        """Read the tasks that match every filter given, in one pass; no filter reads every task.

        Every call builds new tasks. A record a task uses that is not among ``inputs`` is read once
        and shared by every task of the call.

        Args:
            inputs: Keep only tasks that use one of these Records as input, or ``None`` for every
                task. Tasks reuse these objects, which the read leaves as they are.
            split: Keep only tasks of this split, or ``None`` for every task.
            required_modalities: Keep only tasks that declare every one of these input modalities.
            supported_modalities: Keep only tasks whose declared input modalities are all among these.
            load: Which kinds of stored values to load into the objects the read builds. The
                default loads none.

        Returns:
            Concrete tasks in stable ID order.
        """
        with self._as_format_error(preserve_validation=True):
            selected = None if inputs is None else tuple(inputs)
            tasks = self._control_reader().read_tasks(
                selected,
                required_modalities=required_modalities,
                supported_modalities=supported_modalities,
                split=split,
            )
            values = ValueLoad(load, keep=selected or ())
            values.add(tasks)
            values.run()
            return tasks

    def iter_tasks(
        self,
        *,
        inputs: Iterable[Record] | None = None,
        split: Split | str | None = None,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        load: LoadPlan = LoadPlan.ALL_LAZY,
    ) -> Iterator[Task]:
        """Yield the tasks :meth:`read_tasks` reads, a bounded batch at a time.

        Each batch is built and loaded on its own. Its tasks share the objects it builds: a record
        that two of them use is one object, and the signals on one irregular axis share one
        time-offsets handle. Batches share nothing but ``inputs``, so the stream keeps none of a
        batch's objects once it has yielded them: it holds one batch of objects and the selected
        task keys at a time, while the reader keeps compact chunk locators for the signals it has
        resolved until it closes. A record or parent task that tasks of several batches use is
        built once per batch, as separate objects with the same ID, each loaded as far as its own
        batch's plan reaches it.

        Args:
            inputs: Keep only tasks that use one of these Records as input, or ``None`` for every
                task. Tasks reference these ``Record`` objects directly, and the read leaves them
                as they are.
            split: Keep only tasks of this split, or ``None`` for every task. The filter runs in
                DuckDB, so a task outside the split is never built.
            required_modalities: Keep only tasks that declare every one of these input modalities.
            supported_modalities: Keep only tasks whose declared input modalities are all among these.
            load: Which kinds of stored values to load into the objects each batch builds. The
                default loads none. A batch builds but does not load what it reaches only through
                its tasks' ``from_tasks`` parents, which is the parents and their records and
                annotations; a record or annotation that a task of the batch uses as well loads.

        Yields:
            Concrete tasks in stable ID order.
        """
        selected = None if inputs is None else tuple(inputs)
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

    def get(
        self,
        kind: type[E],
        id: str,
        *,
        alias: InputAlias | None = None,
        parent: DatasetRef | None = None,
        load: LoadPlan = LoadPlan.ALL_LAZY,
    ) -> E:
        """Read one object by its ID, from this dataset or imported from a parent.

        ``get(Task, "q1")`` reads the task with ID ``q1``; ``get(AnswerTask, "q1")`` also checks
        that it is an answer task. With ``alias`` or ``parent``, ``id`` is the ID the parent
        dataset gives the object, and the read returns this dataset's imported copy.

        Args:
            kind: ``Record``, ``Signal``, or a task class such as ``Task`` or ``AnswerTask``.
                Sources and annotations are read through the objects that hold them.
            id: The object's ID.
            alias: The input alias of the parent the object was imported from. Excludes ``parent``.
            parent: The exact parent release the object was imported from. Excludes ``alias``.
            load: Which kinds of stored values to load into the object and what it references. The
                default loads none.

        Returns:
            The object. A Signal is part of its Record, which the read builds too.

        Raises:
            TimeFNotFoundError: If the ID, alias, or parent names nothing, or the stored object is
                not a ``kind``.
            TimeFValidationError: If ``kind`` is ``Source``, ``Annotation``, or another kind not
                listed above, ``id`` is not a string, or both ``alias`` and ``parent`` are given.
            TimeFFormatError: If the catalog names another parent release for ``alias`` than the
                manifest does.
        """  # noqa: DOC502 - raised by get_many
        return self.get_many(kind, (id,), alias=alias, parent=parent, load=load)[0]

    def get_many(
        self,
        kind: type[E],
        ids: Iterable[str],
        *,
        alias: InputAlias | None = None,
        parent: DatasetRef | None = None,
        load: LoadPlan = LoadPlan.ALL_LAZY,
    ) -> tuple[E, ...]:
        """Read objects of one kind by their IDs, in one batch.

        The objects form one graph: a Record that two of them reference is one object, and the
        Signals on one irregular axis share one time-offsets handle. ``load`` loads their values in
        one batch.

        Args:
            kind: ``Record``, ``Signal``, or a task class such as ``Task`` or ``AnswerTask``.
                Sources and annotations are read through the objects that hold them.
            ids: The objects' IDs, such as a list or tuple. A single string is rejected, since it
                would read as one ID per character. With ``alias`` or ``parent``, the IDs the parent
                dataset gives the objects.
            alias: The input alias of the parent the objects were imported from. Excludes
                ``parent``.
            parent: The exact parent release the objects were imported from. Excludes ``alias``.
            load: Which kinds of stored values to load into the objects and what they reference.
                The default loads none.

        Returns:
            One object per ID, in the order of ``ids``. A repeated ID returns the same object.

        Raises:
            TimeFNotFoundError: If an ID, the alias, or the parent names nothing, or a stored object
                is not a ``kind``. Nothing is returned then.
            TimeFValidationError: If ``kind`` is ``Source``, ``Annotation``, or another kind not
                listed above, ``ids`` is a string or holds a non-string, or both ``alias`` and
                ``parent`` are given.
            TimeFFormatError: If the catalog names another parent release for ``alias`` than the
                manifest does.
        """  # noqa: DOC502 - TimeFFormatError is raised by _local_ids
        object_kind = _object_kind(kind)
        if isinstance(ids, str):
            raise TimeFValidationError(
                f"get_many takes a collection of IDs, not the string {ids!r}; call get() for one"
            )
        try:
            requested = tuple(ids)
        except TypeError as exc:
            raise TimeFValidationError(f"get_many takes a collection of IDs, got {type(ids).__name__}") from exc
        for identity in requested:
            if not isinstance(identity, str):
                raise TimeFValidationError(f"object IDs are strings, got {type(identity).__name__} {identity!r}")
        if alias is not None and parent is not None:
            raise TimeFValidationError("choose either an input alias or an exact parent release, not both")
        imported_from = None if alias is None and parent is None else self._import_alias(alias, parent)
        with self._as_format_error(preserve_validation=True):
            local = requested if imported_from is None else self._local_ids(imported_from, object_kind, requested)
            unique = tuple(dict.fromkeys(local))
            objects = self._read_by_id(object_kind, unique)
            for identity in unique:
                if not isinstance(objects[identity], kind):
                    raise TimeFNotFoundError(
                        f"no {kind.__name__} {identity!r}: the stored object is a {type(objects[identity]).__name__}"
                    )
            values = ValueLoad(load)
            values.add(objects.values())
            values.run()
            return tuple(cast("E", objects[identity]) for identity in local)

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

    def _import_alias(self, alias: InputAlias | None, parent: DatasetRef | None) -> InputAlias:
        """Return the input alias that ``alias`` or ``parent`` selects, from the manifest's parent cards.

        Returns:
            The alias.

        Raises:
            TimeFValidationError: If the alias is blank or not a string, ``parent`` is not a
                ``DatasetRef``, or the parent release is imported under more than one alias.
            TimeFNotFoundError: If the dataset declares no such alias, or imports from no such
                parent release.
        """
        cards = self._manifest.parent_cards
        dataset_id = self._manifest.dataset_id
        if alias is not None:
            if not isinstance(alias, str) or not alias.strip():
                raise TimeFValidationError(f"an input alias must be a nonblank string, got {alias!r}")
            if alias not in cards:
                raise TimeFNotFoundError(f"{dataset_id} declares no input alias {alias!r}")
            return alias
        if not isinstance(parent, DatasetRef):
            raise TimeFValidationError(f"parent must be an exact DatasetRef, got {parent!r}")
        matches = sorted(name for name, card in cards.items() if card.source == parent)
        if not matches:
            raise TimeFNotFoundError(f"{dataset_id} declares no input from {parent}")
        if len(matches) > 1:
            raise TimeFValidationError(f"{dataset_id} imports {parent} under the aliases {matches}; pass alias=")
        return matches[0]

    def _local_ids(self, alias: InputAlias, kind: ObjectKind, parent_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Translate the IDs a parent gives its objects into the IDs of this dataset's copies.

        Returns:
            The local IDs, one per parent ID, in order.

        Raises:
            TimeFFormatError: If the catalog names another parent release for ``alias`` than the
                manifest does.
            TimeFNotFoundError: If a parent ID names no object imported under ``alias``.
        """
        parent_ref, imported = self._control_reader().imported_ids(alias, kind, dict.fromkeys(parent_ids))
        if parent_ref is not None and parent_ref != str(self._manifest.parent_cards[alias].source):
            raise TimeFFormatError(f"the catalog's parent release for input alias {alias!r} differs from its manifest")
        missing = [identity for identity in parent_ids if identity not in imported]
        if missing:
            raise TimeFNotFoundError(
                f"no {kind.value} {', '.join(map(repr, missing))} imported under input alias {alias!r}"
            )
        return tuple(imported[identity] for identity in parent_ids)

    def _read_by_id(self, kind: ObjectKind, ids: tuple[str, ...]) -> dict[str, Entity]:
        """Build the objects of one kind with these IDs, each once, with every value unloaded.

        Returns:
            The objects by ID.
        """
        objects: dict[str, Entity] = {}
        if not ids:
            return objects
        control = self._control_reader()
        if kind is ObjectKind.RECORD:
            objects.update(zip(ids, control.read_records(ids), strict=True))
        elif kind is ObjectKind.TASK:
            objects.update(control.read_tasks_by_id(ids))
        else:
            objects.update(control.read_signals_by_id(ids))
        return objects

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


def _object_kind(kind: object) -> ObjectKind:
    """Return the catalog tag of a kind of object that :meth:`TimeFReader.get` reads.

    Returns:
        The tag.

    Raises:
        TimeFValidationError: If ``kind`` is ``Source``, ``Annotation``, or anything else that is
            not ``Record``, ``Signal``, a subclass of either, or a task class.
    """
    if isinstance(kind, type):
        for base, tag in _KINDS:
            if issubclass(kind, base):
                return tag
    hint = (
        "; sources and annotations are read through the objects that hold them"
        if kind is Source or kind is Annotation
        else ""
    )
    raise TimeFValidationError(f"get reads a Record, Signal, or task class such as AnswerTask, got {kind!r}{hint}")
