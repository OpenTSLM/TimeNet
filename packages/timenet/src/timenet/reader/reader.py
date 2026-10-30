"""Read TimeF datasets from a DuckDB control plane and a sharded values plane."""

from __future__ import annotations

from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import types as _types
from typing import TYPE_CHECKING, TypeAlias, TypeVar, cast

import pyarrow as pa

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.composition import ImportMap
from timenet.errors import (
    TimeFFormatError,
    TimeFNotFoundError,
    TimeFValidationError,
    TimeNetFileError,
    TimeNetMissingParentError,
)
from timenet.format.checksums import stream_checksum
from timenet.format.control_cache import materialize_control
from timenet.format.control_reader import RECORD_BATCH_SIZE, TASK_BATCH_SIZE, DuckDBControlReader
from timenet.format.stored_array import ChunkLocation, StoredArray
from timenet.format.stored_values import (
    ANNOTATION_COLUMNS,
    TARGET_COLUMNS,
    TASK_COLUMNS,
    PackedValueReader,
    StoredValueColumns,
    materialize_table,
)
from timenet.reader.load_plan import LoadPlan, ValueLoad
from timenet.registry.dependencies import open_local_version
from timenet.types import (
    Annotation,
    DatasetMetadata,
    DatasetSchema,
    EntityId,
    InputModality,
    ObjectKind,
    Task,
    TimeSeriesSpec,
    check_id,
)
from timenet.types.reference_keys import ParentAlias, SourceKey
from timenet.types.references import DatasetRef
from timenet.types.splits import Split
from timenet.values_backends.reader import BaseValuesReader, make_values_reader


if TYPE_CHECKING:
    from pathlib import Path

    from timenet.manifest import FilePart
    from timenet.registry.version import DatasetVersion
    from timenet.types.references import DatasetRef


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
    packed_values: PackedValueReader | None
    """Preserve the shared reader referenced by text and time-offset handles; its cache resets on pickle."""
    chunk_batches: tuple[tuple[int, ...], ...]


class TimeFReader:  # noqa: PLR0904 - one public method per kind of read
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

    @classmethod
    def open(cls, path: str | Path, *, sources_root: str | Path | None = None) -> TimeFReader:
        """Open a local version directory for reading.

        Opening reads only metadata, as :func:`~timenet.registry.dependencies.open_local_version`
        does. The reader opens the catalog on its first read. Close the reader after use, or use it
        as a context manager.

        Args:
            path: The version directory, which holds ``manifest.json`` and ``control.duckdb``.
            sources_root: The registry root to find payload sources under. By default this is the
                root that holds ``path`` as ``<root>/<org>/<name>/<version>``.

        Returns:
            A reader for the version.

        Raises:
            FileNotFoundError: If ``path`` has no manifest.
            TimeFValidationError: If the version has payload sources, is not stored as
                ``<root>/<org>/<name>/<version>``, and no ``sources_root`` is given.
            TimeNetMissingParentError: If a payload source is not under the sources root.
        """  # noqa: DOC502 - raised by open_local_version
        try:
            return cls(open_local_version(path, sources_root=sources_root))
        except TimeNetMissingParentError as error:
            error.hint = (
                "Pass sources_root= with the registry root that holds it, or download the release with "
                "its payload sources through TimeNet.download()."
            )
            raise

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
        self._chunk_locator_batch_by_id = {
            signal_id: index for index, batch in enumerate(self._chunk_locator_batches) for signal_id in batch
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
            TimeNetFileError: If a file is missing, or its size or checksum differs from the manifest
                record that describes it: its own record, or the pin of a payload source file.
        """  # noqa: DOC502 - raised by _verify_files
        release = DatasetRef(dataset_id=self._manifest.dataset_id, version=self._manifest.metadata.dataset_version)
        _verify_files(self._version, self._manifest.files.all_files(), release=release, owner=str(release))
        for key, pinned in self._manifest.payload_sources.items():
            owner = f"payload source {pinned.release} of {release}"
            _verify_files(self._version.resolve_payload_source(key), pinned.files, release=pinned.release, owner=owner)

    @property
    def metadata(self) -> DatasetMetadata:
        """Return the dataset's descriptive metadata."""
        return self._manifest.metadata

    @property
    def schema(self) -> DatasetSchema:
        """Return the dataset's type declaration."""
        return self._manifest.dataset_schema

    def record_ids(self) -> tuple[EntityId, ...]:
        """Return the ID of every Record, without building the Records.

        The IDs come in the release's record order, which `iter_records()` and `read()` also use.
        Every reader of a release returns the same order. In a release with imported Records, the
        order follows where the values are stored, so that a walk over the records reads the stored
        values in sequence. In any other release, the order is by ID.

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

    def iter_tasks(  # noqa: PLR0913 - one argument per task selection axis
        self,
        *,
        inputs: Iterable[Record] | None = None,
        split: Split | str | None = None,
        required_modalities: Iterable[InputModality] | None = None,
        supported_modalities: Iterable[InputModality] | None = None,
        load: LoadPlan = LoadPlan.ALL_LAZY,
        batch_size: int = TASK_BATCH_SIZE,
    ) -> Iterator[Task]:
        """Yield the tasks :meth:`read_tasks` reads, a bounded batch at a time.

        Each batch is built and loaded on its own. Its tasks share the objects it builds: a record
        that two of them use is one object, and the signals on one irregular axis share one
        time-offsets handle. Batches share nothing but ``inputs``, so the stream keeps none of a
        batch's objects once it has yielded them: it holds one batch of objects and that batch's
        task IDs at a time, and selects the next batch's IDs after the last one, while the reader
        keeps compact chunk locators for the signals it has resolved until it closes. A record or
        parent task that tasks of several batches use is built once per batch, as separate objects
        with the same ID, each loaded as far as its own batch's plan reaches it.

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
            batch_size: The most tasks a batch holds. By default, a batch holds up to 5,000 tasks
                (``TASK_BATCH_SIZE``). Pass another value for larger or smaller batches. The
                stream holds one batch at a time, with every value its plan loaded, so with an
                eager plan a smaller batch holds fewer arrays in memory.

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
                batch_size=batch_size,
            ):
                values.add(batch)
                values.run()
                yield from batch
                # Hold no batch while the next one is built.
                del batch

    def get(
        self,
        kind: type[E],
        id: EntityId,
        *,
        parent: ParentAlias | DatasetRef | None = None,
        load: LoadPlan = LoadPlan.ALL_LAZY,
    ) -> E:
        """Read one object by its ID, from this dataset or imported from a parent.

        An ID is an opaque handle. It stays the same across edits of one dataset, and it means
        something only together with that dataset's reference: an object imported from a parent is
        found by the parent's ID, with ``parent`` naming the parent. An ID is not a lookup key for
        content; to find an object by what it is, read objects with :meth:`iter_records` or
        :meth:`read_tasks` and check their ``metadata``.

        ``get(Task, 7)`` reads the task with ID 7; ``get(AnswerTask, 7)`` also checks that it is an
        answer task. With ``parent``, ``id`` is the ID the parent dataset gives the object, and the
        read returns this dataset's imported copy.

        Args:
            kind: ``Record``, ``Signal``, or a task class such as ``Task`` or ``AnswerTask``.
                Sources and annotations are read through the objects that hold them.
            id: The object's ID, a positive int64.
            parent: The parent the object was imported from: the ``ParentAlias`` it was imported
                under, or its exact release as a ``DatasetRef`` when one alias alone imports it.
            load: Which kinds of stored values to load into the object and what it references. The
                default loads none.

        Returns:
            The object. A Signal is part of its Record, which the read builds too.

        Raises:
            TimeFNotFoundError: If the ID or ``parent`` names nothing, or the stored object is not a
                ``kind``.
            TimeFValidationError: If ``kind`` is ``Source``, ``Annotation``, or another kind not
                listed above, ``id`` is not a positive int64, such as a string, ``parent`` is
                neither a ``ParentAlias`` nor a ``DatasetRef``, such as a plain string, or several
                aliases import the ``parent`` release.
            TimeFFormatError: If the catalog names no parent release or another one for the alias
                than the manifest does, or lists the alias or a parent ID more than once.
        """  # noqa: DOC502 - raised by get_many
        return self.get_many(kind, (id,), parent=parent, load=load)[0]

    def get_many(
        self,
        kind: type[E],
        ids: Iterable[EntityId],
        *,
        parent: ParentAlias | DatasetRef | None = None,
        load: LoadPlan = LoadPlan.ALL_LAZY,
    ) -> tuple[E, ...]:
        """Read objects of one kind by their IDs, in one batch.

        IDs are opaque handles, as :meth:`get` describes: stable across edits of this dataset,
        meaningful only with its reference, and no key for content. The objects form one graph: a
        Record that two of them reference is one object, and the Signals on one irregular axis
        share one time-offsets handle. ``load`` loads their values in one batch.

        Args:
            kind: ``Record``, ``Signal``, or a task class such as ``Task`` or ``AnswerTask``.
                Sources and annotations are read through the objects that hold them.
            ids: The objects' IDs, positive int64s in a list, tuple, or other iterable. With
                ``parent``, the IDs the parent dataset gives the objects.
            parent: The parent the objects were imported from: the ``ParentAlias`` they were
                imported under, or its exact release as a ``DatasetRef`` when one alias alone
                imports it.
            load: Which kinds of stored values to load into the objects and what they reference.
                The default loads none.

        Returns:
            One object per ID, in the order of ``ids``. A repeated ID returns the same object.

        Raises:
            TimeFNotFoundError: If an ID or ``parent`` names nothing, or a stored object is not a
                ``kind``. Nothing is returned then.
            TimeFValidationError: If ``kind`` is ``Source``, ``Annotation``, or another kind not
                listed above, ``ids`` is a string or holds anything but positive int64s,
                ``parent`` is neither a ``ParentAlias`` nor a ``DatasetRef``, such as a plain
                string, or several aliases import the ``parent`` release.
            TimeFFormatError: If the catalog names no parent release or another one for the alias
                than the manifest does, or lists the alias or a parent ID more than once.
        """  # noqa: DOC502 - TimeFFormatError is raised by _local_ids
        object_kind = _object_kind(kind)
        if isinstance(ids, str | bytes):
            raise TimeFValidationError(
                f"get_many takes a collection of IDs, not the string {ids!r}; call get() for one"
            )
        try:
            items = tuple(ids)
        except TypeError as exc:
            raise TimeFValidationError(f"get_many takes a collection of IDs, got {type(ids).__name__}") from exc
        requested = tuple(check_id(identity, f"{kind.__name__} id") for identity in items)
        imported_from = None if parent is None else self._import_alias(parent)
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

    def read(self, *, load: LoadPlan = LoadPlan.ALL_LAZY) -> TimeFDataset:
        """Read the whole dataset, loading the kinds of stored values that ``load`` names.

        Every call builds new objects. With the default, every stored value stays behind its value
        handle and reads through to storage on access. With ``LoadPlan.ALL_EAGER``, the dataset
        needs no storage after the reader closes.

        The dataset gets the parent cards and the lineage of the manifest, and the import map of the
        catalog, so a later import into it reuses the copies it holds instead of copying them again.

        Args:
            load: Which kinds of stored values to load into the objects the read builds.

        Returns:
            The dataset.

        Raises:
            TimeFFormatError: If the catalog's parent aliases or their parent releases differ from the
                manifest's parent cards, or the catalog lists an alias or a parent ID more than once.
        """  # noqa: DOC502 - raised by _import_map
        imports = self._import_map()
        control = self._control_reader()
        records = control.read_records()
        occurrences: dict[int, Annotation] = {}
        annotations = control.read_dataset_annotations(self._manifest.dataset_id, by_occurrence=occurrences)
        with self._as_format_error():
            tasks = control.read_tasks(cached_records=records, dataset_annotations=occurrences)
            values = ValueLoad(load)
            values.add((*records, *tasks, *annotations))
            values.run()
        dataset = TimeFDataset.from_parts(
            metadata=self.metadata,
            records=records,
            tasks=tasks,
            schema=self.schema,
            annotations=annotations,
            last_ids=control.read_last_ids(),
        )
        dataset.parent_cards = dict(self._manifest.parent_cards)
        dataset.lineage = {str(card.source): card for card in self._manifest.lineage}
        dataset._imports = imports
        return dataset

    def _import_map(self) -> ImportMap:
        """Return the import map of the catalog, checked against the manifest's parent cards.

        Returns:
            The child ID of each imported object the catalog lists, keyed by parent alias, kind, and
            parent ID.

        Raises:
            TimeFFormatError: If the catalog's parent aliases or their parent releases differ from the
                manifest's parent cards, or the catalog lists an alias or a parent ID more than once.
        """
        with self._as_format_error():
            releases, imports = self._control_reader().import_map()
        cards = {alias: str(card.source) for alias, card in self._manifest.parent_cards.items()}
        if releases != cards:
            raise TimeFFormatError(
                f"the parent aliases and parent releases in the catalog at {self._root} differ from its manifest's "
                f"parent cards: {releases} and {cards}"
            )
        return imports

    def iter_records(
        self,
        record_ids: Iterable[EntityId] | None = None,
        *,
        with_annotations: bool = True,
        load: LoadPlan = LoadPlan.ALL_LAZY,
        batch_size: int = RECORD_BATCH_SIZE,
    ) -> Iterator[Record]:
        """Yield complete records a bounded batch at a time, in a requested ID order or in record order.

        Each batch is built and loaded on its own, before the stream yields it. Its records share
        the objects it builds: the signals on one irregular axis share one time-offsets handle.
        Batches share nothing, so the stream keeps none of a batch's objects once it has yielded
        them: it holds one batch of objects at a time. A record ID that two batches request is
        built once per batch, as separate objects. Every requested ID is checked before the first
        batch is built.

        Args:
            record_ids: IDs to read in result order, or ``None`` for every record in the order of
                :meth:`record_ids`.
            with_annotations: Whether to build annotations on the records and descendants.
            load: Which kinds of stored values to load into the records each batch builds. The
                default loads none.
            batch_size: The most records a batch holds. By default, a batch holds up to 1,000
                records (``RECORD_BATCH_SIZE``). The stream holds one batch at a time, with every
                value its plan loaded, so with an eager plan a smaller batch holds fewer arrays in
                memory.

        Yields:
            Records with the values the plan names loaded. Other stored values stay behind their
            value handles.
        """
        with self._as_format_error(preserve_validation=True):
            for batch in self._control_reader().iter_record_batches(
                record_ids,
                with_annotations=with_annotations,
                batch_size=batch_size,
            ):
                values = ValueLoad(load)
                values.add(batch)
                values.run()
                yield from batch
                # Hold no batch while the next one is built.
                del batch

    def _import_alias(self, parent: ParentAlias | DatasetRef) -> ParentAlias:
        """Return the alias that ``parent`` names among the manifest's parent cards.

        Returns:
            The alias: ``parent`` itself, or the one alias whose parent card holds the release
            ``parent`` names.

        Raises:
            TimeFValidationError: If ``parent`` is neither a ``ParentAlias`` nor a ``DatasetRef``,
                or several aliases import the release.
            TimeFNotFoundError: If the dataset declares no such alias, or imports from no such
                release.
        """
        cards = self._manifest.parent_cards
        dataset_id = self._manifest.dataset_id
        if isinstance(parent, DatasetRef):
            matches = sorted(alias for alias, card in cards.items() if card.source == parent)
            if not matches:
                raise TimeFNotFoundError(f"{dataset_id} declares no parent {parent}")
            if len(matches) > 1:
                raise TimeFValidationError(
                    f"{dataset_id} imports {parent} under the parent aliases {matches}; "
                    "pass parent=ParentAlias(...) to choose one"
                )
            return ParentAlias(matches[0])
        if isinstance(parent, ParentAlias):
            if parent not in cards:
                raise TimeFNotFoundError(f"{dataset_id} declares no parent alias {parent!r}")
            return parent
        raise TimeFValidationError(
            f"parent must be a ParentAlias or a DatasetRef, got {parent!r}; wrap an alias name as "
            "ParentAlias(...) or an exact release as DatasetRef.parse(...)"
        )

    def _local_ids(
        self, alias: ParentAlias, kind: ObjectKind, parent_ids: tuple[EntityId, ...]
    ) -> tuple[EntityId, ...]:
        """Translate the IDs a parent gives its objects into the IDs of this dataset's copies.

        Returns:
            The local IDs, one per parent ID, in order.

        Raises:
            TimeFFormatError: If the catalog names no parent release or another one for ``alias``
                than the manifest does, or lists ``alias`` or a parent ID more than once.
            TimeFNotFoundError: If a parent ID names no object imported under ``alias``.
        """
        parent_ref, imported = self._control_reader().imported_ids(alias, kind, dict.fromkeys(parent_ids))
        if parent_ref is None:
            raise TimeFFormatError(
                f"the catalog names no parent release for parent alias {alias!r}, which its manifest declares"
            )
        if parent_ref != str(self._manifest.parent_cards[alias].source):
            raise TimeFFormatError(f"the catalog's parent release for parent alias {alias!r} differs from its manifest")
        missing = [identity for identity in parent_ids if identity not in imported]
        if missing:
            raise TimeFNotFoundError(
                f"no {kind.value} {', '.join(map(repr, missing))} imported under parent alias {alias!r}"
            )
        return tuple(imported[identity] for identity in parent_ids)

    def _record_ids_of_signals(self, signal_ids: tuple[EntityId, ...]) -> dict[EntityId, EntityId]:
        """Return the ID of the Record that holds each Signal, building no object.

        ``TimeFDataset.import_from`` uses it to read only the Records that hold the Signals it
        imports.

        Returns:
            The Record ID of each Signal, by Signal ID.

        Raises:
            TimeFNotFoundError: If an ID names no stored Signal.
        """  # noqa: DOC502 - raised by the control reader
        with self._as_format_error(preserve_validation=True):
            return self._control_reader().signal_record_ids(signal_ids)

    def _read_tasks_with_records(
        self, task_ids: tuple[EntityId, ...], *, load: LoadPlan
    ) -> tuple[dict[EntityId, Task], dict[EntityId, Record], tuple[Annotation, ...]]:
        """Read tasks by ID, as :meth:`get_many` does, with every Record and dataset occurrence the read built for them.

        ``TimeFDataset.import_from`` uses it, so the Records it copies are the objects that the tasks
        reference, and every annotation the tasks reference lies on one of those Records or is one
        of the dataset occurrences.

        Args:
            task_ids: The IDs of the tasks to read.
            load: Which kinds of stored values to load into the tasks, the tasks they derive from,
                and every Record the read built.

        Returns:
            The tasks by ID, the Records by ID, and the dataset's own occurrences that the tasks
            reference.

        Raises:
            TimeFNotFoundError: If an ID names no stored task.
        """  # noqa: DOC502 - raised by the control reader
        with self._as_format_error(preserve_validation=True):
            tasks, records, on_dataset = self._control_reader()._read_tasks_with_records(task_ids)
            values = ValueLoad(load)
            values.add(tasks.values(), parents=True)
            values.add(records.values())
            values.run()
            return tasks, records, on_dataset

    def _read_by_id(self, kind: ObjectKind, ids: tuple[EntityId, ...]) -> dict[EntityId, Entity]:
        """Build the objects of one kind with these IDs, each once, with every value unloaded.

        Returns:
            The objects by ID.
        """
        objects: dict[EntityId, Entity] = {}
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

    def _make_signal_loader(self, signal_id: EntityId, spec: TimeSeriesSpec, n_values: int) -> StoredArray:
        """Return a picklable, range-aware handle to one Signal's stored values, unloaded.

        The handle checks a read of every chunk against the Signal's declared number of values.
        """
        if signal_id not in self._chunk_locator_batch_by_id:
            if not self._chunk_locator_batches or len(self._chunk_locator_batches[-1]) >= _CHUNK_LOCATOR_BATCH_SIZE:
                self._chunk_locator_batches.append([])
            batch_index = len(self._chunk_locator_batches) - 1
            self._chunk_locator_batches[batch_index].append(signal_id)
            self._chunk_locator_batch_by_id[signal_id] = batch_index
        return StoredArray(self, signal_id, spec, n_values)

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


def _verify_files(version: DatasetVersion, parts: Iterable[FilePart], *, release: DatasetRef, owner: str) -> None:
    """Hash each file of a release and compare it with its manifest record, or its pin.

    Args:
        version: The release that stores the files.
        parts: The records to check, which name each file relative to ``version``.
        release: The release that stores the files, for the error.
        owner: The release, or the payload source, that the files belong to, for the error message.

    Raises:
        TimeNetFileError: If a file is missing, or its size or checksum differs from its record.
    """
    for part in sorted(parts, key=lambda item: item.path):
        expected_size = f"{part.size:,} bytes"
        try:
            handle = version.filesystem.open_input_file(version.resolve_file_path(part.path))
        except FileNotFoundError as exc:
            raise TimeNetFileError(release, owner, part.path, expected_size, None) from exc
        with handle:
            actual_size = handle.size()
            if actual_size != part.size:
                raise TimeNetFileError(release, owner, part.path, expected_size, f"{actual_size:,} bytes")
            actual_checksum = stream_checksum(handle)
        if actual_checksum != part.checksum:
            raise TimeNetFileError(release, owner, part.path, part.checksum, actual_checksum)


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
