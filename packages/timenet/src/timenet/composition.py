"""Producer-side API for building a dataset as a layer over exact parent versions."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from timenet.dataset import Record, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.format.checksums import stream_checksum
from timenet.manifest import (
    DirectDependency,
    LockedDependency,
    ManifestDependencies,
)
from timenet.types import DatasetMetadata, DatasetRef, ObjectKind, ObjectRef, Task


if TYPE_CHECKING:
    from timenet.manifest import Manifest
    from timenet.reader import TimeFReader
    from timenet.registry import BaseRegistry


@dataclass(frozen=True)
class ParentDatasetView:
    """Lazy, read-only view of one exact parent dataset version."""

    alias: str
    reference: DatasetRef
    manifest: Manifest
    _reader: TimeFReader
    _seen_records: dict[str, Record]

    def record_ids(self) -> tuple[str, ...]:
        """Return parent Record IDs without hydrating their signals."""
        return self._reader.record_ids()

    def iter_records(self, record_ids: Iterable[str] | None = None) -> Iterator[Record]:
        """Yield parent Records with lazy values.

        Args:
            record_ids: IDs to read in result order, or ``None`` for every record.

        Yields:
            Parent Records accepted by :meth:`DatasetBuilder.import_record`.
        """
        for record in self._reader.iter_records(record_ids):
            self._seen_records[record.id] = record
            yield record

    def iter_tasks(self, records: Iterable[Record] | None = None) -> Iterator[Task]:
        """Explicitly opt in to parent tasks.

        Parent tasks are never merged automatically. A child connector must call this method and
        choose which tasks to reproduce or extend.

        Args:
            records: Optional parent Records whose tasks to select.

        Yields:
            Parent task objects.
        """
        yield from self._reader.iter_tasks(records)

    def reference_for(self, record: Record) -> ObjectRef:
        """Return the stable reference for a Record yielded by this view.

        Raises:
            TimeFValidationError: If the object did not come from this view.
        """
        if self._seen_records.get(record.id) is not record:
            raise TimeFValidationError(f"Record {record.id!r} was not yielded by parent view {self.alias!r}")
        return ObjectRef(self.reference, ObjectKind.RECORD, record.id)


class DatasetBuilder(TimeFDataset):
    """Mutable producer object for one dataset layer."""

    def __init__(self, context: BuildContext) -> None:
        """Create an empty layer for ``context.metadata``."""
        super().__init__(metadata=context.metadata)
        self._context = context

    def import_record(self, record: Record, *, parent: str) -> Record:
        """Reuse a Record from a parent without copying its hierarchy or values.

        Args:
            record: A Record yielded by the selected parent view.
            parent: Parent alias from the dataset card.

        Returns:
            The imported Record, ready for child tasks and record annotation overlays.
        """
        view = self._context.parent(parent)
        return self.include_record(
            record=record,
            reference=view.reference_for(record),
            parent_alias=parent,
        )


class BuildContext(AbstractContextManager["BuildContext"]):
    """Build-scoped metadata and lazy direct-parent views."""

    def __init__(
        self,
        metadata: DatasetMetadata,
        registry: BaseRegistry,
        parents: Mapping[str, ParentDatasetView],
    ) -> None:
        """Create a context from already opened direct-parent views."""
        self.metadata = metadata
        self._registry = registry
        self._parents = dict(parents)

    @classmethod
    def open(cls, metadata: DatasetMetadata, registry: BaseRegistry) -> BuildContext:
        """Open every exact direct parent in one registry.

        Args:
            metadata: Child dataset metadata containing exact parent declarations.
            registry: Registry that must contain the complete dependency closure.

        Returns:
            The open build context.
        """
        parents: dict[str, ParentDatasetView] = {}
        for parent in metadata.parents:
            reader = registry.open_reader(
                parent.dataset.dataset_id,
                str(parent.dataset.version),
            )
            parents[parent.alias] = ParentDatasetView(
                alias=parent.alias,
                reference=parent.dataset,
                manifest=registry.get_manifest(
                    parent.dataset.dataset_id,
                    str(parent.dataset.version),
                ),
                _reader=reader,
                _seen_records={},
            )
        return cls(metadata, registry, parents)

    def __exit__(self, *_: object) -> None:
        """Close all direct-parent readers."""
        for parent in self._parents.values():
            parent._reader.close()

    def dataset(self) -> DatasetBuilder:
        """Return a new mutable layer builder for this context."""
        return DatasetBuilder(self)

    def parent(self, alias: str) -> ParentDatasetView:
        """Return one direct parent by card alias.

        Raises:
            TimeFValidationError: If the card declares no such alias.
        """
        try:
            return self._parents[alias]
        except KeyError as exc:
            raise TimeFValidationError(f"dataset card declares no parent alias {alias!r}") from exc

    @property
    def parents(self) -> Mapping[str, ParentDatasetView]:
        """Read-only mapping of direct parent aliases to views."""
        return self._parents

    def dependency_lock(self) -> ManifestDependencies:
        """Build direct edges and a flattened, checksummed dependency lock.

        Returns:
            Dependencies ready to attach to the child dataset.

        Raises:
            TimeFValidationError: If two paths resolve one version to different lock metadata.
        """  # noqa: DOC502 - raised by _merge_lock
        direct = tuple(DirectDependency(parent.alias, parent.dataset) for parent in self.metadata.parents)
        locked: dict[DatasetRef, LockedDependency] = {}
        for parent in self._parents.values():
            manifest = parent.manifest
            with self._registry.open_file(
                parent.reference.dataset_id,
                str(parent.reference.version),
                "manifest.json",
            ) as source:
                checksum = stream_checksum(source)
            row = LockedDependency(
                dataset=parent.reference,
                manifest_checksum=checksum,
                size=sum(part.size for part in manifest.files.all_files()),
                counts=manifest.counts,
                license=manifest.metadata.license,
                access=manifest.metadata.access,
            )
            self._merge_lock(locked, row)
            for inherited in manifest.dependencies.lock:
                self._merge_lock(locked, inherited)
        return ManifestDependencies(
            direct=direct,
            lock=tuple(locked[reference] for reference in sorted(locked)),
        )

    @staticmethod
    def _merge_lock(
        locked: dict[DatasetRef, LockedDependency],
        dependency: LockedDependency,
    ) -> None:
        """Merge one lock row and reject conflicting resolutions.

        Raises:
            TimeFValidationError: If the same version has different lock metadata.
        """
        existing = locked.get(dependency.dataset)
        if existing is not None and existing != dependency:
            raise TimeFValidationError(f"dependency {dependency.dataset} resolves to conflicting lock metadata")
        locked[dependency.dataset] = dependency
