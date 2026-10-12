"""Producer-side API for building a dataset as a layer over exact parent versions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

from timenet.dataset import Record, TimeFDataset
from timenet.dataset.composition import import_prefix
from timenet.errors import TimeFValidationError
from timenet.types import DatasetMetadata, DatasetRef, LockedDependency, Task


if TYPE_CHECKING:
    from timenet.manifest import Manifest
    from timenet.reader import TimeFReader
    from timenet.registry import BaseRegistry, ResolvedVersion


class ParentDatasetView:
    """Lazy, read-only view of one exact parent dataset version."""

    def __init__(self, closure: Sequence[ResolvedVersion], reader: TimeFReader) -> None:
        """Wrap an open reader of the parent version.

        Args:
            closure: The verified dependency closure, with the parent last.
            reader: An open reader for the parent version.
        """
        self.manifest: Manifest = closure[-1].manifest
        self.reference = closure[-1].reference
        self.dataset_id = self.reference.dataset_id
        self.lock: tuple[LockedDependency, ...] = tuple(node.lock for node in closure)
        """The parent and its dependencies, pinned by manifest checksum."""
        self._reader = reader

    def record_ids(self) -> tuple[str, ...]:
        """Return parent Record IDs without hydrating their signals."""
        return self._reader.record_ids()

    def iter_records(self, record_ids: Iterable[str] | None = None) -> Iterator[Record]:
        """Yield parent Records with lazy values.

        Args:
            record_ids: IDs to read in result order, or ``None`` for every record.

        Yields:
            Parent Records accepted by :meth:`~timenet.dataset.TimeFDataset.import_record`.
        """
        yield from self._reader.iter_records(record_ids)

    def import_records(self, dataset: TimeFDataset, record_ids: Iterable[str] | None = None) -> tuple[Record, ...]:
        """Import parent records into a child under IDs qualified by this exact release.

        The parent's reader hydrates the records with the ``org/name@version::`` prefix on every
        record, source, signal, axis, and annotation ID, and the child registers them as imports.
        Signal values stay lazy and keep reading the parent's values plane.

        Args:
            dataset: A child whose card declares this parent.
            record_ids: Parent record IDs to import in result order, or ``None`` for every record.

        Returns:
            The imported records as registered in the child, ready for child tasks and overlays.

        Raises:
            TimeFValidationError: If the child does not declare this release, the parent lacks a
                requested record, or an imported ID is already registered.
        """  # noqa: DOC502 - raised by the reader and TimeFDataset.import_record
        return tuple(
            dataset.import_record(record, parent=self.dataset_id)
            for record in self._reader.iter_records(record_ids, prefix=import_prefix(self.reference))
        )

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

    def close(self) -> None:
        """Close the parent reader."""
        self._reader.close()


class BuildContext(AbstractContextManager["BuildContext"]):
    """Build-scoped metadata and lazy direct-parent views."""

    def __init__(self, metadata: DatasetMetadata, parents: Mapping[str, ParentDatasetView]) -> None:
        """Create a context from already opened direct-parent views."""
        self.metadata = metadata
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
            closure = registry.resolve_versions(parent.dataset_id, str(parent.version))
            parents[parent.dataset_id] = ParentDatasetView(closure, registry.open_closure(closure))
        return cls(metadata, parents)

    def __exit__(self, *_: object) -> None:
        """Close all direct-parent readers."""
        for parent in self._parents.values():
            parent.close()

    def parent(self, dataset_id: str) -> ParentDatasetView:
        """Return one direct parent by dataset ID.

        Raises:
            TimeFValidationError: If the card declares no such dataset ID.
        """
        try:
            return self._parents[dataset_id]
        except KeyError as exc:
            raise TimeFValidationError(f"dataset card declares no parent {dataset_id!r}") from exc

    @property
    def parents(self) -> Mapping[str, ParentDatasetView]:
        """Read-only mapping of full parent dataset IDs to views."""
        return self._parents

    def verify_imports(self, dataset: TimeFDataset) -> None:
        """Check imported record ids against their declared parents.

        Args:
            dataset: The converted dataset.

        Raises:
            TimeFValidationError: If a declared parent lacks an imported record id.
        """
        by_parent: dict[str, list[str]] = defaultdict(list)
        for imported in dataset.record_imports.values():
            by_parent[imported.parent_dataset_id].append(imported.parent_record_id)
        for dataset_id, record_ids in by_parent.items():
            missing = sorted(set(record_ids) - set(self.parent(dataset_id).record_ids()))
            if missing:
                raise TimeFValidationError(f"parent {dataset_id!r} does not hold imported record(s) {missing}")

    def dependency_lock(self) -> tuple[LockedDependency, ...]:
        """Combine the parents' dependency locks.

        Returns:
            Direct and transitive parents, pinned by manifest checksum and sorted by reference.

        Raises:
            TimeFValidationError: If two paths pin one version to different checksums.
        """
        locked: dict[DatasetRef, LockedDependency] = {}
        for parent in self._parents.values():
            for dependency in parent.lock:
                existing = locked.get(dependency.dataset)
                if existing is not None and existing != dependency:
                    raise TimeFValidationError(f"dependency {dependency.dataset} resolves to conflicting lock metadata")
                locked[dependency.dataset] = dependency
        return tuple(locked[reference] for reference in sorted(locked))
