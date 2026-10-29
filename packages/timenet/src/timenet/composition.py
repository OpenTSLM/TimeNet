"""Build datasets from exact parent releases with lazy value handles."""

from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, ExitStack

from timenet.dataset import Record, TimeFDataset
from timenet.dataset.composition import clone_record
from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import BaseRegistry
from timenet.types import DatasetMetadata


class ParentDatasetView:
    """Read parent records without loading their signal values."""

    def __init__(self, reader: TimeFReader) -> None:
        self._reader = reader
        self._records: dict[int, Record] = {}
        self._all_records_read = False
        self._by_source: dict[str, Record] = {}

    def record_ids(self) -> tuple[int, ...]:
        """Return the parent's numeric record ids."""
        return self._reader.record_ids()

    def iter_records(self, record_ids: Iterable[int | str] | None = None) -> Iterator[Record]:
        """Yield parent records selected by entity id or upstream identifier.

        Yields:
            Records with lazy signal values.

        Raises:
            TimeFValidationError: If a requested record does not exist.
        """
        requested = None if record_ids is None else tuple(record_ids)
        needs_all = requested is None or any(isinstance(key, str) for key in requested)
        ids = self.record_ids() if needs_all and not self._all_records_read else ()
        if not needs_all and requested is not None:
            ids = tuple(key for key in requested if isinstance(key, int) and key not in self._records)
        for record in self._reader.get_many(Record, (key for key in ids if key not in self._records)):
            self._records[record.id] = record
            source_id = record.metadata.get("source_record_id")
            if isinstance(source_id, str):
                if source_id in self._by_source:
                    raise TimeFValidationError(f"parent repeats upstream record identifier {source_id!r}")
                self._by_source[source_id] = record
        if needs_all:
            self._all_records_read = True
        if requested is None:
            yield from self._records.values()
            return
        for key in requested:
            record = self._by_source.get(key) if isinstance(key, str) else self._records.get(key)
            if record is None:
                raise TimeFValidationError(f"parent has no record {key!r}")
            yield record

    def owns(self, record: Record) -> bool:
        """Return whether this view yielded the record."""
        return self._records.get(record.id) is record


class DatasetBuilder(TimeFDataset):
    """Register separate child hierarchies that share parent value handles."""

    def __init__(self, context: "BuildContext") -> None:
        super().__init__(metadata=context.metadata)
        self._context = context
        self._imports: dict[tuple[str, int], Record] = {}

    def import_record(self, record: Record, *, parent: str) -> Record:
        """Import a parent record once, without changing its parent's hierarchy.

        Returns:
            The child record, with ids assigned in the child dataset.

        Raises:
            TimeFValidationError: If the record did not come from the selected parent view.
        """
        if not self._context.parent(parent).owns(record):
            raise TimeFValidationError(f"record {record.id} was not yielded by parent {parent!r}")
        key = (parent, record.id)
        if key not in self._imports:
            self._imports[key] = self.add_record(record=clone_record(record))
        return self._imports[key]


class BuildContext(AbstractContextManager["BuildContext"]):
    """Keep exact parent readers open for conversion and writing."""

    def __init__(self, metadata: DatasetMetadata, registry: BaseRegistry) -> None:
        self.metadata = metadata
        self._resources = ExitStack()
        self._parents: dict[str, ParentDatasetView] = {}
        try:
            for alias, reference in metadata.parents.items():
                reader = self._resources.enter_context(
                    TimeFReader(registry.open_version(reference.dataset_id, str(reference.version)))
                )
                self._parents[alias] = ParentDatasetView(reader)
        except BaseException:
            self._resources.close()
            raise

    @classmethod
    def open(cls, metadata: DatasetMetadata, registry: BaseRegistry) -> "BuildContext":
        """Open the exact parent releases declared by the card.

        Returns:
            The context that owns all parent readers.
        """
        return cls(metadata, registry)

    def __exit__(self, *_: object) -> None:
        self._resources.close()

    def dataset(self) -> DatasetBuilder:
        """Return an empty child dataset."""
        return DatasetBuilder(self)

    def parent(self, alias: str) -> ParentDatasetView:
        """Return a parent view by its card alias.

        Raises:
            TimeFValidationError: If the card does not declare the alias.
        """
        try:
            return self._parents[alias]
        except KeyError as error:
            raise TimeFValidationError(f"dataset card declares no parent alias {alias!r}") from error
