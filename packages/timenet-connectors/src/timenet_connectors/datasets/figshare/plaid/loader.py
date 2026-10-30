"""Load PLAID's files into one lazy Record per CSV file, and one task per Record."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Self

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, ClassificationTask, Task, TemporalLocalizationTask
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.datasets.figshare.plaid.entries import (
    AggregatedEntry,
    SubmeteredEntry,
    read_aggregated,
    read_submetered,
)
from timenet_connectors.datasets.figshare.plaid.keys import AnnotationKey
from timenet_connectors.datasets.figshare.plaid.measurements import read_measurements
from timenet_connectors.datasets.figshare.plaid.records import ValueLoader, annotate_events, build_record
from timenet_connectors.datasets.figshare.plaid.release import (
    KNOWN_DEFECTS,
    PINNED_COUNTS,
    KnownDefects,
    PinnedCounts,
    Subset,
)
from timenet_connectors.datasets.figshare.plaid.tasks import appliance_task, switching_events_task


@dataclass(frozen=True)
class _Submetered:
    """A submetered file after parsing: its entry and its lazy Record."""

    entry: SubmeteredEntry
    record: Record


@dataclass(frozen=True)
class _Aggregated:
    """An aggregated file after parsing: its entry, its lazy Record, and the event annotations on it."""

    entry: AggregatedEntry
    record: Record
    events: tuple[Annotation, ...]


class PlaidDatasetLoader(DatasetLoader):
    """PLAID's files, read and checked: one lazy Record per CSV file, and its task.

    PLAID defines two tasks, one per Record, as the paper names them. Each has its own method:

    - Appliance type (:meth:`appliance_tasks`): a submetered file's appliance type, as the authors'
      GlobalSIP notebook classifies it.
    - Switching events (:meth:`switching_events_tasks`): an aggregated file's labelled switch-on and
      switch-off events, each with its appliance.

    Neither has a split.

    The loader keeps each file's entry and row count, not its samples. The signals reread their file, so the
    files must stay available until writing ends.
    """

    TASK_TYPES = (ClassificationTask, TemporalLocalizationTask)

    def __init__(self, submetered: tuple[_Submetered, ...], aggregated: tuple[_Aggregated, ...]) -> None:
        self._submetered = submetered
        self._aggregated = aggregated

    @classmethod
    def parse_from(
        cls, refs: Iterable[Path], *, counts: PinnedCounts = PINNED_COUNTS, defects: KnownDefects = KNOWN_DEFECTS
    ) -> Self:
        """Read and check both metadata files and every CSV file's rows, and build one lazy Record per file.

        Args:
            refs: The one directory holding both metadata files and both extracted subset directories.
            counts: What the directory must hold.
            defects: The known defects the directory must hold, and no others.

        Returns:
            The loaded release.

        Raises:
            TimeFValidationError: If ``refs`` isn't one directory, or its counts differ from ``counts``.
        """
        roots = list(refs)
        if len(roots) != 1:
            raise TimeFValidationError(f"PLAID loads from one directory, got {roots}")
        (root,) = roots
        submetered_entries = read_submetered(root / Subset.SUBMETERED.metadata.name)
        aggregated_entries = read_aggregated(root / Subset.AGGREGATED.metadata.name)
        # One loader for the Records of this load, so a file's two signals share one read.
        values = ValueLoader()
        submetered = tuple(
            _Submetered(entry=entry, record=build_record(measurement, values))
            for measurement, entry in zip(
                read_measurements(root, Subset.SUBMETERED, submetered_entries, defects), submetered_entries, strict=True
            )
        )
        aggregated = []
        for measurement, entry in zip(
            read_measurements(root, Subset.AGGREGATED, aggregated_entries, defects), aggregated_entries, strict=True
        ):
            record = build_record(measurement, values)
            aggregated.append(_Aggregated(entry=entry, record=record, events=annotate_events(record, entry)))
        found = PinnedCounts(
            submetered=len(submetered),
            aggregated=len(aggregated),
            events=sum(len(item.events) for item in aggregated),
            households=len({item.entry.household for item in submetered}),
        )
        counts.check(root, found)
        return cls(submetered, tuple(aggregated))

    def dataset_annotations(self) -> tuple[Annotation, ...]:
        """Return the submetered appliance types: all the appliance task needs besides its Record.

        Returns:
            The ``appliance_type_vocabulary`` annotation: the types as the metadata writes them, sorted.
        """
        types = sorted({item.entry.appliance.type for item in self._submetered})
        return (Annotation(key=AnnotationKey.APPLIANCE_TYPE_VOCABULARY, value=types),)

    def records(self) -> Iterator[Record]:
        """Return the submetered Records, then the aggregated ones, each in file number order."""
        return chain((item.record for item in self._submetered), (item.record for item in self._aggregated))

    def tasks(self) -> Iterator[Task]:
        """Return every PLAID task, in stream order.

        Yields:
            Each submetered file's appliance task, then each aggregated file's event task, in file order.
        """
        yield from self.appliance_tasks()
        yield from self.switching_events_tasks()

    def appliance_tasks(self) -> Iterator[ClassificationTask]:
        """Return the appliance type tasks.

        Yields:
            One classification task for each submetered file, in file number order.
        """
        for item in self._submetered:
            yield appliance_task(item.record, item.entry.appliance.type)

    def switching_events_tasks(self) -> Iterator[TemporalLocalizationTask]:
        """Return the switching events tasks.

        Yields:
            One localization task for each aggregated file, in file number order.
        """
        for item in self._aggregated:
            yield switching_events_task(item.record, item.events)
