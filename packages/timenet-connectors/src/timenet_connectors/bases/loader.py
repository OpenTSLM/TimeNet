"""Generic bases for a dataset's loader and its typed source tasks.

A connector deals with the outside world: its card, its download, its pinned source, and the
engine's interface. A :class:`DatasetLoader` deals with the downloaded files: it parses them once,
then hands out the dataset's Records and its tasks as the source defines them. Each source task is a
:class:`DatasetTask`, which knows the TimeNet task it becomes. The connector's ``convert`` parses with
the loader and lets it fill the dataset::

    def convert(self, raw_refs):
        loader = MyDatasetLoader.parse_from(raw_refs)
        return loader.populate(TimeFDataset(metadata=self.metadata()))

The loader returns source tasks, not TimeNet tasks, so it repeats nothing ``TimeFDataset`` does:
``tasks_of`` selects by source task kind, which a TimeNet task type can't tell apart.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, ClassVar, Generic, Self, TypeVar

from timenet.dataset import Record, TimeFDataset
from timenet.types import Annotation, Split, Task


class DatasetTask(ABC):
    """One task as its source defines it, which knows the TimeNet task it becomes.

    A subclass is one kind of task of one dataset, such as TRACE's retrieval task. It holds what the
    task is built from, and builds the TimeNet task on demand, so a loader can keep its tasks small
    and stream them.
    """

    TIMENET_TYPE: ClassVar[type[Task]]
    """The TimeNet task type ``to_timenet_task`` returns."""

    @property
    @abstractmethod
    def split(self) -> Split | None:
        """Return the task's split, as the source assigns it, or ``None`` when the source gives it none."""

    @abstractmethod
    def to_timenet_task(self) -> Task:
        """Return the task as TimeNet stores it, an instance of :attr:`TIMENET_TYPE`."""


TTask = TypeVar("TTask", bound=DatasetTask)
K = TypeVar("K", bound=DatasetTask)


class DatasetLoader(ABC, Generic[TTask]):
    """A dataset's downloaded files, parsed: its Records, its typed source tasks, and its annotations.

    A subclass implements :meth:`parse_from`, :meth:`records` and :meth:`tasks`, declares the kinds
    of task it yields in :attr:`TASK_KINDS`, and lists them in its docstring. It may override
    :meth:`dataset_annotations`. The base provides :meth:`tasks_of`, :meth:`timenet_tasks` and
    :meth:`populate`.
    """

    TASK_KINDS: ClassVar[tuple[type[DatasetTask], ...]]
    """Every kind of task :meth:`tasks` yields, in the order a reader should meet them.

    :meth:`populate` declares their TimeNet types. The writer rejects a streamed task of any other
    type, so a kind missing here fails when the dataset is written.
    """

    @classmethod
    @abstractmethod
    def parse_from(cls, refs: Iterable[Any]) -> Self:
        """Parse and check the downloaded files once.

        Args:
            refs: The references the connector's ``download`` returned.

        Returns:
            The loader, holding what its Records and tasks are built from.
        """

    @abstractmethod
    def records(self) -> Iterator[Record]:
        """Return every Record, in a stable order."""

    @abstractmethod
    def tasks(self) -> Iterator[TTask]:
        """Return every source task, in a stable order. Each call starts again from the first."""

    def dataset_annotations(self) -> Sequence[Annotation]:  # noqa: PLR6301 - the loader interface
        """Return the annotations the dataset holds as a whole, such as a label vocabulary.

        Returns:
            No annotations, unless a subclass overrides this.
        """
        return ()

    def tasks_of(self, kind: type[K], *, split: Split | None = None) -> Iterator[K]:
        """Return the tasks of one kind, optionally of one split, in the order of :meth:`tasks`.

        Args:
            kind: A source task kind from :attr:`TASK_KINDS`, or a base class of several.
            split: Only the tasks of this split, if given.

        Yields:
            The matching tasks.
        """
        for task in self.tasks():
            if isinstance(task, kind) and (split is None or task.split is split):
                yield task

    def timenet_tasks(self) -> Iterator[Task]:
        """Return every task as TimeNet stores it, in the order of :meth:`tasks`."""
        return (task.to_timenet_task() for task in self.tasks())

    def populate(self, dataset: TimeFDataset) -> TimeFDataset:
        """Fill a dataset with this loader's annotations and Records, and stream its tasks.

        Tasks are not held: the dataset calls :meth:`timenet_tasks` each time it reads them.

        Args:
            dataset: A dataset, usually empty, carrying the connector's metadata.

        Returns:
            The same dataset.
        """
        for annotation in self.dataset_annotations():
            dataset.annotate(annotation)
        for record in self.records():
            dataset.add_record(record=record)
        # One TimeNet type per kind, without repeats, in TASK_KINDS order.
        task_types = tuple(dict.fromkeys(kind.TIMENET_TYPE for kind in self.TASK_KINDS))
        dataset.set_task_stream(task_types, self.timenet_tasks)
        return dataset
