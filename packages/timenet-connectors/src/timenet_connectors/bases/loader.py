"""A shared base class that reads a dataset's files and turns them into TimeNet tasks.

A connector gets the files of a dataset from its source, and turns them into Records and tasks.
Most connectors do the second job with the same steps: read the files once, build the Records and
the tasks, and add them to a TimeFDataset. :class:`DatasetLoader` does these steps, so each
connector only writes the parts that are special to its dataset.

The connector still downloads the files. Its ``convert`` gives them to the loader::

    def convert(self, raw_refs):
        loader = MyDatasetLoader.parse_from(raw_refs)
        return loader.populate(TimeFDataset(metadata=self.metadata()))
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, ClassVar, Self

from timenet.dataset import Record, TimeFDataset
from timenet.types import Annotation, Task


class DatasetLoader(ABC):
    """Reads the downloaded files of one dataset and gives its Records, tasks and annotations.

    A subclass must implement :meth:`parse_from`, :meth:`records` and :meth:`tasks`. It must list
    each TimeNet task type that it makes in :attr:`TASK_TYPES`. It can override
    :meth:`dataset_annotations`. This base gives :meth:`populate`.

    When the dataset authors define several tasks, give each one its own method with the name that
    the authors use, such as ``binary_label_tasks``. Then :meth:`tasks` yields from each method. After
    :meth:`populate`, ``TimeFDataset.tasks_of`` and ``TimeFDataset.get_split`` select the tasks.
    """

    TASK_TYPES: ClassVar[tuple[type[Task], ...]]
    """Each TimeNet task type that :meth:`tasks` makes, in the order that a reader sees them first.

    :meth:`populate` gives these types to the dataset. When the dataset is written, the writer stops
    on a task whose type is not in this list.
    """

    @classmethod
    @abstractmethod
    def parse_from(cls, refs: Iterable[Any]) -> Self:
        """Read the downloaded files one time, and make sure that they are correct.

        Args:
            refs: The references that the ``download`` of the connector returned.

        Returns:
            The loader. It keeps the data that its Records and tasks need.
        """

    @abstractmethod
    def records(self) -> Iterator[Record]:
        """Return each Record, always in the same order."""

    @abstractmethod
    def tasks(self) -> Iterator[Task]:
        """Return each TimeNet task, always in the same order. Each call starts again at the first task."""

    def dataset_annotations(self) -> Sequence[Annotation]:  # noqa: PLR6301 - the loader interface
        """Return the annotations of the full dataset, such as the list of its labels.

        Returns:
            No annotations. A subclass can override this.
        """
        return ()

    def populate(self, dataset: TimeFDataset) -> TimeFDataset:
        """Add the annotations, the Records and the tasks of this loader to a dataset.

        The dataset does not keep the tasks. It calls :meth:`tasks` each time that it reads them.

        Args:
            dataset: A dataset, usually empty, with the metadata of the connector.

        Returns:
            The same dataset.
        """
        for annotation in self.dataset_annotations():
            dataset.annotate(annotation)
        for record in self.records():
            dataset.add_record(record=record)
        dataset.set_task_stream(self.TASK_TYPES, self.tasks)
        return dataset
