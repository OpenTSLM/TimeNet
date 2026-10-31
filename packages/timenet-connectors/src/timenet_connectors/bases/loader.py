"""A shared base class that reads a dataset's files and turns them into TimeNet tasks.

A connector gets the files of a dataset from its source, and turns them into Records and tasks.
Most connectors do the second job with the same steps: read the files once, build the Records and
the tasks, and add them to a TimeFDataset. :class:`DatasetLoader` does these steps, so each
connector only writes the parts that are special to its dataset.

The connector still downloads the files. It also adds the annotations that do not come from the
files, such as the source revision. Its ``convert`` gives the files to the loader::

    def convert(self, raw_refs):
        dataset = TimeFDataset(metadata=self.metadata())
        for annotation in PROVENANCE:
            dataset.annotate(annotation)
        return MyDatasetLoader.parse_from(raw_refs).populate(dataset)
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import ClassVar, Generic, Self, TypeVar

from timenet.dataset import Record, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.types import Task


TRaw = TypeVar("TRaw")


class DatasetLoader(ABC, Generic[TRaw]):
    """Reads the downloaded files of one dataset and gives its Records, tasks and annotations.

    ``TRaw`` is the type of the references that the connector downloads, as in
    ``BaseConnector[TRaw]``. A loader names it, for example ``DatasetLoader[HubFileRef]``.

    A subclass must implement :meth:`parse_from`, :meth:`records` and :meth:`tasks`. It must list
    each TimeNet task type that it makes in :attr:`TASK_TYPES`. It can override
    :meth:`annotate_dataset`. This base gives :meth:`populate`.

    The loader attaches each annotation that comes from the files, so a task can refer to it:

    - Attach the annotations of a Record, a Source or a Signal to that object in :meth:`parse_from`.
    - Attach the annotations of the full dataset, such as a label vocabulary, in
      :meth:`annotate_dataset`.
    - Attach the annotations of a task to the task in :meth:`tasks`.

    ``annotate`` attaches a copy and returns it. A task that refers to an annotation must use that
    copy, so keep it.

    When the dataset authors define several tasks, give each one its own method with the name that
    the authors use, such as ``binary_label_tasks``. Then :meth:`tasks` yields from each method.
    """

    TASK_TYPES: ClassVar[tuple[type[Task], ...]]
    """Each TimeNet task type that :meth:`tasks` makes.

    :meth:`populate` gives these types to the dataset, and the dataset schema lists them in this
    order. When the dataset is written, the writer stops on a task whose type is not in this list.
    """

    @classmethod
    @abstractmethod
    def parse_from(cls, refs: Iterable[TRaw]) -> Self:
        """Read the downloaded files one time, and make sure that they are correct.

        Build the Records here, with their annotations, so that :meth:`records` and :meth:`tasks`
        use the same objects.

        Args:
            refs: The references that the ``download`` of the connector returned.

        Returns:
            The loader. It keeps the data that its Records and tasks need.
        """

    @abstractmethod
    def records(self) -> Iterator[Record]:
        """Return each Record once, in the same order on each run.

        The tasks must use these same Record objects, and the same annotation objects in them. A task
        that uses a copy fails when the dataset is written. The order keeps the IDs the same between
        runs.
        """

    @abstractmethod
    def tasks(self) -> Iterator[Task]:
        """Return each TimeNet task, always in the same order. Each call starts again at the first task.

        :meth:`populate` calls :meth:`annotate_dataset` before the dataset reads the tasks. So the
        tasks can use the annotations that :meth:`annotate_dataset` kept. If you call this method
        before :meth:`populate`, those annotations do not exist yet.
        """

    def annotate_dataset(self, dataset: TimeFDataset) -> None:
        """Attach the annotations of the full dataset that come from the files.

        Call ``dataset.annotate`` for each one. When a task refers to an annotation, keep the copy
        that ``dataset.annotate`` returns, and use that copy in :meth:`tasks`. The base attaches
        nothing.

        Args:
            dataset: The dataset that :meth:`populate` fills.
        """

    def populate(self, dataset: TimeFDataset) -> TimeFDataset:
        """Add the annotations, the Records and the tasks of this loader to a dataset.

        The dataset does not keep the tasks. It calls :meth:`tasks` each time that it reads them. So
        ``dataset.tasks``, ``tasks_of`` and ``tasks_for`` stay empty. Use ``iter_tasks`` to read the
        tasks one by one. ``get_split``, ``get_splits`` and ``get_all`` keep each selected task in
        memory.

        Args:
            dataset: A dataset with the metadata of the connector and no tasks. It can already have
                the annotations that the connector adds.

        Returns:
            The same dataset.

        Raises:
            TimeFValidationError: If the dataset already has tasks, or if the loader does not set
                :attr:`TASK_TYPES`. A second stream would replace the first one, and its tasks would
                be lost.
        """
        if dataset.has_task_stream or dataset.tasks:
            raise TimeFValidationError(
                f"populate needs a dataset without tasks, but {dataset.metadata.dataset_id!r} already has tasks"
            )
        task_types = getattr(self, "TASK_TYPES", None)
        if not task_types:
            raise TimeFValidationError(f"{type(self).__name__} must list the task types that it makes in TASK_TYPES")
        self.annotate_dataset(dataset)
        for record in self.records():
            dataset.add_record(record=record)
        dataset.set_task_stream(task_types, self.tasks)
        return dataset
