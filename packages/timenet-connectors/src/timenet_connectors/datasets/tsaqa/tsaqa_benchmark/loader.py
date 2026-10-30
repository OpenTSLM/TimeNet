"""Load TSAQA's three split files into one lazy Record per series and one task per row."""

from collections.abc import Iterable, Iterator, Mapping
from typing import Self

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, AnswerTask, ClassificationTask, Task
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.index import ReleaseIndex
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.keys import AnnotationKey
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.records import ReleaseRecords, SeriesLoader
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.release import PINNED_INVENTORY, InventoryKey, SourceSplit
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.rows import Row
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.tasks import (
    BUILDERS,
    multiple_choice_task,
    puzzling_task,
    true_or_false_task,
)
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.vocabulary import TRUE_OR_FALSE_CHOICES, QuestionFormat


_TRUE_OR_FALSE = "The choices of every true-or-false question: T stands for 'True.' and F for 'False.'"


class TSAQADatasetLoader(DatasetLoader):
    """TSAQA's split files, read and checked: one lazy Record per series, and one task per row.

    Each row is one question of one of the paper's six tasks, in one of its three formats. The format,
    read from the question's text, decides the TimeNet type, so each format has its own method:

    - True-or-false (:meth:`true_or_false_tasks`), a ``ClassificationTask``.
    - Multiple-choice (:meth:`multiple_choice_tasks`), an ``AnswerTask``.
    - Puzzling (:meth:`puzzling_tasks`), an ``AnswerTask``.

    :meth:`tasks` keeps the file order, in which the formats alternate. After ``populate``,
    ``TimeFDataset.get_split`` selects one split. A task's paper task is its ``task`` metadata.

    The loader keeps each row's small fields, not its series, which the Records reread from the split
    files. The files must stay available until writing ends.
    """

    TASK_TYPES = (ClassificationTask, AnswerTask)

    def __init__(self, index: ReleaseIndex, records: ReleaseRecords) -> None:
        self._index = index
        self._records = records

    @classmethod
    def parse_from(
        cls,
        refs: Iterable[HubFileRef],
        *,
        inventory: Mapping[SourceSplit, Mapping[InventoryKey, int]] | None = PINNED_INVENTORY,
    ) -> Self:
        """Read and check each split file once, and build one lazy Record per series of each row.

        ``refs`` are the references the connector's base already checked: each split at most once. A
        malformed row, or a file whose rows differ from ``inventory``, raises ``TimeFValidationError``.

        Args:
            refs: The downloaded split files.
            inventory: Each file's pinned row counts, or ``None`` to skip the check, as test fixtures do.

        Returns:
            The loaded release.

        Raises:
            TimeFValidationError: If a reference names no TSAQA split.
        """
        paths = {}
        for ref in refs:
            if ref.part not in set(SourceSplit):
                raise TimeFValidationError(f"{ref.path}: {ref.part!r} is not a TSAQA split")
            paths[SourceSplit(ref.part)] = ref.path
        index = ReleaseIndex.from_files(paths, inventory)
        records = ReleaseRecords.build(index, SeriesLoader(index.paths))
        return cls(index, records)

    def dataset_annotations(self) -> tuple[Annotation, ...]:  # noqa: PLR6301 - the loader interface
        """Return the true-or-false vocabulary, which every true-or-false task names as its ``target_schema``.

        The connector adds the provenance.

        Returns:
            The one vocabulary annotation.
        """
        return (
            Annotation(
                key=AnnotationKey.TRUE_OR_FALSE_VOCABULARY,
                value=list(TRUE_OR_FALSE_CHOICES),
                description=_TRUE_OR_FALSE,
            ),
        )

    def records(self) -> Iterator[Record]:
        """Return every Record, row by row in file order, whose values stay in the split files until written."""
        return iter(self._records)

    def tasks(self) -> Iterator[Task]:
        """Return one task per row, in file order, of the type its question's format decides.

        Yields:
            Each row's task.
        """
        for row, records in self._rows():
            yield BUILDERS[row.format](row, records, self._index.issues_of(row))

    def true_or_false_tasks(self) -> Iterator[ClassificationTask]:
        """Return the true-or-false tasks, in file order.

        Yields:
            One classification task for each true-or-false row.
        """
        for row, records in self._rows(QuestionFormat.TRUE_OR_FALSE):
            yield true_or_false_task(row, records, self._index.issues_of(row))

    def multiple_choice_tasks(self) -> Iterator[AnswerTask]:
        """Return the multiple-choice tasks, in file order.

        Yields:
            One answer task for each multiple-choice row.
        """
        for row, records in self._rows(QuestionFormat.MULTIPLE_CHOICE):
            yield multiple_choice_task(row, records, self._index.issues_of(row))

    def puzzling_tasks(self) -> Iterator[AnswerTask]:
        """Return the puzzling tasks, in file order.

        Yields:
            One answer task for each puzzle row.
        """
        for row, records in self._rows(QuestionFormat.PUZZLING):
            yield puzzling_task(row, records, self._index.issues_of(row))

    def _rows(self, question_format: QuestionFormat | None = None) -> Iterator[tuple[Row, tuple[Record, ...]]]:
        rows = zip(self._index.rows, self._records.of_rows, strict=True)
        return (pair for pair in rows if question_format is None or pair[0].format is question_format)
