"""Load TSAQA's three split files into one lazy Record per series and one task per row."""

from collections.abc import Iterable, Iterator, Mapping
from typing import Self

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.types import Annotation
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.index import ReleaseIndex
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.keys import AnnotationKey
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.records import ReleaseRecords, SeriesLoader
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.release import PINNED_INVENTORY, InventoryKey, SourceSplit
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.tasks import (
    MultipleChoiceTask,
    PuzzleTask,
    TrueOrFalseTask,
    TSAQATask,
)
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.vocabulary import TRUE_OR_FALSE_CHOICES, QuestionFormat


_KINDS: dict[QuestionFormat, type[TSAQATask]] = {
    QuestionFormat.TRUE_OR_FALSE: TrueOrFalseTask,
    QuestionFormat.MULTIPLE_CHOICE: MultipleChoiceTask,
    QuestionFormat.PUZZLING: PuzzleTask,
}
_TRUE_OR_FALSE = "The choices of every true-or-false question: T stands for 'True.' and F for 'False.'"


class TSAQADatasetLoader(DatasetLoader[TSAQATask]):
    """TSAQA's split files, read and checked: one lazy Record per series, and one task per row.

    Each row is one question of one of the paper's six tasks, in one of its three formats. A task's
    kind is its question's format, read from the question's text:

    - True-or-false (:class:`TrueOrFalseTask`), a ``ClassificationTask``.
    - Multiple-choice (:class:`MultipleChoiceTask`), an ``AnswerTask``.
    - Puzzle (:class:`PuzzleTask`), an ``AnswerTask``.

    Select one kind, or one kind of one split, with ``tasks_of``, such as
    ``tasks_of(PuzzleTask, split=Split.TEST)``. A task's paper task is its ``task`` metadata.

    The loader keeps each row's small fields, not its series, which the Records reread from the split
    files. The files must stay available until writing ends.
    """

    TASK_KINDS = (TrueOrFalseTask, MultipleChoiceTask, PuzzleTask)

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

    def tasks(self) -> Iterator[TSAQATask]:
        """Return one task per row, in file order, of the kind its question's format decides.

        Yields:
            Each row's task.
        """
        for row, records in zip(self._index.rows, self._records.of_rows, strict=True):
            yield _KINDS[row.format](row, records, self._index.issues_of(row))
