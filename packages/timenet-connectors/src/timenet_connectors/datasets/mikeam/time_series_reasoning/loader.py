"""Load the release's six files into its Records and the authors' three tasks."""

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Self

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.types import AnswerTask
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.datasets.mikeam.time_series_reasoning.index import ReleaseIndex
from timenet_connectors.datasets.mikeam.time_series_reasoning.records import ReleaseRecords
from timenet_connectors.datasets.mikeam.time_series_reasoning.release import SourceFile
from timenet_connectors.datasets.mikeam.time_series_reasoning.rows import CounterfactualRow, QuestionRow, read_rows
from timenet_connectors.datasets.mikeam.time_series_reasoning.tasks import (
    counterfactual_task,
    description_task,
    question_task,
)


class TimeSeriesReasoningLoader(DatasetLoader):
    """The release, read and checked: its Records, and one task per row of the authors' task files.

    The authors define three tasks, one per config. Each is an ``AnswerTask``, and each has its own
    method:

    - Description (:meth:`description_tasks`, ``ts2desc_mcq``): pick the series' description, under
      the authors' split.
    - Question (:meth:`question_tasks`, ``qa_mcq``): answer a question about one series. No split.
    - Counterfactual (:meth:`counterfactual_tasks`, ``qa_counterfactual``): answer a question that
      compares a series with its changed version. No split.

    The question tasks stream from their files, which must stay available until writing ends.
    """

    TASK_TYPES = (AnswerTask,)

    def __init__(self, paths: Mapping[SourceFile, Path], index: ReleaseIndex, records: ReleaseRecords) -> None:
        self._paths = paths
        self._index = index
        self._records = records

    @classmethod
    def parse_from(cls, refs: Iterable[HubFileRef], *, pinned: bool = True) -> Self:
        """Read and check the six files once, and build every Record.

        Args:
            refs: One reference per source file, by its :attr:`SourceFile.part`.
            pinned: Whether the files are the pinned release, whose counts are checked. Test fixtures
                aren't.

        Returns:
            The loaded release.

        Raises:
            TimeFValidationError: If a reference names no source file, or a file repeats.
        """
        parts = {file.part: file for file in SourceFile}
        paths: dict[SourceFile, Path] = {}
        for ref in refs:
            file = parts.get(ref.part)
            if file is None or file in paths:
                raise TimeFValidationError(f"{ref.path}: {ref.part!r} is not a source file, or repeats one")
            paths[file] = ref.path
        index = ReleaseIndex.from_files(paths, pinned=pinned)
        return cls(paths, index, ReleaseRecords.build(index))

    def records(self) -> Iterator[Record]:
        """Return the series Records, the rounded copies, then the changed series, each in source order."""
        return iter(self._records)

    def tasks(self) -> Iterator[AnswerTask]:
        """Return every task, in file order.

        Each call rereads the two question files, so the tasks can be iterated more than once.

        Yields:
            The description tasks, then the question tasks, then the counterfactual tasks.
        """
        yield from self.description_tasks()
        yield from self.question_tasks()
        yield from self.counterfactual_tasks()

    def description_tasks(self) -> Iterator[AnswerTask]:
        """Return the description tasks (``ts2desc_mcq``).

        Yields:
            One task for each description row.
        """
        for row in self._index.descriptions:
            yield description_task(row, self._records.of_description(row))

    def question_tasks(self) -> Iterator[AnswerTask]:
        """Return the question tasks (``qa_mcq``), read again from their file.

        Yields:
            One task for each question row.
        """
        for row in read_rows(self._paths[SourceFile.QUESTIONS], SourceFile.QUESTIONS, QuestionRow):
            yield question_task(row, self._records.series[row.uuid])

    def counterfactual_tasks(self) -> Iterator[AnswerTask]:
        """Return the counterfactual tasks (``qa_counterfactual``), read again from their file.

        Yields:
            One task for each counterfactual row.
        """
        for row in read_rows(self._paths[SourceFile.COUNTERFACTUALS], SourceFile.COUNTERFACTUALS, CounterfactualRow):
            changed = self._records.changed[row.uuid, row.counterfactual]
            yield counterfactual_task(row, self._records.series[row.uuid], changed)
