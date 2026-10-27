"""Load the release's six files into its Records and the authors' three tasks."""

from collections.abc import Iterable, Iterator, Mapping
from itertools import chain
from pathlib import Path
from typing import Self

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.datasets.mikeam.time_series_reasoning.index import ReleaseIndex
from timenet_connectors.datasets.mikeam.time_series_reasoning.records import ReleaseRecords
from timenet_connectors.datasets.mikeam.time_series_reasoning.release import SourceFile
from timenet_connectors.datasets.mikeam.time_series_reasoning.rows import CounterfactualRow, QuestionRow, read_rows
from timenet_connectors.datasets.mikeam.time_series_reasoning.tasks import (
    CounterfactualTask,
    DescriptionTask,
    QuestionTask,
    TimeSeriesReasoningTask,
)


class TimeSeriesReasoningLoader(DatasetLoader[TimeSeriesReasoningTask]):
    """The release, read and checked: its Records, and one task per row of the authors' task files.

    The authors define three tasks, one per config:

    - Description (:class:`DescriptionTask`, ``ts2desc_mcq``): pick the series' description, under
      the authors' split.
    - Question (:class:`QuestionTask`, ``qa_mcq``): answer a question about one series. No split.
    - Counterfactual (:class:`CounterfactualTask`, ``qa_counterfactual``): answer a question that
      compares a series with its changed version. No split.

    Select one kind, or one kind of one split, with ``tasks_of``. The question tasks stream from their
    files, which must stay available until writing ends.
    """

    TASK_KINDS = (DescriptionTask, QuestionTask, CounterfactualTask)

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

    def tasks(self) -> Iterator[TimeSeriesReasoningTask]:
        """Return the description tasks, then the question tasks, then the counterfactual tasks, in file order.

        Each call rereads the two question files, so the tasks can be iterated more than once.
        """
        return chain(self._description_tasks(), self._question_tasks(), self._counterfactual_tasks())

    def _description_tasks(self) -> Iterator[DescriptionTask]:
        for row in self._index.descriptions:
            yield DescriptionTask(row, self._records.of_description(row))

    def _question_tasks(self) -> Iterator[QuestionTask]:
        for row in read_rows(self._paths[SourceFile.QUESTIONS], SourceFile.QUESTIONS, QuestionRow):
            yield QuestionTask(row, self._records.series[row.uuid])

    def _counterfactual_tasks(self) -> Iterator[CounterfactualTask]:
        for row in read_rows(self._paths[SourceFile.COUNTERFACTUALS], SourceFile.COUNTERFACTUALS, CounterfactualRow):
            changed = self._records.changed[row.uuid, row.counterfactual]
            yield CounterfactualTask(row, self._records.series[row.uuid], changed)
