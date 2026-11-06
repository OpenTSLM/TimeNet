"""A row whose answer is free text: one AnswerTask, with a Record for each list the question prints.

The Time-MMD forecasting questions and the open-ended questions share this shape. Their question
is free text that may print any number of series, and their answer is text that the dataset keeps
as released.
"""

from dataclasses import dataclass
from typing import Self

from timenet.dataset import TimeFDataset
from timenet.types import AnswerTask
from timenet_connectors.datasets.time_mqa.tsqa.release import PrintedSeries, ReleaseRow, find_printed_series


@dataclass(frozen=True)
class FreeAnswerRow:
    """A row with a free-text answer and the series its question prints."""

    row: ReleaseRow
    series: tuple[PrintedSeries, ...]

    @classmethod
    def from_release_row(cls, row: ReleaseRow) -> Self:
        """Parse a row whose answer is free text.

        Returns:
            The parsed row.
        """
        return cls(row=row, series=find_printed_series(row.question))

    def add_records_and_task_to(self, dataset: TimeFDataset) -> AnswerTask:
        """Add a Record for each list the question prints, and the task that asks the question about them.

        Returns:
            The added task.
        """
        records = tuple(
            dataset.add_record(record=printed.create_record(self.row.record_metadata(f"series_{number}")))
            for number, printed in enumerate(self.series, start=1)
        )
        task = AnswerTask(
            inputs=records,
            prompt=self.row.question_with_series_placeholders(self.series),
            targets=(self.row.answer,),
            metadata=self.row.task_metadata(),
        )
        dataset.add_task(task=task)
        return task
