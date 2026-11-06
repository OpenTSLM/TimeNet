"""Parse questions about missing series values and create editing tasks."""

from dataclasses import dataclass
import re
from typing import Self

from timenet.dataset import TimeFDataset
from timenet.types import TSEditingTask
from timenet_connectors.datasets.time_mqa.tsqa.release import PrintedSeries, ReleaseRow


_IMPUTED = re.compile(
    r"Based on the given information, the full time series with missing value imputed are \[[^\[\]]*\]\."
)


@dataclass(frozen=True)
class ImputationRow:
    """An imputation row with a series that has gaps and a complete series."""

    row: ReleaseRow
    with_gaps: PrintedSeries
    complete: PrintedSeries

    @classmethod
    def from_release_row(cls, row: ReleaseRow) -> Self:
        """Parse an imputation row.

        Returns:
            The parsed row.

        Raises:
            TimeNetConnectorError: If either series is invalid or the answer changes values outside the gaps.
        """  # ruff: ignore[docstring-extraneous-exception] - raised through the ReleaseRow methods
        with_gaps = row.only_printed_series_of_question(with_gaps=True)
        if not with_gaps.has_gaps:
            raise row.malformed("the question prints a series without gaps")
        row.match_of_answer_template(_IMPUTED)
        complete = row.only_printed_series_of_answer()
        if not _fills_the_gaps_of(with_gaps, complete):
            raise row.malformed("the answer does not print the series of the question with its gaps filled")
        return cls(row=row, with_gaps=with_gaps, complete=complete)

    def add_records_and_task_to(self, dataset: TimeFDataset) -> TSEditingTask:
        """Add the incomplete and complete records and the editing task to the dataset.

        Returns:
            The added task.
        """
        with_gaps = dataset.add_record(record=self.with_gaps.create_record(self.row.record_metadata("with_gaps")))
        complete = dataset.add_record(record=self.complete.create_record(self.row.record_metadata("complete")))
        task = TSEditingTask(
            inputs=(with_gaps,),
            targets=(complete,),
            prompt=self.row.question_with_series_placeholders((self.with_gaps,)),
            metadata=self.row.task_metadata(),
        )
        dataset.add_task(task=task)
        return task


def _fills_the_gaps_of(with_gaps: PrintedSeries, complete: PrintedSeries) -> bool:
    """Return whether the complete series fills the gaps without changing other values."""
    if len(complete.values) != len(with_gaps.values):
        return False
    return all(known is None or known == value for known, value in zip(with_gaps.values, complete.values, strict=True))
