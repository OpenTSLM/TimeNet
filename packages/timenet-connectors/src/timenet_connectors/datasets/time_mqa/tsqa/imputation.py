"""The imputation family: a printed series with gaps, and the same series complete.

Each imputation question prints a series with the gap marker in place of 4 to 12 values and asks
for the series with the gaps filled; the answer prints the complete series. Each row becomes two
Records, the series with gaps as a nullable Signal and the complete series, and one TSEditingTask
whose target is the complete Record.
"""

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
    """An imputation row: the printed series with gaps and the complete series of the answer."""

    row: ReleaseRow
    with_gaps: PrintedSeries
    complete: PrintedSeries

    @classmethod
    def from_release_row(cls, row: ReleaseRow) -> Self:
        """Parse an imputation row.

        Returns:
            The parsed row.

        Raises:
            TimeNetConnectorError: If the question does not print one series with gaps, the answer does
                not print the complete series, or the complete series differs from the question's series
                outside the gaps.
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
        """Add the Record with gaps and the complete Record, and the task that asks to fill the gaps.

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
    """Return whether the complete series has the length of the series with gaps and its values outside the gaps."""
    if len(complete.values) != len(with_gaps.values):
        return False
    return all(known is None or known == value for known, value in zip(with_gaps.values, complete.values, strict=True))
