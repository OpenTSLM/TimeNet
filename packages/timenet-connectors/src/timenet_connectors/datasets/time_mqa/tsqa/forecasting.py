"""Parse forecasting questions and create series forecasting or free-answer tasks.

A question that asks for the next values of its printed series becomes a forecasting task from a
history record to a future record. A Time-MMD question, which asks for a prediction in words,
becomes a free-answer task.
"""

from dataclasses import dataclass
import re
from typing import Self

from timenet.dataset import TimeFDataset
from timenet.types import ForecastingTask
from timenet_connectors.datasets.time_mqa.tsqa.free_answer import FreeAnswerRow
from timenet_connectors.datasets.time_mqa.tsqa.release import PrintedSeries, ReleaseRow, TaskRow


_HORIZON = re.compile(r"Please predict the next (?P<horizon>\d+) time series point given information above\.$")
_PREDICTIONS = re.compile(r"Based on the given information, the predictions are \[[^\[\]]*\]\.")


def parse_forecasting_row(row: ReleaseRow) -> TaskRow:
    """Parse a forecasting row as a series forecast or a free-answer row.

    Returns:
        The parsed row.
    """
    if _HORIZON.search(row.question):
        return SeriesForecastingRow.from_release_row(row)
    return FreeAnswerRow.from_release_row(row)


@dataclass(frozen=True)
class SeriesForecastingRow:
    """A forecasting row with a series and its future values."""

    row: ReleaseRow
    history: PrintedSeries
    future: PrintedSeries

    @classmethod
    def from_release_row(cls, row: ReleaseRow) -> Self:
        """Parse a row that asks for the next values of a series.

        Returns:
            The parsed row.

        Raises:
            TimeNetConnectorError: If the series or predictions are invalid, or the prediction count is wrong.
        """  # ruff: ignore[docstring-extraneous-exception] - raised through the ReleaseRow methods
        history = row.only_printed_series_of_question()
        row.match_of_answer_template(_PREDICTIONS)
        future = row.only_printed_series_of_answer()
        horizon = _horizon_of_question(row)
        if len(future.values) != horizon:
            raise row.malformed(f"the question asks for {horizon} values, but the answer prints {len(future.values)}")
        return cls(row=row, history=history, future=future)

    def add_records_and_task_to(self, dataset: TimeFDataset) -> ForecastingTask:
        """Add the history and future records and the forecasting task to the dataset.

        Returns:
            The added task.
        """
        history = dataset.add_record(record=self.history.create_record(self.row.record_metadata("history")))
        future = dataset.add_record(record=self.future.create_record(self.row.record_metadata("future")))
        task = ForecastingTask(
            inputs=(history,),
            targets=(future,),
            prompt=self.row.question_with_series_placeholders((self.history,)),
            metadata=self.row.task_metadata(),
        )
        dataset.add_task(task=task)
        return task


def _horizon_of_question(row: ReleaseRow) -> int:
    """Return the number of future values requested by the question.

    Returns:
        The number of values.

    Raises:
        TimeNetConnectorError: If the question does not ask for the next values.
    """  # ruff: ignore[docstring-extraneous-exception] - raised through malformed()
    match = _HORIZON.search(row.question)
    if match is None:
        raise row.malformed("the question does not ask for the next values of the series")
    return int(match["horizon"])
