"""The forecasting family: the next values of a printed series, or a prediction in words.

Most forecasting questions print a series and ask for its next N values; the answer prints those
values. Such a row becomes two Records, the history and the future, and one ForecastingTask whose
target is the future Record. The other forecasting questions come from Time-MMD: a text about a
market, an epidemic, gasoline prices, or traffic, sometimes with a list of recent values, and a
question that asks for a prediction in words. Such a row is a free-answer row: one AnswerTask whose
target is the released answer text, with a Record for each list of values the question prints.
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
    """Parse a forecasting row as a series forecast if it asks for the next values, else as a free-answer row.

    Returns:
        The parsed row.
    """
    if _HORIZON.search(row.question):
        return SeriesForecastingRow.from_release_row(row)
    return FreeAnswerRow.from_release_row(row)


@dataclass(frozen=True)
class SeriesForecastingRow:
    """A forecasting row that prints a series and asks for its next values."""

    row: ReleaseRow
    history: PrintedSeries
    future: PrintedSeries

    @classmethod
    def from_release_row(cls, row: ReleaseRow) -> Self:
        """Parse a row that asks for the next values of its printed series.

        Returns:
            The parsed row.

        Raises:
            TimeNetConnectorError: If the question does not print one series, the answer does not print
                the predictions, or it prints another number of values than the question asks for.
        """  # ruff: ignore[docstring-extraneous-exception] - raised through the ReleaseRow methods
        history = row.only_printed_series_of_question()
        row.match_of_answer_template(_PREDICTIONS)
        future = row.only_printed_series_of_answer()
        horizon = _horizon_of_question(row)
        if len(future.values) != horizon:
            raise row.malformed(f"the question asks for {horizon} values, but the answer prints {len(future.values)}")
        return cls(row=row, history=history, future=future)

    def add_records_and_task_to(self, dataset: TimeFDataset) -> ForecastingTask:
        """Add the history and the future Records, and the task that asks for the future from the history.

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
    """Return the number of values that the question asks for.

    Returns:
        The horizon.

    Raises:
        TimeNetConnectorError: If the question does not ask for the next values.
    """  # ruff: ignore[docstring-extraneous-exception] - raised through malformed()
    match = _HORIZON.search(row.question)
    if match is None:
        raise row.malformed("the question does not ask for the next values of the series")
    return int(match["horizon"])
