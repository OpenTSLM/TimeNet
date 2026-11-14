"""The anomaly detection family: one printed series, and whether it includes an anomaly.

Each question describes the source of the series, prints the series, and asks whether it has
anomalies. The answer names one of two labels. Each row becomes one Record and one
ClassificationTask whose target is the label.
"""

from dataclasses import dataclass
import re
from typing import Self

from timenet.dataset import TimeFDataset
from timenet.types import ClassificationTask, Label, LabelSet
from timenet_connectors.datasets.time_mqa.tsqa.release import PrintedSeries, ReleaseRow


# The labels are the words of the released answers.
ANOMALY_POINT = LabelSet(
    name="anomaly_point",
    labels=(
        Label(name="Normal Point", description="The series includes no anomaly."),
        Label(name="Anomaly Point", description="The series includes an anomaly."),
    ),
)
_ANSWER = re.compile(r"Based on the given information, this time series includes (?P<label>.+)\.")


@dataclass(frozen=True)
class AnomalyDetectionRow:
    """A row of the anomaly detection file: its printed series and the label of its answer."""

    row: ReleaseRow
    series: PrintedSeries
    label: str

    @classmethod
    def from_release_row(cls, row: ReleaseRow) -> Self:
        """Parse a row of the anomaly detection file.

        Returns:
            The parsed row.

        Raises:
            TimeNetConnectorError: If the question does not print one series, or the answer does not name
                one of the two labels.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by the ReleaseRow methods
        return cls(
            row=row,
            series=row.only_printed_series_of_question(),
            label=row.label_of_answer(_ANSWER, ANOMALY_POINT),
        )

    def add_records_and_task_to(self, dataset: TimeFDataset) -> ClassificationTask:
        """Add the Record of the series and the task that asks whether it includes an anomaly.

        Returns:
            The added task.
        """
        record = dataset.add_record(record=self.series.create_record(self.row.record_metadata("series_1")))
        task = ClassificationTask(
            inputs=(record,),
            prompt=self.row.question_with_series_placeholders((self.series,)),
            targets=(self.label,),
            label_set=ANOMALY_POINT,
            metadata=self.row.task_metadata(),
        )
        dataset.add_task(task=task)
        return task
