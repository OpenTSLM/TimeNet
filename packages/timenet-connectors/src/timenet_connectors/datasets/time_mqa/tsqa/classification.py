"""Parse accelerometer questions and create activity or gait classification tasks.

The classification file mixes two source datasets with fixed question texts. A WISDM question
prints 10 timestamps of a phone accelerometer as 30 values, the X, Y, and Z values of each
timestamp in turn. A Daphnet question prints one timestamp of three body-worn accelerometers as 9
values. Each row becomes one record with one signal per axis and one classification task.
"""

from dataclasses import dataclass
import re
from typing import Self

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.json import JsonMapping
from timenet.types import ClassificationTask, Label, LabelSet, TimeSeriesSpec, ureg
from timenet_connectors.datasets.time_mqa.tsqa.release import PrintedSeries, ReleaseRow


# The labels are the words of the released answers, in the order of the question text.
ACTIVITY = LabelSet(
    name="activity",
    labels=(
        Label(name="Walking"),
        Label(name="Jogging"),
        Label(name="Upstairs", description="Walking up stairs."),
        Label(name="Downstairs", description="Walking down stairs."),
        Label(name="Sitting"),
        Label(name="Standing"),
    ),
)
GAIT_FREEZING = LabelSet(
    name="gait_freezing",
    labels=(
        Label(name="No freeze"),
        Label(name="Freeze", description="Freezing of gait."),
    ),
)
# The WISDM question states that 10 represents 1g, so its values are in meters per second squared.
PHONE_ACCELERATION = TimeSeriesSpec(
    spec_type="accelerometer",
    name="Accelerometer",
    unit_value=ureg.meter / ureg.second**2,
    dtype="float64",
)
# The Daphnet question states its values in milli-g.
BODY_ACCELERATION = TimeSeriesSpec(
    spec_type="accelerometer_milli_g",
    name="Accelerometer",
    unit_value=ureg.Unit("millistandard_gravity"),
    dtype="float64",
)


@dataclass(frozen=True)
class AccelerometerSource:
    """A classification source and the format of its accelerometer data."""

    name: str
    question_start: str
    """The first sentence of every question from the source."""
    answer: re.Pattern[str]
    """The answer pattern, with the label in its ``label`` group."""
    label_set: LabelSet
    spec: TimeSeriesSpec
    time_axis: RegularAxis
    axes: tuple[str, ...]
    """Signal names in the order the question lists the axes at each timestep."""
    timesteps: int

    @property
    def n_values(self) -> int:
        """The total number of axis values in the question."""
        return len(self.axes) * self.timesteps

    def create_record(self, series: PrintedSeries, metadata: JsonMapping) -> Record:
        """Create a record with one signal per axis from values ordered by timestep.

        The question prints the axes of one timestep in turn, so each axis takes every
        ``len(axes)``-th value.

        Returns:
            The record, which is not yet in a dataset.
        """
        signals = tuple(
            Signal.from_values(
                series.values[position :: len(self.axes)], spec=self.spec, name=axis, time_axis=self.time_axis
            )
            for position, axis in enumerate(self.axes)
        )
        return Record(sources=(Source(name="Accelerometer", signals=signals),), metadata=dict(metadata))


WISDM = AccelerometerSource(
    name="wisdm",
    question_start="Following data provides accelerometer data for activity recognition research.",
    answer=re.compile(r"Based on the given information, the activity is (?P<label>.+)\."),
    label_set=ACTIVITY,
    spec=PHONE_ACCELERATION,
    time_axis=RegularAxis.from_rate_hz(20),
    axes=("accelerometer_x", "accelerometer_y", "accelerometer_z"),
    timesteps=10,
)
DAPHNET = AccelerometerSource(
    name="daphnet",
    question_start="This dataset captures gait freezing using wearable accelerometers",
    answer=re.compile(r"Based on the given information, the answer is (?P<label>.+)\."),
    label_set=GAIT_FREEZING,
    spec=BODY_ACCELERATION,
    time_axis=RegularAxis.from_rate_hz(64),
    axes=tuple(
        f"{sensor}_{direction}"
        for sensor in ("ankle", "thigh", "hip")
        for direction in ("forward", "vertical", "lateral")
    ),
    timesteps=1,
)
SOURCES = (WISDM, DAPHNET)


@dataclass(frozen=True)
class ClassificationRow:
    """A classification row with its source, series, and answer label."""

    row: ReleaseRow
    source: AccelerometerSource
    series: PrintedSeries
    label: str

    @classmethod
    def from_release_row(cls, row: ReleaseRow) -> Self:
        """Parse a classification row.

        Returns:
            The parsed row.

        Raises:
            TimeNetConnectorError: If the source, series length, or answer label is invalid.
        """  # ruff: ignore[docstring-extraneous-exception] - raised through the ReleaseRow methods
        source = _source_of_question(row)
        series = row.only_printed_series_of_question()
        if len(series.values) != source.n_values:
            raise row.malformed(f"a {source.name} question prints {source.n_values} values, not {len(series.values)}")
        return cls(row=row, source=source, series=series, label=row.label_of_answer(source.answer, source.label_set))

    def add_records_and_task_to(self, dataset: TimeFDataset) -> ClassificationTask:
        """Add the accelerometer record and classification task to the dataset.

        Returns:
            The added task.
        """
        record = dataset.add_record(record=self.source.create_record(self.series, self.row.record_metadata("series_1")))
        task = ClassificationTask(
            inputs=(record,),
            prompt=self.row.question_with_series_placeholders((self.series,)),
            targets=(self.label,),
            label_set=self.source.label_set,
            metadata=self.row.task_metadata(),
        )
        dataset.add_task(task=task)
        return task


def _source_of_question(row: ReleaseRow) -> AccelerometerSource:
    """Find the source dataset from the start of the question.

    Returns:
        The source.

    Raises:
        TimeNetConnectorError: If the question does not match a source.
    """  # ruff: ignore[docstring-extraneous-exception] - raised through malformed()
    for source in SOURCES:
        if row.question.startswith(source.question_start):
            return source
    raise row.malformed(f"the question starts like none of the sources {[source.name for source in SOURCES]}")
