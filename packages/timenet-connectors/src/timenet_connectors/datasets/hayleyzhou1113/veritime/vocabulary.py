"""Published vocabulary, source contracts, and signal specifications."""

from enum import StrEnum, unique
from fractions import Fraction
import re
from typing import NamedTuple

from timenet.types import (
    Label,
    LabelSet,
    Split,
    TimeSeriesSpec,
    ureg,
)


SERIES_MARKER = "<ts><ts/>"


"""Where a question takes its series. The authors' scripts replace it with the series as text."""


KNOWLEDGE_SEPARATOR = ", "


_MICROSECONDS_PER_SECOND = 1_000_000


# An inferential calculation label: a count, except one two-part row whose label is a decimal amplitude.
QUANTITY_PATTERN = re.compile(r"[0-9]+(\.[0-9]+)?")


@unique
class Family(StrEnum):
    """The paper's two task categories, one folder of the release each."""

    SCENARIO = "scenario"
    KNOWLEDGE = "knowledge"


@unique
class Part(StrEnum):
    """One file of the release: a family's train or test rows."""

    SCENARIO_TRAIN = "scenario_train"
    SCENARIO_TEST = "scenario_test"
    KNOWLEDGE_TRAIN = "knowledge_train"
    KNOWLEDGE_TEST = "knowledge_test"

    @property
    def family(self) -> Family:
        """The family whose rows the file holds."""
        return Family(self.value.split("_")[0])

    @property
    def timef_split(self) -> Split:
        """The TimeNet split of the file's rows: the authors' split."""
        return Split(self.value.split("_")[1])


@unique
class SeriesKind(StrEnum):
    """What a series measures: a ChatTS series, or one knowledge task's signal."""

    SCENARIO = "scenario"
    CTU = "ctu"
    ECG = "ecg"
    EMG = "emg"
    RCW = "rcw"

    @property
    def family(self) -> Family:
        """The family whose rows publish series of the kind."""
        return Family.SCENARIO if self is SeriesKind.SCENARIO else Family.KNOWLEDGE


@unique
class SourceTask(StrEnum):
    """A row's ``task``, as the release spells it: the paper's seven tasks."""

    ANOMALY_DETECTION = "Anomaly detection"
    SCENARIO_ATTRIBUTION = "Scenario attribution"
    INFERENTIAL_CALCULATION = "Inferential calculation"
    CTU = "CTU"
    ECG = "ECG"
    EMG = "EMG"
    RCW = "RCW"

    @property
    def kind(self) -> SeriesKind:
        """The kind of series the task's rows publish."""
        scenario = (SourceTask.ANOMALY_DETECTION, SourceTask.SCENARIO_ATTRIBUTION)
        if self in scenario or self is SourceTask.INFERENTIAL_CALCULATION:
            return SeriesKind.SCENARIO
        return SeriesKind(self.name.lower())

    @property
    def label_set_name(self) -> str:
        """The name of the task's label set, such as ``ecg``."""
        return self.value.lower().replace(" ", "_")


CLASSES = {
    # The authors' extract_label.py takes Yes or No from the ChatTS answer.
    SourceTask.ANOMALY_DETECTION: ("Yes", "No"),
    # Each knowledge question lists its classes as options a), b), and so on, in this order.
    SourceTask.CTU: ("Desktop", "Laptop"),
    SourceTask.ECG: ("normal sinus rhythm", "atrial fibrillation", "other cardiac rhythms", "noise"),
    SourceTask.EMG: ("Healthy", "Myopathy", "Neuropathy"),
    SourceTask.RCW: ("Right Whale Present", "No Right Whale"),
}


"""The classes of each classification task, as the release spells them. Scenario attribution offers its own
options in each question and inferential calculation answers a count, so neither has one."""


LABEL_SETS = {
    task: LabelSet(name=task.label_set_name, labels=tuple(Label(name=name) for name in classes))
    for task, classes in CLASSES.items()
}


"""The label set that every classification task of one source task shares, built from its classes."""


STEP_LABEL_DESCRIPTIONS = {
    "step1_label": "The process label of step 1, the task intent, as published.",
    "step2_label": "The process label of step 2, the key patterns separated by semicolons, as published.",
    "step4_label": "The process label of step 4, the preliminary answer, as published.",
    "step6_label": "The process label of step 6, the final answer, as published.",
}


"""The process labels the authors' RL rewards score, and the description of each task annotation."""


SOURCE_ANSWER = "source_answer"


_SOURCE_ANSWER_DESCRIPTION = "The ChatTS answer the authors extracted the label from, as published."


ANNOTATION_DESCRIPTIONS = {**STEP_LABEL_DESCRIPTIONS, SOURCE_ANSWER: _SOURCE_ANSWER_DESCRIPTION}


class SeriesKindSpec(NamedTuple):
    """What a kind of series measures and how its values are spaced."""

    signal: str
    spec: TimeSeriesSpec
    period_us: Fraction | None
    """The time between values as published, or None where the release gives no rate."""


# The ECG and RCW questions state 300 Hz and 2 kHz, but the series are every second and every eighth
# sample of those, so 150 Hz and 250 Hz. The authors' GitHub copy keeps the originals in its `timeseries`
# column for test rows only, so the train rows are assumed to be decimated the same way. No low-pass
# filter ran first, so RCW is aliased above 125 Hz. Supporting evidence: the median R-R interval of normal
# sinus rhythm rows is 132.5 samples, 68 bpm at 150 Hz, and the RCW average spectrum is flat (README.md).
# CTU states its 2-minute cadence. The ChatTS series are synthetic and EMG states no rate, so both are
# ordinal.
KINDS: dict[SeriesKind, SeriesKindSpec] = {
    SeriesKind.SCENARIO: SeriesKindSpec(
        "series",
        TimeSeriesSpec(spec_type="chatts_series", name="ChatTS series", unit_value=None, dtype="float64"),
        None,
    ),
    # Each CTU series is z-normalised on its own (mean 0, standard deviation 0.9993), so its values are
    # dimensionless and not consumption, though the spec and signal keep the name of the quantity.
    SeriesKind.CTU: SeriesKindSpec(
        "consumption",
        TimeSeriesSpec(
            spec_type="electricity_consumption",
            name="Household electricity consumption",
            unit_value=None,
            dtype="float64",
        ),
        Fraction(120 * _MICROSECONDS_PER_SECOND),
    ),
    SeriesKind.ECG: SeriesKindSpec(
        "ecg",
        TimeSeriesSpec(spec_type="ecg", name="Single-lead ECG", unit_value=None, dtype="float64"),
        Fraction(_MICROSECONDS_PER_SECOND, 150),
    ),
    # The EMG question states the amplitude in mV; the others state no unit.
    SeriesKind.EMG: SeriesKindSpec(
        "emg",
        TimeSeriesSpec(spec_type="emg", name="Needle EMG", unit_value=ureg.millivolt, dtype="float64"),
        None,
    ),
    SeriesKind.RCW: SeriesKindSpec(
        "audio",
        TimeSeriesSpec(spec_type="hydrophone_audio", name="Hydrophone audio", unit_value=None, dtype="float64"),
        Fraction(_MICROSECONDS_PER_SECOND, 250),
    ),
}
