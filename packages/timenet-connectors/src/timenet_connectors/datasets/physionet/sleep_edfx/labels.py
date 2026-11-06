"""The keys, signal specs, and label sets that Sleep-EDFx writes into a dataset."""

from enum import StrEnum

from timenet.types import Label, LabelSet, TimeSeriesSpec, ureg
from timenet_connectors.demographics import SEXES


class AnnotationKey(StrEnum):
    """The key of an annotation this connector builds."""

    SLEEP_STAGE = "sleep_stage"
    LIGHTS_OFF = "lights_off"
    STUDY = "study"
    NIGHT = "night"
    SEX = "sex"
    AGE = "age"
    CONDITION = "condition"
    RECORDING_START_LOCAL = "recording_start_local"
    DEMOGRAPHICS_NOTE = "demographics_note"


class Question(StrEnum):
    """The ``target_name`` of a scalar task."""

    AGE = "age"


class Condition(StrEnum):
    """The drug of each telemetry night. The sheet states it only by the position of its columns."""

    PLACEBO = "placebo"
    TEMAZEPAM = "temazepam"


_EEG = TimeSeriesSpec(spec_type="eeg", name="EEG", unit_value=ureg.microvolt)
_EOG = TimeSeriesSpec(spec_type="eog", name="EOG", unit_value=ureg.microvolt)
_EMG = TimeSeriesSpec(spec_type="emg", name="EMG", unit_value=ureg.microvolt)
_RESPIRATION = TimeSeriesSpec(spec_type="respiration", name="Oro-nasal airflow", unit_value=ureg.dimensionless)
_MARKER = TimeSeriesSpec(spec_type="marker", name="Event marker", unit_value=ureg.dimensionless)
_TEMPERATURE = TimeSeriesSpec(spec_type="temperature", name="Rectal temperature", unit_value=ureg.degC)

# The two studies share four signal names. The rest belong to one study.
SPECS = {
    "EEG Fpz-Cz": _EEG,
    "EEG Pz-Oz": _EEG,
    "EOG horizontal": _EOG,
    "EMG submental": _EMG,
    "Resp oro-nasal": _RESPIRATION,
    "Temp rectal": _TEMPERATURE,
    "Event marker": _MARKER,  # for cassette study
    "Marker": _MARKER,  # for telemetry study
}

# The labels the release writes for a scored epoch. `Sleep stage W` is wake, `Movement time` marks
# a disturbed epoch, and `Sleep stage ?` marks an epoch the technician left unscored.
SLEEP_STAGES = LabelSet(
    name=str(AnnotationKey.SLEEP_STAGE),
    labels=(
        Label(name="Sleep stage W", description="Wake."),
        Label(name="Sleep stage 1"),
        Label(name="Sleep stage 2"),
        Label(name="Sleep stage 3"),
        Label(name="Sleep stage 4"),
        Label(name="Sleep stage R", description="Rapid eye movement (REM) sleep."),
        Label(name="Movement time", description="An epoch that body movement obscures, so it has no stage."),
        Label(name="Sleep stage ?", description="An epoch with no sleep-stage score from data collection."),
    ),
)
STAGE_LABELS = SLEEP_STAGES.names

# The five labels that say the subject was asleep. Only they bound a night.
ASLEEP_LABELS = frozenset(STAGE_LABELS[1:6])

SEX_LABELS = SEXES.names

CONDITION_LABELS = tuple(Condition)
CONDITIONS = LabelSet(name=str(AnnotationKey.CONDITION), labels=tuple(Label(name=str(one)) for one in Condition))
