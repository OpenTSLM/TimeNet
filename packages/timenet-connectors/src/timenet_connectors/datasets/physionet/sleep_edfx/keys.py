"""The names this connector writes into a dataset.

An annotation key and a task question both reach a consumer as strings, and several modules
state the same ones. They are named here so that a rename cannot leave one site behind.
"""

from enum import StrEnum


class AnnotationKey(StrEnum):
    """The key of an annotation this connector builds.

    A key whose annotation carries a span states something about a region of the recording. The
    rest state something about the whole of it.
    """

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
    """What a whole-sample task asks. The value ends the id of that task."""

    AGE = "age"
    SEX = "sex"
    CONDITION = "condition"
    SLEEP_PERIOD = "sleep_period"
    LIGHTS_OFF = "lights_off"


class Condition(StrEnum):
    """The two nights of the telemetry study.

    The sheet states a night's condition in the position of its columns and nowhere in a cell,
    so the value is named here rather than read.
    """

    PLACEBO = "placebo"
    TEMAZEPAM = "temazepam"
