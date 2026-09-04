"""The names this connector writes into a dataset.

An annotation key reaches a consumer as a string, and several modules state the same ones. They
are named here so that a rename cannot leave one site behind.
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
