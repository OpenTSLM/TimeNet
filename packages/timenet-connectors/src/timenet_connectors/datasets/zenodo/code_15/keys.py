"""The names CODE-15% writes into a dataset, so writing and reading them share one spelling."""

from enum import StrEnum, unique


@unique
class AnnotationKey(StrEnum):
    """The key of an annotation the connector builds."""

    SOURCE_URL = "source_url"
    SOURCE_DOI = "source_doi"
    ABNORMALITY_VOCABULARY = "abnormality_vocabulary"
    """The six abnormalities, in the authors' output order: the abnormality tasks' ``target_schema``."""
    SEX = "sex"
    DEATH = "death"
    """Whether the patient died in the follow-up, on the one exam of a patient that states it."""
    FOLLOW_UP = "follow_up"
    """``timey``: years to death, or to the end of follow-up for a survivor, on the same exam as ``death``."""
    NN_PREDICTED_AGE = "nn_predicted_age"
    """The age the authors' network predicts, an age task's annotation."""
    NORMAL_ECG = "normal_ecg"
    """Whether the exam is normal, an abnormality task's annotation."""
