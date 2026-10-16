"""Object kinds shared by catalog references, handles, and stored-value keys."""

from enum import StrEnum, unique


@unique
class ObjectKind(StrEnum):
    """Catalog of tags for TimeF objects."""

    DATASET = "Dataset"
    RECORD = "Record"
    SOURCE = "Source"
    SIGNAL = "Signal"
    TASK = "Task"
    ANNOTATION = "Annotation"
    ANNOTATION_CONTENT = "Content"
    AXIS = "Axis"
