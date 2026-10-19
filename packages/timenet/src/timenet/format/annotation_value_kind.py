"""Tags for annotation values stored in the control catalog."""

from enum import StrEnum, unique


@unique
class AnnotationValueKind(StrEnum):
    """Identify the populated value column of an annotation's content."""

    TEXT = "text"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    TEXT_LIST = "text_list"
