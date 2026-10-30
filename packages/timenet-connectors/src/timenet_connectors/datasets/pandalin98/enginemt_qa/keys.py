"""The names EngineMT-QA writes into a dataset, so writing and reading them share one spelling."""

from enum import StrEnum, unique

from timenet_connectors.datasets.pandalin98.enginemt_qa.questions import Vocabulary


@unique
class AnnotationKey(StrEnum):
    """The key of an annotation the connector builds."""

    SOURCE_REPOSITORY = "source_repository"
    SOURCE_REVISION = "source_revision"


def vocabulary_key(vocabulary: Vocabulary) -> str:
    """Give the key of the dataset annotation that holds a closed question's option letters.

    Returns:
        The key, which each task of the question also states as its ``target_schema``.
    """
    return f"{vocabulary}_vocabulary"


def channel_name(index: int) -> str:
    """Name a ``seq_data`` channel by its index, since the release names none: ``channel_00`` to ``channel_32``.

    The names are zero-padded, so their sorted order is the source's.

    Returns:
        The signal name.
    """
    return f"channel_{index:02d}"
