"""The names CAPTURE-24 writes into a dataset, so writing and reading them share one spelling."""

from enum import StrEnum, unique

from timenet_connectors.datasets.oxwearables.capture24.vocabulary import LabelScheme


@unique
class AnnotationKey(StrEnum):
    """The key of an annotation the connector builds."""

    SOURCE_URL = "source_url"
    SOURCE_SHA256 = "source_sha256"
    AGE = "age"
    """A participant's age band, as ``metadata.csv`` states it, such as ``38-52``."""
    SEX = "sex"
    RECORDING_START_LOCAL = "recording_start_local"
    """The first sample's timestamp, as the source writes it, with no zone. The authors randomized its
    date, so only its time of day is the participant's."""


def scheme_vocabulary(scheme: LabelScheme) -> str:
    """Give the key of the dataset annotation that holds a scheme's classes.

    Returns:
        The key, which each task of the scheme also states as its ``target_schema``.
    """
    return f"{scheme.stem}_vocabulary"
