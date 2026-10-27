"""The keys of the annotations and metadata RATS40K writes, so writing and reading them share one name."""

from enum import StrEnum, unique

from timenet_connectors.datasets.time_ra.rats40k.release import Variant


@unique
class AnnotationKey(StrEnum):
    """The key of an annotation the connector builds."""

    SOURCE_REPOSITORY = "source_repository"
    SOURCE_REVISION = "source_revision"
    SOURCE_DOMAIN = "source_domain"
    """A Record's domain, such as ``Healthcare-ECG``, and its task's only input annotation."""
    LABEL = "label"
    """A task's published ``Label``, kept for the authors' detection metric."""
    ACTION_ID = "action_id"
    """A task's published ``ActionID``, kept for the authors' code-based metric."""


def category_vocabulary(variant: Variant) -> str:
    """Give the key of the dataset annotation that holds a variant's category names.

    Returns:
        The key, which each task of the variant also states as its ``target_schema``.
    """
    return f"{variant}_category_vocabulary"


def request_template(variant: Variant) -> str:
    """Give the key of the dataset annotation that holds a variant's full evaluation prompt.

    Returns:
        The key ``request.authors_request`` reads the template from.
    """
    return f"{variant}_request_template"


CHANNEL_ORDER = "channel_order"
"""The Source metadata key that lists its channels in source order, which TimeF doesn't keep on read."""
