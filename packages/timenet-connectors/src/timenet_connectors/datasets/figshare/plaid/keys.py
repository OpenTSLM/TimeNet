"""The names PLAID writes into a dataset, so writing and reading them share one spelling."""

from enum import StrEnum, unique

from timenet_connectors.datasets.figshare.plaid.entries import EventKind


@unique
class AnnotationKey(StrEnum):
    """The key of an annotation the connector builds."""

    SOURCE_URL = "source_url"
    SOURCE_FILES = "source_files"
    """Each pinned file, as ``<name> sha256:<digest>``."""
    APPLIANCE_TYPE_VOCABULARY = "appliance_type_vocabulary"
    """The submetered appliance types, which the appliance task names as its ``target_schema``."""
    APPLIANCE_ON = "appliance_on"
    """An aggregated file's labelled event: the appliance, by type, switched on at the annotation's point."""
    APPLIANCE_OFF = "appliance_off"
    """An aggregated file's labelled event: the appliance, by type, switched off at the annotation's point."""


EVENT_KEYS = {EventKind.ON: AnnotationKey.APPLIANCE_ON, EventKind.OFF: AnnotationKey.APPLIANCE_OFF}
"""The annotation key of each kind of event."""
