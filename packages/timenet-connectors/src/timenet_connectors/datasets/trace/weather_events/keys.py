"""The names TRACE writes into a dataset, so writing and reading them share one spelling."""

from enum import StrEnum, unique


@unique
class AnnotationKey(StrEnum):
    """The key of an annotation the connector builds."""

    NO_EVENT_LABEL = "no_event_label"
    """The dataset's label for a window followed by no event."""
    EVENT_TYPE_VOCABULARY = "event_type_vocabulary"
    """The dataset's event labels, which the event-type task names as its ``target_schema``."""
    WINDOW_START = "window_start"
    """A Record's first timestamp, as published."""
    WINDOW_END = "window_end"
    """A Record's last timestamp, as published."""
    CHANNEL_DESCRIPTION = "channel_description"
    """A retrieval task's generated description of one channel."""


CHANNEL_ORDER = "channel_order"
"""The Source metadata key that lists its channels in source order, which TimeF doesn't keep on read."""
