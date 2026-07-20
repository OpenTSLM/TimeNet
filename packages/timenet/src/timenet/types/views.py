"""The :class:`View` enum: which slice of a source a sample represents."""

from enum import StrEnum, unique


@unique
class View(StrEnum):
    """Identifies which slice of the source a :class:`~timenet.dataset.Sample` represents."""

    FULL = "full"
    SINGLE_CHANNEL = "single_channel"
    SUBSET = "subset"
    WINDOW = "window"
