"""The :class:`Domain` enum: what kind of data a dataset contains."""

from enum import StrEnum


class Domain(StrEnum):
    """Describes what kind of data a dataset contains. A dataset may declare more than one."""

    HEALTH = "health"
    CARDIOLOGY = "cardiology"
    SLEEP = "sleep"
    ACTIVITY = "activity"
    ECONOMICS = "economics"
    FINANCE = "finance"
    GENERAL = "general"
