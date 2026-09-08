"""The :class:`Domain` enum: what kind of data a dataset contains."""

from enum import StrEnum, unique


@unique
class Domain(StrEnum):
    """Describes what kind of data a dataset contains. A dataset may declare more than one."""

    HEALTH = "health"
    CARDIOLOGY = "cardiology"
    RESPIRATORY = "respiratory"
    SLEEP = "sleep"
    ACTIVITY = "activity"
    MOTION = "motion"
    ECONOMICS = "economics"
    FINANCE = "finance"
    ENVIRONMENT = "environment"
    ENERGY = "energy"
    TRANSPORT = "transport"
    OBSERVABILITY = "observability"
    AUDIO = "audio"
    GENERAL = "general"
