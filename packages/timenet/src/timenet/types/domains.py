"""The :class:`Domain` enum: what kind of data a dataset contains.

Declare every domain that a substantial part of the records belongs to. Put finer labels, such as a
sensor or a benchmark name, in the card's free-form ``tags``.

Each member's docstring gives typical examples and does not claim to be an exhaustive list.
"""

from enum import StrEnum, unique


@unique
class Domain(StrEnum):
    """Describes what kind of data a dataset contains. A dataset may declare more than one."""

    HEALTH = "health"
    """Human physiology and clinical care: biosignals, vital signs, clinical records, glucose.

    The umbrella for ``cardiology``, ``respiratory``, and ``sleep``. Declare it alongside them.
    """
    CARDIOLOGY = "cardiology"
    """The heart: ECG, heart rate, and heart sounds."""
    RESPIRATORY = "respiratory"
    """Breathing and the lungs: respiration, airflow, oxygen saturation, and cough."""
    SLEEP = "sleep"
    """Sleep studies: polysomnography and sleep staging."""
    ACTIVITY = "activity"
    """What people are doing: labeled human activities and daily behavior, usually from wearables.

    Declare ``motion`` as well when the signals record the movement itself.
    """
    MOTION = "motion"
    """Movement of a body or an object: accelerometer, gyroscope, magnetometer, and motion capture."""
    ECONOMICS = "economics"
    """Economic activity: macroeconomic indicators, prices, sales, and demand."""
    FINANCE = "finance"
    """Financial markets: asset prices, returns, trading volumes, and exchange rates."""
    ENVIRONMENT = "environment"
    """The natural world: weather, climate, air quality, and hydrology."""
    ENERGY = "energy"
    """Energy systems: electricity load, generation, power grids, and transformers."""
    TRANSPORT = "transport"
    """Movement of people and goods through transport networks: road traffic, congestion, and ridership.

    Telemetry from a vehicle's own machinery, such as engine health, belongs to ``industrial``.
    """
    OBSERVABILITY = "observability"
    """Software and IT infrastructure: service metrics and cloud monitoring."""
    AUDIO = "audio"
    """Recorded sound: speech, cough, and environmental audio."""
    INDUSTRIAL = "industrial"
    """Machines and industrial processes: engines, turbines, bearings, and production equipment.

    Covers condition monitoring, degradation, fault diagnosis, and remaining useful life, including
    simulated run-to-failure data.
    """
    GENERAL = "general"
    """No particular field: synthetic or demo data, or generic series without a stated origin.

    A corpus that mixes such data with field-specific components declares those domains too.
    """
