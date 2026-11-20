"""The event types of 3W and how its per-second ``class`` labels map to them.

The names and descriptions come from ``dataset/dataset.ini`` of the 3W release. A label of ``0`` is
normal operation, ``k`` (1-9) is the steady state of event ``k``, and ``k + 100`` is its transient
phase. Events 3 (severe slugging) and 4 (flow instability) have no transient phase.
"""

from dataclasses import dataclass
from enum import StrEnum

from timenet.types import Annotation, LabelSet


TRANSIENT_OFFSET = 100
"""The offset between an event's steady-state label and its transient label."""


@dataclass(frozen=True)
class Event:
    """One event type of 3W."""

    code: int
    """The label 3W uses for the event's steady state, which is also its folder name."""
    name: str
    """The event's description in ``dataset.ini``, used as the classification label."""
    has_transient: bool
    """Whether 3W labels a transient phase for this event."""


EVENTS: tuple[Event, ...] = (
    Event(0, "Normal Operation", has_transient=False),
    Event(1, "Abrupt Increase of BSW", has_transient=True),
    Event(2, "Spurious Closure of DHSV", has_transient=True),
    Event(3, "Severe Slugging", has_transient=False),
    Event(4, "Flow Instability", has_transient=False),
    Event(5, "Rapid Productivity Loss", has_transient=True),
    Event(6, "Quick Restriction in PCK", has_transient=True),
    Event(7, "Scaling in PCK", has_transient=True),
    Event(8, "Hydrate in Production Line", has_transient=True),
    Event(9, "Hydrate in Service Line", has_transient=True),
)
EVENTS_BY_CODE = {event.code: event for event in EVENTS}

EVENT_TYPES = LabelSet(
    [event.name for event in EVENTS],
    annotations=[Annotation("name", value="event_type")],
)
"""The closed set of event types a classification task can answer."""


class Phase(StrEnum):
    """The phase a per-second label puts an instant in."""

    NORMAL = "normal"
    TRANSIENT = "transient"
    STEADY = "steady"


def phase_of(label: int, event: Event) -> Phase | None:
    """Return the phase of one per-second label inside an instance of ``event``.

    Args:
        label: The value of the ``class`` column.
        event: The event type of the instance, from its folder.

    Returns:
        The phase, or ``None`` if the label does not belong to this event.
    """
    if label == 0:
        return Phase.NORMAL
    if event.code != 0 and label == event.code:
        return Phase.STEADY
    if event.has_transient and label == event.code + TRANSIENT_OFFSET:
        return Phase.TRANSIENT
    return None
