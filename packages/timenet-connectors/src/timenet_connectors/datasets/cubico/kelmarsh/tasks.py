"""The annotations and tasks of one turbine day, built from the status log of the operator."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby

from timenet.dataset import Record
from timenet.types import (
    US_PER_S,
    Annotation,
    ClassificationTask,
    Label,
    LabelSet,
    ScalarPredictionTask,
    Split,
    Task,
    TemporalLocalizationTask,
    TimeInterval,
    TimePoint,
    unix_us,
)
from timenet_connectors.datasets.cubico.kelmarsh.greenbyte import DAY_US, EventKind, OpenStop, StatusEvent


STOP_KEY = "stop"
STOP_MESSAGE_KEY = "stop_message"
CURTAILMENT_KEY = "curtailment"
OPEN_STOP_KEY = "stop_without_end"
UNKNOWN_REASON = "unknown"

# The reasons are the IEC categories that the operator gives a stop. The category is the only
# reason field that means the same for every message of the log.
STOP_REASONS = {
    "Forced outage": "fault",
    "Scheduled Maintenance": "maintenance",
    "Out of Environmental Specification": "environment",
    "Out of Electrical Specification": "grid",
    "Requested Shutdown": "requested",
    "Technical Standby": "standby",
}
STOPPED = LabelSet(
    annotations=[Annotation("name", value="stopped")],
    labels=(
        Label(name="stopped", description="The status log has a stop on this day."),
        Label(name="running", description="The status log has no stop on this day."),
    ),
)
STOP_REASON = LabelSet(
    annotations=[Annotation("name", value="stop_reason")],
    labels=(
        Label(name="fault", description="IEC category Forced outage: a fault stopped the turbine."),
        Label(name="maintenance", description="IEC category Scheduled Maintenance."),
        Label(
            name="environment",
            description="IEC category Out of Environmental Specification, such as icing or the wind limits.",
        ),
        Label(name="grid", description="IEC category Out of Electrical Specification, such as a grid loss."),
        Label(name="requested", description="IEC category Requested Shutdown: a remote or park stop."),
        Label(name="standby", description="IEC category Technical Standby, such as a battery test or a cable unwind."),
    ),
)
CURTAILED = LabelSet(
    annotations=[Annotation("name", value="curtailed")],
    labels=(
        Label(name="curtailed", description="The status log has a curtailment on this day."),
        Label(name="not_curtailed", description="The status log has no curtailment on this day."),
    ),
)
DOWNTIME_60 = LabelSet(
    annotations=[Annotation("name", value="downtime_60")],
    labels=(
        Label(name="downtime", description="The status log has 60 minutes or more of stop time on this day."),
        Label(
            name="no_downtime",
            description="The status log has no stop on this day and less than 60 minutes of curtailment.",
        ),
    ),
)
DOWNTIME_MINUTES = 60
DOWNTIME_60_PROMPT = (
    "Did the turbine have at least 60 minutes of downtime on this day? "
    "Calm-wind waiting and power limiting do not count."
)
STOPPED_PROMPT = "Did the turbine stop at any time on this day?"
STOPPED_MINUTES_PROMPT = "For how many minutes was the turbine stopped on this day?"
STOP_INTERVALS_PROMPT = "In which intervals was the turbine stopped on this day?"
STOP_REASON_PROMPT = "What is the main reason that the turbine was stopped on this day?"
CURTAILED_PROMPT = "Did the operator limit the power output of the turbine at any time on this day?"

DayEvent = tuple[StatusEvent, int, int]
"""An event and the part of it that lies in one day, as microseconds from the start of the day."""


@dataclass(frozen=True)
class SplitCutoffs:
    """The dates that divide a dataset into train, validation and test, in that order in time.

    All turbines share the dates, because they share the weather. No train day comes after a
    validation day, and no validation day comes after a test day.
    """

    validation: datetime
    """The first UTC day of validation. Earlier days are train."""
    test: datetime
    """The first UTC day of test."""

    def split_of_day(self, day_start_us: int) -> Split:
        """Give the split of a day.

        Args:
            day_start_us: The start of the UTC day, in Unix microseconds.

        Returns:
            The split that holds the day.
        """
        if day_start_us < unix_us(self.validation):
            return Split.TRAIN
        return Split.VALIDATION if day_start_us < unix_us(self.test) else Split.TEST

    def crossing_days(self, intervals: Sequence[tuple[int, int]]) -> set[int]:
        """Give the days of each incident that runs across a cutoff.

        An incident is a set of intervals that overlap or touch. Its days would be in two splits,
        with one event in both, so they get no task.

        Args:
            intervals: The start and the end of each event of one turbine, in Unix microseconds.

        Returns:
            The start of each such day, in Unix microseconds.
        """
        cuts = (unix_us(self.validation), unix_us(self.test))
        days: set[int] = set()
        for start, end in _merge(intervals):
            if any(start < cut < end for cut in cuts):
                days.update(range(start // DAY_US * DAY_US, end, DAY_US))
        return days


def events_of_day(events: Sequence[StatusEvent], day_start_us: int) -> list[DayEvent]:
    """Clip the events to one day.

    Args:
        events: The events of the turbine.
        day_start_us: The start of the UTC day, in Unix microseconds.

    Returns:
        Each event that covers a part of the day, with that part, in start order.
    """
    clipped = [
        (event, max(event.start_us - day_start_us, 0), min(event.end_us - day_start_us, DAY_US)) for event in events
    ]
    return sorted((item for item in clipped if item[1] < item[2]), key=lambda item: (item[1], item[2]))


def annotate_day(record: Record, day_events: Sequence[DayEvent]) -> None:
    """Add one annotation for each curtailment of the day, and two for each stop.

    Args:
        record: The Record of the day.
        day_events: The clipped events of the day, from :func:`events_of_day`.
    """
    for event, start, end in day_events:
        span = TimeInterval.micros(start, end)
        if event.kind is EventKind.CURTAILMENT:
            record.annotate(Annotation(key=CURTAILMENT_KEY, value=event.message, span=span))
            continue
        reason = STOP_REASONS.get(event.iec_category, UNKNOWN_REASON)
        record.annotate(Annotation(key=STOP_KEY, value=reason, span=span))
        record.annotate(Annotation(key=STOP_MESSAGE_KEY, value=event.message, span=span))


def unknown_windows(open_stops: Sequence[OpenStop], ok_times_us: Sequence[int]) -> list[tuple[int, int | None]]:
    """Give the time in which the state of the turbine is not known, for each stop without an end.

    The window runs from the start of the stop to the next ``System OK`` entry of the log. That
    entry is not taken as the end of the stop. It only shows that the labels can be trusted again.

    Args:
        open_stops: The stops without an end time.
        ok_times_us: The ``System OK`` times of the same turbine.

    Returns:
        The start and the end of each window, in Unix microseconds. The end is ``None`` if no
        ``System OK`` entry follows the stop.
    """
    ok_times = sorted(ok_times_us)
    return [(stop.start_us, next((time for time in ok_times if time > stop.start_us), None)) for stop in open_stops]


def annotate_open_stops(record: Record, open_stops: Sequence[OpenStop], day_start_us: int) -> None:
    """Add one point annotation for each stop without an end that starts on the day.

    Args:
        record: The Record of the day.
        open_stops: The stops without an end time, of the turbine.
        day_start_us: The start of the UTC day, in Unix microseconds.
    """
    for stop in open_stops:
        if 0 <= stop.start_us - day_start_us < DAY_US:
            record.annotate(
                Annotation(key=OPEN_STOP_KEY, value=stop.message, span=TimePoint.micros(stop.start_us - day_start_us))
            )


def iter_day_tasks(
    record: Record, day_events: Sequence[DayEvent], *, split: Split, curtailment: bool
) -> Iterator[Task]:
    """Create the tasks of one turbine day.

    Args:
        record: The Record of the day.
        day_events: The clipped events of the day, from :func:`events_of_day`.
        split: The split of the day.
        curtailment: Whether the status log of this turbine and year records curtailment. Without
            such a log, a day without an entry is not known to be free of curtailment.

    Yields:
        The tasks, not yet added to a dataset.
    """
    stops = [item for item in day_events if item[0].kind is EventKind.STOP]
    intervals = _merge([(start, end) for _, start, end in stops])
    stopped_us = sum(end - start for start, end in intervals)
    yield ClassificationTask(
        inputs=(record,),
        targets=("stopped" if stops else "running",),
        label_set=STOPPED,
        prompt=STOPPED_PROMPT,
        split=split,
    )
    yield ScalarPredictionTask(
        inputs=(record,),
        targets=(stopped_us / (60 * US_PER_S),),
        unit="minute",
        target_name="stopped_minutes",
        prompt=STOPPED_MINUTES_PROMPT,
        split=split,
    )
    if stops:
        yield TemporalLocalizationTask(
            inputs=(record,),
            targets=tuple(TimeInterval.micros(start, end) for start, end in intervals),
            prompt=STOP_INTERVALS_PROMPT,
            split=split,
        )
    curtailed_us = sum(
        end - start
        for start, end in _merge(
            [(start, end) for event, start, end in day_events if event.kind is EventKind.CURTAILMENT]
        )
    )
    downtime = _downtime_60(stopped_us, curtailed_us)
    reason = _main_reason(stops)
    if reason is not None:
        yield ClassificationTask(
            inputs=(record,), targets=(reason,), label_set=STOP_REASON, prompt=STOP_REASON_PROMPT, split=split
        )
    if downtime is not None:
        yield ClassificationTask(
            inputs=(record,), targets=(downtime,), label_set=DOWNTIME_60, prompt=DOWNTIME_60_PROMPT, split=split
        )
    if curtailment:
        curtailed = any(event.kind is EventKind.CURTAILMENT for event, _, _ in day_events)
        yield ClassificationTask(
            inputs=(record,),
            targets=("curtailed" if curtailed else "not_curtailed",),
            label_set=CURTAILED,
            prompt=CURTAILED_PROMPT,
            split=split,
        )


def _downtime_60(stopped_us: int, curtailed_us: int) -> str | None:
    """Give the downtime label of a day.

    A day with 60 minutes or more of stop is ``downtime``. A day with no stop and less than 60
    minutes of curtailment is ``no_downtime``. A day with 1 to 59 minutes of stop, or with no stop
    and a long curtailment, gets no label: a short stop is a different question, and a power limit
    is not a stop.

    Args:
        stopped_us: The merged stop time of the day, in microseconds.
        curtailed_us: The merged curtailment time of the day, in microseconds.

    Returns:
        ``downtime``, ``no_downtime``, or ``None`` if the day is in between.
    """
    limit_us = DOWNTIME_MINUTES * 60 * US_PER_S
    if stopped_us >= limit_us:
        return "downtime"
    return "no_downtime" if stopped_us == 0 and curtailed_us < limit_us else None


def _merge(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Join intervals that overlap or touch, so that no stopped time counts twice."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _main_reason(stops: Sequence[DayEvent]) -> str | None:
    """Give the reason with the most stopped time, or ``None`` if a stop of the day has no category."""
    if not stops or any(event.iec_category not in STOP_REASONS for event, _, _ in stops):
        return None
    by_reason = sorted(stops, key=lambda item: STOP_REASONS[item[0].iec_category])
    totals = {
        reason: sum(end - start for _, start, end in group)
        for reason, group in groupby(by_reason, key=lambda item: STOP_REASONS[item[0].iec_category])
    }
    # A tie goes to the reason that comes first in the alphabet, so that a rebuild gives the same label.
    return max(sorted(totals), key=lambda reason: totals[reason])
