"""How a :class:`~timenet.types.Span` is stored in the control plane, and how it comes back.

A span is four columns: the frame it counts in, its start, its exclusive end (null for a point) and
the series it is scoped to. The writer and the reader both call this module, so the two halves of
the contract cannot drift apart. A span must come back as the leaf type its frame and bounds
describe, or it stops comparing equal to the one the writer stored.
"""

from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.types import Span, StepInterval, StepPoint, StepSpan, TimeInterval, TimePoint, TimeSpan


SpanRow = tuple[str, int, int | None, list[str] | None]
"""A span as ``(frame, start_us, end_us, time_series_ids)``, in the column order the tables declare."""


def span_row(span: Span) -> SpanRow:
    """Return a span's stored columns.

    Args:
        span: The span to store.

    Returns:
        The frame, the start, the exclusive end (``None`` for a point), and the scoped series ids.

    Raises:
        TimeFValidationError: If ``span`` is not a concrete time or step span.
    """
    if isinstance(span, StepSpan):
        # A step span counts on exactly one series, stored as a one-element list so both frames
        # share the column.
        return (span.frame, span.start, None if span.is_point else span.exclusive_end, [span.time_series_id])
    if isinstance(span, TimeSpan):
        scope = None if span.time_series_ids is None else list(span.time_series_ids)
        return (span.frame, span.start_us, None if span.is_point else span.exclusive_end, scope)
    raise TimeFValidationError(f"cannot store {span!r}: not a concrete span")


def span_from_row(frame: str, start_us: int, end_us: int | None, time_series_ids: list[str] | None) -> Span:
    """Rebuild a span from its stored columns.

    Args:
        frame: ``"seconds"`` or ``"steps"``.
        start_us: The start bound, in the frame's own unit.
        end_us: The exclusive end, or ``None`` for a point.
        time_series_ids: The series the span is scoped to.

    Returns:
        The span, as the concrete leaf type its frame and bounds describe.

    Raises:
        TimeFFormatError: If the frame is unknown, or a step span stores no series to count on.
    """
    if frame == "steps":
        if not time_series_ids:
            raise TimeFFormatError("a step span stores the one series it counts on, but its id list is empty")
        time_series_id = time_series_ids[0]
        if end_us is None:
            return StepPoint(time_series_id=time_series_id, start=start_us)
        return StepInterval(time_series_id=time_series_id, start=start_us, stop=end_us)
    if frame != "seconds":
        raise TimeFFormatError(f"unknown span frame {frame!r}; expected 'seconds' or 'steps'")
    scope = None if time_series_ids is None else tuple(time_series_ids)
    if end_us is None:
        return TimePoint(start_us=start_us, time_series_ids=scope)
    return TimeInterval(start_us=start_us, end_us=end_us, time_series_ids=scope)
