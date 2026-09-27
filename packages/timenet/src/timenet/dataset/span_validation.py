"""Validate annotation and task spans against Signal windows."""

import warnings

from timenet.dataset.time_series import Signal
from timenet.errors import SpanOutsideWindowWarning, TimeFValidationError
from timenet.types import Span, StepSpan, TimeInterval, TimeOrigin, TimeSpan


def check_span_within_window(  # noqa: PLR0912, PLR0913 (span rules have several branches)
    label: str,
    span: Span,
    time_series: tuple[Signal, ...],
    record_id: str,
    time_span: TimeInterval | None = None,
    *,
    warn_when_outside: bool = True,
    origins: dict[str, TimeOrigin] | None = None,
) -> None:
    """Reject a span its targeted series cannot place, by the three-part scoping rule.

    A span's bounds are in the source recording timeline, the same frame as each series' axis. Where it
    is allowed to fall depends on what it is scoped to:

    - **Scoped** to named ``time_series_ids``: it claims to apply to every one of them, so it must lie
      inside the *intersection* of their windows. Falling outside even one of those windows breaks
      that claim.
    - **Unscoped** (``time_series_ids`` is ``None``) on a record that declares a ``time_span``: checked
      against that session span. This is how a recording that spans a sensor gap says so. It lets an
      event fall in the gap on purpose, for example a note taken while every sensor was briefly off.
    - **Unscoped** with no ``time_span``: checked against the *union* of the timed series' windows.
      This rejects an event landing in an unrecorded gap instead of silently accepting it. That way, a
      convex hull of the series never masks a hole in the data. Timeless (ordinal) series carry no
      window and drop out.

    Shared by a task's ``scope`` and an annotation so the two never disagree about what a span may cover.

    Args:
        label: Human-readable label for the span, used in the error message.
        span: The span to check.
        time_series: The series the span is checked against.
        record_id: The owning record's id, for the error message.
        time_span: The record's declared session span, if any, consulted only for an unscoped span.
        warn_when_outside: Warn and keep the span when it leaves its window, rather than raise.

    Raises:
        TimeFValidationError: If a series id is unknown. If a scoped span names a timeless series
            or ones whose windows do not overlap. If the record has no timeline for an unscoped
            span. If the span falls outside the window the rule selects, when
            ``warn_when_outside`` is False. If a step span names a series with a timeline, or it runs
            past its steps.
    """
    if isinstance(span, StepSpan):
        ts = next((t for t in time_series if t.id == span.time_series_id), None)
        if ts is None:
            raise TimeFValidationError(
                f"{label} references unknown time_series_id {span.time_series_id!r} on record {record_id!r}"
            )
        if ts.span_us is not None:  # axis-fit: steps only on a series with no timeline
            raise TimeFValidationError(
                f"{label} counts in steps, but series {ts.id!r} on record {record_id!r} has a "
                f"timeline; name the region in seconds instead"
            )
        if span.exclusive_end > ts.n_values:
            raise TimeFValidationError(
                f"{label} runs past the {ts.n_values} steps of series {ts.id!r} on record {record_id!r}: got {span!r}"
            )
        return
    if not isinstance(span, TimeSpan):  # Span is abstract. Only time and step spans reach here
        raise TimeFValidationError(f"{label} is not a concrete span: {span!r}")
    scope = span.time_series_ids
    covered = {ts.id: ts.span_us for ts in time_series if scope is None or ts.id in scope}
    if origins is not None:
        clocks = {id(origins[series_id]) for series_id, window in covered.items() if window is not None}
        if len(clocks) > 1:
            raise TimeFValidationError(f"{label} spans signals on different source clocks in record {record_id!r}")
    for series_id in scope or ():
        if series_id not in covered:
            raise TimeFValidationError(
                f"{label} references unknown time_series_id {series_id!r} on record {record_id!r}"
            )

    if scope is not None:
        windows = [window for window in covered.values() if window is not None]
        if len(windows) < len(covered):
            timeless = sorted(sid for sid, window in covered.items() if window is None)
            raise TimeFValidationError(
                f"{label} is scoped to {timeless}, which have no timeline, so a time-valued region means "
                f"nothing on them; scope it to the series that do"
            )
        start = max(window[0] for window in windows)
        end = min(window[1] for window in windows)
        if start >= end:
            raise TimeFValidationError(
                f"{label} is scoped to series whose windows do not overlap on record {record_id!r}, so no "
                f"region lies inside all of them"
            )
        _reject_outside(label, span, start, end, record_id, warn_when_outside=warn_when_outside)
        return

    if time_span is not None:
        _reject_outside(
            label, span, time_span.start_us, time_span.end_us, record_id, warn_when_outside=warn_when_outside
        )
        return

    windows = sorted(window for window in covered.values() if window is not None)
    if not windows:
        raise TimeFValidationError(
            f"{label} is a time-valued region, but record {record_id!r} has no timeline to place it against: "
            f"no timed series and no time_span. Use a static annotation, or declare a time_span"
        )
    _reject_outside_union(label, span, windows, record_id, warn_when_outside=warn_when_outside)


def _reject_outside(  # noqa: PLR0913 (the sixth is the keyword-only guard)
    label: str, span: TimeSpan, start: int, end: int, record_id: str, *, warn_when_outside: bool = True
) -> None:
    """Reject a time span that runs past the half-open window ``[start, end)``.

    Args:
        label: Human-readable label for the span, used in the message.
        span: The span to check.
        start: The first microsecond of the window.
        end: One microsecond past the window.
        record_id: The owning record's id, for the message.
        warn_when_outside: Warn and keep the span when it leaves its window, rather than raise.

    Raises:
        TimeFValidationError: If the span starts before ``start`` or ends after ``end``, and
            ``warn_when_outside`` is False.
    """
    if span.start_us < start or span.exclusive_end > end:
        message = (
            f"{label} falls outside record {record_id!r} span ({start}, {end}) us: got {span!r}; span "
            f"times are in the source recording timeline"
        )
        if warn_when_outside:
            warnings.warn(f"{message}. It is kept as it was given.", SpanOutsideWindowWarning, stacklevel=2)
            return

        raise TimeFValidationError(message)


def _reject_outside_union(
    label: str,
    span: TimeSpan,
    windows: list[tuple[int, int]],
    record_id: str,
    *,
    warn_when_outside: bool = True,
) -> None:
    """Reject a span not covered by the union of ``windows``.

    With no gaps the union is one contiguous window, so this is the same bounds check as a scope. With
    gaps the span must fall entirely within one of the merged windows. This rejects a span that lands
    in a gap.

    Args:
        label: Human-readable label for the span, used in the message.
        span: The span to check.
        windows: The windows of the timed series, sorted by start.
        record_id: The owning record's id, for the message.
        warn_when_outside: Warn and keep the span when it leaves its window, rather than raise.

    Raises:
        TimeFValidationError: If the span runs past the windows or falls in a gap between them, and
            ``warn_when_outside`` is False.
    """
    merged: list[tuple[int, int]] = []
    for start, end in windows:  # sorted by start, half-open, so windows that touch are contiguous
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    if len(merged) == 1:
        _reject_outside(label, span, merged[0][0], merged[0][1], record_id, warn_when_outside=warn_when_outside)
        return
    if not any(start <= span.start_us and span.exclusive_end <= end for start, end in merged):
        message = (
            f"{label} falls in a gap between the recorded windows of record {record_id!r} {merged}: got "
            f"{span!r}. Declare a time_span if the session spans the gap"
        )
        if warn_when_outside:
            warnings.warn(f"{message}. It is kept as it was given.", SpanOutsideWindowWarning, stacklevel=2)
            return

        raise TimeFValidationError(message)
