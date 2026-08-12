"""The :class:`Sample` type: one logical unit of time-series data."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import cast

import numpy as np
import pyarrow as pa

from timenet.dataset.time_series import TimeSeries
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, Span, new_id
from timenet.types.clock import check_int64, unix_us


def check_span_within_window(
    label: str,
    span: Span,
    time_series: tuple[TimeSeries, ...],
    sample_id: str,
) -> None:
    """Reject a span its targeted series cannot place, by the scoping rule.

    A span's bounds are in the source recording timeline, the same frame as each series' axis. Where it
    is allowed to fall depends on what it is scoped to:

    - **Scoped** to named ``time_series_ids``: it claims to apply to every one of them, so it must lie
      inside the *intersection* of their windows. Falling outside even one would be misleading.
    - **Unscoped** (``time_series_ids`` is ``None``): checked against the *union* of the timed series'
      windows. An event landing in an unrecorded gap is rejected rather than silently accepted, so a
      convex hull of the series never masks a hole in the data. Timeless (ordinal) series carry no
      window and drop out.

    Shared by a task's ``scope`` and an annotation so the two never disagree about what a span may cover.

    Args:
        label: Human-readable label for the span, used in the error message.
        span: The span to check.
        time_series: The series the span is checked against.
        sample_id: The owning sample's id, for the error message.

    Raises:
        TimeFValidationError: If a series id is unknown, a scoped span names a timeless series or ones
            whose windows do not overlap, the sample has no timeline for an unscoped span, or the span
            falls outside the window the rule selects.
    """
    scope = span.time_series_ids
    covered = {ts.time_series_id: ts.span_us for ts in time_series if scope is None or ts.time_series_id in scope}
    for series_id in scope or ():
        if series_id not in covered:
            raise TimeFValidationError(
                f"{label} references unknown time_series_id {series_id!r} on sample {sample_id!r}"
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
                f"{label} is scoped to series whose windows do not overlap on sample {sample_id!r}, so no "
                f"region lies inside all of them"
            )
        _reject_outside(label, span, start, end, sample_id)
        return

    windows = sorted(window for window in covered.values() if window is not None)
    if not windows:
        raise TimeFValidationError(
            f"{label} is a time-valued region, but sample {sample_id!r} has no timeline to place it against: "
            f"no timed series. Use a static annotation instead"
        )
    _reject_outside_union(label, span, windows, sample_id)


def _exclusive_end(span: Span) -> int:
    """Return the exclusive upper bound of ``span``: its ``end`` for an interval, ``start + 1`` for a point.

    The window is half-open ``[start, end)``, so a point's exclusive end is ``start + 1`` and a point
    sitting exactly on an excluded upper bound is caught.
    """
    return span.start + 1 if span.is_point else cast("int", span.end)


def _reject_outside(label: str, span: Span, start: int, end: int, sample_id: str) -> None:
    """Reject a span that runs past the half-open window ``[start, end)``.

    Raises:
        TimeFValidationError: If the span starts before ``start`` or ends after ``end``.
    """
    if span.start < start or _exclusive_end(span) > end:
        raise TimeFValidationError(
            f"{label} ({span.start}, {span.end}) us falls outside sample {sample_id!r} span "
            f"({start}, {end}) us; span times are in the source recording timeline"
        )


def _reject_outside_union(label: str, span: Span, windows: list[tuple[int, int]], sample_id: str) -> None:
    """Reject a span not covered by the union of ``windows``.

    With no gaps the union is one contiguous window, so this is the same bounds check as a scope. With
    gaps the span must fall entirely within one of the merged windows; one landing in a gap is rejected.

    Raises:
        TimeFValidationError: If the span runs past the windows or falls in a gap between them.
    """
    merged: list[tuple[int, int]] = []
    for start, end in windows:  # sorted by start; half-open, so windows that touch are contiguous
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    if len(merged) == 1:
        _reject_outside(label, span, merged[0][0], merged[0][1], sample_id)
        return
    if not any(start <= span.start and _exclusive_end(span) <= end for start, end in merged):
        raise TimeFValidationError(
            f"{label} ({span.start}, {span.end}) us falls in a gap between the recorded windows of sample "
            f"{sample_id!r} {merged}"
        )


@dataclass(kw_only=True)
class Sample:
    """One logical unit of time-series data: a recording, a session, a sensor bundle, a market window.

    Created via :meth:`~timenet.dataset.TimeFDataset.add_sample`. Mutable so ``task_ids`` and
    ``annotations`` can be populated after construction.
    """

    time_series: tuple[TimeSeries, ...]
    """The logical :class:`TimeSeries` streams the sample uses."""
    sample_id: str = field(default_factory=new_id)
    """Unique id for the sample (default: an auto-generated uuid7)."""
    subject_ids: tuple[str, ...] = ()
    """Subjects this sample belongs to (empty for subject-less domains)."""
    task_ids: tuple[str, ...] = ()
    """Ids of the tasks attached to this sample."""
    annotations: tuple[Annotation, ...] = ()
    """Annotations attached to the sample."""
    start_time: datetime | int | None = None
    """Wall-clock timestamp that this sample's relative time zero refers to, for every series and
    annotation on it. Pass a timezone-aware :class:`~datetime.datetime` or whole Unix microseconds;
    construction normalizes either one to microseconds, so a constructed sample holds an ``int``.
    ``None`` means no wall-clock reference exists; never fabricate one.

    A bare float is refused, because seconds and microseconds are both plausible readings of it. When
    the source really does hand over seconds, convert at the call site so the unit is visible::

        start_time=datetime(2026, 8, 5, tzinfo=timezone.utc)   # 1_785_888_000_000_000
        start_time=seconds_to_us(1)                            # 1_000_000, one second past the epoch
        start_time=1_000_000                                   # the same moment, written directly
    """

    def __post_init__(self) -> None:
        """Normalize ``start_time`` to whole Unix microseconds and range-check it.

        Both steps delegate their contract: ``unix_us`` rejects a naive datetime or a bare float, and
        ``check_int64`` rejects an anchor past the int64 microsecond column, each raising
        :class:`~timenet.errors.TimeFValidationError`.
        """
        if self.start_time is None:
            return
        anchor = unix_us(self.start_time)
        check_int64("Sample.start_time", anchor)
        self.start_time = anchor

    @property
    def has_absolute_time(self) -> bool:
        """Whether this sample's relative timeline has a Unix-time anchor."""
        return self.start_time is not None

    def add_annotation(self, annotation: Annotation) -> Annotation:
        """Attach an annotation to the sample and return it.

        Args:
            annotation: The annotation to attach.

        Returns:
            The attached annotation (the same instance).

        Raises:
            TimeFValidationError: If the annotation's span references a series not on this sample, a
                scoped span names a timeless series, or the span falls outside the window its scope
                selects (the intersection of named series, or the union of the series' windows).
        """  # noqa: DOC502 (raised by check_span_within_window, not directly here)
        if annotation.span is not None:
            check_span_within_window(
                f"annotation {annotation.key!r}",
                annotation.span,
                self.time_series,
                self.sample_id,
            )
        self.annotations = (*self.annotations, annotation)
        return annotation

    def to_arrow(self) -> pa.Array:
        """Read the sole channel's values as an Arrow array, for the common single-channel sample.

        Returns:
            The single :class:`TimeSeries`' values as a 1-D Arrow array.

        Raises:
            ValueError: If the sample has more than one channel; read ``time_series[i]`` explicitly then.
        """
        if len(self.time_series) != 1:
            raise ValueError(
                f"Sample.to_arrow() needs a single-channel sample, but this one has "
                f"{len(self.time_series)} series; read sample.time_series[i].to_arrow() instead"
            )
        return self.time_series[0].to_arrow()

    def to_numpy(self) -> np.ndarray:
        """Read the sole channel's values as a NumPy array (materializes :meth:`to_arrow`).

        Returns:
            The single :class:`TimeSeries`' values as a 1-D ``np.ndarray``.
        """
        return self.to_arrow().to_numpy(zero_copy_only=False)
