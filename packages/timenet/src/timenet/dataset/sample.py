"""The :class:`Sample` type: one logical unit of time-series data."""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pyarrow as pa

from timenet.dataset.time_series import TimeSeries
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, Span, StepSpan, TimeInterval, TimePoint, TimeSpan, new_id
from timenet.types.clock import check_int64, offset_us, unix_us


def check_span_within_window(
    label: str,
    span: Span,
    time_series: tuple[TimeSeries, ...],
    sample_id: str,
    time_span: TimeInterval | None = None,
) -> None:
    """Reject a span its targeted series cannot place, by the three-part scoping rule.

    A span's bounds are in the source recording timeline, the same frame as each series' axis. Where it
    is allowed to fall depends on what it is scoped to:

    - **Scoped** to named ``time_series_ids``: it claims to apply to every one of them, so it must lie
      inside the *intersection* of their windows. Falling outside even one would be misleading.
    - **Unscoped** (``time_series_ids`` is ``None``) on a sample that declares a ``time_span``: checked
      against that session span. This is how a recording spanning a sensor gap says so, letting an event
      fall in the gap on purpose (a note taken while every sensor was briefly off).
    - **Unscoped** with no ``time_span``: checked against the *union* of the timed series' windows. An
      event landing in an unrecorded gap is rejected rather than silently accepted, so a convex hull of
      the series never masks a hole in the data. Timeless (ordinal) series carry no window and drop out.

    Shared by a task's ``scope`` and an annotation so the two never disagree about what a span may cover.

    Args:
        label: Human-readable label for the span, used in the error message.
        span: The span to check.
        time_series: The series the span is checked against.
        sample_id: The owning sample's id, for the error message.
        time_span: The sample's declared session span, if any; consulted only for an unscoped span.

    Raises:
        TimeFValidationError: If a series id is unknown, a scoped span names a timeless series or ones
            whose windows do not overlap, the sample has no timeline for an unscoped span, or the span
            falls outside the window the rule selects; or a step span names a series with a timeline or
            runs past its steps.
    """
    if isinstance(span, StepSpan):
        ts = next((t for t in time_series if t.time_series_id == span.time_series_id), None)
        if ts is None:
            raise TimeFValidationError(
                f"{label} references unknown time_series_id {span.time_series_id!r} on sample {sample_id!r}"
            )
        if ts.span_us is not None:  # axis-fit: steps only on a series with no timeline
            raise TimeFValidationError(
                f"{label} counts in steps, but series {ts.time_series_id!r} on sample {sample_id!r} has a "
                f"timeline; name the region in seconds instead"
            )
        if span.exclusive_end > ts.n_values:
            raise TimeFValidationError(
                f"{label} runs past the {ts.n_values} steps of series {ts.time_series_id!r} on sample "
                f"{sample_id!r}: got {span!r}"
            )
        return
    if not isinstance(span, TimeSpan):  # Span is abstract; only time and step spans reach here
        raise TimeFValidationError(f"{label} is not a concrete span: {span!r}")
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

    if time_span is not None:
        _reject_outside(label, span, time_span.start_us, time_span.end_us, sample_id)
        return

    windows = sorted(window for window in covered.values() if window is not None)
    if not windows:
        raise TimeFValidationError(
            f"{label} is a time-valued region, but sample {sample_id!r} has no timeline to place it against: "
            f"no timed series and no time_span. Use a static annotation, or declare a time_span"
        )
    _reject_outside_union(label, span, windows, sample_id)


def _reject_outside(label: str, span: TimeSpan, start: int, end: int, sample_id: str) -> None:
    """Reject a time span that runs past the half-open window ``[start, end)``.

    Raises:
        TimeFValidationError: If the span starts before ``start`` or ends after ``end``.
    """
    if span.start_us < start or span.exclusive_end > end:
        raise TimeFValidationError(
            f"{label} falls outside sample {sample_id!r} span ({start}, {end}) us: got {span!r}; span "
            f"times are in the source recording timeline"
        )


def _reject_outside_union(label: str, span: TimeSpan, windows: list[tuple[int, int]], sample_id: str) -> None:
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
    if not any(start <= span.start_us and span.exclusive_end <= end for start, end in merged):
        raise TimeFValidationError(
            f"{label} falls in a gap between the recorded windows of sample {sample_id!r} {merged}: got "
            f"{span!r}. Declare a time_span if the session spans the gap"
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
    time_span: TimeInterval | None = None
    """The session's overall span on the source recording timeline: an :class:`~timenet.types.TimeInterval`
    covering the whole sample, or ``None``. Declare it when the series have gaps and an event may fall in
    one (a note taken while every sensor was briefly off); an unscoped span is then checked against it
    rather than against the union of the series' windows. Its ``time_series_ids`` must be ``None``, and
    it must contain every series' window."""

    def __post_init__(self) -> None:
        """Normalize ``start_time`` to whole Unix microseconds and validate ``time_span``.

        ``start_time`` delegates its contract: ``unix_us`` rejects a naive datetime or a bare float, and
        ``check_int64`` rejects an anchor past the int64 microsecond column. ``time_span``, when set,
        must be a whole-sample :class:`~timenet.types.TimeInterval` that contains every series' window.

        Raises:
            TimeFValidationError: If ``time_span`` is not a whole-sample ``TimeInterval`` or does not
                contain some series' window.
        """
        if self.start_time is not None:
            anchor = unix_us(self.start_time)
            check_int64("Sample.start_time", anchor)
            self.start_time = anchor
        if self.time_span is not None:
            if not isinstance(self.time_span, TimeInterval):
                raise TimeFValidationError(
                    f"Sample.time_span must be a TimeInterval covering the whole sample, got {self.time_span!r}"
                )
            if self.time_span.time_series_ids is not None:
                raise TimeFValidationError(
                    "Sample.time_span covers the whole sample, so its time_series_ids must be None"
                )
            for ts in self.time_series:
                window = ts.span_us
                if window is not None and (window[0] < self.time_span.start_us or window[1] > self.time_span.end_us):
                    raise TimeFValidationError(
                        f"Sample.time_span ({self.time_span.start_us}, {self.time_span.end_us}) us must contain "
                        f"every series' window, but {ts.time_series_id!r} covers {window} us"
                    )

    @property
    def has_absolute_time(self) -> bool:
        """Whether this sample's relative timeline has a Unix-time anchor."""
        return self.start_time is not None

    def time_point(self, at: datetime, *, time_series_ids: tuple[str, ...] | None = None) -> TimePoint:
        """Build a :class:`~timenet.types.TimePoint` at a wall-clock moment on this sample's timeline.

        Places ``at`` on the recording timeline against this sample's own ``start_time``, so the caller
        never repeats the anchor. A span's bounds are offsets on that timeline, so this needs an
        anchored sample; ``offset_us`` raises if the sample has no ``start_time``.

        Args:
            at: The wall-clock moment, timezone-aware.
            time_series_ids: Series the point is scoped to; ``None`` covers every series.

        Returns:
            The point, in microseconds from this sample's relative zero.
        """
        return TimePoint(start_us=offset_us(at, self.start_time), time_series_ids=time_series_ids)

    def time_interval(
        self, start: datetime, end: datetime, *, time_series_ids: tuple[str, ...] | None = None
    ) -> TimeInterval:
        """Build a :class:`~timenet.types.TimeInterval` between two wall-clock moments on this timeline.

        Places ``start`` and ``end`` on the recording timeline against this sample's own ``start_time``;
        ``offset_us`` raises if the sample has no ``start_time`` to measure against.

        Args:
            start: Wall-clock start, timezone-aware.
            end: Wall-clock end, exclusive and timezone-aware.
            time_series_ids: Series the interval is scoped to; ``None`` covers every series.

        Returns:
            The half-open interval, in microseconds from this sample's relative zero.
        """
        return TimeInterval(
            start_us=offset_us(start, self.start_time),
            end_us=offset_us(end, self.start_time),
            time_series_ids=time_series_ids,
        )

    def add_annotation(self, annotation: Annotation) -> Annotation:
        """Attach an annotation to the sample and return it.

        Args:
            annotation: The annotation to attach.

        Returns:
            The attached annotation (the same instance).

        Raises:
            TimeFValidationError: If the annotation's span references a series not on this sample, a
                scoped span names a timeless series, or the span falls outside the window its scope
                selects (the intersection of named series, the sample's ``time_span``, or the union of
                the series' windows).
        """  # noqa: DOC502 (raised by check_span_within_window, not directly here)
        if annotation.span is not None:
            check_span_within_window(
                f"annotation {annotation.key!r}",
                annotation.span,
                self.time_series,
                self.sample_id,
                self.time_span,
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
