"""The :class:`Sample` type: one logical unit of time-series data."""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pyarrow as pa

from timenet.dataset.time_series import TimeSeries
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, Span, SpanFrame, new_id
from timenet.types.clock import unix_us


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def check_span_within_window(label: str, span: Span, time_series: tuple[TimeSeries, ...], sample_id: str) -> None:
    """Reject a span whose series are unknown, timeless, or that falls outside the sample's span.

    A seconds span's bounds are in the source recording timeline, the same frame as each series' axis,
    so it is checked against the sample's overall span: the bounding interval from the earliest start
    to the latest end of the targeted series. This is deliberately the bounding interval, not a
    per-series or gap-free rule, so an annotation can mark a time offset that falls in a gap between two
    series' windows (a note logged between one sensor coming off and another going on, say). Shared
    by a task's ``scope`` and an annotation so the two never disagree about what a span may cover.

    A steps span instead counts in each named series' own steps, so it is bounded by their length
    rather than any timeline. This is the only span that resolves on an ordinal series, which reports
    no window at all.

    Args:
        label: Human-readable label for the span, used in the error message.
        span: The span to check.
        time_series: The series the span is checked against.
        sample_id: The owning sample's id, for the error message.

    Raises:
        TimeFValidationError: If a series id is unknown; if a steps span runs past a named series'
            steps; or, for a seconds span, if a targeted series has no timeline or the span falls
            outside the covered window.
    """
    scope = span.time_series_ids
    # A span with no time_series_ids covers every series on the sample.
    in_scope = [ts for ts in time_series if scope is None or ts.time_series_id in scope]
    covered = {ts.time_series_id: ts.span_us for ts in in_scope}
    for series_id in scope or ():
        if series_id not in covered:
            raise TimeFValidationError(
                f"{label} references unknown time_series_id {series_id!r} on sample {sample_id!r}"
            )
    if not covered:
        return
    if span.frame is SpanFrame.STEPS:
        # A steps span always names its series (the span validates that), so every one is in_scope. Its
        # exclusive end must fit each series' step count; n_values is that count, and is what an ordinal
        # series has in place of a window.
        last = span.start + 1 if span.is_point else span.end
        for ts in in_scope:
            if last is not None and last > ts.n_values:
                raise TimeFValidationError(
                    f"{label} ({span.start}, {span.end}) runs past the {ts.n_values} steps of series "
                    f"{ts.time_series_id!r} on sample {sample_id!r}"
                )
        return
    windows = [w for w in covered.values() if w is not None]
    if len(windows) < len(covered):
        timeless = sorted(sid for sid, w in covered.items() if w is None)
        raise TimeFValidationError(
            f"{label} covers {timeless}, which have no timeline at all, so a time-valued region "
            f"means nothing on them. Scope it to the series that do"
        )
    start = min(w[0] for w in windows)  # bounding span of the targeted series; gaps between them stay valid
    end = max(w[1] for w in windows)
    # The window is half-open [start, end). A point's own exclusive end is start + 1; an interval's
    # is its end. Either is outside when it runs past the window, which catches a point sitting
    # exactly on the excluded upper bound.
    span_end = span.start + 1 if span.is_point else span.end
    if span.start < start or (span_end is not None and span_end > end):
        raise TimeFValidationError(
            f"{label} ({span.start}, {span.end}) us falls outside sample {sample_id!r} span "
            f"({start}, {end}) us; span times are in the source recording timeline"
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
        """Validate the intrinsic per-sample invariants.

        Raises:
            TimeFValidationError: If ``start_time`` is neither a timezone-aware datetime nor whole
                Unix microseconds, or does not fit int64.
        """
        if self.start_time is None:
            return
        anchor = unix_us(self.start_time)
        if not (_INT64_MIN <= anchor <= _INT64_MAX):
            raise TimeFValidationError(f"Sample.start_time must fit int64 microseconds, got {anchor}")
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
                targeted series has no timeline, the span falls outside the covered window, or a
                trial-level interval annotation is added when the sample's series do not share a
                common window.
        """
        if annotation.span is not None:
            check_span_within_window(
                f"annotation {annotation.key!r}", annotation.span, self.time_series, self.sample_id
            )
            if not annotation.span.is_point and annotation.span.time_series_ids is None:
                # A trial-level interval applies to every series at once, so they must agree on the
                # window it is measured against. The shared check has already refused any timeless
                # series, so every window here is a concrete pair.
                windows = {ts.span_us for ts in self.time_series}
                if len(windows) != 1:
                    raise TimeFValidationError(
                        f"trial-level interval annotation {annotation.key!r} requires a common "
                        "window across the sample's time_series"
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
