"""The :class:`Span` geometry primitive: one point or interval on a recording timeline.

A span is the shape shared by every time-localized field a task carries: the ``scope`` a task is asked
about and the target regions a :class:`~timenet.types.TemporalLocalizationTask` predicts are the same
type run in opposite directions. Keeping it one frozen dataclass means the bounds and channel checks
live in one place instead of being restated per task.

The three annotation shapes map directly onto task geometry: a static annotation corresponds to no
span (``scope=None``), a point annotation to :meth:`Span.point`, and an interval annotation to
:meth:`Span.interval`.
"""

from dataclasses import dataclass
from typing import Self

from timenet.errors import TimeFValidationError


@dataclass(frozen=True, kw_only=True)
class Span:
    """A point (``end_s is None``) or half-open interval ``[start_s, end_s)``, optionally per-channel.

    Times are in the **source recording timeline**, the same frame as
    :attr:`~timenet.dataset.TimeSeries.t_start_s`, so a span stays meaningful on a windowed sample that
    starts partway into the recording. That a span actually falls inside the sample it is attached to is
    checked by :meth:`~timenet.dataset.TimeFDataset.add_task`, which has the sample to check against.
    """

    start_s: float
    """Start of the interval, or the time offset itself, in recording seconds."""
    end_s: float | None = None
    """End of the interval in recording seconds, exclusive; ``None`` makes the span a point."""
    time_series_ids: tuple[str, ...] | None = None
    """Series the span is scoped to; ``None`` covers every series in the sample."""

    def __post_init__(self) -> None:
        """Reject a non-positive interval or an explicitly empty ``time_series_ids``.

        Raises:
            TimeFValidationError: If ``end_s`` is not strictly greater than ``start_s``, or if
                ``time_series_ids`` is ``()`` rather than ``None`` or non-empty.
        """
        if self.end_s is not None and self.end_s <= self.start_s:
            raise TimeFValidationError(f"Span end_s ({self.end_s}) must be > start_s ({self.start_s})")
        if self.time_series_ids is not None and not self.time_series_ids:
            raise TimeFValidationError("Span time_series_ids must be None (whole sample) or non-empty, got ()")

    @classmethod
    def point(
        cls,
        start_s: float,
        *,
        time_series_ids: tuple[str, ...] | None = None,
    ) -> Self:
        """Construct a span marking one time offset.

        Args:
            start_s: The time offset in recording seconds.
            time_series_ids: Series the point is scoped to; ``None`` covers every series.

        Returns:
            A point span whose ``end_s`` is ``None``.
        """
        return cls(start_s=start_s, time_series_ids=time_series_ids)

    @classmethod
    def interval(
        cls,
        start_s: float,
        end_s: float,
        *,
        time_series_ids: tuple[str, ...] | None = None,
    ) -> Self:
        """Construct a half-open interval span.

        Args:
            start_s: Start of the interval in recording seconds.
            end_s: End of the interval in recording seconds, exclusive.
            time_series_ids: Series the interval is scoped to; ``None`` covers every series.

        Returns:
            The interval span.
        """
        return cls(start_s=start_s, end_s=end_s, time_series_ids=time_series_ids)

    @property
    def is_point(self) -> bool:
        """Whether the span marks a time offset rather than a bounded interval."""
        return self.end_s is None
