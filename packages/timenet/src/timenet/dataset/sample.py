"""The :class:`Sample` type: one logical unit of time-series data."""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pyarrow as pa

from timenet.dataset.time_series import TimeSeries
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, IntervalAnnotation, PointAnnotation, View, new_id
from timenet.types.clock import unix_us


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


@dataclass(kw_only=True)
class Sample:
    """One logical unit of time-series data: a recording, a session, a sensor bundle, a market window.

    Created via :meth:`~timenet.dataset.TimeFDataset.add_sample`. Mutable so ``task_ids`` and
    ``annotations`` can be populated after construction.
    """

    time_series: tuple[TimeSeries, ...]
    """The logical :class:`TimeSeries` streams the sample uses."""
    view: View = View.FULL
    """Which slice of the source this sample represents (defaults to the full recording)."""
    sample_id: str = field(default_factory=new_id)
    """Unique id for the sample (default: an auto-generated uuid7)."""
    subject_ids: tuple[str, ...] = ()
    """Subjects this sample belongs to (empty for subject-less domains)."""
    task_ids: tuple[str, ...] = ()
    """Ids of the tasks attached to this sample."""
    annotations: tuple[Annotation, ...] = ()
    """Annotations attached to the sample."""
    start_time: datetime | int | None = None
    """Wall-clock instant that this sample's relative time zero refers to, for every series and
    annotation on it. Pass a timezone-aware :class:`~datetime.datetime` or whole Unix microseconds;
    either normalizes to microseconds on construction, so the field always holds an ``int``
    afterwards. ``None`` means no wall-clock reference exists; never fabricate one.

    A bare float is refused, because seconds and microseconds are both plausible readings of it. When
    the source really does hand over seconds, convert at the call site so the unit is visible::

        start_time=datetime(2026, 8, 5, tzinfo=timezone.utc)   # 1_785_888_000_000_000
        start_time=seconds_to_us(1)                            # 1_000_000, one second past the epoch
        start_time=1_000_000                                   # the same instant, written directly
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
        object.__setattr__(self, "start_time", anchor)

    @property
    def has_absolute_time(self) -> bool:
        """Whether this sample's relative timeline has a Unix-time anchor."""
        return self.start_time is not None

    def add_annotation(self, annotation: Annotation) -> Annotation:
        """Attach an annotation to the sample and return it.

        Args:
            annotation: A ``StaticAnnotation``, ``PointAnnotation``, or ``IntervalAnnotation``.

        Returns:
            The attached annotation (the same instance).

        Raises:
            ValueError: If a temporal annotation's ``time_series_ids`` references a series not on this
                sample, or a trial-level ``IntervalAnnotation`` is added when the sample's series do
                not share a common ``(t_start_s, t_end_s)`` span.
        """
        if isinstance(annotation, PointAnnotation | IntervalAnnotation):
            series_ids = {ts.time_series_id for ts in self.time_series}
            if annotation.time_series_ids is not None:
                # An empty tuple is rejected by the annotation's own __post_init__, so by here
                # time_series_ids is guaranteed non-empty and only the ids need checking.
                for series_id in annotation.time_series_ids:
                    if series_id not in series_ids:
                        raise ValueError(
                            f"annotation {annotation.key!r} references unknown time_series_id {series_id!r}"
                        )
            elif isinstance(annotation, IntervalAnnotation):
                spans = {(ts.t_start_s, ts.t_end_s) for ts in self.time_series}
                if len(spans) != 1:
                    raise ValueError(
                        f"trial-level IntervalAnnotation {annotation.key!r} requires a common "
                        "(t_start_s, t_end_s) span across the sample's time_series"
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
