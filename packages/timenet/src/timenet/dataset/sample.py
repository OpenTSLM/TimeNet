"""The :class:`Sample` type: one logical unit of time-series data."""

from dataclasses import dataclass, field

from timenet.dataset.time_series import TimeSeries
from timenet.types import Annotation, IntervalAnnotation, PointAnnotation, View, new_id


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


@dataclass(kw_only=True)
class Sample:
    """One logical unit of time-series data: a recording, a session, a sensor bundle, a market window.

    Created via :meth:`~timenet.dataset.TimeFDataset.add_sample`. Mutable so ``task_ids`` and
    ``annotations`` can be populated after construction.
    """

    time_series: tuple[TimeSeries, ...]
    """One :class:`TimeSeries` per channel the sample uses."""
    view: View
    """Which slice of the source this sample represents."""
    sample_id: str = field(default_factory=new_id)
    """Unique id for the sample (default: an auto-generated uuid7)."""
    subject_ids: tuple[str, ...] = ()
    """Subjects this sample belongs to (empty for subject-less domains)."""
    task_ids: tuple[str, ...] = ()
    """Ids of the tasks attached to this sample."""
    annotations: tuple[Annotation, ...] = ()
    """Annotations attached to the sample."""
    t0_unix_ns: int | None = None
    """Wall-clock anchor: the Unix time (UTC, integer nanoseconds) that relative time zero refers to,
    for every series and annotation on this sample. ``None`` means no wall-clock reference exists;
    never fabricate one."""

    def __post_init__(self) -> None:
        """Validate the intrinsic per-sample invariants.

        Raises:
            ValueError: If ``t0_unix_ns`` does not fit a signed 64-bit integer.
        """
        if self.t0_unix_ns is not None and not (_INT64_MIN <= self.t0_unix_ns <= _INT64_MAX):
            raise ValueError(f"Sample.t0_unix_ns must fit int64, got {self.t0_unix_ns}")

    def add_annotation(self, annotation: Annotation) -> Annotation:
        """Attach an annotation to the sample and return it.

        Args:
            annotation: A ``StaticAnnotation``, ``PointAnnotation``, or ``IntervalAnnotation``.

        Returns:
            The attached annotation (the same instance).

        Raises:
            ValueError: If a temporal annotation's ``time_series_ids`` is empty or references a series
                not on this sample, or a trial-level ``IntervalAnnotation`` is added when the sample's
                series do not share a common ``(t_start_s, t_end_s)`` span.
        """
        if isinstance(annotation, PointAnnotation | IntervalAnnotation):
            series_ids = {ts.time_series_id for ts in self.time_series}
            if annotation.time_series_ids is not None:
                if not annotation.time_series_ids:
                    raise ValueError(
                        f"annotation {annotation.key!r} has an empty time_series_ids; use None for a "
                        "trial-level annotation"
                    )
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
