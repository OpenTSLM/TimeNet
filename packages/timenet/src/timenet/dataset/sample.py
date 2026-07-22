"""The :class:`Sample` type: one logical unit of time-series data."""

from dataclasses import dataclass, field

from timenet.dataset.time_series import TimeSeries
from timenet.types import Annotation, IntervalAnnotation, PointAnnotation, View, new_id


@dataclass(kw_only=True)
class Sample:
    """One logical unit of time-series data: a recording, a session, a sensor bundle, a market window.

    Created via :meth:`~timenet.dataset.TimeFDataset.add_sample`. Mutable so ``task_ids`` and
    ``annotations`` can be populated after construction.
    """

    time_series: tuple[TimeSeries, ...]
    """The logical :class:`TimeSeries` streams the sample uses."""
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
