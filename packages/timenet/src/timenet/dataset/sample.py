"""The :class:`Sample` type: one logical unit of time-series data."""

from dataclasses import dataclass, field
import uuid

from timenet.dataset.time_series import TimeSeries
from timenet.types import Annotation, IntervalAnnotation, PointAnnotation, View


@dataclass(kw_only=True)
class Sample:
    """One logical unit of time-series data: a recording, a session, a sensor bundle, a market window.

    Created via :meth:`~timenet.dataset.TimeFDataset.add_sample`. Mutable so ``task_ids`` and
    ``annotations`` can be populated after construction.
    """

    time_series: tuple[TimeSeries, ...]
    view: View
    sample_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject_ids: tuple[str, ...] = ()
    task_ids: tuple[str, ...] = ()
    annotations: tuple[Annotation, ...] = ()

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
