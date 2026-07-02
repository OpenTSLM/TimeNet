"""Annotations: contextual metadata attached to a sample.

An annotation is either **static** (sample-scoped, time-independent context such as demographics or a
ticker symbol) or **temporal** (anchored to a point or bounded interval in the original recording
timeline). All three shapes are flat frozen dataclasses carrying ``key`` / ``unit`` / ``description`` as
instance fields so connectors can author them directly (or subclass with field defaults for reuse) and
:class:`~timenet.reader.TimeFReader` can reconstruct the identical instances from the manifest without
runtime class synthesis. :class:`AnnotationDescriptor` is the type-level projection stored in the schema
and manifest.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
import uuid


class AnnotationType(StrEnum):
    """The three annotation shapes; the discriminator stored on disk and in the manifest."""

    STATIC = "static"
    POINT = "point"
    INTERVAL = "interval"


@dataclass(frozen=True, kw_only=True)
class Annotation:
    """Base for all annotations. Not instantiated directly; use one of the three shapes below."""

    key: str
    value: Any = None
    unit: str | None = None
    description: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        """Canonicalize a sequence ``value`` to a list.

        ``value_type`` is ``"list"`` for any sequence and the manifest stores it as a JSON array,
        which decodes back to a list. Normalizing a tuple here keeps an in-memory annotation equal to
        its read-back form (the reader's field-for-field guarantee). Subclasses that override
        ``__post_init__`` must call ``super().__post_init__()``.
        """
        if isinstance(self.value, tuple):
            object.__setattr__(self, "value", list(self.value))


@dataclass(frozen=True, kw_only=True)
class StaticAnnotation(Annotation):
    """Sample-scoped, time-independent context. Carries a required ``value`` and no time fields."""

    value: Any = None

    def __post_init__(self) -> None:
        """Reject a missing value.

        Raises:
            ValueError: If ``value`` is ``None``. Redeclaring the field without a default does not
                remove the base's inherited ``None`` default at runtime, so the check is explicit.
        """
        super().__post_init__()
        if self.value is None:
            raise ValueError("StaticAnnotation requires a value")


@dataclass(frozen=True, kw_only=True)
class PointAnnotation(Annotation):
    """Anchored to a single instant in the original recording timeline."""

    start_time_s: float
    time_series_ids: tuple[str, ...] | None = None


@dataclass(frozen=True, kw_only=True)
class IntervalAnnotation(Annotation):
    """Anchored to a bounded interval in the original recording timeline."""

    start_time_s: float
    end_time_s: float
    time_series_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Reject a non-positive interval.

        Raises:
            ValueError: If ``end_time_s`` is not strictly greater than ``start_time_s``.
        """
        super().__post_init__()
        if self.end_time_s <= self.start_time_s:
            raise ValueError(
                f"IntervalAnnotation end_time_s ({self.end_time_s}) must be > start_time_s ({self.start_time_s})"
            )


ANNOTATION_BASES: dict[AnnotationType, type[Annotation]] = {
    AnnotationType.STATIC: StaticAnnotation,
    AnnotationType.POINT: PointAnnotation,
    AnnotationType.INTERVAL: IntervalAnnotation,
}


def annotation_type_of(annotation: Annotation) -> AnnotationType:
    """Return the :class:`AnnotationType` shape of an annotation instance.

    Args:
        annotation: The annotation to classify.

    Returns:
        The matching :class:`AnnotationType`.

    Raises:
        ValueError: If ``annotation`` is not one of the three concrete shapes.
    """
    if isinstance(annotation, IntervalAnnotation):
        return AnnotationType.INTERVAL
    if isinstance(annotation, PointAnnotation):
        return AnnotationType.POINT
    if isinstance(annotation, StaticAnnotation):
        return AnnotationType.STATIC
    raise ValueError(f"not a concrete annotation shape: {type(annotation).__name__}")


@dataclass(frozen=True)
class AnnotationDescriptor:
    """Type-level projection of an annotation key, stored in the schema and manifest."""

    key: str
    annotation_type: AnnotationType
    value_type: str | None = None
    unit: str | None = None
    description: str | None = None
