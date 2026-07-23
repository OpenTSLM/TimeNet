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
from enum import StrEnum, unique
from typing import Any

from timenet.errors import TimeFValidationError
from timenet.types.ids import new_id


@unique
class AnnotationType(StrEnum):
    """The three annotation shapes; the discriminator stored on disk and in the manifest."""

    STATIC = "static"
    POINT = "point"
    INTERVAL = "interval"


@dataclass(frozen=True, kw_only=True)
class Annotation:
    """Base for all annotations. Not instantiated directly; use one of the three shapes below."""

    key: str
    """Name identifying the annotation."""
    value: Any = None
    """The annotation's payload value."""
    unit: str | None = None
    """Optional physical unit of ``value``."""
    description: str | None = None
    """Optional human-readable description of the annotation."""
    id: str = field(default_factory=new_id)
    """Unique identifier, a UUIDv7 string by default."""

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
    """Required payload value; must not be ``None``."""

    def __post_init__(self) -> None:
        """Reject a missing value.

        Raises:
            TimeFValidationError: If ``value`` is ``None``. Redeclaring the field without a default does not
                remove the base's inherited ``None`` default at runtime, so the check is explicit.
        """
        super().__post_init__()
        if self.value is None:
            raise TimeFValidationError("StaticAnnotation requires a value")


@dataclass(frozen=True, kw_only=True)
class PointAnnotation(Annotation):
    """Anchored to a single instant in the original recording timeline."""

    start_time_s: float
    """Instant on the original recording timeline, in seconds."""
    time_series_ids: tuple[str, ...] | None = None
    """Series this annotation targets; ``None`` covers the whole sample."""

    def __post_init__(self) -> None:
        """Reject an explicitly empty ``time_series_ids``.

        Raises:
            TimeFValidationError: If ``time_series_ids`` is ``()`` rather than ``None`` or non-empty.
        """
        super().__post_init__()
        if self.time_series_ids is not None and not self.time_series_ids:
            raise TimeFValidationError(
                "PointAnnotation time_series_ids must be None (whole sample) or non-empty, got ()"
            )


@dataclass(frozen=True, kw_only=True)
class IntervalAnnotation(Annotation):
    """Anchored to a bounded interval in the original recording timeline."""

    start_time_s: float
    """Interval start on the recording timeline, in seconds."""
    end_time_s: float
    """Interval end in seconds; must exceed ``start_time_s``."""
    time_series_ids: tuple[str, ...] | None = None
    """Series this annotation targets; ``None`` covers the whole sample."""

    def __post_init__(self) -> None:
        """Reject a non-positive interval or an explicitly empty ``time_series_ids``.

        Raises:
            TimeFValidationError: If ``end_time_s`` is not strictly greater than ``start_time_s``, or
                if ``time_series_ids`` is ``()`` rather than ``None`` or non-empty.
        """
        super().__post_init__()
        if self.end_time_s <= self.start_time_s:
            raise TimeFValidationError(
                f"IntervalAnnotation end_time_s ({self.end_time_s}) must be > start_time_s ({self.start_time_s})"
            )
        if self.time_series_ids is not None and not self.time_series_ids:
            raise TimeFValidationError(
                "IntervalAnnotation time_series_ids must be None (whole sample) or non-empty, got ()"
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
    """Name of the annotation this descriptor projects."""
    annotation_type: AnnotationType
    """Which of the three annotation shapes this key uses."""
    value_type: str | None = None
    """Manifest value-type tag (bool, int, float, str, or list)."""
    unit: str | None = None
    """Optional physical unit of the value."""
    description: str | None = None
    """Optional human-readable description of the annotation."""


def value_type_of(value: Any) -> str | None:
    """Return the manifest ``value_type`` tag for an annotation value.

    Args:
        value: The annotation's value.

    Returns:
        One of ``"bool" | "int" | "float" | "str" | "list"``, or ``None`` for a pure marker.

    Raises:
        TypeError: If ``value`` is a non-null value of an unsupported type.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # bool before int: bool is an int subclass
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list | tuple):
        return "list"
    raise TypeError(f"unsupported annotation value type: {type(value).__name__}")
