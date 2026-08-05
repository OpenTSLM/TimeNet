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

import pint

from timenet.errors import TimeFValidationError
from timenet.types.ids import new_id
from timenet.types.spans import IntervalSpan, PointSpan
from timenet.types.units import normalize_unit


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
    unit: str | pint.Unit | None = None
    """Optional physical unit of ``value`` — a unit string (e.g. ``"years"``) or a :class:`pint.Unit`.
    Validated against the shared registry on construction; an unrecognized string raises ``ValueError``,
    and a ``pint.Unit`` is stored as its canonical name."""
    description: str | None = None
    """Optional human-readable description of the annotation."""
    id: str = field(default_factory=new_id)
    """Unique identifier, a UUIDv7 string by default."""

    def __post_init__(self) -> None:
        """Canonicalize a sequence ``value`` to a list and normalize ``unit`` against the registry.

        ``value_type`` is ``"list"`` for any sequence and the manifest stores it as a JSON array,
        which decodes back to a list. Normalizing a tuple here (and the unit to a string) keeps an
        in-memory annotation equal to its read-back form (the reader's field-for-field guarantee).
        Subclasses that override ``__post_init__`` must call ``super().__post_init__()``.
        """
        if isinstance(self.value, tuple):
            object.__setattr__(self, "value", list(self.value))
        object.__setattr__(self, "unit", normalize_unit(self.unit))


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
    """Anchored to a single instant in the original recording timeline.

    The instant, and which series it applies to, are a :class:`~timenet.types.PointSpan`. That is the
    same type a task's ``scope`` uses, so a region means one thing across the format::

        PointAnnotation(key="stimulus", span=PointSpan.seconds(0.5))
    """

    span: PointSpan
    """Where on the recording timeline this annotation sits, and which series it targets."""

    def __post_init__(self) -> None:
        """Reject a span that is not a point.

        The annotation is typed to a point, so a caller cannot get this wrong. A span rebuilt from a
        file can be, since its shape comes from the bounds on disk.

        Raises:
            TimeFValidationError: If ``span`` carries an end, making it an interval.
        """
        super().__post_init__()
        if not self.span.is_point:
            raise TimeFValidationError(
                f"PointAnnotation {self.key!r} needs a point span, got one ending at {self.span.end}"
            )


@dataclass(frozen=True, kw_only=True)
class IntervalAnnotation(Annotation):
    """Anchored to a bounded interval in the original recording timeline.

    The interval, and which series it applies to, are an :class:`~timenet.types.IntervalSpan`::

        IntervalAnnotation(key="artifact", span=IntervalSpan.seconds(0.0, 0.25))
    """

    span: IntervalSpan
    """Where on the recording timeline this annotation sits, and which series it targets."""

    def __post_init__(self) -> None:
        """Reject a span that is not an interval.

        The annotation is typed to an interval, so a caller cannot get this wrong. A span rebuilt
        from a file can be, since its shape comes from the bounds on disk.

        Raises:
            TimeFValidationError: If ``span`` is a point, which names an instant rather than a region.
        """
        super().__post_init__()
        if self.span.is_point:
            raise TimeFValidationError(
                f"IntervalAnnotation {self.key!r} needs an interval span, got a point at {self.span.start}"
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
