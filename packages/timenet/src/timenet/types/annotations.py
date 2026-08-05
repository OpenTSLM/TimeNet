"""Annotations: contextual metadata attached to a sample.

One :class:`Annotation` class covers every case, and its optional ``span`` says how it sits in time:
absent for sample-scoped, time-independent context such as demographics or a ticker symbol, a
:class:`~timenet.types.spans.PointSpan` for one time offset, an
:class:`~timenet.types.spans.IntervalSpan` for a bounded region of the original recording timeline. It
is a flat frozen dataclass carrying ``key`` / ``unit`` / ``description`` as
instance fields so connectors can author it directly (or subclass with field defaults for reuse) and
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
from timenet.types.spans import Span
from timenet.types.units import normalize_unit


@unique
class AnnotationType(StrEnum):
    """The shape an annotation key takes across the dataset; a schema-level projection in the manifest."""

    STATIC = "static"
    POINT = "point"
    INTERVAL = "interval"


@dataclass(frozen=True, kw_only=True)
class Annotation:
    """Contextual metadata attached to a sample, optionally anchored to a region of its timeline.

    An annotation carries a ``value``, a ``span``, or both. With a span it says where on the
    recording timeline it applies and which series it targets; without one it is sample-scoped
    context that has no place in time, like a subject's age. The span's own shape says whether it
    marks a time offset or covers a stretch, so there is one class here rather than one per shape::

        Annotation(key="age", value=64)
        Annotation(key="stimulus", span=PointSpan.seconds(0.5))
        Annotation(key="artifact", value="motion", span=IntervalSpan.seconds(0.0, 0.25))
    """

    key: str
    """Name identifying the annotation."""
    value: Any = None
    """The annotation's payload value. ``None`` makes it a pure marker, which needs a ``span`` to
    mark something."""
    span: Span | None = None
    """Where on the recording timeline this annotation applies, and which series it targets.
    ``None`` means it is sample-scoped context with no place in time."""
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

        Raises:
            TimeFValidationError: If ``span`` is set to something that is not a span, or the
                annotation has neither a value nor a span.
        """
        if isinstance(self.value, tuple):
            object.__setattr__(self, "value", list(self.value))
        object.__setattr__(self, "unit", normalize_unit(self.unit))
        if self.span is not None and not isinstance(self.span, Span):
            raise TimeFValidationError(
                f"annotation {self.key!r} span must be a PointSpan or an IntervalSpan, got {type(self.span).__name__}"
            )
        if self.value is None and self.span is None:
            raise TimeFValidationError(
                f"annotation {self.key!r} has neither a value nor a span, so it says nothing. Give it "
                f"a value, or a span to mark a region of the timeline"
            )


def annotation_type_of(annotation: Annotation) -> AnnotationType:
    """Return the :class:`AnnotationType` shape of an annotation instance.

    Args:
        annotation: The annotation to classify.

    Returns:
        The matching :class:`AnnotationType`, derived from the span rather than from a class.
    """
    if annotation.span is None:
        return AnnotationType.STATIC
    return AnnotationType.POINT if annotation.span.is_point else AnnotationType.INTERVAL


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
