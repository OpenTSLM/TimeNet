"""Contextual metadata attached to a record.

One :class:`Annotation` class covers every case. Its optional ``span`` says how the annotation sits
in time. An absent span marks record-scoped context with no place in time, such as demographics or a
ticker symbol. A :class:`~timenet.types.spans.TimePoint` marks one time offset. A
:class:`~timenet.types.spans.TimeInterval` marks a bounded region of the original recording timeline.
:class:`Annotation` is a flat frozen dataclass. It carries ``key``, ``unit``, and ``description`` as
instance fields. A connector can author it directly, or subclass it with field defaults for reuse.
:class:`~timenet.reader.TimeFReader` can rebuild identical instances from the manifest without runtime
class synthesis. :class:`AnnotationDescriptor` is the type-level projection stored in the schema and
manifest.
"""

from dataclasses import dataclass, field, replace
from enum import StrEnum, unique
from typing import Annotated, Any, Protocol

import pint
from pydantic import BeforeValidator, StrictStr

from timenet.errors import TimeFValidationError
from timenet.json import JsonMapping, validate_json_mapping
from timenet.types._model import TimeFModel
from timenet.types.ids import new_id
from timenet.types.payloads import Payload
from timenet.types.references import ObjectRef
from timenet.types.spans import TimeInterval, TimePoint, TimeSpan
from timenet.types.units import normalize_unit
from timenet.types.wire import ValueType


@unique
class AnnotationType(StrEnum):
    """Name the shape an annotation key takes across the dataset. It is a schema-level projection in the manifest."""

    STATIC = "static"
    POINT = "point"
    INTERVAL = "interval"


@dataclass(frozen=True, kw_only=True, init=False)
class Annotation:
    """Contextual metadata attached to a record, optionally anchored to a region of its timeline.

    An annotation carries a ``value``, a ``span``, or both. With a span, it says where on the
    Record timeline it applies. Its owner determines which Signals it covers. Without a span, it is record-scoped
    context that has no place in time, like a subject's age. The span's own shape says whether the
    annotation marks a time offset or covers a stretch. One class covers every shape::

        Annotation(key="age", value=64)
        Annotation(key="stimulus", span=TimePoint.seconds(0.5))
        Annotation(key="artifact", value="motion", span=TimeInterval.seconds(0.0, 0.25))
    """

    key: str
    """Name identifying the annotation."""
    value: Payload[Any] | None = None
    """An explicit Payload handle after construction. ``None`` makes it a pure marker, which needs a ``span`` to
    mark something."""
    span: TimePoint | TimeInterval | None = None
    """Where on the Record timeline this annotation applies. Its owner determines its scope.
    ``None`` means it is record-scoped context with no place in time."""
    unit: str | pint.Unit | None = None
    """Optional physical unit of ``value``. Give a unit string (for example ``"years"``) or a
    :class:`pint.Unit`. Construction validates the unit against the shared registry. An unrecognized
    string raises ``ValueError``. Construction stores a ``pint.Unit`` as its canonical name."""
    description: str | None = None
    """Optional human-readable description of the annotation."""
    source: str | None = None
    """Optional per-instance provenance (the rater, method, or model behind this value). Unlike
    ``description``, which the descriptor holds once per key, ``source`` can differ between
    annotations that share a key."""
    id: str = field(default_factory=new_id)
    """Reusable content identifier, a UUIDv7 string by default."""
    content_origin: ObjectRef | None = field(default=None, compare=False)
    occurrence_origin: ObjectRef | None = field(default=None, compare=False)
    occurrence_id: str | None = field(default=None, compare=False)
    """Identity of one attachment. It is assigned by ``annotate()``."""
    metadata: JsonMapping = field(default_factory=dict)
    """Optional metadata that is part of the reusable content."""
    confidence: float | None = field(default=None, compare=False)
    """Optional confidence for this particular application."""
    occurrence_metadata: JsonMapping = field(default_factory=dict, compare=False)
    """Optional metadata for this particular application."""

    def __init__(  # noqa: PLR0913 - annotation content and occurrence fields
        self,
        *,
        key: str,
        value: Any = None,
        span: TimePoint | TimeInterval | None = None,
        unit: str | pint.Unit | None = None,
        description: str | None = None,
        source: str | None = None,
        id: str | None = None,
        occurrence_id: str | None = None,
        content_origin: ObjectRef | None = None,
        occurrence_origin: ObjectRef | None = None,
        metadata: JsonMapping | None = None,
        confidence: float | None = None,
        occurrence_metadata: JsonMapping | None = None,
    ) -> None:
        values = {
            "key": key,
            "value": value,
            "span": span,
            "unit": unit,
            "description": description,
            "source": source,
            "id": id or new_id(),
            "occurrence_id": occurrence_id,
            "content_origin": content_origin,
            "occurrence_origin": occurrence_origin,
            "metadata": {} if metadata is None else metadata,
            "confidence": confidence,
            "occurrence_metadata": {} if occurrence_metadata is None else occurrence_metadata,
        }
        for name, content in values.items():
            object.__setattr__(self, name, content)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Wrap native values in immutable payloads and normalize the unit.

        ``value_type`` is ``"list"`` for any sequence, and the manifest stores it as a JSON array that
        decodes back to a list. Payload reads return a list, and construction normalizes the unit to a string. That
        keeps an in-memory annotation equal to its read-back form, the reader's field-for-field
        guarantee.

        Raises:
            TimeFValidationError: If the caller sets ``span`` to something that is not a span, or
                the annotation has neither a value nor a span.
        """
        if self.value is not None and not isinstance(self.value, Payload):
            object.__setattr__(self, "value", Payload.from_value(self.value))
        object.__setattr__(self, "unit", normalize_unit(self.unit))
        object.__setattr__(self, "metadata", validate_json_mapping(self.metadata))
        object.__setattr__(
            self,
            "occurrence_metadata",
            validate_json_mapping(self.occurrence_metadata),
        )
        if self.span is not None and not isinstance(self.span, TimeSpan):
            raise TimeFValidationError(
                f"annotation {self.key!r} span must be a TimePoint or a TimeInterval, got {type(self.span).__name__}"
            )
        if self.span is not None and self.span.time_series_ids is not None:
            raise TimeFValidationError(
                "annotation scope comes from its owner; attach it to each selected Signal "
                "instead of setting time_series_ids"
            )
        if self.value is None and self.span is None:
            raise TimeFValidationError(
                f"annotation {self.key!r} has neither a value nor a span, so it says nothing. Give it "
                f"a value, or a span to mark a region of the timeline"
            )

    def read_value(self) -> Any:
        """Resolve the native value, or return None for a pure marker.

        Returns:
            The annotation value.
        """
        return None if self.value is None else self.value.read()

    @property
    def content_id(self) -> str:
        """Return the identity of the reusable annotation content."""
        return self.id

    @property
    def _content_fields(self) -> tuple[object, ...]:
        """Return the reusable payload, without occurrence-specific fields."""
        return (self.key, self.value, self.unit, self.description, self.metadata)

    @property
    def name(self) -> str:
        """Return the annotation's public name."""
        return self.key

    def _new_occurrence(self) -> "Annotation":
        """Return an internal copy representing one new attachment occurrence."""
        return replace(self, occurrence_id=new_id())


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


class SupportsAnnotate(Protocol):
    """Anything an annotation occurrence can be attached to.

    ``annotate`` binds a fresh occurrence of ``annotation`` to this object and returns it; the
    returned occurrence is what a Task names in ``input_annotations`` or ``target_annotations``.
    ``annotations`` lists every occurrence attached so far. Records, Sources, Signals, Tasks, and
    the dataset itself implement it. A :class:`~timenet.dataset.SignalSelection` does not: it fans
    one annotation out over several Signals and returns every occurrence it created. The contract is
    checked statically; there is no runtime ``isinstance`` support because ``annotations`` is a
    property.
    """

    @property
    def annotations(self) -> tuple[Annotation, ...]:
        """Return the occurrences attached to this object."""
        ...

    def annotate(self, annotation: Annotation) -> Annotation:
        """Attach one occurrence of ``annotation`` and return it."""
        ...


class AnnotationDescriptor(TimeFModel):
    """Type-level projection of an annotation key, stored in the schema and manifest."""

    key: StrictStr
    """Name of the annotation this descriptor projects."""
    annotation_type: AnnotationType
    """Which of the three annotation shapes this key uses."""
    value_type: ValueType | None = None
    """Manifest value-type tag (bool, int, float, str, list, or map)."""
    unit: Annotated[StrictStr | None, BeforeValidator(normalize_unit)] = None
    """Optional physical unit of the value."""
    description: StrictStr | None = None
    """Optional human-readable description of the annotation."""


def value_type_of(value: Any) -> ValueType | None:
    """Return the manifest ``value_type`` tag for an annotation value.

    Args:
        value: The annotation's value.

    Returns:
        One of ``"bool" | "int" | "float" | "str" | "list"``, or ``None`` for a pure marker. A list
        holds strings only, such as the vocabulary of a classification target.

    Raises:
        TimeFValidationError: If ``value`` is a non-null value of an unsupported type.
    """
    if value is None:
        return None
    if isinstance(value, Payload):
        if value.descriptor.kind == "array":
            raise TimeFValidationError("annotation values cannot contain array payloads")
        return value.descriptor.kind
    # bool before int, since bool is an int subclass.
    tags: tuple[tuple[type | object, str], ...] = (
        (bool, "bool"),
        (int, "int"),
        (float, "float"),
        (str, "str"),
    )
    for value_type, tag in tags:
        if isinstance(value, value_type):
            return tag
    if isinstance(value, list | tuple):
        if all(isinstance(item, str) for item in value):
            return "list"
        raise TimeFValidationError("annotation list values must hold strings only")
    raise TimeFValidationError(f"unsupported annotation value type: {type(value).__name__}")
