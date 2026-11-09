"""Contextual metadata attached to a record.

One :class:`Annotation` class covers every case. Its optional ``span`` says how the annotation sits
in time. An absent span marks record-scoped context with no place in time, such as demographics or a
ticker symbol. A :class:`~timenet.types.spans.TimePoint` marks one time offset. A
:class:`~timenet.types.spans.TimeInterval` marks a bounded region of the original recording timeline.
:class:`Annotation` is a flat, frozen, final dataclass. It carries ``key``, ``unit``, and
``description`` as instance fields. A connector authors it directly, and reuses content through a
constant, or a builder function for content that varies. Because no subclass exists,
:class:`~timenet.reader.TimeFReader` rebuilds instances equal to the ones written, without runtime
class synthesis. :class:`AnnotationDescriptor` is the type-level projection stored in the schema and
manifest.
"""

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum, unique
import json
from typing import Annotated, Any, NoReturn, Protocol, cast, final

import pint
from pydantic import BeforeValidator, StrictStr

from timenet.errors import TimeFValidationError
from timenet.json import JsonMapping, validate_json_mapping
from timenet.types._fields import ValueField
from timenet.types._model import TimeFModel
from timenet.types.spans import TimeInterval, TimePoint, TimeSpan
from timenet.types.text import InlineText, InlineTextList, TextHandle, TextListHandle
from timenet.types.units import normalize_unit
from timenet.types.wire import ValueType


@unique
class AnnotationType(StrEnum):
    """Name the shape an annotation key takes across the dataset. It is a schema-level projection in the manifest."""

    STATIC = "static"
    POINT = "point"
    INTERVAL = "interval"


AnnotationValue = TextHandle | TextListHandle | bool | int | float | None

AnnotationValueInput = str | list[str] | tuple[str, ...] | TextHandle | TextListHandle | bool | int | float
"""A value that a mapping passed to ``add_annotations`` can hold. A sequence of strings must be a
``list`` or a ``tuple``."""


def _refuse_change(*_args: object, **_kwargs: object) -> NoReturn:
    """Refuse a change to frozen annotation metadata.

    Raises:
        TypeError: Always.
    """
    raise TypeError("annotation metadata cannot change; build a new Annotation instead")


class _FrozenJsonObject(dict):
    """A JSON object that cannot change. It compares equal to a ``dict``, and ``json.dumps`` writes it as one."""

    __slots__ = ()

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _refuse_change

    def __reduce__(self) -> tuple[type, tuple[dict]]:
        # Unpickling a dict subclass would otherwise set each item, which this class refuses.
        return (_FrozenJsonObject, (dict(self),))


class _FrozenJsonArray(list):
    """A JSON array that cannot change. It compares equal to a ``list``, and ``json.dumps`` writes it as one."""

    __slots__ = ()

    __setitem__ = __delitem__ = __iadd__ = __imul__ = _refuse_change
    append = clear = extend = insert = pop = remove = reverse = sort = _refuse_change

    def __reduce__(self) -> tuple[type, tuple[list]]:
        # Unpickling a list subclass would otherwise append each item, which this class refuses.
        return (_FrozenJsonArray, (list(self),))


def _frozen(value: object) -> object:
    """Return a copy of a validated JSON value in which no object or array can change."""
    if isinstance(value, dict):
        return _FrozenJsonObject({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenJsonArray(_frozen(item) for item in value)
    return value


def _frozen_metadata(value: object) -> JsonMapping:
    """Validate a metadata mapping and return a copy of it that cannot change.

    Returns:
        The frozen copy.

    Raises:
        TimeFValidationError: If the value is not a JSON object with text keys.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by validate_json_mapping
    return cast("JsonMapping", _frozen(validate_json_mapping(value)))


def _annotation_value(value: Any) -> AnnotationValue:
    """Normalize text values to handles and reject unsupported annotation values.

    Returns:
        A text handle, a native scalar, or an absent marker value.
    """
    value_type_of(value)
    if isinstance(value, str):
        return InlineText(value)
    if isinstance(value, list | tuple):
        return InlineTextList(list(value))
    return value


@final
@dataclass(frozen=True, kw_only=True)
class Annotation:
    """Contextual metadata attached to a record, optionally anchored to a region of its timeline.

    An annotation carries a ``value``, a ``span``, or both. With a span, it says where on the
    Record timeline it applies. Its owner determines which Signals it covers. Without a span, it is record-scoped
    context that has no place in time, like a subject's age. The span's own shape says whether the
    annotation marks a time offset or covers a stretch. One class covers every shape::

        Annotation(key="age", value=64)
        Annotation(key="stimulus", span=TimePoint.seconds(0.5))
        Annotation(key="artifact", value="motion", span=TimeInterval.seconds(0.0, 0.25))

    The class is final. Reuse content as a constant, or build content that varies with a function::

        HEALTHY_ANNOTATION = Annotation(key="condition", value="healthy")


        def quality_annotation(level: Literal["low", "medium", "high"]) -> Annotation:
            return Annotation(key="quality", value=level)
    """

    key: str
    """Name identifying the annotation."""
    value: ValueField[Any, AnnotationValue] = field(default=ValueField(_annotation_value, None))
    """A native scalar or a text handle. Call ``read()`` for string and string-list values.

    ``None`` makes it a pure marker, which needs a ``span`` to mark something.
    """
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
    metadata: JsonMapping = field(default_factory=dict)
    """Optional metadata that is part of the reusable content. The annotation keeps a copy that
    cannot change: it compares equal to a ``dict``, and changing it raises ``TypeError``."""
    confidence: float | None = field(default=None, compare=False)
    """Optional confidence for this particular application."""
    occurrence_metadata: JsonMapping = field(default_factory=dict, compare=False)
    """Optional metadata for this particular application. It cannot change, as ``metadata``."""

    # cls is typed as a plain type: with the class final, a checker would otherwise report a
    # subclass twice, once as a final-class error and once as a bad argument to this method.
    def __init_subclass__(cls: type) -> None:
        """Refuse every subclass. Field defaults on a subclass would bypass value normalization.

        Raises:
            TypeError: Always, naming the refused subclass.
        """
        raise TypeError(
            f"Annotation can't be subclassed, so {cls.__name__} is refused. Reuse a constant "
            "Annotation, or write a builder function that returns one"
        )

    def __post_init__(self) -> None:
        """Normalize units and validate the annotation's span and content.

        The value field wraps strings and string sequences in text handles. Call ``value.read()``
        to obtain the native string or list. Numeric and boolean values remain native scalars.

        Raises:
            TimeFValidationError: If the caller sets ``span`` to something that is not a span, or
                the annotation has neither a value nor a span.
        """
        object.__setattr__(self, "unit", normalize_unit(self.unit))
        # Freeze both metadata mappings, so that the content of an annotation cannot change once
        # it is built, also through a dict or a list that the caller still holds.
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))
        object.__setattr__(self, "occurrence_metadata", _frozen_metadata(self.occurrence_metadata))
        if self.span is not None and not isinstance(self.span, TimeSpan):
            raise TimeFValidationError(
                f"annotation {self.key!r} span must be a TimePoint or a TimeInterval, got {type(self.span).__name__}"
            )
        if self.span is not None and self.span.time_series is not None:
            raise TimeFValidationError(
                "annotation scope comes from its owner; attach it to each selected Signal "
                "instead of setting time_series"
            )
        if self.value is None and self.span is None:
            raise TimeFValidationError(
                f"annotation {self.key!r} has neither a value nor a span, so it says nothing. Give it "
                f"a value, or a span to mark a region of the timeline"
            )

    @property
    def _content_key(self) -> tuple[Hashable, ...]:
        """The reusable payload, without occurrence-specific fields, as a hashable key.

        The writer stores annotations with equal keys as one content row. The value type is part of
        the key, so ``True``, ``1`` and ``1.0`` differ. A float compares by its bits, so ``0.0`` and
        ``-0.0`` differ, and every NaN is equal. A text value gives its handle's content key, which
        needs no read, so equal keys still mean equal text.
        """
        value = self.value
        content: Hashable
        if isinstance(value, TextHandle | TextListHandle):
            content = value._content_key()
        elif isinstance(value, float):
            content = float.hex(value)
        else:
            content = value
        metadata = json.dumps(self.metadata, sort_keys=True)
        return (self.key, value_type_of(value), content, self.unit, self.description, metadata)

    @property
    def name(self) -> str:
        """The annotation's public name."""
        return self.key

    def _new_occurrence(self) -> "Annotation":
        """Return an internal copy representing one new attachment occurrence."""
        return replace(self)


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
        """The occurrences attached to this object."""
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
        One of ``"bool" | "int" | "float" | "str" | "list"``, or ``None`` for a marker with no value.
        A list holds only strings, such as the names of activities in a recording.

    Raises:
        TimeFValidationError: If ``value`` is a non-null value of an unsupported type.
    """
    if isinstance(value, TextHandle):
        return "str"
    if isinstance(value, TextListHandle):
        return "list"
    if value is None:
        return None
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


def _annotation_item(key: object, value: Any) -> Annotation:
    """Return an annotation from a mapping entry with contextual validation.

    Raises:
        TimeFValidationError: If the key or value is invalid for a mapping entry.
    """
    if not isinstance(key, str):
        raise TimeFValidationError(f"an annotation key must be text, got {key!r}")
    if isinstance(value, Annotation):
        raise TimeFValidationError(
            f"annotation {key!r}: mapping values must be annotation values, not Annotation objects; "
            f"pass mapping.values() or a list of annotations instead"
        )
    if value is None:
        raise TimeFValidationError(f"annotation {key!r}: a mapping value cannot be None")
    try:
        return Annotation(key=key, value=value)
    except TimeFValidationError as error:
        raise TimeFValidationError(f"annotation {key!r}: {error}") from error


def annotation_batch(
    annotations: Iterable["Annotation"] | Mapping[str, AnnotationValueInput],
) -> tuple["Annotation", ...]:
    """Convert annotation objects or a mapping of keys to values into a checked tuple of annotations.

    A mapping gives one plain annotation per item, ``Annotation(key=key, value=value)``, in the
    mapping's order. Build an :class:`Annotation` yourself for a unit, a description, or metadata.

    Args:
        annotations: The annotations, or a mapping from each key to its value.

    Returns:
        The annotations, in the order given.

    Raises:
        TimeFValidationError: If a mapping key is not text, a mapping value is ``None``, an
            :class:`Annotation`, or not valid for an annotation, or an item of the iterable is not an
            :class:`Annotation`.
    """
    if isinstance(annotations, Mapping):
        return tuple(_annotation_item(key, value) for key, value in annotations.items())
    if isinstance(annotations, Annotation):
        raise TimeFValidationError("pass a sequence of Annotation objects; attach a single one with annotate()")
    if annotations is None or isinstance(annotations, str | bytes):
        raise TimeFValidationError(
            f"pass a sequence of Annotation objects or a mapping, not {type(annotations).__name__}"
        )
    batch = list(annotations)
    for item in batch:
        if not isinstance(item, Annotation):
            raise TimeFValidationError(f"each batch item must be an Annotation; got {type(item).__name__}")
    return tuple(batch)
